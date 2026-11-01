"""
Runs every (response, judge) pair against OpenRouter with a fixed number
of requests in flight, retrying what's worth retrying, recording every
outcome as it lands, and resuming cleanly when re-run.

Each sampled response goes to two judges: the textual judge (no image;
scores interpretation, consistency and, where the task has a reference
description, knowledge) and the visual judge (gets the image; scores
visual-evidence). The judge kind plays the part
that the model plays in the evaluation runner, so progress, failure
thresholds and resume all work per judge.

How failures are handled (same as weedvlm.evaluation.runner):
    - transient errors (rate limits, 5xx, timeouts) are retried with
      exponential backoff; a 429 also pauses every worker
    - a reply that isn't JSON with every score an integer from 0 to 4 is
      retried on a separate budget
    - a job that still fails is recorded as status "failed" with the
      reason, and the run carries on
    - a judge that fails `model_failure_threshold` jobs in a row is
      dropped for the rest of the run (its jobs stay pending)
    - a bad API key or exhausted credits stops the whole run
    - Ctrl-C stops taking new work and lets in-flight requests finish;
      a second Ctrl-C abandons them
    Anything not recorded as "ok" is asked again on the next run.

Results go to <output_dir>/<run_name>/judgements.jsonl, one JSON object
per job, keyed by (response_id, kind); the last line for a key wins.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import random
import signal
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from weedvlm.evaluation.openrouter import ChatReply, OpenRouterClient, RequestFailure
from weedvlm.evaluation.parsing import check_fields, extract_json_object
from weedvlm.evaluation.progress import ModelProgress, Progress
from weedvlm.evaluation.runner import FatalRunError, load_api_key
from weedvlm.evaluation.store import model_fingerprint
from weedvlm.reasoning.config import ModelConfig, ReasoningConfig
from weedvlm.reasoning.prompt import SCORE_LEVELS, JudgeBuilder, JudgeKind, ReasoningPrompt
from weedvlm.reasoning.runner import JudgeItem

log = logging.getLogger("weedvlm.reasoning")

JUDGEMENTS_JSONL = "judgements.jsonl"
RUN_JSON = "run.json"
PROGRESS_JSON = "progress.json"
RUN_LOG = "run.log"

JUDGE_KINDS: tuple[JudgeKind, ...] = ("textual", "visual")
PROGRESS_INTERVAL_SECONDS = 5.0
SCORE_RANGE = (0, SCORE_LEVELS - 1)


@dataclass
class JudgeJob:
    item: JudgeItem
    kind: JudgeKind
    model: ModelConfig


@dataclass
class JudgeRunResult:
    location: str
    state: str  # finished / stopped / interrupted / aborted
    per_kind: dict[str, ModelProgress]
    top_errors: list[tuple[str, int]] = field(default_factory=list)
    fatal_message: str | None = None

    @property
    def failed(self) -> int:
        return sum(m.failed for m in self.per_kind.values())

    @property
    def pending(self) -> int:
        return sum(m.total - m.answered - m.failed for m in self.per_kind.values())

    @property
    def ok(self) -> bool:
        return self.state == "finished" and self.failed == 0


# -- the store -----------------------------------------------------------


class JudgementStore:
    """The run directory: judgements appended as they land (one flushed
    line each, so a killed run loses nothing), plus run.json and
    progress.json."""

    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        self._file = self.path(JUDGEMENTS_JSONL).open("a", encoding="utf-8")

    def path(self, name: str) -> Path:
        return self.run_dir / name

    @property
    def location(self) -> str:
        return str(self.path(JUDGEMENTS_JSONL))

    def load(self) -> dict[tuple[int, str], dict[str, Any]]:
        records: dict[tuple[int, str], dict[str, Any]] = {}
        path = self.path(JUDGEMENTS_JSONL)
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                log.warning("%s line %d isn't valid JSON, ignoring it", path, number)
                continue
            records[(record["response_id"], record["kind"])] = record
        return records

    def append(self, record: dict[str, Any]) -> None:
        self._file.write(json.dumps(record) + "\n")
        self._file.flush()

    def read_run_info(self) -> dict[str, Any] | None:
        path = self.path(RUN_JSON)
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def write_run_info(self, data: dict[str, Any]) -> None:
        self.path(RUN_JSON).write_text(json.dumps(data, indent=2), encoding="utf-8")

    def write_progress(self, snapshot: dict[str, Any]) -> None:
        tmp = self.path(PROGRESS_JSON + ".tmp")
        tmp.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
        tmp.replace(self.path(PROGRESS_JSON))

    def close(self) -> None:
        self._file.close()


# -- setup -----------------------------------------------------------------


def judge_models(config: ReasoningConfig) -> dict[JudgeKind, ModelConfig]:
    return {"textual": config.textual_model, "visual": config.visual_model}


def _system_prompts(builder: JudgeBuilder, items: list[JudgeItem]) -> dict[JudgeKind, str]:
    sample = items[0]
    return {kind: builder.build(sample, kind).system for kind in JUDGE_KINDS}


def _fingerprints(
    config: ReasoningConfig, builder: JudgeBuilder, items: list[JudgeItem]
) -> dict[str, str]:
    # max_tokens is deliberately left out, as in the evaluation runner:
    # raising it to re-ask cut-off replies is what resuming is for.
    systems = _system_prompts(builder, items)
    return {
        kind: model_fingerprint(
            system_prompt_sha256=hashlib.sha256(systems[kind].encode()).hexdigest(),
            judge_model=model.id,
            temperature=config.temperature_for(model),
            extra_body=model.extra_body,
        )
        for kind, model in judge_models(config).items()
    }


def _check_resumable(store: JudgementStore, fingerprints: dict[str, str]) -> None:
    previous = (store.read_run_info() or {}).get("fingerprints", {})
    changed = [kind for kind, fp in fingerprints.items() if previous.get(kind, fp) != fp]
    if changed:
        raise FatalRunError(
            f"{store.location} already holds judgements made with a different judge model, "
            f"request settings or system prompt for: {', '.join(changed)}. Mixing them would "
            "make the scores incomparable -- use a new run_name, or delete that run."
        )


def _setup_logging(path: Path) -> None:
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.handlers[:] = [handler]
    log.setLevel(logging.INFO)
    log.propagate = False


def plan_jobs(
    config: ReasoningConfig, items: list[JudgeItem], store: JudgementStore
) -> tuple[list[JudgeJob], dict[str, ModelProgress]]:
    """Every (response, judge) not already judged ok, response-major so
    that a run cut short has covered both judges evenly. Also seeds the
    per-judge progress with what earlier sessions finished."""
    existing = store.load()
    models = judge_models(config)
    progress = {kind: ModelProgress(total=len(items)) for kind in JUDGE_KINDS}
    jobs: list[JudgeJob] = []
    for item in items:
        for kind in JUDGE_KINDS:
            record = existing.get((item.response_id, kind))
            if record is not None and record["status"] == "ok":
                progress[kind].already_done += 1
                continue
            jobs.append(JudgeJob(item, kind, models[kind]))
    return jobs, progress


# -- the run -----------------------------------------------------------------


class JudgeRunner:
    def __init__(
        self,
        config: ReasoningConfig,
        *,
        client: OpenRouterClient,
        builder: JudgeBuilder,
        store: JudgementStore,
        progress: Progress,
        stop: asyncio.Event,
    ):
        self.config = config
        self.client = client
        self.builder = builder
        self.store = store
        self.progress = progress
        self.stop = stop
        self.fatal_message: str | None = None
        self.error_counts: Counter[str] = Counter()
        self._consecutive_failures: Counter[str] = Counter()
        self._tripped: set[str] = set()

    async def run_jobs(self, jobs: list[JudgeJob]) -> None:
        queue: asyncio.Queue[JudgeJob] = asyncio.Queue()
        for job in jobs:
            queue.put_nowait(job)
        workers = [
            asyncio.create_task(self._worker(queue))
            for _ in range(min(self.config.concurrency, len(jobs)))
        ]
        await asyncio.gather(*workers)

    async def _worker(self, queue: asyncio.Queue[JudgeJob]) -> None:
        while not self.stop.is_set():
            try:
                job = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            if job.kind in self._tripped:
                continue
            try:
                await self._process(job)
            except FatalRunError as error:
                self.fatal_message = self.fatal_message or str(error)
                log.error("fatal: %s", error)
                self.stop.set()

    async def _sleep(self, seconds: float) -> bool:
        """Sleeps unless the run is stopping. False if it was cut short."""
        try:
            await asyncio.wait_for(self.stop.wait(), timeout=seconds)
        except TimeoutError:
            return True
        return False

    def _backoff(self, retry_number: int, retry_after: float | None) -> float:
        retry = self.config.retry
        delay = min(retry.backoff_max_seconds, retry.backoff_base_seconds * 2 ** (retry_number - 1))
        delay *= 0.5 + random.random() / 2  # jitter, so workers don't retry in lockstep
        return max(delay, retry_after or 0.0)

    async def _process(self, job: JudgeJob) -> None:
        item, kind = job.item, job.kind
        self.progress.started(kind)
        try:
            record = await self._judge(job)
        except BaseException:
            self.progress.abandoned(kind)
            raise
        if record is None:  # stopping mid-job; stays pending
            self.progress.abandoned(kind)
            return

        self.store.append(record)
        ok = record["status"] == "ok"
        self.progress.finished(
            kind,
            ok=ok,
            retries=record["attempts"] - 1,
            cost_usd=record.get("cost_usd"),
            tokens=(record.get("usage") or {}).get("total_tokens"),
        )
        if ok:
            self._consecutive_failures[kind] = 0
            return

        message = record["error"]["message"]
        self.error_counts[f"{kind}: {message[:160]}"] += 1
        log.warning(
            "FAILED %s judge of response %s after %d attempt(s): [%s] %s",
            kind, item.response_id, record["attempts"], record["error"]["kind"], message,
        )
        self._consecutive_failures[kind] += 1
        threshold = self.config.retry.model_failure_threshold
        if threshold and self._consecutive_failures[kind] >= threshold:
            self._tripped.add(kind)
            self.progress.models[kind].stopped_early = True
            log.error(
                "%s judge failed %d jobs in a row (last: %s); skipping its remaining jobs",
                kind, threshold, message,
            )

    def _prepare(self, job: JudgeJob) -> tuple[ReasoningPrompt, list[dict[str, Any]]]:
        """The prompt and chat messages for a job. Blocking (reads the
        database, renders and encodes the image), so run in a thread."""
        prompt = self.builder.build(job.item, job.kind)
        content: list[dict[str, Any]] = []
        if prompt.attach_image:
            data, media_type = self.builder.image_for(job.item)
            url = f"data:{media_type};base64,{base64.b64encode(data).decode()}"
            content.append({"type": "image_url", "image_url": {"url": url}})
        content.append({"type": "text", "text": prompt.user_text})
        messages = [
            {"role": "system", "content": prompt.system},
            {"role": "user", "content": content},
        ]
        return prompt, messages

    async def _judge(self, job: JudgeJob) -> dict[str, Any] | None:
        item, kind, model = job.item, job.kind, job.model
        retry = self.config.retry
        prompt, messages = await asyncio.to_thread(self._prepare, job)

        base = {
            "response_id": item.response_id,
            "kind": kind,
            "judge_model": model.id,
            "run": item.run,
            "model": item.model,
            "question_id": item.question_id,
            "task": item.task,
            "question_type": item.question_type,
            "image_id": item.image_id,
            "match": item.match,
        }

        attempts = 0
        transient_retries = 0
        parse_retries = 0
        retry_log: list[str] = []
        reply: ChatReply | None = None

        while True:
            attempts += 1
            reply = None
            try:
                reply = await self.client.chat(
                    model=model.id,
                    messages=messages,
                    temperature=self.config.temperature_for(model),
                    max_tokens=self.config.max_tokens_for(model),
                    extra_body=model.extra_body,
                    timeout=self.config.defaults.timeout_seconds,
                )
            except RequestFailure as failure:
                error = {
                    "kind": failure.kind,
                    "message": failure.message,
                    "http_status": failure.http_status,
                }
                if failure.kind == "fatal":
                    raise FatalRunError(
                        f"{model.id}: HTTP {failure.http_status}: {failure.message}"
                    ) from failure
                if failure.kind == "transient" and transient_retries < retry.max_retries:
                    transient_retries += 1
                    delay = self._backoff(transient_retries, failure.retry_after)
                    retry_log.append(f"{failure.message[:120]} (retry in {delay:.1f}s)")
                    log.info("retry %s judge of %s: %s", kind, item.response_id, retry_log[-1])
                    if not await self._sleep(delay):
                        return None
                    continue
                return self._failed(base, error, attempts, retry_log, reply)

            result, problems = self._interpret(reply, prompt)
            if result is not None:
                return {
                    **base,
                    "status": "ok",
                    **result,
                    "raw_response": reply.content,
                    "attempts": attempts,
                    "retry_log": retry_log,
                    **self._reply_fields(reply),
                }

            error = {"kind": "parse", "message": "; ".join(problems), "http_status": None}
            truncated = reply.finish_reason == "length"  # retrying can't fix a token budget
            if not truncated and parse_retries < retry.max_parse_retries:
                parse_retries += 1
                retry_log.append(f"unusable reply: {error['message'][:120]}")
                log.info("retry %s judge of %s: %s", kind, item.response_id, retry_log[-1])
                if not await self._sleep(self._backoff(1, None)):
                    return None
                continue
            return self._failed(base, error, attempts, retry_log, reply)

    @staticmethod
    def _interpret(
        reply: ChatReply, prompt: ReasoningPrompt
    ) -> tuple[dict[str, Any] | None, list[str]]:
        """({"scores": ..., "notes": ...}, no problems) for a usable reply,
        else (None, the reasons it wasn't usable). Notes are optional:
        empty, "none" and "n/a" all become None."""
        parsed = extract_json_object(reply.content) if reply.content.strip() else None
        if parsed is None:
            if reply.finish_reason == "length":
                return None, [
                    "reply was cut off at max_tokens before a complete JSON answer (raise "
                    "max_tokens for this judge; reasoning models spend it on thinking)"
                ]
            return None, ["reply was empty" if not reply.content.strip() else "no JSON object found in the reply"]
        problems = check_fields(parsed, prompt.required_fields)
        scores: dict[str, int] = {}
        low, high = SCORE_RANGE
        for name in prompt.required_fields:
            value = parsed.get(name)
            if isinstance(value, float) and value.is_integer():
                value = int(value)
            if isinstance(value, str) and value.strip().isdigit():
                value = int(value)
            if isinstance(value, bool) or not isinstance(value, int):
                if value is not None:
                    problems.append(f"{name!r} must be an integer, got {value!r}")
            elif not low <= value <= high:
                problems.append(f"{name!r} must be {low}-{high}, got {value}")
            else:
                scores[name] = value
        if problems:
            return None, problems
        notes = parsed.get("notes")
        notes = notes.strip() if isinstance(notes, str) else None
        if not notes or notes.lower().rstrip(".") in ("none", "n/a", "na"):
            notes = None
        return {"scores": scores, "notes": notes}, []

    @staticmethod
    def _reply_fields(reply: ChatReply) -> dict[str, Any]:
        return {
            "finish_reason": reply.finish_reason,
            "latency_seconds": round(reply.latency_seconds, 3),
            "usage": reply.usage,
            "cost_usd": reply.cost_usd,
            "generation_id": reply.generation_id,
            "provider": reply.provider,
            "judged_at": datetime.now(UTC).isoformat(),
        }

    def _failed(
        self,
        base: dict[str, Any],
        error: dict[str, Any],
        attempts: int,
        retry_log: list[str],
        reply: ChatReply | None,
    ) -> dict[str, Any]:
        return {
            **base,
            "status": "failed",
            "raw_response": reply.content if reply else None,
            "error": error,
            "attempts": attempts,
            "retry_log": retry_log,
            **(self._reply_fields(reply) if reply else {}),
        }


async def run_judges(
    config: ReasoningConfig, items: list[JudgeItem], builder: JudgeBuilder
) -> JudgeRunResult:
    if not items:
        raise ValueError("no items to judge")
    builder.check_descriptions(items)
    api_key = load_api_key(config.api_key_env)

    store = JudgementStore(config.run_dir())
    _setup_logging(store.path(RUN_LOG))
    fingerprints = _fingerprints(config, builder, items)
    _check_resumable(store, fingerprints)
    jobs, per_kind = plan_jobs(config, items, store)

    previous = store.read_run_info() or {}
    store.write_run_info(
        {
            "run_name": config.run_name,
            "config": config.model_dump(mode="json"),
            "fingerprints": {**previous.get("fingerprints", {}), **fingerprints},
            "sessions": [*previous.get("sessions", []), datetime.now(UTC).isoformat()],
        }
    )

    already = sum(m.already_done for m in per_kind.values())
    print(
        f"Run '{config.run_name}': {len(items)} responses x {len(JUDGE_KINDS)} judges = "
        f"{len(items) * len(JUDGE_KINDS)} requests, {already} already judged, {len(jobs)} to go "
        f"(concurrency {config.concurrency}). Results: {store.location}",
        flush=True,
    )
    if not jobs:
        store.close()
        print("Nothing to do.")
        return JudgeRunResult(store.location, "finished", per_kind)

    stop = asyncio.Event()
    progress = Progress(per_kind, write_snapshot=store.write_progress)
    client = OpenRouterClient(
        base_url=config.base_url, api_key=api_key, max_connections=config.concurrency
    )
    runner = JudgeRunner(
        config, client=client, builder=builder, store=store, progress=progress, stop=stop
    )

    main_task = asyncio.current_task()
    interrupted = False

    def on_sigint() -> None:
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            stop.set()
            log.warning("interrupt: finishing in-flight requests (Ctrl-C again to abandon them)")
            progress.detach_display()
            print(
                "\nStopping after in-flight requests finish... (Ctrl-C again to abandon them)",
                flush=True,
            )
        elif main_task is not None:
            main_task.cancel()

    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGINT, on_sigint)
    reporter = asyncio.create_task(progress.run(PROGRESS_INTERVAL_SECONDS))
    state = "finished"
    try:
        await runner.run_jobs(jobs)
    except asyncio.CancelledError:
        state = "interrupted"
    finally:
        reporter.cancel()
        loop.remove_signal_handler(signal.SIGINT)
        await client.aclose()
        if runner.fatal_message:
            state = "aborted"
        elif interrupted:
            state = "interrupted"
        elif any(m.stopped_early for m in per_kind.values()):
            state = "stopped"
        progress.finish(state)
        store.close()
        log.info("run %s", state)

    return JudgeRunResult(
        store.location,
        state,
        per_kind,
        top_errors=runner.error_counts.most_common(5),
        fatal_message=runner.fatal_message,
    )
