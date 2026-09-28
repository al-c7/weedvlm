"""
Runs every (question, model) pair against OpenRouter with a fixed number
of requests in flight, retrying what's worth retrying, recording every
outcome as it lands, and resuming cleanly when re-run.

How failures are handled:
    - transient errors (rate limits, 5xx, timeouts) are retried with
      exponential backoff; a 429 also pauses every worker
    - a reply with no usable JSON answer is retried on a separate budget
    - a question that still fails is recorded as status "failed" with the
      reason, and the run carries on
    - a model that fails `model_failure_threshold` questions in a row is
      dropped for the rest of the run (its questions stay pending)
    - a bad API key or exhausted credits stops the whole run
    - Ctrl-C stops taking new work and lets in-flight requests finish;
      a second Ctrl-C abandons them
    Anything not recorded as "ok" is asked again on the next run.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import mimetypes
import os
import random
import signal
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from weedvlm.evaluation.config import EvaluationConfig, ModelConfig
from weedvlm.evaluation.openrouter import ChatReply, OpenRouterClient, RequestFailure
from weedvlm.evaluation.parsing import check_fields, extract_json_object
from weedvlm.evaluation.progress import ModelProgress, Progress
from weedvlm.evaluation.prompts import (
    PROMPT_VERSION,
    Prompt,
    build_prompt,
    density_rules_text,
    load_density_config,
)
from weedvlm.evaluation.questions import load_questions
from weedvlm.evaluation.store import (
    RUN_LOG,
    ResponseError,
    ResponseRecord,
    RunStore,
    model_fingerprint,
)
from weedvlm.types.questions import QuestionBase

REPO_ROOT = Path(__file__).resolve().parents[3]

log = logging.getLogger("weedvlm.evaluation")


class FatalRunError(Exception):
    """Something only the user can fix (bad key, no credits, or a resumed
    run whose settings changed)."""


@dataclass
class Job:
    task: str
    question: QuestionBase
    model: ModelConfig


@dataclass
class RunResult:
    run_dir: Path
    state: str  # finished / stopped / interrupted / aborted
    per_model: dict[str, ModelProgress]
    top_errors: list[tuple[str, int]] = field(default_factory=list)
    fatal_message: str | None = None

    @property
    def failed(self) -> int:
        return sum(m.failed for m in self.per_model.values())

    @property
    def pending(self) -> int:
        return sum(m.total - m.answered - m.failed for m in self.per_model.values())

    @property
    def ok(self) -> bool:
        return self.state == "finished" and self.failed == 0


# -- setup ------------------------------------------------------------


def load_api_key(env_name: str) -> str:
    """From the environment, else from a .env file in the working
    directory or repo root."""
    if key := os.environ.get(env_name):
        return key
    for candidate in (Path.cwd() / ".env", REPO_ROOT / ".env"):
        if not candidate.is_file():
            continue
        for line in candidate.read_text(encoding="utf-8").splitlines():
            line = line.strip().removeprefix("export ")
            name, sep, value = line.partition("=")
            if sep and name.strip() == env_name and (value := value.strip().strip("'\"")):
                return value
    raise FatalRunError(
        f"No OpenRouter API key: set {env_name} in the environment or in a .env file "
        f"({REPO_ROOT / '.env'})."
    )


def _fingerprints(config: EvaluationConfig) -> dict[str, str]:
    # max_tokens is deliberately left out: it only decides whether a long
    # reply is cut off, so raising it to re-ask truncated questions is
    # exactly what resuming is for.
    return {
        m.id: model_fingerprint(
            prompt_version=PROMPT_VERSION,
            temperature=config.temperature_for(m),
            extra_body=m.extra_body,
        )
        for m in config.models
    }


def _check_resumable(store: RunStore, fingerprints: dict[str, str]) -> None:
    previous = (store.read_run_json() or {}).get("model_fingerprints", {})
    changed = [model for model, fp in fingerprints.items() if previous.get(model, fp) != fp]
    if changed:
        raise FatalRunError(
            f"{store.run_dir} already holds answers from different request settings (or prompt "
            f"version) for: {', '.join(changed)}. Mixing them would make the results "
            "incomparable -- use a new run_name, or delete that run directory."
        )


def _image_unchanged_since(path: Path, iso_timestamp: str) -> bool:
    """A stored answer only stands if its image hasn't been re-rendered
    since (question ids are reused when the benchmark is regenerated, but
    the picture, and e.g. localisation's box numbering, may differ)."""
    try:
        modified = datetime.fromtimestamp(path.stat().st_mtime, UTC)
    except OSError:
        return False
    return modified <= datetime.fromisoformat(iso_timestamp)


def _read_image(path: Path) -> tuple[str, str]:
    """(data URL, sha256) for an image file."""
    data = path.read_bytes()
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(data).decode()}", hashlib.sha256(data).hexdigest()


def _setup_logging(path: Path) -> None:
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.handlers[:] = [handler]
    log.setLevel(logging.INFO)
    log.propagate = False


# -- the run ----------------------------------------------------------


class Runner:
    def __init__(
        self,
        config: EvaluationConfig,
        *,
        client: OpenRouterClient,
        store: RunStore,
        progress: Progress,
        density_rules: str | None,
        stop: asyncio.Event,
    ):
        self.config = config
        self.client = client
        self.store = store
        self.progress = progress
        self.density_rules = density_rules
        self.stop = stop
        self.fatal_message: str | None = None
        self.error_counts: Counter[str] = Counter()
        self._consecutive_failures: Counter[str] = Counter()
        self._tripped: set[str] = set()

    async def run_jobs(self, jobs: list[Job]) -> None:
        queue: asyncio.Queue[Job] = asyncio.Queue()
        for job in jobs:
            queue.put_nowait(job)
        workers = [
            asyncio.create_task(self._worker(queue)) for _ in range(min(self.config.concurrency, len(jobs)))
        ]
        await asyncio.gather(*workers)

    async def _worker(self, queue: asyncio.Queue[Job]) -> None:
        while not self.stop.is_set():
            try:
                job = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            if job.model.id in self._tripped:
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

    async def _process(self, job: Job) -> None:
        question, model = job.question, job.model
        self.progress.started(model.id)
        try:
            record = await self._answer(job)
        except BaseException:
            self.progress.abandoned(model.id)
            raise
        if record is None:  # stopping mid-question; stays pending
            self.progress.abandoned(model.id)
            return

        self.store.append(record)
        ok = record.status == "ok"
        self.progress.finished(
            model.id,
            ok=ok,
            retries=record.attempts - 1,
            cost_usd=record.cost_usd,
            tokens=(record.usage or {}).get("total_tokens"),
        )
        if ok:
            self._consecutive_failures[model.id] = 0
            return

        assert record.error is not None
        self.error_counts[f"{model.id}: {record.error.message[:160]}"] += 1
        log.warning(
            "FAILED %s x %s after %d attempt(s): [%s] %s",
            model.id, question.id, record.attempts, record.error.kind, record.error.message,
        )
        self._consecutive_failures[model.id] += 1
        threshold = self.config.retry.model_failure_threshold
        if threshold and self._consecutive_failures[model.id] >= threshold:
            self._tripped.add(model.id)
            self.progress.models[model.id].stopped_early = True
            log.error(
                "%s failed %d questions in a row (last: %s); skipping its remaining questions",
                model.id, threshold, record.error.message,
            )

    async def _answer(self, job: Job) -> ResponseRecord | None:
        task, question, model = job.task, job.question, job.model
        retry = self.config.retry
        prompt = build_prompt(question, density_rules=self.density_rules)

        image_path = Path(question.image_path)
        data_url, image_sha256 = await asyncio.to_thread(_read_image, image_path)
        messages = self._messages(prompt, data_url)

        base = {
            "question_id": question.id,
            "model": model.id,
            "task": task,
            "question_type": question.question_type.value,
            "annotated": question.annotated,
            "benchmark_class": question.benchmark_class,
            "dataset_name": question.source.dataset_name,
            "source_image_id": question.source.image_id,
            "image_path": str(image_path),
            "image_sha256": image_sha256,
            "prompt_version": PROMPT_VERSION,
        }

        attempts = 0
        transient_retries = 0
        parse_retries = 0
        retry_log: list[str] = []
        error: ResponseError
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
                error = ResponseError(
                    kind=failure.kind, message=failure.message, http_status=failure.http_status
                )
                if failure.kind == "fatal":
                    raise FatalRunError(
                        f"{model.id}: HTTP {failure.http_status}: {failure.message}"
                    ) from failure
                if failure.kind == "transient" and transient_retries < retry.max_retries:
                    transient_retries += 1
                    delay = self._backoff(transient_retries, failure.retry_after)
                    retry_log.append(f"{failure.message[:120]} (retry in {delay:.1f}s)")
                    log.info("retry %s x %s: %s", model.id, question.id, retry_log[-1])
                    if not await self._sleep(delay):
                        return None
                    continue
                return self._failed(base, error, attempts, retry_log, reply)

            parsed, problems = self._interpret(reply, prompt)
            if parsed is not None:
                return ResponseRecord(
                    **base,
                    status="ok",
                    parsed=parsed,
                    raw_response=reply.content,
                    warnings=problems,
                    attempts=attempts,
                    retry_log=retry_log,
                    **self._reply_fields(reply),
                )

            error = ResponseError(kind="parse", message="; ".join(problems))
            truncated = reply.finish_reason == "length"  # retrying can't fix a token budget
            if not truncated and parse_retries < retry.max_parse_retries:
                parse_retries += 1
                retry_log.append(f"unusable reply: {error.message[:120]}")
                log.info("retry %s x %s: %s", model.id, question.id, retry_log[-1])
                if not await self._sleep(self._backoff(1, None)):
                    return None
                continue
            return self._failed(base, error, attempts, retry_log, reply)

    @staticmethod
    def _messages(prompt: Prompt, data_url: str) -> list[dict[str, Any]]:
        return [
            {"role": "system", "content": prompt.system},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": prompt.user_text},
                ],
            },
        ]

    @staticmethod
    def _interpret(reply: ChatReply, prompt: Prompt) -> tuple[dict[str, Any] | None, list[str]]:
        """(parsed answer, warnings) for a usable reply, else
        (None, reasons it wasn't usable)."""
        parsed = extract_json_object(reply.content) if reply.content.strip() else None
        if parsed is None:
            if reply.finish_reason == "length":
                return None, [
                    "reply was cut off at max_tokens before a complete JSON answer (raise "
                    "max_tokens for this model; reasoning models spend it on thinking)"
                ]
            return None, ["reply was empty" if not reply.content.strip() else "no JSON object found in the reply"]
        problems = check_fields(parsed, prompt.required_fields)
        if "answer" in parsed and isinstance(parsed["answer"], dict | list):
            return None, ["'answer' must be a single value, not an object or list"]
        if any("'answer'" in problem for problem in problems):
            return None, problems
        return parsed, problems

    @staticmethod
    def _reply_fields(reply: ChatReply) -> dict[str, Any]:
        return {
            "finish_reason": reply.finish_reason,
            "latency_seconds": round(reply.latency_seconds, 3),
            "usage": reply.usage,
            "cost_usd": reply.cost_usd,
            "generation_id": reply.generation_id,
            "provider": reply.provider,
        }

    def _failed(
        self,
        base: dict[str, Any],
        error: ResponseError,
        attempts: int,
        retry_log: list[str],
        reply: ChatReply | None,
    ) -> ResponseRecord:
        return ResponseRecord(
            **base,
            status="failed",
            raw_response=reply.content if reply else None,
            error=error,
            attempts=attempts,
            retry_log=retry_log,
            **(self._reply_fields(reply) if reply else {}),
        )


def plan_jobs(
    config: EvaluationConfig,
    questions: list[tuple[str, QuestionBase]],
    store: RunStore,
) -> tuple[list[Job], dict[str, ModelProgress]]:
    """Every (question, model) pair not already answered, question-major
    so that a run cut short has covered every model evenly. Also seeds
    per-model progress with what earlier sessions already finished."""
    existing = store.load_records()
    progress = {m.id: ModelProgress(total=len(questions)) for m in config.models}
    jobs: list[Job] = []
    stale = 0
    for task, question in questions:
        for model in config.models:
            record = existing.get((model.id, question.id))
            if record is not None and record.status == "ok":
                if _image_unchanged_since(Path(question.image_path), record.created_at):
                    progress[model.id].already_done += 1
                    continue
                stale += 1
            jobs.append(Job(task, question, model))
    if stale:
        print(
            f"{stale} earlier answer(s) are for images that were re-rendered since; asking again.",
        )
    return jobs, progress


async def run_evaluation(config: EvaluationConfig) -> RunResult:
    api_key = load_api_key(config.api_key_env)
    questions = load_questions(config)

    density_rules = None
    if config.tasks.density:
        density_rules = density_rules_text(load_density_config(config.resolved_generation_config()))

    run_dir = config.output_dir / config.run_name
    store = RunStore(run_dir)
    _setup_logging(store.path(RUN_LOG))

    fingerprints = _fingerprints(config)
    _check_resumable(store, fingerprints)
    jobs, per_model = plan_jobs(config, questions, store)

    previous = store.read_run_json() or {}
    store.write_run_json(
        {
            "run_name": config.run_name,
            "prompt_version": PROMPT_VERSION,
            "config": config.model_dump(mode="json"),
            "model_fingerprints": {**previous.get("model_fingerprints", {}), **fingerprints},
            "sessions": [*previous.get("sessions", []), datetime.now(UTC).isoformat()],
        }
    )

    total_pairs = len(questions) * len(config.models)
    already = sum(m.already_done for m in per_model.values())
    print(
        f"Run '{config.run_name}': {len(questions)} questions x {len(config.models)} model(s) = "
        f"{total_pairs} requests, {already} already answered, {len(jobs)} to go "
        f"(concurrency {config.concurrency}). Results: {run_dir}/",
        flush=True,
    )
    order = [(m.id, q.id) for _, q in questions for m in config.models]

    if not jobs:
        count = store.write_consolidated(order)
        print(f"Nothing to do; {count} responses in {store.path('responses.json')}")
        return RunResult(run_dir, "finished", per_model)

    stop = asyncio.Event()
    progress = Progress(per_model, write_snapshot=store.write_progress)
    client = OpenRouterClient(
        base_url=config.base_url, api_key=api_key, max_connections=config.concurrency
    )
    runner = Runner(
        config,
        client=client,
        store=store,
        progress=progress,
        density_rules=density_rules,
        stop=stop,
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
    reporter = asyncio.create_task(progress.run(config.progress_interval_seconds))
    state = "finished"
    try:
        await runner.run_jobs(jobs)
    except asyncio.CancelledError:
        state = "interrupted"
    finally:
        reporter.cancel()
        loop.remove_signal_handler(signal.SIGINT)
        await client.aclose()
        store.close()
        if runner.fatal_message:
            state = "aborted"
        elif interrupted:
            state = "interrupted"
        elif any(m.stopped_early for m in per_model.values()):
            state = "stopped"
        progress.finish(state)
        count = store.write_consolidated(order)
        log.info("run %s: %d responses consolidated", state, count)

    return RunResult(
        run_dir,
        state,
        per_model,
        top_errors=runner.error_counts.most_common(5),
        fatal_message=runner.fatal_message,
    )
