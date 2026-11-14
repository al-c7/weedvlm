"""
Result storage for a run, everything inside <output_dir>/<run_name>/:

    responses.jsonl    append-only log, one JSON response per line, flushed
                       as each finishes -- the durable record, and what a
                       resumed run reads to know what's already done
    responses.parquet  the latest attempt per question x model, one flat
                       row each (see docs/DATABASE.md), written when each
                       session of the run ends
    run.json           the config the run was started with, and a
                       fingerprint of each model's request settings
    progress.json      live status, rewritten every few seconds
    run.log            errors and retries, one line each

Parquet can't be appended to row by row, which is why the jsonl log stays
the record while a run is going. To analyse every run at once:
    pl.scan_parquet("results/*/responses.parquet")

A response never embeds the ground truth; it relates back to its question
by `question_id` (plus enough of the question's identity to be useful on
its own), so scoring is a join against the questions.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import polars as pl
from pydantic import BaseModel, Field

from weedvlm.storage.parquet import write_atomic

if TYPE_CHECKING:
    from weedvlm.evaluation.config import EvaluationConfig

RESPONSES_JSONL = "responses.jsonl"
RESPONSES_PARQUET = "responses.parquet"
RUN_JSON = "run.json"
PROGRESS_JSON = "progress.json"
RUN_LOG = "run.log"


class ResponseError(BaseModel):
    # transient / request / fatal (see openrouter.RequestFailure), or
    # "parse" when a reply arrived but held no usable JSON answer.
    kind: str
    message: str
    http_status: int | None = None


class ResponseRecord(BaseModel):
    # -- which question, answered by which model
    question_id: str
    model: str
    task: str
    question_type: str
    annotated: bool
    benchmark_class: str
    dataset_name: str
    source_image_id: int
    image_path: str
    image_sha256: str | None = None
    prompt_version: int

    status: Literal["ok", "failed"]
    # The model's JSON object: `answer`, `reasoning`, and for density
    # `estimated_weed_count` / `estimated_weed_coverage_percent`.
    # Localisation has `bboxes` and `response_format` in place of `answer`.
    parsed: dict[str, Any] | None = None
    # The reply verbatim (kept even when it couldn't be parsed).
    raw_response: str | None = None
    warnings: list[str] = Field(default_factory=list)
    error: ResponseError | None = None

    # Attempts made, and why each retried one was retried.
    attempts: int = 1
    retry_log: list[str] = Field(default_factory=list)

    finish_reason: str | None = None
    latency_seconds: float | None = None
    usage: dict[str, Any] | None = None
    cost_usd: float | None = None
    generation_id: str | None = None
    provider: str | None = None
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


def model_fingerprint(**settings: Any) -> str:
    """Identifies everything about a model's requests that would change
    its answers, so a resumed run can refuse to mix two settings."""
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:16]


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


class RunStore:
    """A run kept as JSON files in its run directory."""

    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self._jsonl_path = run_dir / RESPONSES_JSONL
        self._file = None

    def path(self, name: str) -> Path:
        return self.run_dir / name

    @property
    def location(self) -> str:
        """Where the responses are, for messages."""
        return f"{self.run_dir}/"

    # -- resume -------------------------------------------------------

    def load_records(self) -> dict[tuple[str, str], ResponseRecord]:
        """Latest record per (model, question_id). A trailing line cut
        short by a crash is ignored (its question is simply asked
        again)."""
        latest: dict[tuple[str, str], ResponseRecord] = {}
        if not self._jsonl_path.exists():
            return latest
        with self._jsonl_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    record = ResponseRecord.model_validate_json(line)
                except ValueError:
                    continue
                latest[(record.model, record.question_id)] = record
        return latest

    def read_run_info(self) -> dict[str, Any] | None:
        """What write_run_info last stored, None for a run not yet started."""
        path = self.path(RUN_JSON)
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def write_run_info(self, data: dict[str, Any]) -> None:
        """Stores the run's settings: run_name, prompt_version, config,
        model_fingerprints and sessions."""
        _atomic_write(self.path(RUN_JSON), json.dumps(data, indent=2))

    # -- writing ------------------------------------------------------

    def append(self, record: ResponseRecord) -> None:
        if self._file is None:
            self._file = self._jsonl_path.open("a", encoding="utf-8")
        self._file.write(record.model_dump_json() + "\n")
        self._file.flush()

    def write_progress(self, snapshot: dict[str, Any]) -> None:
        _atomic_write(self.path(PROGRESS_JSON), json.dumps(snapshot, indent=2))

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def write_consolidated(self, order: list[tuple[str, str]]) -> int:
        """responses.parquet: the latest record per (model, question_id),
        in question order then model order. Returns how many were written."""
        latest = self.load_records()
        run = self.run_dir.name
        # Keep the ids already given out (a run migrated from SQLite has
        # its original ones), so tables joined on them stay valid.
        existing: dict[tuple[str, str], int] = {}
        if self.path(RESPONSES_PARQUET).exists():
            previous = pl.read_parquet(self.path(RESPONSES_PARQUET), columns=["model", "question_id", "id"])
            existing = {(m, q): i for m, q, i in previous.iter_rows()}
        rows = [response_row(run, latest[key], existing.get(key)) for key in order if key in latest]
        write_atomic(pl.DataFrame(rows, schema=RESPONSE_SCHEMA), self.path(RESPONSES_PARQUET))
        return len(rows)


def _json_or_none(value: Any) -> str | None:
    return None if value is None else json.dumps(value)


def _text_or_none(value: Any) -> str | None:
    """A parsed reply's field as text for its own column; non-strings
    (a numeric answer, a list of boxes) as JSON."""
    if value is None or isinstance(value, str):
        return value
    return json.dumps(value)


def response_id(run: str, model: str, question_id: str) -> int:
    """A response's `id`: stable across sessions and re-consolidation, and
    unique across runs, so tables joined on it (the scored responses, the
    reasoning judgements) stay valid. Runs migrated from SQLite keep their
    original small integer ids instead; these are 62-bit, so never clash."""
    digest = hashlib.sha256(f"{run}\x1f{model}\x1f{question_id}".encode()).digest()
    return int.from_bytes(digest[:8]) >> 2


RESPONSE_SCHEMA = {
    "id": pl.Int64,
    "run": pl.String,
    "model": pl.String,
    "question_id": pl.String,
    "status": pl.String,
    "answer": pl.String,
    "reasoning": pl.String,
    "parsed": pl.String,
    "raw_response": pl.String,
    "warnings": pl.String,
    "error_kind": pl.String,
    "error_message": pl.String,
    "error_http_status": pl.Int64,
    "attempts": pl.Int64,
    "retry_log": pl.String,
    "finish_reason": pl.String,
    "latency_seconds": pl.Float64,
    "usage": pl.String,
    "total_tokens": pl.Int64,
    "cost_usd": pl.Float64,
    "generation_id": pl.String,
    "provider": pl.String,
    "image_sha256": pl.String,
    "prompt_version": pl.Int64,
    "created_at": pl.String,
}


def response_row(run: str, record: ResponseRecord, id: int | None = None) -> dict[str, Any]:
    """A record as one flat responses.parquet row: the answer and
    reasoning pulled out of `parsed`, the error split into columns, nested
    values as JSON text."""
    parsed = record.parsed or {}
    error = record.error
    return {
        "id": response_id(run, record.model, record.question_id) if id is None else id,
        "run": run,
        "model": record.model,
        "question_id": record.question_id,
        "status": record.status,
        "answer": _text_or_none(parsed.get("answer", parsed.get("bboxes"))),
        "reasoning": _text_or_none(parsed.get("reasoning")),
        "parsed": _json_or_none(record.parsed),
        "raw_response": record.raw_response,
        "warnings": json.dumps(record.warnings),
        "error_kind": error.kind if error else None,
        "error_message": error.message if error else None,
        "error_http_status": error.http_status if error else None,
        "attempts": record.attempts,
        "retry_log": json.dumps(record.retry_log),
        "finish_reason": record.finish_reason,
        "latency_seconds": record.latency_seconds,
        "usage": _json_or_none(record.usage),
        "total_tokens": (record.usage or {}).get("total_tokens"),
        "cost_usd": record.cost_usd,
        "generation_id": record.generation_id,
        "provider": record.provider,
        "image_sha256": record.image_sha256,
        "prompt_version": record.prompt_version,
        "created_at": record.created_at,
    }


def open_run_store(config: EvaluationConfig) -> RunStore:
    """The store for a run about to start (or resume). Creates the run
    directory."""
    run_dir = config.output_dir / config.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return RunStore(run_dir)
