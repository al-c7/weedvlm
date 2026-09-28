"""
Result storage for a run, all inside <output_dir>/<run_name>/:

    responses.jsonl  append-only log, one JSON response per line, flushed
                     as each finishes -- the durable record, and what a
                     resumed run reads to know what's already done
    responses.json   the same responses as one JSON array (latest attempt
                     per question x model), written when the run ends
    run.json         the config the run was started with, and a
                     fingerprint of each model's request settings
    progress.json    live status, rewritten every few seconds
    run.log          errors and retries, one line each

A response never embeds the ground truth; it relates back to its question
by `question_id` (plus enough of the question's identity to be useful on
its own), so scoring is a join against the question files.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

RESPONSES_JSONL = "responses.jsonl"
RESPONSES_JSON = "responses.json"
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
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        self._jsonl_path = run_dir / RESPONSES_JSONL
        self._file = None

    def path(self, name: str) -> Path:
        return self.run_dir / name

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

    def read_run_json(self) -> dict[str, Any] | None:
        path = self.path(RUN_JSON)
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def write_run_json(self, data: dict[str, Any]) -> None:
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
        """responses.json: the latest record per (model, question_id), in
        question order then model order. Returns how many were written."""
        latest = self.load_records()
        records = [latest[key].model_dump(mode="json") for key in order if key in latest]
        _atomic_write(self.path(RESPONSES_JSON), json.dumps(records, indent=2))
        return len(records)
