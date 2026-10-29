"""
Result storage for a run, in either storage format (see
weedvlm.storage.config).

RunStore is the JSON format, everything inside <output_dir>/<run_name>/:

    responses.jsonl  append-only log, one JSON response per line, flushed
                     as each finishes -- the durable record, and what a
                     resumed run reads to know what's already done
    responses.json   the same responses as one JSON array (latest attempt
                     per question x model), written when the run ends
    run.json         the config the run was started with, and a
                     fingerprint of each model's request settings
    progress.json    live status, rewritten every few seconds
    run.log          errors and retries, one line each

SqliteRunStore keeps the responses and the run's settings in the benchmark
database instead (the runs and responses tables, docs/DATABASE.md), next to
the questions they answer. The run directory then only holds progress.json
and run.log.

A response never embeds the ground truth; it relates back to its question
by `question_id` (plus enough of the question's identity to be useful on
its own), so scoring is a join against the questions.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

from weedvlm.storage import database

if TYPE_CHECKING:
    from weedvlm.evaluation.config import EvaluationConfig
    from weedvlm.evaluation.scoring import ScoredResponse

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
        """responses.json: the latest record per (model, question_id), in
        question order then model order. Returns how many were written."""
        latest = self.load_records()
        records = [latest[key].model_dump(mode="json") for key in order if key in latest]
        _atomic_write(self.path(RESPONSES_JSON), json.dumps(records, indent=2))
        return len(records)

    # -- scores -------------------------------------------------------

    def write_scores(self, scored: list[ScoredResponse], summary: dict[str, Any]) -> str:
        """Stores a scoring pass (see score-evaluation.py): one row per
        scored response, and `summary`, the aggregate metrics with how
        they were produced. Returns what was written, for a message."""
        import pandas as pd

        self.path("scores.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        frame = pd.DataFrame([vars(s) for s in scored])
        frame.to_csv(self.path("scored-responses.csv"), index=False)

        # Open-ended answers judged wrong, to check for name variants the
        # matcher missed.
        mismatches = frame[
            frame["suite"].str.startswith("species_id/open_ended") & (frame["outcome"] == "incorrect")
        ]
        self.path("open-ended-mismatches.json").write_text(
            json.dumps(
                mismatches[["question_id", "model", "expected", "answer"]].to_dict("records"),
                indent=2,
            ),
            encoding="utf-8",
        )
        return (
            "scores.json, scored-responses.csv and open-ended-mismatches.json to "
            f"{self.run_dir}/"
        )


def _json_or_none(value: Any) -> str | None:
    return None if value is None else json.dumps(value)


def _text_or_none(value: Any) -> str | None:
    """A parsed reply's field as text for its own column; non-strings
    (a numeric answer, a list of boxes) as JSON."""
    if value is None or isinstance(value, str):
        return value
    return json.dumps(value)


# ScoredResponse fields stored in the scores table. The rest of a
# ScoredResponse repeats what the questions and responses tables hold.
_SCORE_COLUMNS = (
    "suite",
    "outcome",
    "predicted_label",
    "chance",
    "true_count",
    "est_count",
    "count_error",
    "true_coverage_pct",
    "est_coverage_pct",
    "coverage_error_pp",
    "category_rank_error",
    "true_box_count",
    "pred_box_count",
    "box_count_error",
    "matched_iou",
    "localisation_score",
    "bbox_format",
    "pred_boxes",
)


class SqliteRunStore(RunStore):
    """A run kept in the benchmark database. run_dir, where progress.json
    and run.log still go, is only needed while a run is in progress."""

    def __init__(self, database_path: Path, run_name: str, run_dir: Path | None = None):
        if run_dir is not None:
            super().__init__(run_dir)
        self.database_path = database_path
        self.run_name = run_name
        self._conn = database.connect(database_path)

    @property
    def location(self) -> str:
        return f"{self.database_path} (run '{self.run_name}')"

    def load_records(self) -> dict[tuple[str, str], ResponseRecord]:
        rows = self._conn.execute(
            "SELECT r.*, q.task, q.question_type, q.annotated, q.benchmark_class, "
            "q.dataset_name, q.source_image_id, i.path AS image_path "
            "FROM responses r JOIN questions q ON q.id = r.question_id "
            "JOIN images i ON i.id = q.image_id WHERE r.run = ?",
            (self.run_name,),
        )
        return {(row["model"], row["question_id"]): _record_from_row(row) for row in rows}

    def read_run_info(self) -> dict[str, Any] | None:
        row = self._conn.execute("SELECT * FROM runs WHERE name = ?", (self.run_name,)).fetchone()
        if row is None:
            return None
        return {
            "run_name": row["name"],
            "prompt_version": row["prompt_version"],
            "config": json.loads(row["config"]),
            "model_fingerprints": json.loads(row["model_fingerprints"]),
            "sessions": json.loads(row["sessions"]),
        }

    def write_run_info(self, data: dict[str, Any]) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT INTO runs (name, prompt_version, config, model_fingerprints, sessions) "
                "VALUES (?, ?, ?, ?, ?) ON CONFLICT (name) DO UPDATE SET "
                "prompt_version = excluded.prompt_version, config = excluded.config, "
                "model_fingerprints = excluded.model_fingerprints, sessions = excluded.sessions",
                (
                    self.run_name,
                    data["prompt_version"],
                    json.dumps(data["config"]),
                    json.dumps(data["model_fingerprints"]),
                    json.dumps(data["sessions"]),
                ),
            )

    def append(self, record: ResponseRecord) -> None:
        self.append_many([record])

    def append_many(self, records: Iterable[ResponseRecord]) -> None:
        """Stores each record as the answer to its (model, question),
        replacing an earlier attempt -- and dropping that attempt's score,
        which was for a different answer. Committed together."""
        with self._conn:
            for record in records:
                row = self._response_row(record)
                assignments = ", ".join(
                    f"{column} = excluded.{column}"
                    for column in row
                    if column not in ("run", "model", "question_id")
                )
                (response_id,) = self._conn.execute(
                    f"INSERT INTO responses ({', '.join(row)}) "
                    f"VALUES ({', '.join(':' + column for column in row)}) "
                    f"ON CONFLICT (run, model, question_id) DO UPDATE SET {assignments} "
                    "RETURNING id",
                    row,
                ).fetchone()
                self._conn.execute("DELETE FROM scores WHERE response_id = ?", (response_id,))

    def _response_row(self, record: ResponseRecord) -> dict[str, Any]:
        parsed = record.parsed or {}
        error = record.error
        return {
            "run": self.run_name,
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

    def question_ids(self) -> set[str]:
        """Ids of the questions in the database, which are the only ones
        a response can be stored for."""
        return {row["id"] for row in self._conn.execute("SELECT id FROM questions")}

    def close(self) -> None:
        self._conn.close()

    def write_consolidated(self, order: list[tuple[str, str]]) -> int:
        """Nothing to consolidate -- the table already holds one row per
        (model, question). Returns how many of `order` are answered."""
        stored = {
            (row["model"], row["question_id"])
            for row in self._conn.execute(
                "SELECT model, question_id FROM responses WHERE run = ?", (self.run_name,)
            )
        }
        return sum(key in stored for key in order)

    def write_scores(self, scored: list[ScoredResponse], summary: dict[str, Any]) -> str:
        with self._conn:
            self._conn.execute(
                "DELETE FROM scores WHERE response_id IN (SELECT id FROM responses WHERE run = ?)",
                (self.run_name,),
            )
            self._conn.executemany(
                f"INSERT INTO scores (response_id, {', '.join(_SCORE_COLUMNS)}) "
                f"SELECT id, {', '.join(':' + column for column in _SCORE_COLUMNS)} "
                "FROM responses WHERE run = :run AND model = :model AND question_id = :question_id",
                [vars(s) | {"run": self.run_name} for s in scored],
            )
            self._conn.execute(
                "INSERT OR REPLACE INTO scoring_runs (run, scored_at, aliases_file, "
                "skipped_missing_questions, skipped_changed_images, metrics) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    self.run_name,
                    summary["scored_at"],
                    summary["aliases_file"],
                    summary["skipped_missing_questions"],
                    summary["skipped_changed_images"],
                    json.dumps(summary["models"]),
                ),
            )
        return f"the scores and scoring_runs tables of {self.location}"


def _record_from_row(row: sqlite3.Row) -> ResponseRecord:
    error = None
    if row["error_kind"] is not None:
        error = ResponseError(
            kind=row["error_kind"],
            message=row["error_message"],
            http_status=row["error_http_status"],
        )
    return ResponseRecord(
        question_id=row["question_id"],
        model=row["model"],
        task=row["task"],
        question_type=row["question_type"],
        annotated=bool(row["annotated"]),
        benchmark_class=row["benchmark_class"],
        dataset_name=row["dataset_name"],
        source_image_id=row["source_image_id"],
        image_path=row["image_path"],
        image_sha256=row["image_sha256"],
        prompt_version=row["prompt_version"],
        status=row["status"],
        parsed=json.loads(row["parsed"]) if row["parsed"] is not None else None,
        raw_response=row["raw_response"],
        warnings=json.loads(row["warnings"]),
        error=error,
        attempts=row["attempts"],
        retry_log=json.loads(row["retry_log"]),
        finish_reason=row["finish_reason"],
        latency_seconds=row["latency_seconds"],
        usage=json.loads(row["usage"]) if row["usage"] is not None else None,
        cost_usd=row["cost_usd"],
        generation_id=row["generation_id"],
        provider=row["provider"],
        created_at=row["created_at"],
    )


def open_run_store(config: EvaluationConfig) -> RunStore:
    """The store for a run about to start (or resume), in the format its
    config names. Creates the run directory."""
    run_dir = config.output_dir / config.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    if config.storage.format == "sqlite":
        return SqliteRunStore(config.resolved_database(), config.run_name, run_dir)
    return RunStore(run_dir)
