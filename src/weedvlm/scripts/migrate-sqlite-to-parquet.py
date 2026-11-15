"""
One-off: converts a SQLite benchmark database (the old storage format) to
the parquet layout (docs/DATABASE.md).

    <questions-dir>/questions.parquet        from the questions + images tables
    <questions-dir>/generation-config.yaml   from meta.generation_config, if not already there
    <results-dir>/<run>/run.json             from the runs table
    <results-dir>/<run>/responses.jsonl      from the responses table, so the run can be resumed
    <results-dir>/<run>/responses.parquet    the same, keeping each response's original `id`

Keeping the ids means the scored responses parquet and the reasoning
judgements (which refer to responses by that id) still join to the new
files. Images whose bytes were stored in the database but whose file is
gone are written back out to their path. The database itself is only read.

It refuses to touch a run directory that already has responses.jsonl or
responses.parquet, unless --force is given.

Usage:
    uv run src/weedvlm/scripts/migrate-sqlite-to-parquet.py questions/benchmark.db
    uv run src/weedvlm/scripts/migrate-sqlite-to-parquet.py old.db --questions-dir questions --results-dir results
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import polars as pl

from weedvlm.evaluation.store import (
    PROGRESS_JSON,
    RESPONSE_SCHEMA,
    RESPONSES_JSONL,
    RESPONSES_PARQUET,
    RUN_JSON,
    ResponseError,
    ResponseRecord,
    response_row,
)
from weedvlm.generation.config import GENERATION_CONFIG_NAME
from weedvlm.storage.parquet import QUESTION_SCHEMA, QUESTIONS_FILE, check_rows, write_atomic


def _loads(text: str | None):
    return None if text is None else json.loads(text)


def migrate_questions(conn: sqlite3.Connection, questions_dir: Path) -> int:
    images = conn.execute("SELECT id, path, sha256, data IS NOT NULL AS stored FROM images").fetchall()
    restored = 0
    for image in images:
        path = Path(image["path"])
        if not path.exists():
            if not image["stored"]:
                sys.exit(f"error: {path} is missing and its bytes aren't in the database")
            (data,) = conn.execute("SELECT data FROM images WHERE id = ?", (image["id"],)).fetchone()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            restored += 1
    if restored:
        print(f"Restored {restored} image(s) from the database to disk")

    columns = [c for c in QUESTION_SCHEMA if c not in ("image_path", "image_sha256")]
    rows = [
        dict(row)
        for row in conn.execute(
            f"SELECT {', '.join('q.' + c for c in columns)}, i.path AS image_path, "
            "i.sha256 AS image_sha256 FROM questions q JOIN images i ON i.id = q.image_id "
            "ORDER BY q.task, q.position"
        )
    ]
    for row in rows:
        row["annotated"] = bool(row["annotated"])
    check_rows(rows)
    write_atomic(pl.DataFrame(rows, schema=QUESTION_SCHEMA), questions_dir / QUESTIONS_FILE)
    print(f"Wrote {len(rows)} questions ({len(images)} images) to {questions_dir / QUESTIONS_FILE}")

    config_path = questions_dir / GENERATION_CONFIG_NAME
    meta = conn.execute("SELECT value FROM meta WHERE key = 'generation_config'").fetchone()
    if meta is not None and not config_path.exists():
        config_path.write_text(meta["value"], encoding="utf-8")
        print(f"Wrote {config_path}")
    elif meta is None and not config_path.exists():
        print(f"warning: no generation config recorded; put one at {config_path} before evaluating")
    return len(rows)


def _record(row: sqlite3.Row) -> ResponseRecord:
    error = None
    if row["error_kind"] is not None:
        error = ResponseError(
            kind=row["error_kind"], message=row["error_message"], http_status=row["error_http_status"]
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
        parsed=_loads(row["parsed"]),
        raw_response=row["raw_response"],
        warnings=json.loads(row["warnings"]),
        error=error,
        attempts=row["attempts"],
        retry_log=json.loads(row["retry_log"]),
        finish_reason=row["finish_reason"],
        latency_seconds=row["latency_seconds"],
        usage=_loads(row["usage"]),
        cost_usd=row["cost_usd"],
        generation_id=row["generation_id"],
        provider=row["provider"],
        created_at=row["created_at"],
    )


def migrate_run(conn: sqlite3.Connection, run: sqlite3.Row, results_dir: Path, force: bool) -> int:
    name = run["name"]
    run_dir = results_dir / name
    if not force and any((run_dir / f).exists() for f in (RESPONSES_JSONL, RESPONSES_PARQUET)):
        sys.exit(f"error: {run_dir} already holds responses; pass --force to overwrite")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / RUN_JSON).write_text(
        json.dumps(
            {
                "run_name": name,
                "prompt_version": run["prompt_version"],
                "config": json.loads(run["config"]),
                "model_fingerprints": json.loads(run["model_fingerprints"]),
                "sessions": json.loads(run["sessions"]),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    responses = conn.execute(
        "SELECT r.*, q.task, q.question_type, q.annotated, q.benchmark_class, q.dataset_name, "
        "q.source_image_id, i.path AS image_path FROM responses r "
        "JOIN questions q ON q.id = r.question_id JOIN images i ON i.id = q.image_id "
        "WHERE r.run = ? ORDER BY q.task, q.position, r.model",
        (name,),
    ).fetchall()
    rows = []
    with (run_dir / RESPONSES_JSONL).open("w", encoding="utf-8") as log:
        for row in responses:
            record = _record(row)
            log.write(record.model_dump_json() + "\n")
            rows.append(response_row(name, record, id=row["id"]))
    write_atomic(pl.DataFrame(rows, schema=RESPONSE_SCHEMA), run_dir / RESPONSES_PARQUET)
    (run_dir / PROGRESS_JSON).unlink(missing_ok=True)
    print(f"  {name}: {len(rows)} responses -> {run_dir}/")
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("database", type=Path, help="The SQLite benchmark database to convert")
    parser.add_argument(
        "--questions-dir", type=Path, default=None, help="Default: the database's directory"
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--force", action="store_true", help="Overwrite existing run directories")
    args = parser.parse_args()

    if not args.database.exists():
        parser.error(f"{args.database} not found")
    conn = sqlite3.connect(f"file:{args.database}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    migrate_questions(conn, args.questions_dir or args.database.parent)
    runs = conn.execute("SELECT * FROM runs ORDER BY name").fetchall()
    print(f"Migrating {len(runs)} run(s):")
    total = sum(migrate_run(conn, run, args.results_dir, args.force) for run in runs)
    (expected,) = conn.execute("SELECT COUNT(*) FROM responses").fetchone()
    conn.close()
    if total != expected:
        print(f"error: wrote {total} responses but the database has {expected}", file=sys.stderr)
        return 1
    print(f"Done: {total} responses. The database was not modified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
