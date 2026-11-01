"""
Copies a benchmark stored as JSON -- the question files written by
generate-all-questions.py, and the run directories written by
run-evaluation.py -- into a SQLite benchmark database (docs/DATABASE.md).
The JSON files are left untouched.

Questions come from every task file present in --questions-dir, with the
images they use and the generation-config.yaml alongside them. Each run
under --results-dir (any directory holding a run.json) is then copied in:
its settings, and the latest response to each (model, question).

A response can only be stored for a question that is in the database, so
responses to questions the benchmark no longer contains are left behind,
and counted in the output. Responses given for an image that has since
been re-rendered are copied, but are stale: score-evaluation.py refuses
to score them unless told to --skip-stale.

Scores aren't copied; run score-evaluation.py --database on each migrated
run to produce them.

Usage:
    uv run src/weedvlm/scripts/migrate-to-sqlite.py
    uv run src/weedvlm/scripts/migrate-to-sqlite.py --run gemini-full-run --no-images
    uv run src/weedvlm/scripts/migrate-to-sqlite.py --skip-questions --run a-later-run
"""

import argparse
import json
import sys
from pathlib import Path

from weedvlm.evaluation.store import RUN_JSON, RunStore, SqliteRunStore
from weedvlm.generation.config import GENERATION_CONFIG_NAME
from weedvlm.storage import database
from weedvlm.storage.config import DEFAULT_DATABASE_NAME
from weedvlm.types.questions import TASK_FILES, QuestionBase, parse_question


def _load_json_questions(questions_dir: Path) -> list[tuple[str, QuestionBase]]:
    questions: list[tuple[str, QuestionBase]] = []
    for task, filename in TASK_FILES.items():
        path = questions_dir / filename
        if not path.exists():
            print(f"  {task}: no {filename}, skipped")
            continue
        parsed = [parse_question(raw) for raw in json.loads(path.read_text(encoding="utf-8"))]
        questions.extend((task, question) for question in parsed)
        print(f"  {task}: {len(parsed)} questions")
    return questions


def _migrate_questions(conn, questions_dir: Path, *, store_images: bool) -> None:
    print(f"Questions from {questions_dir}:")
    questions = _load_json_questions(questions_dir)
    if not questions:
        raise FileNotFoundError(f"No question files found in {questions_dir}")

    missing = sorted({str(q.image_path) for _, q in questions if not Path(q.image_path).exists()})
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} question image(s) don't exist, e.g. {missing[0]}"
        )

    config_path = questions_dir / GENERATION_CONFIG_NAME
    generation_config = config_path.read_text(encoding="utf-8") if config_path.exists() else None
    if generation_config is None:
        print(
            f"  warning: no {GENERATION_CONFIG_NAME}; density runs will need `generation_config` "
            "set in their evaluation config"
        )

    num_images = database.write_benchmark(
        conn, questions, generation_config=generation_config, store_images=store_images
    )
    print(
        f"  wrote {len(questions)} questions and {num_images} images "
        f"({'stored' if store_images else 'by path only'})"
    )


def _migrate_run(run_dir: Path, database_path: Path, hashes: dict[str, str]) -> None:
    source = RunStore(run_dir)
    info = source.read_run_info()
    assert info is not None  # only directories with a run.json get here
    records = source.load_records()

    target = SqliteRunStore(database_path, run_dir.name)
    # The migrated run lives in the database from here on, so that's
    # where resuming it should read and write.
    info["config"]["storage"] = {"format": "sqlite", "database": str(database_path)}
    target.write_run_info(info)

    known = target.question_ids()
    kept = [record for record in records.values() if record.question_id in known]
    target.append_many(kept)

    target.close()

    stale = sum(
        record.image_sha256 is not None and record.image_sha256 != hashes[record.image_path]
        for record in kept
        if record.image_path in hashes
    )

    notes = []
    if len(kept) < len(records):
        notes.append(f"{len(records) - len(kept)} left behind (question not in the benchmark)")
    if stale:
        notes.append(f"{stale} stale (image changed since)")
    print(f"  {run_dir.name}: {len(kept)} responses" + (f"; {', '.join(notes)}" if notes else ""))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--questions-dir", type=Path, default=Path("questions"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--database",
        type=Path,
        default=None,
        help=f"Database to write. Default: {DEFAULT_DATABASE_NAME} inside --questions-dir.",
    )
    parser.add_argument(
        "--run",
        action="append",
        dest="runs",
        default=None,
        help="Name of a run under --results-dir to migrate (repeatable). Default: all of them.",
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Record each image's path and hash only, rather than storing its bytes",
    )
    parser.add_argument(
        "--skip-questions",
        action="store_true",
        help="Leave the questions already in the database as they are; only add runs",
    )
    parser.add_argument("--skip-runs", action="store_true", help="Only migrate the questions")
    args = parser.parse_args()

    database_path = args.database or args.questions_dir / DEFAULT_DATABASE_NAME
    try:
        conn = database.connect(database_path, create=not args.skip_questions)
        if not args.skip_questions:
            _migrate_questions(conn, args.questions_dir, store_images=not args.no_images)
        hashes = database.image_hashes(conn)
        conn.close()

        if not args.skip_runs:
            if args.runs is not None:
                run_dirs = [args.results_dir / name for name in args.runs]
                not_runs = [str(d) for d in run_dirs if not (d / RUN_JSON).exists()]
                if not_runs:
                    raise FileNotFoundError(f"Not a run directory: {', '.join(not_runs)}")
            else:
                run_dirs = sorted(p.parent for p in args.results_dir.glob(f"*/{RUN_JSON}"))
            print(f"Runs from {args.results_dir}:")
            for run_dir in run_dirs:
                _migrate_run(run_dir, database_path, hashes)
    except (database.BenchmarkInUseError, FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    print(f"Done: {database_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
