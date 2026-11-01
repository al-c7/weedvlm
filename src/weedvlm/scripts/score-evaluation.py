"""
Scores the responses from run-evaluation.py against the benchmark's ground
truth: an outcome per response, and from those accuracy (with 95%
intervals), macro-F1 and unsure rate per task and image condition, plus
density's count/coverage errors, the multiple-choice vs open-ended gap for
species ID, localisation's box count and IoU scores, and cost/latency.

Each response is joined to its question by id, using the benchmark the run
was configured with. If a question's image has changed since the model saw
it -- the benchmark was regenerated -- the answer key may no longer match,
so scoring stops rather than producing wrong numbers (--skip-stale drops
those responses instead).

A run stored as JSON is named by its directory, and gets written next to
its responses, in <output_dir>/<run_name>/:
    scores.json                 every metric, plus per-class and per-dataset
                                accuracy
    scored-responses.csv        one row per response with its outcome, for
                                notebooks/pandas
    open-ended-mismatches.json  open-ended answers judged wrong, to check for
                                name variants the matcher missed

A run stored in a SQLite benchmark database is named by its run name, with
--database. Its scores go into the same database: the scores table (one row
per response) and scoring_runs (the metrics). See docs/DATABASE.md.

Open-ended species answers are matched by normalised name (see
weedvlm.evaluation.scoring.open_ended_correct). Add spelling/common-name
variants the mismatches reveal with --aliases, a YAML mapping of the
benchmark's name to accepted alternatives:

    Portulaca oleracea: [common purslane, purslane]

Localisation boxes are read in the format each reply states in its
`response_format`. For a model that describes its format badly, set it
with --bbox-format (or `bbox_format` on the model in the run's config):

    --bbox-format google/gemini-2.5-flash=yxyx_1000

Usage:
    uv run src/weedvlm/scripts/score-evaluation.py results/gemini-full-run
    uv run src/weedvlm/scripts/score-evaluation.py results/gemini-full-run --aliases aliases.yaml
    uv run src/weedvlm/scripts/score-evaluation.py gemini-full-run --database questions/benchmark.db
"""

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml
from pydantic import ValidationError

from weedvlm.evaluation.bboxes import parse_bbox_format
from weedvlm.evaluation.config import EvaluationConfig
from weedvlm.evaluation.metrics import compute_metrics
from weedvlm.evaluation.questions import open_benchmark
from weedvlm.evaluation.scoring import load_aliases, score_run
from weedvlm.evaluation.store import RunStore, SqliteRunStore


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "run",
        help="A run directory, e.g. results/gemini-full-run -- or, with --database, a run name",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=None,
        help="Score a run stored in this SQLite benchmark database, rather than one stored as JSON",
    )
    parser.add_argument("--aliases", type=Path, default=None, help="YAML of accepted name variants")
    parser.add_argument(
        "--bbox-format",
        action="append",
        default=[],
        metavar="MODEL=FORMAT",
        help="Localisation box format for a model, <order>_<scale> (order xyxy/yxyx/xywh/cxcywh, "
        "scale pixel/unit/100/1000), instead of its replies' response_format. Repeatable.",
    )
    parser.add_argument(
        "--questions-dir",
        type=Path,
        default=None,
        help="Question files to score against (default: the run's configured questions_dir). "
        "JSON runs only.",
    )
    parser.add_argument(
        "--skip-stale",
        action="store_true",
        help="Drop responses whose question or image changed since they were answered, "
        "instead of refusing to score",
    )
    args = parser.parse_args()

    try:
        if args.database is not None:
            if args.questions_dir is not None:
                raise ValueError("--questions-dir doesn't apply with --database")
            store: RunStore = SqliteRunStore(args.database, args.run)
        else:
            store = RunStore(Path(args.run))
        run_info = store.read_run_info()
        if run_info is None:
            raise FileNotFoundError(f"No run found at {store.location}")
        config = EvaluationConfig.model_validate(run_info["config"])
        if args.database is not None:
            # Wherever the database was when the run was made, it's here now.
            config.storage.format = "sqlite"
            config.storage.database = args.database
        if args.questions_dir is not None:
            config.questions_dir = args.questions_dir
        records = store.load_records()
        if not records:
            raise ValueError(f"{store.location} has no responses yet")

        aliases = {}
        if args.aliases is not None:
            aliases = load_aliases(yaml.safe_load(args.aliases.read_text(encoding="utf-8")))

        bbox_formats = {m.id: m.bbox_format for m in config.models if m.bbox_format}
        for override in args.bbox_format:
            model, sep, fmt = override.rpartition("=")
            if not sep or not model:
                raise ValueError(f"--bbox-format {override!r} isn't MODEL=FORMAT")
            parse_bbox_format(fmt)
            bbox_formats[model] = fmt

        print(f"Scoring {len(records)} responses from {store.location} ...", file=sys.stderr)
        benchmark = open_benchmark(config)
        joined = score_run(records, benchmark, aliases, bbox_formats=bbox_formats)
        benchmark.close()
    except (ValidationError, FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    if joined.missing_questions or joined.changed_images:
        print(
            f"warning: {len(joined.missing_questions)} response(s) are for questions no longer in "
            f"the benchmark, and {len(joined.changed_images)} for images that changed since the "
            "model saw them (the benchmark was regenerated?).",
            file=sys.stderr,
        )
        if not args.skip_stale:
            print(
                "Refusing to score against a possibly different answer key. Score against the "
                "original questions (--questions-dir for a JSON run), or pass --skip-stale to "
                "drop these.",
                file=sys.stderr,
            )
            return 2
    if not joined.scored:
        print("error: nothing left to score", file=sys.stderr)
        return 2

    written = store.write_scores(
        joined.scored,
        {
            "run": args.run if args.database is not None else Path(args.run).name,
            "scored_at": datetime.now(UTC).isoformat(),
            "aliases_file": str(args.aliases) if args.aliases else None,
            "skipped_missing_questions": len(joined.missing_questions),
            "skipped_changed_images": len(joined.changed_images),
            "models": compute_metrics(joined.scored),
        },
    )
    store.close()
    print(f"Scored {len(joined.scored)} responses. Wrote {written}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
