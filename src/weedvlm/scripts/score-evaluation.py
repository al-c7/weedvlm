"""
Scores the responses from run-evaluation.py against the benchmark's ground
truth and reports accuracy (with 95% intervals), macro-F1 and unsure rate
per task and image condition, plus density's count/coverage errors, the
multiple-choice vs open-ended gap for species ID, and cost/latency.

Each response is joined to its question by id, using the question files
the run was configured with (recorded in its run.json). If a question's
image has changed since the model saw it -- the benchmark was regenerated
-- the answer key may no longer match, so scoring stops rather than
producing wrong numbers (--skip-stale drops those responses instead).

Writes next to the responses, in <output_dir>/<run_name>/:
    scores.json                 every metric in the report, plus per-class and
                                per-dataset accuracy
    scored-responses.csv        one row per response with its outcome, for
                                notebooks/pandas
    open-ended-mismatches.json  open-ended answers judged wrong, to check for
                                name variants the matcher missed
    summary.png                 the headline tables (accuracy per suite, multiple
                                choice vs open-ended, density errors, cost) as a
                                figure

Open-ended species answers are matched by normalised name (see
weedvlm.evaluation.scoring.open_ended_correct). Add spelling/common-name
variants the review file reveals with --aliases, a YAML mapping of the
benchmark's name to accepted alternatives:

    Portulaca oleracea: [common purslane, purslane]

Usage:
    uv run src/weedvlm/score-evaluation.py results/gemini-full-run
    uv run src/weedvlm/score-evaluation.py results/gemini-full-run --aliases aliases.yaml
"""

import argparse
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import yaml
from pydantic import ValidationError

from weedvlm.evaluation.config import EvaluationConfig
from weedvlm.evaluation.figure import render_summary_figure
from weedvlm.evaluation.metrics import compute_metrics
from weedvlm.evaluation.questions import load_questions
from weedvlm.evaluation.report import render_report
from weedvlm.evaluation.scoring import load_aliases, score_run
from weedvlm.evaluation.store import RunStore


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("run_dir", type=Path, help="A run directory, e.g. results/gemini-full-run")
    parser.add_argument("--aliases", type=Path, default=None, help="YAML of accepted name variants")
    parser.add_argument(
        "--questions-dir",
        type=Path,
        default=None,
        help="Question files to score against (default: the run's configured questions_dir)",
    )
    parser.add_argument(
        "--skip-stale",
        action="store_true",
        help="Drop responses whose question or image changed since they were answered, "
        "instead of refusing to score",
    )
    args = parser.parse_args()

    try:
        store = RunStore(args.run_dir)
        run_json = store.read_run_json()
        if run_json is None:
            raise FileNotFoundError(f"{args.run_dir / 'run.json'} not found -- is this a run directory?")
        config = EvaluationConfig.model_validate(run_json["config"])
        if args.questions_dir is not None:
            config.questions_dir = args.questions_dir
        records = store.load_records()
        if not records:
            raise ValueError(f"{args.run_dir} has no responses yet")

        aliases = {}
        if args.aliases is not None:
            aliases = load_aliases(yaml.safe_load(args.aliases.read_text(encoding="utf-8")))

        print(f"Scoring {len(records)} responses from {args.run_dir} ...", file=sys.stderr)
        joined = score_run(records, load_questions(config), aliases)
    except (ValidationError, FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    if joined.missing_questions or joined.changed_images:
        print(
            f"warning: {len(joined.missing_questions)} response(s) are for questions no longer in "
            f"{config.questions_dir}, and {len(joined.changed_images)} for images that changed "
            "since the model saw them (the benchmark was regenerated?).",
            file=sys.stderr,
        )
        if not args.skip_stale:
            print(
                "Refusing to score against a possibly different answer key. Score against the "
                "original questions with --questions-dir, or pass --skip-stale to drop these.",
                file=sys.stderr,
            )
            return 2
    if not joined.scored:
        print("error: nothing left to score", file=sys.stderr)
        return 2

    metrics = compute_metrics(joined.scored)
    failures: dict[str, list[tuple[str, int]]] = {}
    for model in metrics:
        counts = Counter(
            f"[{r.error.kind}] {r.error.message[:110]}"
            for (m, _), r in records.items()
            if m == model and r.error is not None
        )
        failures[model] = counts.most_common(5)

    (args.run_dir / "scores.json").write_text(
        json.dumps(
            {
                "run": args.run_dir.name,
                "scored_at": datetime.now(UTC).isoformat(),
                "aliases_file": str(args.aliases) if args.aliases else None,
                "skipped_missing_questions": len(joined.missing_questions),
                "skipped_changed_images": len(joined.changed_images),
                "models": metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    frame = pd.DataFrame([vars(s) for s in joined.scored])
    frame.to_csv(args.run_dir / "scored-responses.csv", index=False)

    mismatches = frame[(frame["suite"].str.startswith("species_id/open_ended")) & (frame["outcome"] == "incorrect")]
    (args.run_dir / "open-ended-mismatches.json").write_text(
        json.dumps(
            mismatches[["question_id", "model", "expected", "answer"]].to_dict("records"), indent=2
        ),
        encoding="utf-8",
    )

    print(render_report(metrics, failures))
    render_summary_figure(metrics, args.run_dir.name, args.run_dir / "summary.png")
    print(
        "Wrote scores.json, scored-responses.csv, open-ended-mismatches.json and summary.png "
        f"to {args.run_dir}/"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
