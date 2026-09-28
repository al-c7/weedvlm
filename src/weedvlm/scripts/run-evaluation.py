"""
Runs the generated benchmark against VLMs on OpenRouter.

Which tasks run, which models, how many requests are in flight at once and
how failures are retried are all set in a YAML config (see
evaluation.example.yaml at the repo root). Every (question, model) pair is
sent the question's image plus a prompt asking for a JSON reply -- an
`answer` (an option verbatim, or a species) and a two-sentence `reasoning`
-- and each response is stored, keyed by question id, under
<output_dir>/<run_name>/ (responses.jsonl as it goes, responses.json when
done). The terminal shows live progress; progress.json holds the same for
anything else that wants to watch.

Re-running with the same run_name resumes: answered questions are
skipped, failed or unfinished ones are asked again. Ctrl-C stops cleanly
(twice to abandon in-flight requests) and is resumable the same way.

The API key is read from OPENROUTER_API_KEY (environment or .env).

Usage:
    uv run src/weedvlm/run-evaluation.py --config evaluation.example.yaml
    uv run src/weedvlm/run-evaluation.py --run-name gpt-first-look --max-questions-per-task 5

Exit status: 0 if every request succeeded, 1 if any failed or were left
unfinished, 2 for a configuration problem, 130 if interrupted.
"""

import argparse
import asyncio
import sys
from pathlib import Path

from pydantic import ValidationError

from weedvlm.evaluation.config import (
    MAX_CONCURRENCY,
    load_evaluation_config,
    resolve_config_path,
)
from weedvlm.evaluation.runner import FatalRunError, RunResult, run_evaluation


def _print_summary(result: RunResult) -> None:
    print(f"\nRun {result.state}. Results in {result.run_dir}/")
    for model, m in result.per_model.items():
        pending = m.total - m.answered - m.failed
        print(
            f"  {model}: {m.answered}/{m.total} answered"
            + (f", {m.failed} failed" if m.failed else "")
            + (f", {pending} not attempted" if pending else "")
            + f", ${m.cost_usd:.4f} this session"
            + ("  [dropped: too many failures in a row]" if m.stopped_early else "")
        )
    if result.top_errors:
        print("Most common failures:")
        for message, count in result.top_errors:
            print(f"  {count:>4} x {message}")
    if result.fatal_message:
        print(f"Stopped early: {result.fatal_message}")
    if result.pending or result.failed:
        print("Re-run the same command to retry what's left.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML config file (default: evaluation.example.yaml at the repo root)",
    )
    parser.add_argument("--run-name", default=None, help="Overrides the config's run_name")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help=f"Overrides the config's concurrency (1-{MAX_CONCURRENCY})",
    )
    parser.add_argument(
        "--max-questions-per-task",
        type=int,
        default=None,
        help="Only run a seeded random subset of this many questions per task (smoke test)",
    )
    args = parser.parse_args()

    try:
        config = load_evaluation_config(resolve_config_path(args.config))
        overrides = {
            "run_name": args.run_name,
            "concurrency": args.concurrency,
            "max_questions_per_task": args.max_questions_per_task,
        }
        overrides = {k: v for k, v in overrides.items() if v is not None}
        if overrides:
            config = type(config).model_validate({**config.model_dump(), **overrides})
        result = asyncio.run(run_evaluation(config))
    except (ValidationError, FileNotFoundError, ValueError, FatalRunError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130

    _print_summary(result)
    if result.state == "interrupted":
        return 130
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
