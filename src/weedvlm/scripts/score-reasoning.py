"""
Runs the generated benchmark against VLMs on OpenRouter.

Evaluates the post-hoc reasoning provided by the VLM models on the WeedVLM
benchmark.

The tasks to evaluate, the set of results to evaluate, and the model to use
for the evaluation are set in a YAML configuration file (see 
reasoning.example.yaml at the repo root). The judge VLM does not know which
model produced a specific response. The VLM is provided with the original
question text, and the source image, as well as any additional required
ground truth. The responses are loaded into a JSON file at...

The API key is read from OPENROUTER_API_KEY.

Usage:
    uv run src/weedvlm/score-reasoning.py --config ...

Exit status: 0 if every request succeeded, 1 if any failed or were left
unfinished, 2 for a configuration problem, 130 if interrupted.
"""

import argparse
import asyncio
import sys
from pathlib import Path
import random

from pydantic import ValidationError

from weedvlm.reasoning.config import (
    MAX_CONCURRENCY,
    load_reasoning_config,
    resolve_reasoning_path,
)

from weedvlm.reasoning.runner import get_question_samples, JudgeItem
from weedvlm.reasoning.judge import JUDGE_KINDS, run_judges
from weedvlm.reasoning.prompt import JudgeBuilder, load_species_descriptions
from weedvlm.storage.database import connect

from weedvlm.evaluation.runner import FatalRunError

def spread(items: list[JudgeItem], limit: int, seed: int = 0) -> list[JudgeItem]:
    """`limit` items taking one group at a time in turn, where a group is a
    task, question type and correct/incorrect, and items within a group are
    shuffled so models are mixed. A small smoke or pilot run therefore
    touches every task and includes incorrect responses."""
    groups: dict[tuple, list[JudgeItem]] = {}
    for item in items:
        groups.setdefault((item.task, item.question_type, item.match), []).append(item)
    rng = random.Random(seed)
    for queue in groups.values():
        rng.shuffle(queue)
    picked: list[JudgeItem] = []
    while len(picked) < limit and any(groups.values()):
        for queue in groups.values():
            if queue and len(picked) < limit:
                picked.append(queue.pop())
    return picked


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
        "--reasoning-fraction",
        default=None,
        help="Overrides the config's reasoning_fraction: run an evenly sampled share of the "
        "benchmark, e.g. 0.5 or 1/3",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Smoke test or pilot: judge only this many sampled responses, spread round-robin across tasks and correct/incorrect",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build every prompt (and render every image) without sending anything",
    )
    args = parser.parse_args()

    try:
        config = load_reasoning_config(resolve_reasoning_path(args.config))
        overrides = {
            "run_name": args.run_name,
            "concurrency": args.concurrency,
            "reasoning_fraction": args.reasoning_fraction,
        }
        overrides = {k: v for k, v in overrides.items() if v is not None}
        if overrides:
            config = type(config).model_validate({**config.model_dump(), **overrides})
        print(f"Configuration is as follows:\n{config}")
        results = get_question_samples(
            str(config.results_file),
            runs=config.responses_to_evaluate,
            fraction=config.reasoning_fraction,
            wrong_per_task=config.wrong_per_task,
            tasks=config.enabled_tasks(),
        )
        config.run_dir().mkdir(parents=True, exist_ok=True)
        conn = connect(config.resolved_database())
        builder = JudgeBuilder(
            conn,
            str(config.resolved_generation_config()),
            load_species_descriptions(config.species_descriptions),
        )
        builder.check_descriptions(results)
        # How many total questions do we have to review?
        print(f"Questions for review for frac={config.reasoning_fraction}: {len(results)}")

        if args.limit is not None:
            results = spread(results, args.limit)
            print(f"--limit {args.limit}: judging {len(results)} responses")

        if args.dry_run:
            for item in results:
                for kind in JUDGE_KINDS:
                    prompt = builder.build(item, kind)
                    if prompt.attach_image:
                        builder.image_for(item)
            print(f"Dry run: built {len(results) * len(JUDGE_KINDS)} prompts without errors")
            return 0

        result = asyncio.run(run_judges(config, results, builder))
        if result.fatal_message:
            print(f"error: {result.fatal_message}", file=sys.stderr)
        for message, count in result.top_errors:
            print(f"  {count}x {message}", file=sys.stderr)
        if result.state == "interrupted":
            return 130
        return 0 if result.ok else 1

    except (ValidationError, FileNotFoundError, ValueError, FatalRunError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130
    return 0

if __name__ == "__main__":
    sys.exit(main())