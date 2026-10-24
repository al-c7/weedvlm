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

from weedvlm.reasoning.runner import get_question_samples, get_example

from weedvlm.evaluation.runner import FatalRunError, RunResult, run_evaluation

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
            "./notebooks/weedvlm_scored_responses.parquet",
            runs=config.responses_to_evaluate,
            fraction=config.reasoning_fraction
        )
        # How many total questions do we have to review?
        print(f"Questions for review for frac={config.reasoning_fraction}: {len(results)}")

        # Print out some examples
        print(get_example("localisation", results))
        print(get_example("species_id", results))

    except (ValidationError, FileNotFoundError, ValueError, FatalRunError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130
    return 0

if __name__ == "__main__":
    sys.exit(main())