"""
Prints the rationale rubric for every task, read straight from RUBRICS in
weedvlm.reasoning.prompt (the same data the judge system prompts are built
from). Nothing is sent and no API key is needed.

Usage:
    uv run src/weedvlm/scripts/show-rubrics.py
    uv run src/weedvlm/scripts/show-rubrics.py --task density
"""

import argparse

from weedvlm.reasoning.prompt import RUBRICS, Criterion


def _print_criterion(name: str, criterion: Criterion) -> None:
    print(f"\n  {name.replace('_', ' ').title()}")
    if criterion.preamble:
        print(f"    {criterion.preamble}")
    for score, text in enumerate(criterion.levels):
        print(f"    {score}: {text}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--task", choices=sorted(RUBRICS), help="Only print this task's rubric")
    args = parser.parse_args()

    tasks = [args.task] if args.task else list(RUBRICS)
    for task in tasks:
        rubric = RUBRICS[task]
        print("=" * 80)
        print(f"TASK: {task}")
        print("=" * 80)
        print(f"General: {rubric.general}")
        _print_criterion("interpretation", rubric.interpretation)
        _print_criterion("visual_evidence", rubric.visual_evidence)
        if rubric.knowledge is not None:
            _print_criterion("knowledge", rubric.knowledge)
        _print_criterion("consistency", rubric.consistency)
        print()


if __name__ == "__main__":
    main()
