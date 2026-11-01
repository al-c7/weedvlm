"""
Appends the reasoning scores from score-reasoning.py to the scored responses
parquet table (notebooks/weedvlm_scored_responses.parquet by default), for
up to two judges side by side.

Each judge is one or more reasoning run directories, given with --judge1 and
--judge2. Each judgement is joined to its response by `response_id`, which is
the table's `id`. For each judge the table gains, prefixed `judge1_` or
`judge2_`:
    interpretation, consistency, knowledge, visual_evidence
                    the scores (knowledge only exists for questions that name
                    a species; visual evidence only comes from the visual
                    judge)
    textual_model   the model that gave the textual scores
    visual_model    the model that gave the visual evidence score
    evaluated       true for every response kept for this judge, false otherwise
A response the judge did not sample has null scores and models.

A response is kept only if every judge given has both its textual and its
visual judgement for it. If any judgement is missing or failed, the response
is left out for all judges (null scores and models, evaluated false), so that
the judges are always compared on the same responses. The ids left out are
printed.

Only judgements with status ok are used. If a response has several for the
same judge kind within one judge (several directories, or a resumed run), the
latest by `judged_at` wins. A judgement whose response is missing from the
table, or whose run or question differs from it, stops the script rather than
appending scores to the wrong rows.

Running it again replaces the judge1_ and judge2_ columns (and the
reasoning_ columns an earlier version of this script wrote) rather than
adding to them, so it is safe to re-run with different directories.
Judgements written before grounding was renamed visual-evidence are read as
visual evidence.

Usage:
    uv run src/weedvlm/scripts/append-reasoning-scores.py --judge1 reasoning-results/gemini-judge --judge2 reasoning-results/glm-heavy
    uv run src/weedvlm/scripts/append-reasoning-scores.py --judge1 reasoning-results/a reasoning-results/b --output scored.parquet
"""

import argparse
import sys
from pathlib import Path

import polars as pl

JUDGE_PREFIXES = ("judge1_", "judge2_")
# Columns written by an earlier version of this script.
LEGACY_PREFIX = "reasoning_"
# Names the judges used before the rename, to their current name.
LEGACY_SCORE_NAMES = {"grounding": "visual-evidence"}
KINDS = ("textual", "visual")
KEYS = ["response_id", "run", "question_id"]


def load_judgements(directories: list[Path], prefix: str) -> pl.DataFrame:
    """One row per response with the `prefix` score and model columns, from
    the judgements.jsonl in each directory."""
    frames = []
    for directory in directories:
        path = directory / "judgements.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"No judgements.jsonl in {directory}")
        frames.append(pl.read_ndjson(path, infer_schema_length=None))
    judgements = pl.concat(frames, how="diagonal_relaxed").filter(pl.col("status") == "ok")
    if judgements.is_empty():
        raise ValueError(f"{prefix}: no judgements with status ok")

    latest = judgements.sort("judged_at").unique(["response_id", "kind"], keep="last")
    long = (
        latest.select(*KEYS, "scores")
        .unnest("scores")
        .unpivot(index=KEYS, variable_name="score")
        .drop_nulls("value")
        .with_columns(
            pl.concat_str(
                pl.lit(prefix),
                pl.col("score").replace(LEGACY_SCORE_NAMES).str.replace_all("-", "_"),
            ).alias("score")
        )
    )
    scores = long.pivot(on="score", index=KEYS, values="value", aggregate_function="first")
    models = latest.select(*KEYS, "kind", "judge_model").pivot(
        on="kind", index=KEYS, values="judge_model", aggregate_function="first"
    )
    models = models.with_columns(
        [pl.lit(None, dtype=pl.String).alias(kind) for kind in KINDS if kind not in models.columns]
    ).rename({kind: f"{prefix}{kind}_model" for kind in KINDS})
    return models.join(scores, on=KEYS, how="left").sort("response_id")


def check_against_table(judge: pl.DataFrame, table: pl.DataFrame, name: str, results: Path) -> str | None:
    """A message if any judgement is for a response the table lacks or has
    under a different run or question, otherwise None."""
    joined = judge.join(
        table.select("id", "run", "question_id"), left_on="response_id", right_on="id", how="left", suffix="_table"
    )
    missing = joined.filter(pl.col("run_table").is_null() & pl.col("question_id_table").is_null())
    if not missing.is_empty():
        return (
            f"{name}: {missing.height} judged responses are not in {results}, "
            f"e.g. response_id {missing['response_id'].head(5).to_list()}"
        )
    mismatched = joined.filter(
        (pl.col("run") != pl.col("run_table")) | (pl.col("question_id") != pl.col("question_id_table"))
    )
    if not mismatched.is_empty():
        return (
            f"{name}: {mismatched.height} judgements disagree with the table on run or question, "
            f"e.g. response_id {mismatched['response_id'].head(5).to_list()}; "
            "the results table may have been regenerated since they were judged"
        )
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--judge1",
        nargs="+",
        type=Path,
        required=True,
        metavar="DIR",
        help="Reasoning run directories holding judge 1's judgements.jsonl",
    )
    parser.add_argument(
        "--judge2",
        nargs="+",
        type=Path,
        default=None,
        metavar="DIR",
        help="Reasoning run directories holding judge 2's judgements.jsonl",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("notebooks/weedvlm_scored_responses.parquet"),
        help="The scored responses parquet to read (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to write the table (default: overwrite --results)",
    )
    args = parser.parse_args()

    sources = {"judge1_": args.judge1}
    if args.judge2:
        sources["judge2_"] = args.judge2
    try:
        judges = {prefix: load_judgements(dirs, prefix) for prefix, dirs in sources.items()}
        table = pl.read_parquet(args.results)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    table = table.drop(
        [c for c in table.columns if c.startswith((LEGACY_PREFIX, *JUDGE_PREFIXES))]
    )
    for prefix, judge in judges.items():
        if problem := check_against_table(judge, table, prefix.rstrip("_"), args.results):
            print(f"error: {problem}", file=sys.stderr)
            return 1

    complete = [
        set(
            judge.filter(
                pl.col(f"{prefix}textual_model").is_not_null()
                & pl.col(f"{prefix}visual_model").is_not_null()
            )["response_id"]
        )
        for prefix, judge in judges.items()
    ]
    kept = set.intersection(*complete)
    left_out = sorted(set.union(*(set(judge["response_id"]) for judge in judges.values())) - kept)
    judges = {
        prefix: judge.filter(pl.col("response_id").is_in(list(kept)))
        for prefix, judge in judges.items()
    }

    output = table
    for prefix, judge in judges.items():
        columns = [c for c in judge.columns if c.startswith(prefix)]
        output = output.join(
            judge.select("response_id", *columns), left_on="id", right_on="response_id", how="left"
        ).with_columns(pl.col("id").is_in(judge["response_id"].implode()).alias(f"{prefix}evaluated"))
    assert output.height == table.height, "join changed the number of rows"

    destination = args.output or args.results
    temporary = destination.with_name(destination.name + ".tmp")
    output.write_parquet(temporary)
    temporary.replace(destination)

    print(f"Wrote {destination}: {output.height} rows")
    if left_out:
        print(f"  left out, missing a judgement from at least one judge: response_id {left_out}")
    for prefix in judges:
        columns = [c for c in output.columns if c.startswith(prefix)]
        print(f"  {prefix.rstrip('_')}: {output[f'{prefix}evaluated'].sum()} responses scored")
        for column in columns:
            if column != f"{prefix}evaluated":
                print(f"    {column}: {output[column].drop_nulls().len()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
