"""
Confidence-interval helpers for the evaluation results.

    wilson_table        proportions (accuracy, error-flag and category rates), Wilson 95% CI
    bootstrap_ci        any statistic of a frame (a mean, a kappa, ...), percentile bootstrap
    bootstrap_by        bootstrap_ci per group (e.g. per model)
    paired_diff_ci      model A minus model B on the same questions, bootstrapped over questions

Resampling unit: the bootstrap resamples whole clusters. Each question uses its
own source image, so within one model the cluster can simply be the response
(`cluster=None`). Pass a cluster column when rows are not independent, e.g.
when pooling models, where the same question appears once per model.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import polars as pl

from weedvlm.evaluation.metrics import wilson_interval

DEFAULT_BOOTSTRAPS = 2000
DEFAULT_SEED = 69420


def wilson_table(
    df: pl.DataFrame,
    by: str | Sequence[str],
    success: pl.Expr,
    z: float = 1.96,
) -> pl.DataFrame:
    """Per group: n, k (rows where `success` is true), rate and the Wilson interval.

    Null `success` values are dropped from n, so they aren't counted as failures.
    e.g. wilson_table(loc, "model", pl.col("loc_outcome") == "correct")
    """
    by = [by] if isinstance(by, str) else list(by)
    counts = (
        df.group_by(by, maintain_order=True)
        .agg(n=success.drop_nulls().len(), k=success.drop_nulls().sum())
        .sort(by)
    )
    bounds = [
        wilson_interval(int(k), int(n), z) if n else (None, None)
        for k, n in zip(counts["k"], counts["n"])
    ]
    return counts.with_columns(
        rate=pl.when(pl.col("n") > 0).then(pl.col("k") / pl.col("n")),
        lo=pl.Series([b[0] for b in bounds], dtype=pl.Float64),
        hi=pl.Series([b[1] for b in bounds], dtype=pl.Float64),
    )


def _resample_indices(
    cluster_rows: list[np.ndarray], rng: np.random.Generator
) -> np.ndarray:
    picks = rng.integers(0, len(cluster_rows), size=len(cluster_rows))
    return np.concatenate([cluster_rows[i] for i in picks])


def _clusters(df: pl.DataFrame, cluster: str | pl.Expr | None) -> list[np.ndarray]:
    if cluster is None:
        return [np.array([i]) for i in range(df.height)]
    keys = df.select(cluster.alias("_c") if isinstance(cluster, pl.Expr) else pl.col(cluster).alias("_c"))
    grouped = keys.with_row_index("_i").group_by("_c").agg(pl.col("_i"))
    return [np.asarray(rows, dtype=np.int64) for rows in grouped["_i"].to_list()]


def bootstrap_ci(
    df: pl.DataFrame,
    stat: Callable[[pl.DataFrame], float | None],
    cluster: str | pl.Expr | None = None,
    n_boot: int = DEFAULT_BOOTSTRAPS,
    alpha: float = 0.05,
    seed: int = DEFAULT_SEED,
) -> tuple[float | None, float | None, float | None]:
    """(estimate, lo, hi): `stat` on the full frame and its percentile
    bootstrap interval. `stat` takes a frame and returns a number, e.g.
        lambda d: d["score"].mean()
        lambda d: cohen_kappa_score(d["a"], d["b"], weights="quadratic")
    Resamples with replacement over clusters (rows if `cluster` is None).
    Replicates where `stat` returns None/NaN (e.g. a degenerate kappa) are skipped."""
    if df.is_empty():
        return None, None, None
    rng = np.random.default_rng(seed)
    rows = _clusters(df, cluster)
    replicates = []
    for _ in range(n_boot):
        value = stat(df[_resample_indices(rows, rng)])
        if value is not None and np.isfinite(value):
            replicates.append(value)
    estimate = stat(df)
    if len(replicates) < n_boot * 0.9:
        # Too many degenerate resamples for the interval to be trusted.
        return estimate, None, None
    lo, hi = np.quantile(replicates, [alpha / 2, 1 - alpha / 2])
    return estimate, float(lo), float(hi)


def bootstrap_by(
    df: pl.DataFrame,
    by: str | Sequence[str],
    stat: Callable[[pl.DataFrame], float | None],
    cluster: str | pl.Expr | None = None,
    **kwargs,
) -> pl.DataFrame:
    """bootstrap_ci for each group, as a frame of by-columns, n, estimate, lo, hi."""
    by = [by] if isinstance(by, str) else list(by)
    rows = []
    for key, part in df.group_by(by, maintain_order=True):
        estimate, lo, hi = bootstrap_ci(part, stat, cluster, **kwargs)
        rows.append({**dict(zip(by, key)), "n": part.height, "estimate": estimate, "lo": lo, "hi": hi})
    return pl.DataFrame(rows).sort(by)


def paired_diff_ci(
    df: pl.DataFrame,
    group: str,
    a: str,
    b: str,
    value: str,
    id_col: str = "question_id",
    n_boot: int = DEFAULT_BOOTSTRAPS,
    alpha: float = 0.05,
    seed: int = DEFAULT_SEED,
) -> tuple[float | None, float | None, float | None, int]:
    """(mean of a - b, lo, hi, n questions) on questions both groups answered,
    bootstrapped over questions. `value` is a number per row (a 0/1 correct
    flag gives the accuracy difference). Tighter and fairer than comparing two
    separate intervals, and the right way to compare against the Sol subset:
    only the questions Sol saw are used.
    e.g. paired_diff_ci(r, "model", "openai/gpt-6-luna", "google/gemma-4-31b-it", "correct")"""
    wide = (
        df.filter(pl.col(group).is_in([a, b]))
        .select(id_col, group, value)
        .pivot(on=group, index=id_col, values=value, aggregate_function="first")
        .drop_nulls()
    )
    if wide.is_empty():
        return None, None, None, 0
    diff = (wide[a].cast(pl.Float64) - wide[b].cast(pl.Float64)).to_numpy()
    rng = np.random.default_rng(seed)
    boots = diff[rng.integers(0, len(diff), size=(n_boot, len(diff)))].mean(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(diff.mean()), float(lo), float(hi), len(diff)
