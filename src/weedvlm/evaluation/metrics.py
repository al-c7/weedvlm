"""
Aggregates scored responses into per-model, per-suite metrics (see
weedvlm.evaluation.scoring.suite_name for what a suite is).

Accuracy is reported two ways, because failed requests (no usable
response) say more about the run's settings than about the model:
    accuracy            correct / questions the model actually answered
    accuracy_all        correct / every question, failures counting as wrong
Each has a 95% Wilson interval where it's a proportion of answered
questions. "Unsure" and invalid answers count as wrong in both.

Macro-F1 (docs/SPECIES-ID.md: to handle class imbalance) is averaged over
the ground-truth classes that occur, treating unsure/invalid answers as
a prediction of no class -- they cost recall but never precision.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from weedvlm.evaluation.scoring import DENSITY_CATEGORIES, ScoredResponse

# Suites where "which class was predicted" is meaningful enough for F1.
# (Open-ended answers are free text; grounded VQA's answer is a box
# number, not a class.)
_F1_SUITES = ("species_id/multiple_choice", "fine_grained", "density")


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    if n == 0:
        return None
    p = successes / n
    denominator = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def macro_f1(expected: pd.Series, predicted: pd.Series) -> float | None:
    """Macro-averaged F1 over the classes in `expected`. `predicted` is
    NaN/None where no class was predicted."""
    scores = []
    for cls in expected.unique():
        true_positive = int(((expected == cls) & (predicted == cls)).sum())
        false_positive = int(((expected != cls) & (predicted == cls)).sum())
        false_negative = int(((expected == cls) & (predicted != cls)).sum())
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(2 * true_positive / denominator if denominator else 0.0)
    return sum(scores) / len(scores) if scores else None


def _accuracy_block(group: pd.DataFrame) -> dict[str, Any]:
    answered = group[group["status"] == "ok"]
    n_total, n_answered = len(group), len(answered)
    correct = int((answered["outcome"] == "correct").sum())
    interval = wilson_interval(correct, n_answered)
    return {
        "n_total": n_total,
        "n_answered": n_answered,
        "n_failed": n_total - n_answered,
        "correct": correct,
        "incorrect": int((answered["outcome"] == "incorrect").sum()),
        "unsure": int((answered["outcome"] == "unsure").sum()),
        "invalid": int((answered["outcome"] == "invalid").sum()),
        "accuracy": correct / n_answered if n_answered else None,
        "accuracy_ci95": list(interval) if interval else None,
        "accuracy_all": correct / n_total if n_total else None,
        "unsure_rate": int((answered["outcome"] == "unsure").sum()) / n_answered if n_answered else None,
    }


def _breakdown(group: pd.DataFrame, by: str) -> dict[str, dict[str, Any]]:
    answered = group[group["status"] == "ok"]
    result = {}
    for key, part in answered.groupby(by):
        correct = int((part["outcome"] == "correct").sum())
        result[str(key)] = {"n": len(part), "correct": correct, "accuracy": correct / len(part)}
    return result


def _density_block(group: pd.DataFrame) -> dict[str, Any]:
    answered = group[group["status"] == "ok"]

    def error_stats(column: str) -> dict[str, Any]:
        values = answered[column].dropna()
        return {
            "n": len(values),
            "mean_abs_error": float(values.abs().mean()) if len(values) else None,
            "median_abs_error": float(values.abs().median()) if len(values) else None,
            "mean_signed_error": float(values.mean()) if len(values) else None,
        }

    confusion = {
        truth: {
            pred: int(((answered["expected"] == truth) & (answered["predicted_label"] == pred)).sum())
            for pred in DENSITY_CATEGORIES
        }
        for truth in DENSITY_CATEGORIES
    }
    ranks = answered["category_rank_error"].dropna()
    return {
        "count": error_stats("count_error"),
        "coverage_pp": error_stats("coverage_error_pp"),
        "mean_category_distance": float(ranks.mean()) if len(ranks) else None,
        "confusion_truth_by_predicted": confusion,
    }


def _suite_metrics(suite: str, group: pd.DataFrame) -> dict[str, Any]:
    metrics = _accuracy_block(group)
    chance = group["chance"].dropna()
    metrics["chance_accuracy"] = float(chance.mean()) if len(chance) else None

    answered = group[group["status"] == "ok"]
    if suite.startswith(_F1_SUITES) and len(answered):
        metrics["macro_f1"] = macro_f1(answered["expected"], answered["predicted_label"])
    else:
        metrics["macro_f1"] = None

    metrics["by_class"] = _breakdown(group, "benchmark_class")
    metrics["by_dataset"] = _breakdown(group, "dataset")
    if suite.startswith("density"):
        metrics["density"] = _density_block(group)
    return metrics


def _mc_vs_open_ended(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """For each image condition, the same questions asked both ways: how
    much of the multiple-choice accuracy survives without the options
    (the point of the open-ended ablation)."""
    result = {}
    open_ended = frame[frame["suite"].str.startswith("species_id/open_ended")]
    multiple_choice = frame[frame["suite"].str.startswith("species_id/multiple_choice")]
    paired = open_ended.merge(
        multiple_choice, left_on="ablation_of", right_on="question_id", suffixes=("_oe", "_mc")
    )
    paired = paired[(paired["status_oe"] == "ok") & (paired["status_mc"] == "ok")]
    for annotated, part in paired.groupby("annotated_oe"):
        result["boxed" if annotated else "unannotated"] = {
            "n_pairs": len(part),
            "multiple_choice_accuracy": float((part["outcome_mc"] == "correct").mean()),
            "open_ended_accuracy": float((part["outcome_oe"] == "correct").mean()),
        }
    return result


def _usage_block(group: pd.DataFrame) -> dict[str, Any]:
    ok = group[group["status"] == "ok"]
    failed = group[group["status"] == "failed"]
    return {
        "requests": len(group),
        "failed": len(failed),
        "total_cost_usd": float(group["cost_usd"].sum()),
        "mean_cost_usd": float(group["cost_usd"].mean()) if group["cost_usd"].notna().any() else None,
        "mean_total_tokens": float(group["total_tokens"].mean()) if group["total_tokens"].notna().any() else None,
        "mean_latency_seconds": float(ok["latency_seconds"].mean()) if len(ok) else None,
        "retried_requests": int((group["attempts"] > 1).sum()),
    }


def compute_metrics(scored: list[ScoredResponse]) -> dict[str, Any]:
    frame = pd.DataFrame([vars(s) for s in scored])
    models: dict[str, Any] = {}
    for model, model_frame in frame.groupby("model"):
        models[str(model)] = {
            "usage": _usage_block(model_frame),
            "suites": {
                str(suite): _suite_metrics(str(suite), group)
                for suite, group in model_frame.groupby("suite")
            },
            "species_id_multiple_choice_vs_open_ended": _mc_vs_open_ended(model_frame),
        }
    return models
