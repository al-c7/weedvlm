"""Summary figure of the metrics from weedvlm.evaluation.metrics: a stack
of tables (accuracy per suite, multiple choice vs open-ended, density
errors, cost/latency), one column group per model, saved as an image."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_HEADER_BG = "#2f3e46"
_BEST_BG = "#d8ecd8"
_STRIPE_BG = "#f3f5f6"
_ROW_HEIGHT = 0.34  # inches
_HEADER_ROWS = 2.0  # a header row is this many body rows tall (two-line headers)
_TITLE_INCHES = 0.55  # room above each table for its title


def _pct(value: float | None, digits: int = 1) -> str:
    return "--" if value is None else f"{100 * value:.{digits}f}%"


def _num(value: float | None, digits: int = 2) -> str:
    return "--" if value is None else f"{value:.{digits}f}"


def _accuracy_cell(suite: dict[str, Any] | None) -> str:
    if suite is None:
        return "--"
    if suite["accuracy"] is None:
        return f"all {suite['n_failed']} failed"
    low, high = suite["accuracy_ci95"]
    text = f"{100 * suite['accuracy']:.1f}%  ({100 * low:.0f}-{100 * high:.0f})"
    return f"{text}  [{suite['n_failed']} failed]" if suite["n_failed"] else text


def _accuracy_table(models: dict[str, Any]) -> tuple[list[str], list[list[str]], dict[int, set[int]]]:
    """Rows are suites; per model an accuracy (95% CI) and a macro-F1 column.
    The third value maps row -> columns to highlight (best accuracy per row)."""
    suites = sorted({name for m in models.values() for name in m["suites"]})
    header = ["Suite", "n", "Chance"]
    for model in models:
        header += [f"{model}\naccuracy (95% CI)", "macro-F1"]

    rows, best = [], {}
    for r, suite in enumerate(suites):
        per_model = [m["suites"].get(suite) for m in models.values()]
        first = next(s for s in per_model if s is not None)
        row = [suite, str(first["n_total"]), _pct(first["chance_accuracy"], 0)]
        for s in per_model:
            row += [_accuracy_cell(s), _num(s["macro_f1"]) if s else "--"]
        rows.append(row)

        # Ranked with failed requests counted as wrong, so a model that
        # only answered a handful of questions can't win on those alone.
        accuracies = [s["accuracy_all"] if s else -1 for s in per_model]
        if sum(a >= 0 for a in accuracies) > 1:
            top = max(accuracies)
            best[r] = {3 + 2 * i for i, a in enumerate(accuracies) if a == top}
    return header, rows, best


def _mc_vs_open_ended_table(models: dict[str, Any]) -> tuple[list[str], list[list[str]]]:
    header = ["Species ID, same questions asked both ways", "pairs"]
    for model in models:
        header += [f"{model}\nmultiple choice", "open-ended", "drop"]

    conditions = sorted({c for m in models.values() for c in m["species_id_multiple_choice_vs_open_ended"]})
    rows = []
    for condition in conditions:
        pairs = [m["species_id_multiple_choice_vs_open_ended"].get(condition) for m in models.values()]
        first = next(p for p in pairs if p is not None)
        row = [condition, str(first["n_pairs"])]
        for p in pairs:
            if p is None:
                row += ["--", "--", "--"]
                continue
            drop = p["multiple_choice_accuracy"] - p["open_ended_accuracy"]
            row += [
                _pct(p["multiple_choice_accuracy"]),
                _pct(p["open_ended_accuracy"]),
                f"{100 * drop:+.1f} pp",
            ]
        rows.append(row)
    return header, rows


def _density_table(models: dict[str, Any]) -> tuple[list[str], list[list[str]]]:
    header = [
        "Model",
        "Condition",
        "Category\ndistance",
        "Count MAE",
        "Count bias",
        "Coverage MAE (pp)",
        "Coverage bias (pp)",
    ]
    rows = []
    for model, m in models.items():
        for suite, s in sorted(m["suites"].items()):
            if "density" not in s:
                continue
            d = s["density"]
            rows.append(
                [
                    model,
                    suite.split("/", 1)[1],
                    _num(d["mean_category_distance"]),
                    _num(d["count"]["mean_abs_error"], 1),
                    f"{d['count']['mean_signed_error']:+.1f}",
                    _num(d["coverage_pp"]["mean_abs_error"], 1),
                    f"{d['coverage_pp']['mean_signed_error']:+.1f}",
                ]
            )
    return header, rows


def _usage_table(models: dict[str, Any]) -> tuple[list[str], list[list[str]]]:
    header = ["Model", "Responses", "Failed", "Total cost", "Cost / request", "Mean tokens", "Mean latency"]
    rows = []
    for model, m in models.items():
        u = m["usage"]
        rows.append(
            [
                model,
                str(u["requests"]),
                str(u["failed"]),
                f"${u['total_cost_usd']:.2f}",
                f"${_num(u['mean_cost_usd'], 4)}",
                _num(u["mean_total_tokens"], 0),
                f"{_num(u['mean_latency_seconds'], 1)}s",
            ]
        )
    return header, rows


def _draw_table(
    ax: plt.Axes,
    title: str,
    header: list[str],
    rows: list[list[str]],
    highlight: dict[int, set[int]] | None = None,
) -> None:
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=8)
    if not rows:
        ax.text(0, 0.5, "no data", fontsize=9, color="gray")
        return

    table = ax.table(cellText=rows, colLabels=header, cellLoc="center", bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.auto_set_column_width(list(range(len(header))))
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#d0d5d8")
        if row == 0:
            cell.set_facecolor(_HEADER_BG)
            cell.set_text_props(color="white", fontweight="bold")
            cell.set_height(_HEADER_ROWS / (len(rows) + _HEADER_ROWS))
        else:
            cell.set_height(1 / (len(rows) + _HEADER_ROWS))
            cell.set_facecolor(_STRIPE_BG if row % 2 == 0 else "white")
            if highlight and col in highlight.get(row - 1, ()):
                cell.set_facecolor(_BEST_BG)
                cell.set_text_props(fontweight="bold")


def render_summary_figure(models: dict[str, Any], run_name: str, out_path: Path) -> None:
    sections = [
        ("Accuracy by suite (shaded: best model per row, failed requests counted as wrong)", *_accuracy_table(models)),
        ("Multiple choice vs open-ended", *_mc_vs_open_ended_table(models), None),
        ("Density errors", *_density_table(models), None),
        ("Cost and latency", *_usage_table(models), None),
    ]

    heights = [(len(rows) + _HEADER_ROWS) * _ROW_HEIGHT + _TITLE_INCHES for _, _, rows, _ in sections]
    fig, axes = plt.subplots(
        len(sections),
        1,
        figsize=(13, sum(heights) + 1.0),
        gridspec_kw={"height_ratios": heights},
    )
    fig.suptitle(
        f"{run_name}: evaluation summary", x=0.02, ha="left", fontsize=15, fontweight="bold"
    )
    for ax, (title, header, rows, highlight) in zip(axes, sections):
        _draw_table(ax, title, header, rows, highlight)

    fig.text(
        0.02,
        0.005,
        "Accuracy = correct / answered; unsure and invalid answers count as wrong. "
        "Intervals are 95% Wilson. Bias is predicted minus true.",
        fontsize=8,
        color="gray",
    )
    fig.tight_layout(rect=(0.01, 0.02, 0.99, 0.96), h_pad=1.6)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
