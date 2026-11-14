"""Density estimation outcomes split by annotated vs unannotated questions (tables 15 and 16)."""

import matplotlib.pyplot as plt
import polars as pl
from matplotlib import font_manager

from density_gvqa_plots import (COVERAGE_TOLERANCE, DENSITY_OUTCOMES, HERE, INK, MODEL_LABELS, MUTED,
                                SHORT_LABELS, save, stack, style_axis)

DATA = HERE / "reasoning-results.parquet"
CATEGORY_OUTCOMES = {"correct": ("Correct", DENSITY_OUTCOMES["correct"][1]), "incorrect": ("Incorrect", "#E3E3E3")}
MEASURES = {"category": "Category", "count": "Count", "coverage": "Coverage"}
ALL_OUTCOMES = {**DENSITY_OUTCOMES, "incorrect": CATEGORY_OUTCOMES["incorrect"]}


def load() -> dict[tuple[int, str, str], dict[str, float]]:
    df = pl.read_parquet(DATA).filter(pl.col("task") == "density").with_columns(
        category=pl.when(pl.col("match")).then(pl.lit("correct")).otherwise(pl.lit("incorrect")),
        count=pl.when(pl.col("est_count") == pl.col("true_weed_count")).then(pl.lit("correct"))
        .when(pl.col("est_count") > pl.col("true_weed_count")).then(pl.lit("over_estimated"))
        .otherwise(pl.lit("under_estimated")),
        coverage=pl.when(pl.col("abs_coverage_error_pp") < COVERAGE_TOLERANCE).then(pl.lit("correct"))
        .when(pl.col("coverage_error_pp") > 0).then(pl.lit("over_estimated"))
        .otherwise(pl.lit("under_estimated")),
    )
    props = {}
    for annotated in (1, 0):
        for model in MODEL_LABELS:
            sub = df.filter((pl.col("model") == model) & (pl.col("annotated") == annotated))
            for measure in MEASURES:
                props[annotated, model, measure] = {
                    o: sub.filter(pl.col(measure) == o).height / sub.height for o in ALL_OUTCOMES}
            print(annotated, MODEL_LABELS[model], sub.height,
                  {m: {o: round(v, 4) for o, v in props[annotated, model, m].items() if v} for m in MEASURES})
    return props


def draw(ax, props, annotated: int, title: str) -> None:
    width, gap = 0.27, 0.015
    offsets = {m: (i - 1) * (width + gap) for i, m in enumerate(MEASURES)}
    tick_x, tick_labels = [], []
    for j, model in enumerate(MODEL_LABELS):
        for measure, name in MEASURES.items():
            x = j + offsets[measure]
            outcomes = CATEGORY_OUTCOMES if measure == "category" else DENSITY_OUTCOMES
            stack(ax, x, width, props[annotated, model, measure], outcomes, 8.5, 0.05, 1)
            tick_x.append(x)
            tick_labels.append(name)
        ax.text(j, -0.1, MODEL_LABELS[model], ha="center", va="top", fontsize=11, color=INK,
                transform=ax.get_xaxis_transform())
    ax.set_xticks(tick_x)
    ax.set_xticklabels(tick_labels, fontsize=8.5, color=MUTED)
    ax.set_xlim(-0.6, len(MODEL_LABELS) - 0.4)
    style_axis(ax, "Proportion of responses", title, 14, 34, 10)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c in ALL_OUTCOMES.values()]
    ax.legend(handles, [l for l, _ in ALL_OUTCOMES.values()], loc="lower left", bbox_to_anchor=(0, 1.0),
              ncol=len(ALL_OUTCOMES), frameon=False, fontsize=11, handlelength=1.1, handleheight=1.1,
              columnspacing=1.0, handletextpad=0.5, borderaxespad=0.2)


def main() -> None:
    for weight in ("Regular", "Bold"):
        font_manager.fontManager.addfont(f"/usr/share/fonts/liberation-sans-fonts/LiberationSans-{weight}.ttf")
    plt.rcParams.update({"font.family": "Liberation Sans", "text.color": INK, "axes.labelcolor": INK,
                         "xtick.color": INK, "ytick.color": MUTED})
    props = load()
    for annotated, name, title in ((1, "annotated", "Weed density estimation, annotated questions"),
                                   (0, "unannotated", "Weed density estimation, unannotated questions")):
        fig, ax = plt.subplots(figsize=(11, 5.6), dpi=200)
        draw(ax, props, annotated, title)
        fig.tight_layout()
        save(fig, f"density_outcomes_{name}")


if __name__ == "__main__":
    main()
