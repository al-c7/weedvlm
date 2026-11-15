"""Density estimation and grounded VQA charts, plus a fine-grained / grounded VQA / density summary row."""

from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
from matplotlib import font_manager

from fine_grained_outcomes_plot import OUTCOMES as FINE_OUTCOMES
from fine_grained_outcomes_plot import load_proportions as load_fine_grained

HERE = Path(__file__).parent
SCORED = HERE / "weedvlm_scored_responses.parquet"

MODEL_LABELS = {
    "anthropic/claude-haiku-5.5": "Claude Haiku 5.5",
    "deepseek/deepseek-v4.1-flash": "DeepSeek V4.1 Flash",
    "google/gemma-4-31b-it": "Gemma 4 31B",
    "openai/gpt-6-luna": "GPT-6 Luna",
    "openai/gpt-6.1-sol": "GPT-6.1 Sol (subset)",
}
SHORT_LABELS = {
    "anthropic/claude-haiku-5.5": "Haiku 5.5",
    "deepseek/deepseek-v4.1-flash": "DeepSeek\nV4.1 Flash",
    "google/gemma-4-31b-it": "Gemma 4\n31B",
    "openai/gpt-6-luna": "GPT-6\nLuna",
    "openai/gpt-6.1-sol": "GPT-6.1 Sol\n(subset)",
}
DENSITY_OUTCOMES = {
    "correct": ("Correct", "#B5D8B0"),
    "over_estimated": ("Over-estimated", "#CDBBE6"),
    "under_estimated": ("Under-estimated", "#A9CCE3"),
}
DENSITY_MEASURES = {"count": "Count", "coverage": "Coverage"}
ACCURACY_COLOUR = "#B5D8B0"
COVERAGE_TOLERANCE = 3.0  # percentage points, as in evaluation.ipynb
INK, MUTED, GRID = "#2B2B2B", "#6B6B6B", "#E6E6E6"


def load_density() -> dict[tuple[str, str], dict[str, float]]:
    df = pl.read_parquet(SCORED).filter(pl.col("task") == "density").with_columns(
        count=pl.when(pl.col("est_count") == pl.col("true_weed_count")).then(pl.lit("correct"))
        .when(pl.col("est_count") > pl.col("true_weed_count")).then(pl.lit("over_estimated"))
        .otherwise(pl.lit("under_estimated")),
        coverage=pl.when(pl.col("abs_coverage_error_pp") < COVERAGE_TOLERANCE).then(pl.lit("correct"))
        .when(pl.col("coverage_error_pp") > 0).then(pl.lit("over_estimated"))
        .otherwise(pl.lit("under_estimated")),
    )
    props: dict[tuple[str, str], dict[str, float]] = {}
    for model in MODEL_LABELS:
        sub = df.filter(pl.col("model") == model)
        for measure in DENSITY_MEASURES:
            props[model, measure] = {o: sub.filter(pl.col(measure) == o).height / sub.height for o in DENSITY_OUTCOMES}
            print("density", MODEL_LABELS[model], measure, sub.height,
                  {o: round(v, 3) for o, v in props[model, measure].items()})
    return props


def load_gvqa_accuracy() -> dict[str, float]:
    df = pl.read_parquet(SCORED).filter(pl.col("task") == "grounded_vqa")
    acc = {m: df.filter(pl.col("model") == m)["match"].mean() for m in MODEL_LABELS}
    print("grounded vqa", {MODEL_LABELS[m]: round(v, 3) for m, v in acc.items()})
    return acc


def style_axis(ax, ylabel: str, title: str, title_size: float, pad: float, tick_size: float) -> None:
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=tick_size)
    ax.set_ylabel(ylabel, fontsize=tick_size + 1)
    ax.set_title(title, fontsize=title_size, fontweight="bold", pad=pad, loc="left")
    ax.yaxis.grid(True, color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#BBBBBB")
    ax.tick_params(axis="both", length=0)


def add_legend(ax, outcomes: dict, fontsize: float) -> None:
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c in outcomes.values()]
    ax.legend(handles, [l for l, _ in outcomes.values()], loc="lower left", bbox_to_anchor=(0, 1.0),
              ncol=len(outcomes), frameon=False, fontsize=fontsize, handlelength=1.1, handleheight=1.1,
              columnspacing=1.0, handletextpad=0.5, borderaxespad=0.2)


def stack(ax, x: float, width: float, props: dict[str, float], outcomes: dict, fontsize: float, min_label: float,
          decimals: int = 1) -> None:
    bottom = 0.0
    for key, (_, colour) in outcomes.items():
        h = props[key]
        ax.bar(x, h, width, bottom=bottom, color=colour, edgecolor="white", linewidth=1.2, zorder=3)
        if h >= min_label:
            ax.text(x, bottom + h / 2, f"{h:.{decimals}%}",
                    ha="center", va="center", fontsize=fontsize, color=INK)
        bottom += h


def draw_fine_grained(ax, props, small: bool = False) -> None:
    labels = SHORT_LABELS if small else MODEL_LABELS
    for j, model in enumerate(MODEL_LABELS):
        stack(ax, j, 0.6, props[model], FINE_OUTCOMES, 12.5 if small else 10, 0.05 if small else 0.04, 0 if small else 1)
    ax.set_xticks(range(len(MODEL_LABELS)))
    ax.set_xticklabels([labels[m] for m in MODEL_LABELS], fontsize=11.5 if small else 11)
    style_axis(ax, "Proportion of responses", "Fine-grained identification" if small else
               "Fine-grained identification: response breakdown by model",
               17 if small else 14, 44 if small else 34, 12 if small else 10)
    add_legend(ax, FINE_OUTCOMES, 12.5 if small else 11)


def draw_gvqa(ax, acc, small: bool = False) -> None:
    labels = SHORT_LABELS if small else MODEL_LABELS
    for j, model in enumerate(MODEL_LABELS):
        ax.bar(j, acc[model], 0.6, color=ACCURACY_COLOUR, edgecolor="white", linewidth=1.2, zorder=3)
        ax.text(j, acc[model] + 0.015, f"{acc[model]:.1%}", ha="center", va="bottom",
                fontsize=13 if small else 10, color=INK)
    ax.set_xticks(range(len(MODEL_LABELS)))
    ax.set_xticklabels([labels[m] for m in MODEL_LABELS], fontsize=11.5 if small else 11)
    style_axis(ax, "Accuracy", "Grounded weed VQA" if small else "Grounded weed VQA accuracy by model",
               17 if small else 14, 44 if small else 34, 12 if small else 10)
    if small:  # keep the legend row so the three panel titles line up
        add_legend(ax, {"correct": ("Accuracy", ACCURACY_COLOUR)}, 12.5)


def draw_density(ax, props, small: bool = False) -> None:
    labels = SHORT_LABELS if small else MODEL_LABELS
    width, gap = (0.44, 0.01) if small else (0.36, 0.02)
    offsets = {m: (i - 0.5) * (width + gap) for i, m in enumerate(DENSITY_MEASURES)}
    tick_x, tick_labels = [], []
    for j, model in enumerate(MODEL_LABELS):
        for measure, name in DENSITY_MEASURES.items():
            x = j + offsets[measure]
            stack(ax, x, width, props[model, measure], DENSITY_OUTCOMES, 10.5 if small else 9, 0.09 if small else 0.04, 0 if small else 1)
            tick_x.append(x)
            tick_labels.append(name[:5] + "." if small and name == "Coverage" else name)
        ax.text(j, -0.19 if small else -0.1, labels[model], ha="center", va="top",
                fontsize=11.5 if small else 11, color=INK, transform=ax.get_xaxis_transform())
    ax.set_xticks(tick_x)
    ax.set_xticklabels(tick_labels, fontsize=10.5 if small else 9, color=MUTED, rotation=90 if small else 0)
    ax.set_xlim(-0.6, len(MODEL_LABELS) - 0.4)
    style_axis(ax, "Proportion of responses", "Weed density estimation" if small else
               "Weed density estimation: response breakdown by model",
               17 if small else 14, 44 if small else 34, 12 if small else 10)
    add_legend(ax, DENSITY_OUTCOMES, 12 if small else 11)


def save(fig, name: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"{name}.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    for weight in ("Regular", "Bold"):
        font_manager.fontManager.addfont(f"/usr/share/fonts/liberation-sans-fonts/LiberationSans-{weight}.ttf")
    plt.rcParams.update({"font.family": "Liberation Sans", "text.color": INK, "axes.labelcolor": INK,
                         "xtick.color": INK, "ytick.color": MUTED})

    fine, density, gvqa = load_fine_grained(), load_density(), load_gvqa_accuracy()

    fig, ax = plt.subplots(figsize=(10, 5.4), dpi=200)
    draw_density(ax, density)
    fig.tight_layout()
    save(fig, "density_outcomes")

    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=200)
    draw_gvqa(ax, gvqa)
    fig.tight_layout()
    save(fig, "grounded_vqa_accuracy")

    fig, axes = plt.subplots(1, 3, figsize=(16, 8), dpi=200)
    draw_fine_grained(axes[0], fine, small=True)
    draw_gvqa(axes[1], gvqa, small=True)
    draw_density(axes[2], density, small=True)
    for ax in axes[1:]:
        ax.set_ylabel("")
    fig.tight_layout(w_pad=2)
    save(fig, "fine_grained_gvqa_density_summary")


if __name__ == "__main__":
    main()
