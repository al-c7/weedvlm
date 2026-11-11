"""Localisation outcome charts: all categories side by side, and correct vs stacked failures."""

from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
from matplotlib import font_manager

HERE = Path(__file__).parent
SCORED = HERE / "weedvlm_scored_responses.parquet"

MODEL_LABELS = {
    "anthropic/claude-haiku-5.5": "Claude Haiku 5.5",
    "deepseek/deepseek-v4.1-flash": "DeepSeek V4.1 Flash",
    "google/gemma-4-31b-it": "Gemma 4 31B",
    "openai/gpt-6-luna": "GPT-6 Luna",
    "openai/gpt-6.1-sol": "GPT-6.1 Sol (subset)",
}
CATEGORIES = {
    "correct": ("Correct", "#B5D8B0"),
    "miss": ("Miss", "#F2AFA6"),
    "poor_placement": ("Poor placement", "#F7DA94"),
    "under_detect": ("Under-detect", "#A9CCE3"),
    "over_detect": ("Over-detect", "#CDBBE6"),
}
FAILURES = [c for c in CATEGORIES if c != "correct"]
INK, MUTED, GRID = "#2B2B2B", "#6B6B6B", "#E6E6E6"
BOXES = pl.List(pl.List(pl.Float64))


def load_rates() -> dict[str, dict[str, float]]:
    """Percentage of each model's localisation responses in each outcome category."""
    df = pl.read_parquet(SCORED).filter(pl.col("task") == "localisation")
    df = df.with_columns(
        n_true=pl.col("boxes").str.json_decode(BOXES).list.len(),
        n_pred=pl.col("pred_boxes").str.json_decode(BOXES).list.len(),
    ).with_columns(
        loc_outcome=pl.when(pl.col("matched_iou").is_null()).then(pl.lit("miss"))
        .when(pl.col("matched_iou") < 0.5).then(pl.lit("poor_placement"))
        .when(pl.col("n_pred") > pl.col("n_true")).then(pl.lit("over_detect"))
        .when(pl.col("n_pred") < pl.col("n_true")).then(pl.lit("under_detect"))
        .otherwise(pl.lit("correct"))
    )
    counts = df.group_by("model", "loc_outcome").len(name="n")
    rates: dict[str, dict[str, float]] = {}
    for model in MODEL_LABELS:
        sub = counts.filter(pl.col("model") == model)
        total = sub["n"].sum()
        by_cat = dict(zip(sub["loc_outcome"], sub["n"]))
        rates[model] = {c: 100 * by_cat.get(c, 0) / total for c in CATEGORIES}
        print(MODEL_LABELS[model], total, {c: round(v, 1) for c, v in rates[model].items()})
    return rates


def style(ax, title: str, ymax: float) -> None:
    ax.set_xticks(range(len(MODEL_LABELS)))
    ax.set_xticklabels(list(MODEL_LABELS.values()), fontsize=11)
    ax.set_ylim(0, ymax)
    ticks = range(0, int(ymax), 20 if ymax > 60 else 10)
    ax.set_yticks(list(ticks))
    ax.set_yticklabels([f"{t}%" for t in ticks], fontsize=10)
    ax.set_ylabel("% of responses", fontsize=11)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=34, loc="left")
    ax.yaxis.grid(True, color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#BBBBBB")
    ax.tick_params(axis="both", length=0)


def legend(ax, keys: list[str]) -> None:
    handles = [plt.Rectangle((0, 0), 1, 1, color=CATEGORIES[k][1]) for k in keys]
    ax.legend(
        handles, [CATEGORIES[k][0] for k in keys], loc="lower left", bbox_to_anchor=(0, 1.0),
        ncol=len(keys), frameon=False, fontsize=11, handlelength=1.2, handleheight=1.2,
    )


def grouped(rates) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.2), dpi=200)
    keys = list(CATEGORIES)
    width = 0.15
    for i, k in enumerate(keys):
        xs = [j + (i - (len(keys) - 1) / 2) * (width + 0.01) for j in range(len(rates))]
        vals = [rates[m][k] for m in rates]
        bars = ax.bar(xs, vals, width, color=CATEGORIES[k][1], edgecolor="white", linewidth=1.2, zorder=3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}", ha="center", va="bottom", fontsize=8.5, color=INK)
    style(ax, "Weed localisation outcomes by model", 90)
    legend(ax, keys)
    fig.tight_layout()
    save(fig, "localisation_outcomes_grouped")


def stacked(rates) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.2), dpi=200)
    width = 0.36
    for j, m in enumerate(rates):
        xc, xf = j - width / 2 - 0.015, j + width / 2 + 0.015
        v = rates[m]["correct"]
        ax.bar(xc, v, width, color=CATEGORIES["correct"][1], edgecolor="white", linewidth=1.5, zorder=3)
        ax.text(xc, v + 1, f"{v:.1f}%", ha="center", va="bottom", fontsize=10, color=INK)
        bottom = 0.0
        for k in FAILURES:
            h = rates[m][k]
            ax.bar(xf, h, width, bottom=bottom, color=CATEGORIES[k][1], edgecolor="white", linewidth=1.5, zorder=3)
            if h >= 3:
                ax.text(xf, bottom + h / 2, f"{h:.1f}", ha="center", va="center", fontsize=9, color=INK)
            bottom += h
        ax.text(xf, bottom + 1, f"{bottom:.1f}%", ha="center", va="bottom", fontsize=10, color=INK)
    style(ax, "Weed localisation: correct vs. failure breakdown", 90)
    legend(ax, list(CATEGORIES))
    fig.tight_layout()
    save(fig, "localisation_outcomes_stacked")


def save(fig, name: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"{name}.{ext}", bbox_inches="tight", facecolor="white")


def main() -> None:
    for weight in ("Regular", "Bold"):
        font_manager.fontManager.addfont(f"/usr/share/fonts/liberation-sans-fonts/LiberationSans-{weight}.ttf")
    plt.rcParams.update({"font.family": "Liberation Sans", "text.color": INK, "axes.labelcolor": INK,
                         "xtick.color": INK, "ytick.color": MUTED})
    rates = load_rates()
    grouped(rates)
    stacked(rates)


if __name__ == "__main__":
    main()
