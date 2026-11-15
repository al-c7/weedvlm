"""Grouped bar chart of species ID accuracy (MCQ vs open-ended) per model."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager

HERE = Path(__file__).parent
SCORED = HERE / "weedvlm_scored_responses.parquet"
OUT = HERE / "species_id_accuracy_by_model"

MODEL_LABELS = {
    "anthropic/claude-haiku-5.5": "Claude Haiku 5.5",
    "deepseek/deepseek-v4.1-flash": "DeepSeek V4.1 Flash",
    "google/gemma-4-31b-it": "Gemma 4 31B",
    "openai/gpt-6-luna": "GPT-6 Luna",
    "openai/gpt-6.1-sol": "GPT-6.1 Sol",
}
FORMATS = {
    "species_id_multiple_choice": ("Multiple choice", "#A9CCE3"),
    "species_id_open_ended": ("Open-ended", "#F5C6A5"),
}
INK = "#2B2B2B"
MUTED = "#6B6B6B"
GRID = "#E6E6E6"


def load_accuracy() -> pd.DataFrame:
    df = pd.read_parquet(SCORED)
    df = df[df["task"] == "species_id"]
    acc = (
        df.groupby(["model", "question_type"])["match"]
        .agg(accuracy="mean", n="size")
        .reset_index()
    )
    return acc


def main() -> None:
    font_manager.fontManager.addfont(
        "/usr/share/fonts/liberation-sans-fonts/LiberationSans-Regular.ttf"
    )
    font_manager.fontManager.addfont(
        "/usr/share/fonts/liberation-sans-fonts/LiberationSans-Bold.ttf"
    )
    plt.rcParams.update(
        {
            "font.family": "Liberation Sans",
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": MUTED,
        }
    )

    acc = load_accuracy()
    models = list(MODEL_LABELS)
    n_fmt = len(FORMATS)
    width = 0.36

    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=200)
    for i, (qtype, (label, colour)) in enumerate(FORMATS.items()):
        vals = []
        for m in models:
            row = acc[(acc["model"] == m) & (acc["question_type"] == qtype)]
            vals.append(row["accuracy"].iloc[0] * 100)
        xs = [j + (i - (n_fmt - 1) / 2) * (width + 0.03) for j in range(len(models))]
        bars = ax.bar(
            xs, vals, width, label=label, color=colour, edgecolor="white", linewidth=1.5, zorder=3
        )
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + 1.2,
                f"{v:.1f}%",
                ha="center",
                va="bottom",
                fontsize=10,
                color=INK,
            )

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([MODEL_LABELS[m] for m in models], fontsize=11)
    ax.set_ylim(0, 105)
    ax.set_yticks(range(0, 101, 20))
    ax.set_yticklabels([f"{t}%" for t in range(0, 101, 20)], fontsize=10)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title("Species identification accuracy by model", fontsize=14, fontweight="bold", pad=34, loc="left")
    ax.yaxis.grid(True, color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#BBBBBB")
    ax.tick_params(axis="both", length=0)
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0, 1.0),
        ncol=2,
        frameon=False,
        fontsize=11,
        handlelength=1.2,
        handleheight=1.2,
    )

    fig.tight_layout()
    fig.savefig(OUT.with_suffix(".png"), bbox_inches="tight", facecolor="white")
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    print(acc.to_string())


if __name__ == "__main__":
    main()
