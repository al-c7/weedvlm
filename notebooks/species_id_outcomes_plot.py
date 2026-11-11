"""Species ID: proportion of correct / incorrect / unsure responses, MCQ and open-ended per model."""

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
FORMATS = {
    "species_id_multiple_choice": "MCQ",
    "species_id_open_ended": "Open-ended",
}
OUTCOMES = {
    "correct": ("Correct", "#B5D8B0"),
    "incorrect": ("Incorrect", "#F2AFA6"),
    "unsure": ("Unsure", "#F7DA94"),
}
# Mirrors _UNSURE in src/weedvlm/evaluation/scoring.py
UNSURE = ["unsure", "unknown", "not sure", "i don't know", "i do not know", "don't know", "cannot tell", "can't tell"]
INK, MUTED, GRID = "#2B2B2B", "#6B6B6B", "#E6E6E6"


def load_proportions() -> dict[tuple[str, str], dict[str, float]]:
    df = pl.read_parquet(SCORED).filter(pl.col("task") == "species_id")
    df = df.with_columns(
        outcome=pl.when(pl.col("answer").str.strip_chars().str.to_lowercase().is_in(UNSURE)).then(pl.lit("unsure"))
        .when(pl.col("match")).then(pl.lit("correct"))
        .otherwise(pl.lit("incorrect"))
    )
    props: dict[tuple[str, str], dict[str, float]] = {}
    for model in MODEL_LABELS:
        for qtype in FORMATS:
            sub = df.filter((pl.col("model") == model) & (pl.col("question_type") == qtype))
            props[model, qtype] = {o: sub.filter(pl.col("outcome") == o).height / sub.height for o in OUTCOMES}
            print(MODEL_LABELS[model], FORMATS[qtype], sub.height,
                  {o: round(v, 3) for o, v in props[model, qtype].items()})
    return props


def main() -> None:
    for weight in ("Regular", "Bold"):
        font_manager.fontManager.addfont(f"/usr/share/fonts/liberation-sans-fonts/LiberationSans-{weight}.ttf")
    plt.rcParams.update({"font.family": "Liberation Sans", "text.color": INK, "axes.labelcolor": INK,
                         "xtick.color": INK, "ytick.color": MUTED})

    props = load_proportions()
    fig, ax = plt.subplots(figsize=(11, 5.4), dpi=200)
    width, gap = 0.36, 0.02
    offsets = {q: (i - 0.5) * (width + gap) for i, q in enumerate(FORMATS)}

    tick_x, tick_labels = [], []
    for j, model in enumerate(MODEL_LABELS):
        for qtype, fmt in FORMATS.items():
            x = j + offsets[qtype]
            bottom = 0.0
            for key, (_, colour) in OUTCOMES.items():
                h = props[model, qtype][key]
                ax.bar(x, h, width, bottom=bottom, color=colour, edgecolor="white", linewidth=1.5, zorder=3)
                if h >= 0.04:
                    ax.text(x, bottom + h / 2, f"{h:.1%}", ha="center", va="center", fontsize=9, color=INK)
                bottom += h
            tick_x.append(x)
            tick_labels.append(fmt)
        ax.text(j, -0.1, MODEL_LABELS[model], ha="center", va="top", fontsize=11, color=INK,
                transform=ax.get_xaxis_transform())

    ax.set_xticks(tick_x)
    ax.set_xticklabels(tick_labels, fontsize=9, color=MUTED)
    ax.set_xlim(-0.6, len(MODEL_LABELS) - 0.4)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=10)
    ax.set_ylabel("Proportion of responses", fontsize=11)
    ax.set_title("Species identification: response breakdown by model",
                 fontsize=14, fontweight="bold", pad=34, loc="left")
    ax.yaxis.grid(True, color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#BBBBBB")
    ax.tick_params(axis="both", length=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c in OUTCOMES.values()]
    ax.legend(handles, [l for l, _ in OUTCOMES.values()], loc="lower left", bbox_to_anchor=(0, 1.0),
              ncol=3, frameon=False, fontsize=11, handlelength=1.2, handleheight=1.2)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"species_id_outcomes.{ext}", bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
