"""Share of incorrect responses carrying each error flag (report Table 17), strict vs lenient."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = Path(__file__).parent
OUT = HERE / "error_flags_by_model"

# Values transcribed from Table 17 of the report (% of incorrect responses).
# Per model: (strict, lenient) for perception, knowledge, evaluation; then unsure.
TABLE = {
    "Claude Haiku 5.5": {"perception": (11.8, 34.9), "knowledge": (19.7, 30.9), "evaluation": (0.6, 7.1), "unsure": 49.3},
    "DeepSeek V4.1 Flash": {"perception": (29.7, 58.7), "knowledge": (31.6, 41.3), "evaluation": (0.0, 2.4), "unsure": 21.5},
    "Gemma 4 31B": {"perception": (43.1, 67.3), "knowledge": (48.4, 54.0), "evaluation": (0.3, 2.8), "unsure": 6.3},
    "GPT-6 Luna": {"perception": (32.9, 58.2), "knowledge": (35.7, 45.4), "evaluation": (0.0, 1.4), "unsure": 15.7},
    "GPT-6.1 Sol (subset)": {"perception": (25.4, 49.3), "knowledge": (22.1, 29.3), "evaluation": (0.0, 4.0), "unsure": 23.9},
}
# Strict is the darker shade, lenient the lighter shade of each flag's hue.
FLAGS = {
    "perception": ("Perception", "#8DBAD9", "#C5DDEE"),
    "knowledge": ("Knowledge", "#EFA97B", "#F9D9C3"),
    "evaluation": ("Evaluation", "#B79DDB", "#DDD0EF"),
}
UNSURE_COLOUR = "#F2CF72"
INK, MUTED, GRID = "#2B2B2B", "#6B6B6B", "#E6E6E6"


def main() -> None:
    for weight in ("Regular", "Bold"):
        font_manager.fontManager.addfont(f"/usr/share/fonts/liberation-sans-fonts/LiberationSans-{weight}.ttf")
    plt.rcParams.update({"font.family": "Liberation Sans", "text.color": INK, "axes.labelcolor": INK,
                         "xtick.color": INK, "ytick.color": MUTED})

    # (key, which value, colour) for each bar in a group, in display order
    bars = []
    for key, (_, strict, lenient) in FLAGS.items():
        bars += [(key, 0, strict), (key, 1, lenient)]
    bars.append(("unsure", None, UNSURE_COLOUR))

    width, gap, pair_gap = 0.115, 0.008, 0.035
    offsets, x = [], 0.0
    for i, (key, _, _) in enumerate(bars):
        offsets.append(x)
        x += width + gap + (pair_gap if i % 2 == 1 and i < len(bars) - 1 else 0)
    centre = (offsets[0] + offsets[-1]) / 2
    offsets = [o - centre for o in offsets]

    fig, ax = plt.subplots(figsize=(13, 5.6), dpi=200)
    for j, row in enumerate(TABLE.values()):
        for off, (key, idx, colour) in zip(offsets, bars):
            v = row[key] if idx is None else row[key][idx]
            ax.bar(j + off, v, width, color=colour, edgecolor="white", linewidth=1.2, zorder=3)
            ax.text(j + off, v + 0.8, f"{v:.1f}", ha="center", va="bottom", fontsize=8, color=INK)

    ax.set_xticks(range(len(TABLE)))
    ax.set_xticklabels(list(TABLE), fontsize=11)
    ax.set_ylim(0, 75)
    ax.set_yticks(range(0, 71, 10))
    ax.set_yticklabels([f"{t}%" for t in range(0, 71, 10)], fontsize=10)
    ax.set_ylabel("Share of incorrect responses", fontsize=11)
    ax.set_title("Error flags among incorrect responses, by model", fontsize=14, fontweight="bold", pad=38, loc="left")
    ax.yaxis.grid(True, color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#BBBBBB")
    ax.tick_params(axis="both", length=0)

    handles, labels = [], []
    for _, (label, strict, lenient) in FLAGS.items():
        handles += [plt.Rectangle((0, 0), 1, 1, color=strict), plt.Rectangle((0, 0), 1, 1, color=lenient)]
        labels += [f"{label} (strict)", f"{label} (lenient)"]
    handles.append(plt.Rectangle((0, 0), 1, 1, color=UNSURE_COLOUR))
    labels.append("Unsure")
    ax.legend(handles, labels, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=7, frameon=False, fontsize=9.5,
              handlelength=1.1, handleheight=1.1, columnspacing=1.0, handletextpad=0.5, borderaxespad=0.2)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT.with_suffix(f".{ext}"), bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
