import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


METRICS = [
    "Source Recall",
    "Contextual Recall",
    "Entity Exact Recall@10",
    "Entity Recall@10",
    "Relation Connectivity Recall@10",
    "Answer Correctness",
    "Faithfulness",
    "Source Provenance Recall",
]

SHORT_METRICS = [
    "SR",
    "CR",
    "EER@10",
    "ER@10",
    "RCR@10",
    "AC",
    "FS",
    "SPR",
]

DELTA_METRICS = [
    "Entity Recall@10",
    "Contextual Recall",
    "Entity Exact Recall@10",
    "Source Provenance Recall",
    "Source Recall",
    "Faithfulness",
    "Relation Connectivity Recall@10",
    "Answer Correctness",
]

GPT = np.array([0.9855, 0.7797, 0.3125, 0.5138, 0.4150, 0.5739, 0.9865, 0.9855])
GEMMA = np.array([0.9783, 0.7174, 0.2858, 0.3681, 0.4963, 0.6870, 0.9909, 0.9783])

MODEL_NAMES = ["OpenAI GPT-5.4 mini", "Google Gemma 4 31B"]

OUT_DIR = Path(__file__).resolve().parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Reuse these semantic colors in future thesis figures.
# Hybrid / GraphRAG is always blue; Naive / vector-only RAG is always orange.
COLORS = {
    "blue": "#4E79A7",
    "orange": "#F28E2B",
    "hybrid": "#4E79A7",
    "naive": "#F28E2B",
    "grid": "#EAEAEA",
    "axis": "#B8B8B8",
    "text": "#5F5F5F",
    "title": "#3F3F3F",
    "fill_blue": "#AFC6DD",
    "fill_orange": "#F8C999",
}

plt.rcParams.update({
    "font.family": "Arial",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "axes.titlesize": 14,
    "axes.labelsize": 10.5,
    "xtick.labelsize": 9.0,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9.5,
    "figure.titlesize": 14,
    "axes.edgecolor": COLORS["axis"],
    "axes.labelcolor": COLORS["text"],
    "xtick.color": COLORS["text"],
    "ytick.color": COLORS["text"],
    "text.color": COLORS["text"],
    "axes.titlecolor": COLORS["title"],
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": COLORS["grid"],
    "grid.linewidth": 0.8,
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
})


def save_chart(fig, basename):
    fig.savefig(OUT_DIR / f"{basename}.png", dpi=450, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_DIR / f"{basename}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_DIR / f"{basename}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def grouped_bar_chart():
    x = np.arange(len(METRICS))
    width = 0.36

    fig, ax = plt.subplots(figsize=(6.9, 3.9))
    bars_gpt = ax.bar(
        x - width / 2,
        GPT,
        width,
        color=COLORS["blue"],
        label=MODEL_NAMES[0],
    )
    bars_gemma = ax.bar(
        x + width / 2,
        GEMMA,
        width,
        color=COLORS["orange"],
        label=MODEL_NAMES[1],
    )

    for bars, values, color in [
        (bars_gpt, GPT, COLORS["blue"]),
        (bars_gemma, GEMMA, COLORS["orange"]),
    ]:
        is_blue = color == COLORS["blue"]
        for bar, value in zip(bars, values):
            is_high = value >= 0.9
            ax.annotate(
                f"{value:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, value),
                xytext=(0, (-5 if is_blue else -18) if is_high else (4 if is_blue else 13)),
                textcoords="offset points",
                ha="center",
                va="top" if is_high else "bottom",
                fontsize=7.7,
                color="white" if is_high else color,
            )

    fig.suptitle("All-Metric Performance by Model", y=0.98, fontweight="bold")
    ax.set_ylabel("Score (0–1)")
    ax.set_xticks(x)
    ax.set_xticklabels(SHORT_METRICS)
    ax.set_ylim(0, 1.12)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")
    ax.set_axisbelow(True)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.055),
        ncol=2,
        frameon=False,
        handlelength=1.4,
        columnspacing=1.6,
    )
    fig.subplots_adjust(top=0.82, bottom=0.15)
    save_chart(fig, "figure_model_metric_comparison")


def delta_chart():
    metric_to_delta = dict(zip(METRICS, GPT - GEMMA))
    deltas = np.array([metric_to_delta[metric] for metric in DELTA_METRICS])
    colors = np.where(deltas >= 0, COLORS["blue"], COLORS["orange"])
    y = np.arange(len(DELTA_METRICS))

    fig, ax = plt.subplots(figsize=(6.9, 4.15))
    bars = ax.barh(y, deltas, color=colors, height=0.72)

    for bar, delta in zip(bars, deltas):
        offset = 3 if delta >= 0 else -3
        ax.annotate(
            f"{delta:+.4f}",
            xy=(delta, bar.get_y() + bar.get_height() / 2),
            xytext=(offset, 0),
            textcoords="offset points",
            ha="left" if delta >= 0 else "right",
            va="center",
            fontsize=9.0,
            color=COLORS["text"],
        )

    ax.axvline(0, color="#666666", linewidth=0.9)
    fig.suptitle("Score Difference by Metric", y=0.98, fontweight="bold")
    ax.set_title(
        "Positive = GPT-5.4 mini leads   ·   Negative = Gemma 4 31B leads",
        pad=10,
        fontsize=9.2,
        fontweight="normal",
        color=COLORS["text"],
    )
    ax.set_xlabel("Difference in score (GPT-5.4 mini − Gemma 4 31B)")
    ax.set_yticks(y)
    ax.set_yticklabels(DELTA_METRICS)
    ax.invert_yaxis()
    ax.set_xlim(-0.16, 0.18)
    ax.set_xticks(np.arange(-0.15, 0.181, 0.05))
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    ax.set_axisbelow(True)
    fig.subplots_adjust(top=0.82)
    save_chart(fig, "figure_model_metric_differences")


def radar_chart():
    angles = np.linspace(0, 2 * np.pi, len(METRICS), endpoint=False)
    closed_angles = np.concatenate([angles, angles[:1]])
    gpt_closed = np.concatenate([GPT, GPT[:1]])
    gemma_closed = np.concatenate([GEMMA, GEMMA[:1]])

    fig, ax = plt.subplots(figsize=(6.9, 6.0), subplot_kw={"polar": True})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.plot(
        closed_angles,
        gpt_closed,
        color=COLORS["blue"],
        linewidth=2.1,
        marker="o",
        markersize=4.5,
        label=MODEL_NAMES[0],
    )
    ax.fill(closed_angles, gpt_closed, color=COLORS["fill_blue"], alpha=0.42)
    ax.plot(
        closed_angles,
        gemma_closed,
        color=COLORS["orange"],
        linewidth=2.1,
        marker="s",
        markersize=4.2,
        label=MODEL_NAMES[1],
    )
    ax.fill(closed_angles, gemma_closed, color=COLORS["fill_orange"], alpha=0.35)

    radar_labels = ["SR", "CR", "EER@10", "ER@10", "RCR@10", "AC", "FS", "SPR"]
    ax.set_xticks(angles)
    ax.set_xticklabels(radar_labels, fontsize=9.0)
    ax.tick_params(axis="x", pad=10)
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0.2, 1.01, 0.2))
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8.5)
    ax.set_rlabel_position(67.5)
    ax.grid(color="#D8D8D8", linewidth=0.8)
    ax.spines["polar"].set_color(COLORS["axis"])
    ax.set_title("Capability Profile by Model", pad=24, fontweight="bold")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.14),
        ncol=2,
        frameon=False,
        handlelength=2.0,
        columnspacing=1.6,
    )
    save_chart(fig, "figure_model_capability_radar")


if __name__ == "__main__":
    grouped_bar_chart()
    delta_chart()
    radar_chart()
    print(f"Charts written to: {OUT_DIR}")
