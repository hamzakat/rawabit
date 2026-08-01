import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


OUT_DIR = Path(__file__).resolve().parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "graphrag": "#4E79A7",
    "vector": "#76B7E5",
    "grid": "#EAEAEA",
    "axis": "#B8B8B8",
    "text": "#5F5F5F",
    "title": "#3F3F3F",
}

plt.rcParams.update({
    "font.family": "Arial",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "axes.titlesize": 14,
    "axes.labelsize": 10.5,
    "xtick.labelsize": 9.5,
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


PRIMARY_METRICS = [
    "Answer Correctness",
    "Faithfulness",
    "Source Provenance Hit",
    "Source Provenance Recall",
    "Source Provenance Precision",
]
PRIMARY_SHORT = ["AC", "FS", "SPH", "SPR", "SPP"]
GRAPHRAG_PRIMARY = np.array([0.5739, 0.9865, 1.0000, 0.9855, 0.5014])
VECTOR_PRIMARY = np.array([0.4783, 0.9819, 0.9130, 0.8986, 0.4290])


def save_chart(fig, basename):
    fig.savefig(OUT_DIR / f"{basename}.png", dpi=450, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_DIR / f"{basename}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_DIR / f"{basename}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def add_bar_labels(ax, bars, values, color, digits=3):
    for bar, value in zip(bars, values):
        is_high = value >= 0.9
        ax.annotate(
            f"{value:.{digits}f}",
            xy=(bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, -6 if is_high else 5),
            textcoords="offset points",
            ha="center",
            va="top" if is_high else "bottom",
            fontsize=8.8,
            color="white" if is_high else color,
        )


def primary_metric_comparison():
    x = np.arange(len(PRIMARY_SHORT))
    width = 0.34

    fig, ax = plt.subplots(figsize=(6.9, 3.9))
    bars_graph = ax.bar(
        x - width / 2,
        GRAPHRAG_PRIMARY,
        width,
        color=COLORS["graphrag"],
        label="GraphRAG (Hybrid)",
    )
    bars_vector = ax.bar(
        x + width / 2,
        VECTOR_PRIMARY,
        width,
        color=COLORS["vector"],
        label="Vector-only (Naive)",
    )
    add_bar_labels(ax, bars_graph, GRAPHRAG_PRIMARY, COLORS["graphrag"])
    add_bar_labels(ax, bars_vector, VECTOR_PRIMARY, COLORS["vector"])

    deltas = GRAPHRAG_PRIMARY - VECTOR_PRIMARY
    for index, delta in enumerate(deltas):
        ax.text(
            index,
            0.055,
            f"Δ {delta:+.4f}",
            ha="center",
            va="bottom",
            fontsize=8.2,
            color=COLORS["text"],
            fontweight="bold",
        )

    fig.suptitle("Generation Quality on In-Scope Queries", y=0.98, fontweight="bold")
    ax.set_ylabel("Score (0–1)")
    ax.set_xticks(x)
    ax.set_xticklabels(PRIMARY_SHORT)
    ax.set_ylim(0, 1.10)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")
    ax.set_axisbelow(True)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.055),
        ncol=2,
        frameon=False,
        handlelength=1.5,
        columnspacing=1.8,
    )
    fig.subplots_adjust(top=0.82, bottom=0.15)
    save_chart(fig, "figure_generation_primary_metric_comparison")


def primary_metric_radar():
    angles = np.linspace(0, 2 * np.pi, len(PRIMARY_METRICS), endpoint=False)
    closed_angles = np.concatenate([angles, angles[:1]])
    graph_closed = np.concatenate([GRAPHRAG_PRIMARY, GRAPHRAG_PRIMARY[:1]])
    vector_closed = np.concatenate([VECTOR_PRIMARY, VECTOR_PRIMARY[:1]])

    fig, ax = plt.subplots(figsize=(6.9, 5.8), subplot_kw={"polar": True})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.plot(
        closed_angles,
        graph_closed,
        color=COLORS["graphrag"],
        linewidth=2.25,
        marker="o",
        markersize=5,
        label="GraphRAG (Hybrid)",
    )
    ax.fill(closed_angles, graph_closed, color=COLORS["graphrag"], alpha=0.18)
    ax.plot(
        closed_angles,
        vector_closed,
        color=COLORS["vector"],
        linewidth=2.25,
        marker="s",
        markersize=4.8,
        label="Vector-only (Naive)",
    )
    ax.fill(closed_angles, vector_closed, color=COLORS["vector"], alpha=0.20)

    ax.set_xticks(angles)
    ax.set_xticklabels(PRIMARY_SHORT, fontsize=9.5)
    ax.tick_params(axis="x", pad=10)
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0.2, 1.01, 0.2))
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8.5)
    ax.set_rlabel_position(54)
    ax.grid(color="#D8D8D8", linewidth=0.8)
    ax.spines["polar"].set_color(COLORS["axis"])
    ax.set_title("Generation Quality Profile", pad=24, fontweight="bold")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=2,
        frameon=False,
        handlelength=2.0,
        columnspacing=1.8,
    )
    save_chart(fig, "figure_generation_primary_metric_radar")


def primary_metric_delta():
    deltas = GRAPHRAG_PRIMARY - VECTOR_PRIMARY
    order = np.argsort(deltas)[::-1]
    labels = np.array(PRIMARY_METRICS)[order]
    ordered_deltas = deltas[order]
    y = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(6.9, 3.8))
    bars = ax.barh(y, ordered_deltas, color=COLORS["graphrag"], height=0.64)
    for bar, delta in zip(bars, ordered_deltas):
        ax.annotate(
            f"+{delta:.4f}",
            xy=(delta, bar.get_y() + bar.get_height() / 2),
            xytext=(4, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=9.0,
            color=COLORS["text"],
        )

    fig.suptitle("GraphRAG Improvement in Generation Quality", y=0.98, fontweight="bold")
    ax.set_title("Difference in score (Hybrid − Naive)", pad=10, fontsize=9.5, fontweight="normal")
    ax.set_xlabel("Delta score")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.11)
    ax.set_xticks(np.arange(0, 0.111, 0.02))
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    ax.set_axisbelow(True)
    fig.subplots_adjust(top=0.79, left=0.31)
    save_chart(fig, "figure_generation_primary_metric_delta")


def quality_by_analysis_type():
    analysis_types = ["Link", "Event", "Flow"]
    graph = {
        "Answer Correctness": np.array([0.433, 0.420, 0.700]),
        "Faithfulness": np.array([0.983, 0.958, 1.000]),
        "Source Provenance Recall": np.array([0.944, 1.000, 1.000]),
    }
    vector = {
        "Answer Correctness": np.array([0.383, 0.460, 0.680]),
        "Faithfulness": np.array([1.000, 0.983, 1.000]),
        "Source Provenance Recall": np.array([0.944, 1.000, 1.000]),
    }

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.5), sharey=True)
    x = np.arange(len(analysis_types))
    width = 0.34

    for ax, metric in zip(axes, graph):
        bars_graph = ax.bar(
            x - width / 2,
            graph[metric],
            width,
            color=COLORS["graphrag"],
            label="GraphRAG (Hybrid)",
        )
        bars_vector = ax.bar(
            x + width / 2,
            vector[metric],
            width,
            color=COLORS["vector"],
            label="Vector-only (Naive)",
        )
        ax.set_title(metric, fontsize=10.0, pad=9, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(analysis_types, rotation=25, ha="right")
        ax.set_ylim(0, 1.08)
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.grid(True, axis="y")
        ax.grid(False, axis="x")
        ax.set_axisbelow(True)

        for bars, values, color in [
            (bars_graph, graph[metric], COLORS["graphrag"]),
            (bars_vector, vector[metric], COLORS["vector"]),
        ]:
            for bar, value in zip(bars, values):
                is_high = value >= 0.9
                ax.annotate(
                    f"{value:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, value),
                    xytext=(0, -4 if is_high else 4),
                    textcoords="offset points",
                    ha="center",
                    va="top" if is_high else "bottom",
                    fontsize=7.1,
                    color="white" if is_high else color,
                    rotation=90,
                )

    axes[0].set_ylabel("Score (0–1)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("Generation Quality by Analysis Type", y=0.99, fontweight="bold")
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.92),
        ncol=2,
        frameon=False,
        handlelength=1.5,
        columnspacing=1.8,
    )
    fig.subplots_adjust(top=0.72, bottom=0.22, left=0.09, right=0.99, wspace=0.20)
    save_chart(fig, "figure_generation_quality_by_analysis_type")


def out_of_scope_honesty():
    metrics = [
        "Correct Refusal /\nGEval Correctness",
        "Faithfulness",
        "Source Provenance\nPrecision",
    ]
    graphrag = np.array([0.957, 0.964, 0.500])
    vector = np.array([1.000, 0.857, 0.333])
    x = np.arange(len(metrics))
    width = 0.34

    fig, ax = plt.subplots(figsize=(6.9, 3.8))
    bars_graph = ax.bar(
        x - width / 2,
        graphrag,
        width,
        color=COLORS["graphrag"],
        label="GraphRAG (Hybrid)",
    )
    bars_vector = ax.bar(
        x + width / 2,
        vector,
        width,
        color=COLORS["vector"],
        label="Vector-only (Naive)",
    )
    add_bar_labels(ax, bars_graph, graphrag, COLORS["graphrag"])
    add_bar_labels(ax, bars_vector, vector, COLORS["vector"])

    deltas = graphrag - vector
    for index, delta in enumerate(deltas):
        ax.text(
            index,
            0.055,
            f"Δ {delta:+.3f}",
            ha="center",
            va="bottom",
            fontsize=8.7,
            color=COLORS["text"],
            fontweight="bold",
        )

    fig.suptitle("Generation Honesty on Out-of-Scope Queries", y=0.98, fontweight="bold")
    ax.set_title("n = 7 queries per mode", pad=10, fontsize=9.5, fontweight="normal")
    ax.set_ylabel("Score (0–1)")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.10)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")
    ax.set_axisbelow(True)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=2,
        frameon=False,
        handlelength=1.5,
        columnspacing=1.8,
    )
    fig.subplots_adjust(top=0.75, bottom=0.19)
    save_chart(fig, "figure_generation_out_of_scope_honesty")


if __name__ == "__main__":
    primary_metric_comparison()
    primary_metric_radar()
    primary_metric_delta()
    quality_by_analysis_type()
    out_of_scope_honesty()
    print(f"Charts written to: {OUT_DIR}")
