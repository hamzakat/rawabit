import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
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
    "heat_low": "#F3F7FA",
    "heat_high": "#4E79A7",
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


def save_chart(fig, basename):
    fig.savefig(OUT_DIR / f"{basename}.png", dpi=450, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_DIR / f"{basename}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_DIR / f"{basename}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def add_bar_labels(ax, bars, values, color):
    for bar, value in zip(bars, values):
        ax.annotate(
            f"{value:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9.0,
            color=color,
        )


def primary_metric_comparison():
    metrics = [
        "Source\nRecall",
        "Source\nPrecision",
        "Contextual\nRecall",
        "Contextual\nPrecision",
    ]
    graphrag = np.array([0.986, 0.501, 0.780, 0.839])
    vector = np.array([0.899, 0.470, 0.623, 0.763])

    x = np.arange(len(metrics))
    width = 0.34

    fig, ax = plt.subplots(figsize=(6.9, 3.9))
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

    fig.suptitle("Retrieval Quality on In-Scope Queries", y=0.98, fontweight="bold")
    ax.set_ylabel("Score (0–1)")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.08)
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
    fig.subplots_adjust(top=0.82, bottom=0.17)
    save_chart(fig, "figure_retrieval_primary_metric_comparison")


def primary_metric_delta():
    metrics = [
        "Contextual Recall",
        "Source Recall",
        "Contextual Precision",
        "Source Precision",
    ]
    deltas = np.array([0.157, 0.087, 0.076, 0.031])
    y = np.arange(len(metrics))

    fig, ax = plt.subplots(figsize=(6.9, 3.55))
    bars = ax.barh(y, deltas, color=COLORS["graphrag"], height=0.62)

    for bar, delta in zip(bars, deltas):
        ax.annotate(
            f"+{delta:.3f}",
            xy=(delta, bar.get_y() + bar.get_height() / 2),
            xytext=(4, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=9.2,
            color=COLORS["text"],
        )

    fig.suptitle("GraphRAG Improvement over Vector-Only Retrieval", y=0.98, fontweight="bold")
    ax.set_title("Difference in score (Hybrid − Naive)", pad=10, fontsize=9.5, fontweight="normal")
    ax.set_xlabel("Delta score")
    ax.set_yticks(y)
    ax.set_yticklabels(metrics)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.18)
    ax.set_xticks(np.arange(0, 0.181, 0.03))
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    ax.set_axisbelow(True)
    fig.subplots_adjust(top=0.78, left=0.27)
    save_chart(fig, "figure_retrieval_primary_metric_delta")


def primary_metric_radar():
    metrics = [
        "Source Recall",
        "Source Precision",
        "Contextual Recall",
        "Contextual Precision",
    ]
    graphrag = np.array([0.986, 0.501, 0.780, 0.839])
    vector = np.array([0.899, 0.470, 0.623, 0.763])

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
    closed_angles = np.concatenate([angles, angles[:1]])
    graphrag_closed = np.concatenate([graphrag, graphrag[:1]])
    vector_closed = np.concatenate([vector, vector[:1]])

    fig, ax = plt.subplots(figsize=(6.9, 5.7), subplot_kw={"polar": True})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.plot(
        closed_angles,
        graphrag_closed,
        color=COLORS["graphrag"],
        linewidth=2.25,
        marker="o",
        markersize=5,
        label="GraphRAG (Hybrid)",
    )
    ax.fill(closed_angles, graphrag_closed, color=COLORS["graphrag"], alpha=0.18)
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
    ax.set_xticklabels(metrics, fontsize=9.5)
    ax.tick_params(axis="x", pad=11)
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0.2, 1.01, 0.2))
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8.5)
    ax.set_rlabel_position(45)
    ax.grid(color="#D8D8D8", linewidth=0.8)
    ax.spines["polar"].set_color(COLORS["axis"])

    ax.set_title("Primary Retrieval Metric Profile", pad=24, fontweight="bold")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=2,
        frameon=False,
        handlelength=2.0,
        columnspacing=1.8,
    )
    save_chart(fig, "figure_retrieval_primary_metric_radar")


def graph_specific_metrics():
    metrics = ["Entity Exact\nRecall@10", "Entity\nRecall@10", "Relation Connectivity\nRecall@10"]
    values = np.array([0.313, 0.514, 0.415])
    x = np.arange(len(metrics))

    fig, ax = plt.subplots(figsize=(6.9, 3.65))
    bars = ax.bar(x, values, width=0.55, color=COLORS["graphrag"])
    add_bar_labels(ax, bars, values, COLORS["graphrag"])

    fig.suptitle("Graph-Specific Retrieval Quality", y=0.98, fontweight="bold")
    ax.set_title(
        "Reported for GraphRAG only; vector-only retrieval returns no graph objects",
        pad=10,
        fontsize=9.2,
        fontweight="normal",
    )
    ax.set_ylabel("Recall@10")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 0.62)
    ax.set_yticks(np.arange(0, 0.61, 0.1))
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")
    ax.set_axisbelow(True)
    fig.subplots_adjust(top=0.78, bottom=0.19)
    save_chart(fig, "figure_retrieval_graph_specific_metrics")


def analysis_type_heatmap():
    analysis_types = [
        "Link (n = 6)",
        "Temporal (n = 5)",
        "Flow (n = 5)",
        "Factual Verification (n = 7)",
    ]
    metrics = [
        "Source\nRecall",
        "Contextual\nRecall",
        "Entity Exact\nRecall@10",
        "Entity\nRecall@10",
        "Relation Connectivity\nRecall@10",
    ]
    values = np.array([
        [0.944, 0.667, 0.301, 0.438, 0.266],
        [1.000, 0.920, 0.367, 0.500, 0.590],
        [1.000, 0.667, 0.227, 0.320, 0.100],
        [1.000, 0.857, 0.345, 0.726, 0.643],
    ])

    cmap = LinearSegmentedColormap.from_list(
        "thesis_blue",
        [COLORS["heat_low"], "#BFD5E5", COLORS["heat_high"]],
    )

    fig, ax = plt.subplots(figsize=(6.9, 3.7))
    image = ax.imshow(values, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels(metrics)
    ax.set_yticks(np.arange(len(analysis_types)))
    ax.set_yticklabels(analysis_types)
    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True, length=0)

    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            text_color = "white" if value >= 0.68 else COLORS["title"]
            ax.text(
                column,
                row,
                f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=9.2,
                color=text_color,
                fontweight="bold" if value >= 0.9 else "normal",
            )

    for spine in ax.spines.values():
        spine.set_visible(False)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.025)
    colorbar.set_label("Score", color=COLORS["text"])
    colorbar.outline.set_edgecolor(COLORS["axis"])
    colorbar.ax.tick_params(colors=COLORS["text"], labelsize=8.5)

    fig.suptitle("GraphRAG Retrieval Quality by Analysis Type", y=0.98, fontweight="bold")
    fig.subplots_adjust(top=0.84, bottom=0.22, left=0.24, right=0.93)
    save_chart(fig, "figure_retrieval_quality_by_analysis_type")


if __name__ == "__main__":
    primary_metric_comparison()
    primary_metric_delta()
    primary_metric_radar()
    graph_specific_metrics()
    analysis_type_heatmap()
    print(f"Charts written to: {OUT_DIR}")
