import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


BASE_DIR = r"./"
FIGURES_DIR = os.path.join(BASE_DIR, "figures")


@dataclass(frozen=True)
class Node:
    title: str
    subtitle: str
    xy: tuple[float, float]
    width: float = 0.25
    height: float = 0.10


def _add_node(ax, node: Node, facecolor: str, edgecolor: str = "#2b2b2b"):
    x, y = node.xy
    rect = FancyBboxPatch(
        (x, y),
        node.width,
        node.height,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.2,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(rect)

    ax.text(
        x + node.width / 2,
        y + node.height * 0.66,
        node.title,
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
    )
    ax.text(
        x + node.width / 2,
        y + node.height * 0.30,
        node.subtitle,
        ha="center",
        va="center",
        fontsize=10,
    )


def _arrow(ax, start: tuple[float, float], end: tuple[float, float]):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="->", lw=1.6, color="#2b2b2b"),
    )


def generate_workflow_figure(output_path: str | None = None) -> str:
    fig, ax = plt.subplots(figsize=(14.5, 8.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.suptitle(
        "End-to-End Workflow of Lightweight Malicious URL Detection and Interpretability",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    nodes = {
        "sources": Node(
            title="Data Collection",
            subtitle="Malicious URLs + Benign URLs",
            xy=(0.06, 0.80),
            width=0.28,
        ),
        "clean": Node(
            title="Data Cleaning",
            subtitle="Normalize scheme, remove malformed",
            xy=(0.06, 0.64),
            width=0.28,
        ),
        "features": Node(
            title="Feature Engineering",
            subtitle="18 lightweight URL features",
            xy=(0.06, 0.48),
            width=0.28,
        ),
        "split": Node(
            title="Dataset Split",
            subtitle="Stratified 70/30 (fixed seed)",
            xy=(0.06, 0.32),
            width=0.28,
        ),
        "train": Node(
            title="Model Training",
            subtitle="Linear baseline and tree boosting",
            xy=(0.40, 0.64),
            width=0.28,
        ),
        "eval": Node(
            title="Model Evaluation",
            subtitle="Accuracy, Precision, Recall, F1, AUC",
            xy=(0.40, 0.48),
            width=0.28,
        ),
        "shap": Node(
            title="SHAP Interpretability",
            subtitle="Global importance + local explanations",
            xy=(0.40, 0.32),
            width=0.28,
        ),
        "bench": Node(
            title="Efficiency Benchmark",
            subtitle="Warm-up + repeats + trimmed mean",
            xy=(0.74, 0.48),
            width=0.22,
        ),
        "report": Node(
            title="Reporting",
            subtitle="Figures, tables, and discussion",
            xy=(0.74, 0.32),
            width=0.22,
        ),
    }

    color_data = "#e8f1ff"
    color_model = "#eaf7ea"
    color_xai = "#fff4e6"
    color_bench = "#f2f2f2"

    _add_node(ax, nodes["sources"], facecolor=color_data)
    _add_node(ax, nodes["clean"], facecolor=color_data)
    _add_node(ax, nodes["features"], facecolor=color_data)
    _add_node(ax, nodes["split"], facecolor=color_data)

    _add_node(ax, nodes["train"], facecolor=color_model)
    _add_node(ax, nodes["eval"], facecolor=color_model)

    _add_node(ax, nodes["shap"], facecolor=color_xai)

    _add_node(ax, nodes["bench"], facecolor=color_bench)
    _add_node(ax, nodes["report"], facecolor=color_bench)

    def _bottom_center(n: Node):
        return (n.xy[0] + n.width / 2, n.xy[1])

    def _top_center(n: Node):
        return (n.xy[0] + n.width / 2, n.xy[1] + n.height)

    def _right_center(n: Node):
        return (n.xy[0] + n.width, n.xy[1] + n.height / 2)

    def _left_center(n: Node):
        return (n.xy[0], n.xy[1] + n.height / 2)

    _arrow(ax, _bottom_center(nodes["sources"]), _top_center(nodes["clean"]))
    _arrow(ax, _bottom_center(nodes["clean"]), _top_center(nodes["features"]))
    _arrow(ax, _bottom_center(nodes["features"]), _top_center(nodes["split"]))

    _arrow(ax, _right_center(nodes["split"]), _left_center(nodes["train"]))
    _arrow(ax, _bottom_center(nodes["train"]), _top_center(nodes["eval"]))
    _arrow(ax, _bottom_center(nodes["eval"]), _top_center(nodes["shap"]))

    _arrow(ax, _right_center(nodes["eval"]), _left_center(nodes["bench"]))
    _arrow(ax, _bottom_center(nodes["bench"]), _top_center(nodes["report"]))
    _arrow(ax, _right_center(nodes["shap"]), _left_center(nodes["report"]))

    ax.text(0.06, 0.22, "Notes: All steps use a fixed random seed for reproducibility.", fontsize=10)

    if not os.path.exists(FIGURES_DIR):
        os.makedirs(FIGURES_DIR)

    if output_path is None:
        output_path = os.path.join(FIGURES_DIR, "workflow_overview.png")

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Workflow figure saved to {output_path}")
    return output_path


def main():
    generate_workflow_figure()


if __name__ == "__main__":
    main()
