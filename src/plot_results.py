# -*- coding: utf-8 -*-
"""
plot_results.py

Only generate:
- comparison_validation_metric_curves.png
- baseline_confusion_matrices.png
- innovation_confusion_matrices.png

Changes:
1) remove loss figure
2) remove baseline-only / innovation-only metric figures
3) reduce the blank space above comparison subplots
4) square confusion matrices
5) one colorbar on the right of each confusion matrix
6) support normalized confusion matrices
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


VAL_METRIC_KEYS = [
    "mean_seg_iou",
    "mean_dice",
    "mean_bbox_iou",
    "det_acc@0.5",
    "top1_acc",
    "macro_f1",
]

PRETTY_NAMES = {
    "mean_seg_iou": "Segmentation IoU",
    "mean_dice": "Dice Score",
    "mean_bbox_iou": "BBox IoU",
    "det_acc@0.5": "Detection Acc@0.5",
    "top1_acc": "Top-1 Accuracy",
    "macro_f1": "Macro F1",
}


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def set_plot_style():
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 13.5,
        "axes.labelsize": 11.5,
        "legend.fontsize": 10.5,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.linewidth": 1.0,
        "lines.linewidth": 2.0,
        "grid.linewidth": 0.8,
        "grid.alpha": 0.40,
        "axes.grid": True,
        "grid.linestyle": "-",
        "legend.frameon": True,
        "legend.framealpha": 0.92,
    })


def get_epochs(train_log):
    return [row["epoch"] for row in train_log]


def get_series(train_log, key):
    return np.array([row.get(key, np.nan) for row in train_log], dtype=float)


def plot_val_metric_comparison(base_log, innov_log, out_path):
    epochs_b = get_epochs(base_log)
    epochs_i = get_epochs(innov_log)

    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.6))
    axes = axes.flatten()

    for ax, key in zip(axes, VAL_METRIC_KEYS):
        ax.plot(epochs_b, get_series(base_log, key), label="Baseline")
        ax.plot(epochs_i, get_series(innov_log, key), label="Innovation")
        ax.set_title(PRETTY_NAMES.get(key, key))
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Score")
        ax.set_ylim(bottom=0)

    handles, labels = axes[0].get_legend_handles_labels()

    # Title and legend closer together
    fig.suptitle(
        "Validation Metric Curves: Baseline vs Innovation",
        y=0.965,
        fontsize=15
    )
    fig.legend(
        handles, labels,
        loc="upper center",
        ncol=2,
        frameon=True,
        bbox_to_anchor=(0.5, 0.935),
        borderaxespad=0.15
    )

    # THIS is the key part:
    # move the whole subplot area upward to reduce the large blank band
    fig.subplots_adjust(
        left=0.07,
        right=0.975,
        bottom=0.08,
        top=0.84,     # increase this to push subplots upward
        wspace=0.22,
        hspace=0.32
    )

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def draw_confusion_matrix(ax, cm, class_names, title, normalize=True, cmap="viridis"):
    cm = np.array(cm, dtype=float)

    if normalize:
        row_sum = cm.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        cm_show = cm / row_sum
    else:
        cm_show = cm

    im = ax.imshow(cm_show, interpolation="nearest", aspect="equal", cmap=cmap)
    ax.set_title(title, pad=8)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    thresh = cm_show.max() * 0.6 if cm_show.size > 0 else 0.5
    n = cm_show.shape[0]
    for i in range(n):
        for j in range(n):
            if i == j:  # only show diagonal values
                txt = f"{cm_show[i, j]:.2f}" if normalize else f"{int(cm_show[i, j])}"
                ax.text(
                    j, i, txt,
                    ha="center", va="center",
                    color="white" if cm_show[i, j] > thresh else "black",
                    fontsize=9,
                    fontweight="bold"
                )

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3.2%", pad=0.04)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label("Normalized Value" if normalize else "Count")


def plot_confusion_pair(metrics_val, metrics_test, model_name, class_names, out_path, normalize=True):
    cm_val = metrics_val["confusion_matrix"]
    cm_test = metrics_test["confusion_matrix"]

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.6))

    draw_confusion_matrix(
        axes[0], cm_val, class_names,
        f"{model_name} - Validation",
        normalize=normalize
    )
    draw_confusion_matrix(
        axes[1], cm_test, class_names,
        f"{model_name} - Test",
        normalize=normalize
    )

    fig.suptitle(f"{model_name}: Confusion Matrices", y=0.955, fontsize=15)
    fig.subplots_adjust(
        left=0.06,
        right=0.97,
        bottom=0.12,
        top=0.83,
        wspace=0.03
    )

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project_root",
        type=str,
        default="..",
        help="Path to project root. If you run this inside src/, keep default '..'."
    )
    parser.add_argument(
        "--class_names",
        nargs="*",
        default=[str(i) for i in range(10)],
        help="Optional class names."
    )
    parser.add_argument(
        "--normalize_cm",
        action="store_true",
        help="Normalize confusion matrices row-wise."
    )
    args = parser.parse_args()

    set_plot_style()

    project_root = Path(args.project_root).resolve()
    results_dir = project_root / "results"
    fig_dir = results_dir / "figures"
    ensure_dir(fig_dir)

    baseline_dir = results_dir / "baseline_rgb"
    innov_dir = results_dir / "innovation_rgb_light"

    base_log = load_json(baseline_dir / "train_log.json")
    base_val = load_json(baseline_dir / "metrics_val.json")
    base_test = load_json(baseline_dir / "metrics_test.json")

    innov_log = load_json(innov_dir / "train_log.json")
    innov_val = load_json(innov_dir / "metrics_val.json")
    innov_test = load_json(innov_dir / "metrics_test.json")

    # Only one comparison metric figure
    plot_val_metric_comparison(
        base_log,
        innov_log,
        fig_dir / "comparison_validation_metric_curves.png"
    )

    # Only confusion matrices
    plot_confusion_pair(
        base_val,
        base_test,
        "Baseline",
        args.class_names,
        fig_dir / "baseline_confusion_matrices.png",
        normalize=args.normalize_cm
    )

    plot_confusion_pair(
        innov_val,
        innov_test,
        "Innovation",
        args.class_names,
        fig_dir / "innovation_confusion_matrices.png",
        normalize=args.normalize_cm
    )

    print("Saved figures to:", fig_dir)
    print(" - comparison_validation_metric_curves.png")
    print(" - baseline_confusion_matrices.png")
    print(" - innovation_confusion_matrices.png")


if __name__ == "__main__":
    main()