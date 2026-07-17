#!/usr/bin/env python3
"""Generate report-ready plots and error examples from completed experiments."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib")
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


LABELS = ("N0", "N75", "NFull")
BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
PURPLE = "#CC79A7"


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Create experiment result figures.")
    parser.add_argument("--project-dir", type=Path, default=project_dir)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_dir / "results" / "figures",
    )
    return parser.parse_args()


def read_json(path: Path) -> object:
    with path.open(encoding="utf-8") as json_file:
        return json.load(json_file)


def configure_plotting() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 200,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def plot_training_curves(
    simple_history: list[dict[str, float]],
    resnet_history: list[dict[str, float]],
    output: Path,
) -> None:
    histories = (("Simple CNN", simple_history), ("ResNet18", resnet_history))
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)

    for column, (model_name, history) in enumerate(histories):
        epochs = [row["epoch"] for row in history]
        axes[0, column].plot(
            epochs, [row["train_loss"] for row in history], color=BLUE, label="Train"
        )
        axes[0, column].plot(
            epochs, [row["val_loss"] for row in history], color=ORANGE, label="Validation"
        )
        axes[0, column].set_title(f"{model_name}: cross-entropy loss")
        axes[0, column].set_xlabel("Epoch")
        axes[0, column].set_ylabel("Loss")
        axes[0, column].legend(frameon=False)

        axes[1, column].plot(
            epochs,
            [row["train_macro_f1"] for row in history],
            color=BLUE,
            label="Train",
        )
        axes[1, column].plot(
            epochs,
            [row["val_macro_f1"] for row in history],
            color=ORANGE,
            label="Validation",
        )
        best_row = max(history, key=lambda row: row["val_macro_f1"])
        axes[1, column].scatter(
            [best_row["epoch"]],
            [best_row["val_macro_f1"]],
            color=GREEN,
            zorder=3,
            label=f"Best val: {best_row['val_macro_f1']:.3f}",
        )
        axes[1, column].set_title(f"{model_name}: macro-F1")
        axes[1, column].set_xlabel("Epoch")
        axes[1, column].set_ylabel("Macro-F1")
        axes[1, column].set_ylim(0, 1)
        axes[1, column].legend(frameon=False)

    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison(
    simple_metrics: dict[str, float],
    resnet_metrics: dict[str, float],
    output: Path,
) -> None:
    metric_names = ("Accuracy", "Macro precision", "Macro recall", "Macro-F1")
    metric_keys = ("accuracy", "macro_precision", "macro_recall", "macro_f1")
    simple_values = [100 * simple_metrics[key] for key in metric_keys]
    resnet_values = [100 * resnet_metrics[key] for key in metric_keys]

    positions = np.arange(len(metric_names))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    simple_bars = ax.bar(
        positions - width / 2,
        simple_values,
        width,
        color=ORANGE,
        label="Simple CNN",
    )
    resnet_bars = ax.bar(
        positions + width / 2,
        resnet_values,
        width,
        color=BLUE,
        label="ResNet18",
    )
    ax.bar_label(simple_bars, fmt="%.1f", padding=3)
    ax.bar_label(resnet_bars, fmt="%.1f", padding=3)
    ax.set_xticks(positions, metric_names)
    ax.set_ylabel("Test score (%)")
    ax.set_ylim(0, 75)
    ax.set_title("Held-out test performance")
    ax.legend(frameon=False, ncols=2)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(matrix: list[list[int]], output: Path) -> None:
    counts = np.asarray(matrix, dtype=int)
    percentages = counts / counts.sum(axis=1, keepdims=True) * 100
    fig, ax = plt.subplots(figsize=(6.2, 5.4), constrained_layout=True)
    image = ax.imshow(percentages, cmap="Blues", vmin=0, vmax=100)

    for row in range(len(LABELS)):
        for column in range(len(LABELS)):
            text_color = "white" if percentages[row, column] >= 55 else "black"
            ax.text(
                column,
                row,
                f"{counts[row, column]}\n{percentages[row, column]:.1f}%",
                ha="center",
                va="center",
                color=text_color,
                fontweight="bold" if row == column else "normal",
            )

    ax.set_xticks(range(len(LABELS)), LABELS)
    ax.set_yticks(range(len(LABELS)), LABELS)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    ax.set_title("ResNet18 test confusion matrix")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Percentage within actual class")
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_error_cases(project_dir: Path, predictions_path: Path, output: Path) -> None:
    with predictions_path.open(newline="", encoding="utf-8") as csv_file:
        errors = [
            row
            for row in csv.DictReader(csv_file)
            if row["correct"].lower() == "false"
        ]

    def predicted_confidence(row: dict[str, str]) -> float:
        return float(row[f"prob_{row['predicted_label']}"])

    errors.sort(key=predicted_confidence, reverse=True)
    selected = errors[:12]
    fig, axes = plt.subplots(3, 4, figsize=(13, 8.7), constrained_layout=True)

    for axis, row in zip(axes.flat, selected, strict=True):
        image_path = project_dir / row["filepath"]
        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")
            axis.imshow(image)
        confidence = 100 * predicted_confidence(row)
        axis.set_title(
            f"{row['true_label']} → {row['predicted_label']} ({confidence:.1f}%)\n"
            f"plot {row['plot_id']}",
            color="#A40000",
        )
        axis.axis("off")

    fig.suptitle("ResNet18: highest-confidence test errors", fontsize=13)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    project_dir = args.project_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_plotting()

    simple_history = read_json(project_dir / "models" / "simple_cnn" / "history.json")
    resnet_history = read_json(project_dir / "models" / "resnet18" / "history.json")
    simple_metrics = read_json(project_dir / "results" / "simple_cnn" / "metrics.json")
    resnet_metrics = read_json(project_dir / "results" / "resnet18" / "metrics.json")

    plot_training_curves(
        simple_history,
        resnet_history,
        output_dir / "training_curves.png",
    )
    plot_model_comparison(
        simple_metrics,
        resnet_metrics,
        output_dir / "model_comparison.png",
    )
    plot_confusion_matrix(
        resnet_metrics["confusion_matrix"],
        output_dir / "resnet18_confusion_matrix.png",
    )
    plot_error_cases(
        project_dir,
        project_dir / "results" / "resnet18" / "predictions.csv",
        output_dir / "resnet18_error_cases.png",
    )

    for path in sorted(output_dir.glob("*.png")):
        print(path)


if __name__ == "__main__":
    main()
