#!/usr/bin/env python3
"""Evaluate a saved maize classifier on the held-out test set."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch import nn

from dataset import LABELS, MaizeNitrogenDataset
from model import build_model
from train import select_device


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Evaluate a saved model.")
    parser.add_argument("--project-dir", type=Path, default=project_dir)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=project_dir / "models" / "simple_cnn" / "best_model.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_dir / "results" / "simple_cnn",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    return parser.parse_args()


def collect_predictions(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[float, list[int], list[int], list[list[float]]]:
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    targets: list[int] = []
    predictions: list[int] = []
    probabilities: list[list[float]] = []

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            total_loss += criterion(logits, labels).item()
            batch_probabilities = logits.softmax(dim=1)

            targets.extend(labels.cpu().tolist())
            predictions.extend(batch_probabilities.argmax(dim=1).cpu().tolist())
            probabilities.extend(batch_probabilities.cpu().tolist())

    return total_loss / len(targets), targets, predictions, probabilities


def calculate_metrics(
    loss: float, targets: list[int], predictions: list[int]
) -> tuple[dict[str, object], list[list[int]], str]:
    label_indices = list(range(len(LABELS)))
    matrix = confusion_matrix(targets, predictions, labels=label_indices)
    report_dict = classification_report(
        targets,
        predictions,
        labels=label_indices,
        target_names=LABELS,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        targets,
        predictions,
        labels=label_indices,
        target_names=LABELS,
        digits=4,
        zero_division=0,
    )
    metrics: dict[str, object] = {
        "test_loss": loss,
        "accuracy": accuracy_score(targets, predictions),
        "macro_precision": precision_score(
            targets,
            predictions,
            labels=label_indices,
            average="macro",
            zero_division=0,
        ),
        "macro_recall": recall_score(
            targets,
            predictions,
            labels=label_indices,
            average="macro",
            zero_division=0,
        ),
        "macro_f1": f1_score(
            targets,
            predictions,
            labels=label_indices,
            average="macro",
            zero_division=0,
        ),
        "per_class": {
            label: {
                key: value
                for key, value in report_dict[label].items()
                if key in ("precision", "recall", "f1-score", "support")
            }
            for label in LABELS
        },
        "confusion_matrix": matrix.tolist(),
    }
    return metrics, matrix.tolist(), report_text


def write_predictions(
    path: Path,
    dataset: MaizeNitrogenDataset,
    targets: list[int],
    predictions: list[int],
    probabilities: list[list[float]],
) -> None:
    fieldnames = (
        "filepath",
        "filename",
        "plot_id",
        "true_index",
        "true_label",
        "predicted_index",
        "predicted_label",
        "correct",
        "prob_N0",
        "prob_N75",
        "prob_NFull",
    )
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row, target, prediction, probability in zip(
            dataset.records, targets, predictions, probabilities, strict=True
        ):
            writer.writerow(
                {
                    "filepath": row["filepath"],
                    "filename": row["filename"],
                    "plot_id": row["plot_id"],
                    "true_index": target,
                    "true_label": LABELS[target],
                    "predicted_index": prediction,
                    "predicted_label": LABELS[prediction],
                    "correct": target == prediction,
                    "prob_N0": f"{probability[0]:.8f}",
                    "prob_N75": f"{probability[1]:.8f}",
                    "prob_NFull": f"{probability[2]:.8f}",
                }
            )


def write_confusion_matrix(path: Path, matrix: list[list[int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["actual\\predicted", *LABELS])
        for label, row in zip(LABELS, matrix, strict=True):
            writer.writerow([label, *row])


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    checkpoint_path = args.checkpoint.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    if tuple(checkpoint["labels"]) != LABELS:
        raise ValueError(
            f"Checkpoint labels {checkpoint['labels']} do not match {LABELS}."
        )

    config = checkpoint["config"]
    dataset = MaizeNitrogenDataset(
        args.project_dir,
        "test",
        image_size=int(config["image_size"]),
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    model = build_model(
        checkpoint["model_name"],
        pretrained=False,
        freeze_backbone=bool(config["freeze_backbone"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    loss, targets, predictions, probabilities = collect_predictions(
        model, loader, device
    )
    metrics, matrix, report_text = calculate_metrics(loss, targets, predictions)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)
    (output_dir / "classification_report.txt").write_text(
        report_text, encoding="utf-8"
    )
    write_predictions(
        output_dir / "predictions.csv",
        dataset,
        targets,
        predictions,
        probabilities,
    )
    write_confusion_matrix(output_dir / "confusion_matrix.csv", matrix)

    print(f"checkpoint: {checkpoint_path}")
    print(f"best validation epoch: {checkpoint['epoch']}")
    print(f"device: {device}")
    print(f"test samples: {len(dataset)}")
    print(f"test loss: {metrics['test_loss']:.4f}")
    print(f"accuracy: {metrics['accuracy']:.4f}")
    print(f"macro precision: {metrics['macro_precision']:.4f}")
    print(f"macro recall: {metrics['macro_recall']:.4f}")
    print(f"macro F1: {metrics['macro_f1']:.4f}")
    print("confusion matrix (rows=actual, columns=predicted):")
    print(f"labels: {LABELS}")
    for row in matrix:
        print(row)
    print(f"results: {output_dir}")


if __name__ == "__main__":
    main()
