"""Evaluation metrics and report figures shared by RMOF-Net scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from rmof_data import CLASS_NAMES


def metrics_from_predictions(
    labels: Sequence[int], predictions: Sequence[int]
) -> Dict[str, float]:
    """Compute the classification and ordinal metrics reported by the project."""
    labels_array = np.asarray(labels)
    predictions_array = np.asarray(predictions)
    f1s = f1_score(
        labels_array,
        predictions_array,
        labels=[0, 1, 2],
        average=None,
        zero_division=0,
    )
    deficiency_labels = labels_array != 2
    deficiency_predictions = predictions_array != 2
    matrix = confusion_matrix(labels_array, predictions_array, labels=[0, 1, 2])
    endpoint_total = int(matrix[0].sum() + matrix[2].sum())
    endpoint_errors = int(matrix[0, 2] + matrix[2, 0])
    return {
        "accuracy": float(accuracy_score(labels_array, predictions_array)),
        "macro_f1": float(
            f1_score(labels_array, predictions_array, average="macro", zero_division=0)
        ),
        "f1_deficiency": float(
            f1_score(deficiency_labels, deficiency_predictions, zero_division=0)
        ),
        "f1_n0": float(f1s[0]),
        "f1_n75": float(f1s[1]),
        "f1_nfull": float(f1s[2]),
        "ordinal_mae": float(np.abs(labels_array - predictions_array).mean()),
        "endpoint_confusion": endpoint_errors,
        "endpoint_confusion_rate": (
            float(endpoint_errors / endpoint_total) if endpoint_total else 0.0
        ),
    }


def save_confusion(matrix: np.ndarray, target: Path, title: str) -> None:
    """Save a consistently labelled three-class confusion matrix."""
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(4.6, 4.0))
    image = axis.imshow(matrix, cmap="Blues")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    axis.set(
        xticks=range(3),
        yticks=range(3),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
    )
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title(title)
    figure.colorbar(image, ax=axis, shrink=0.8)
    figure.tight_layout()
    figure.savefig(target, dpi=200)
    plt.close(figure)
