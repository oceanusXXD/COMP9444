#!/usr/bin/env python3
"""Generate Grad-CAM explanations for the trained ResNet18 classifier."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib")
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as functional
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from torch import nn

from dataset import IMAGENET_MEAN, IMAGENET_STD, LABELS, MaizeNitrogenDataset
from model import build_model
from train import select_device


class GradCAM:
    """Compute Grad-CAM for one convolutional target layer."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self.handle = target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(
        self,
        module: nn.Module,
        inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        del module, inputs
        self.activations = output
        output.register_hook(self._save_gradient)

    def _save_gradient(self, gradient: torch.Tensor) -> None:
        self.gradients = gradient

    def __call__(
        self, inputs: torch.Tensor, class_index: int | None = None
    ) -> tuple[np.ndarray, torch.Tensor]:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(inputs)
        if class_index is None:
            class_index = int(logits.argmax(dim=1).item())
        logits[0, class_index].backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations and gradients.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        heatmap = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        heatmap = functional.interpolate(
            heatmap,
            size=inputs.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        heatmap = heatmap[0, 0]
        heatmap -= heatmap.min()
        maximum = heatmap.max()
        if maximum > 0:
            heatmap /= maximum
        return heatmap.detach().cpu().numpy(), logits.detach()

    def close(self) -> None:
        self.handle.remove()


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Create Grad-CAM report figures.")
    parser.add_argument("--project-dir", type=Path, default=project_dir)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=project_dir / "models" / "resnet18" / "best_model.pt",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=project_dir / "results" / "resnet18" / "predictions.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_dir / "results" / "gradcam",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    return parser.parse_args()


def read_predictions(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def confidence(row: dict[str, str]) -> float:
    return float(row[f"prob_{row['predicted_label']}"])


def select_examples(
    predictions: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    correct_examples: list[dict[str, str]] = []
    for label in LABELS:
        candidates = [
            row
            for row in predictions
            if row["correct"].lower() == "true" and row["true_label"] == label
        ]
        candidates.sort(key=confidence, reverse=True)
        correct_examples.extend(candidates[:3])

    error_examples = [
        row for row in predictions if row["correct"].lower() == "false"
    ]
    error_examples.sort(key=confidence, reverse=True)
    return correct_examples, error_examples[:12]


def denormalize(image: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(IMAGENET_MEAN, dtype=image.dtype).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=image.dtype).view(3, 1, 1)
    image = image.cpu() * std + mean
    return image.clamp(0, 1).permute(1, 2, 0).numpy()


def make_overlay(heatmap: np.ndarray) -> np.ndarray:
    overlay = plt.get_cmap("turbo")(heatmap)
    overlay[..., 3] = np.clip(heatmap * 0.62, 0.0, 0.62)
    return overlay


def plot_examples(
    examples: list[dict[str, str]],
    dataset: MaizeNitrogenDataset,
    gradcam: GradCAM,
    device: torch.device,
    output: Path,
    rows: int,
    columns: int,
    title: str,
    errors: bool,
) -> None:
    index_by_filename = {
        row["filename"]: index for index, row in enumerate(dataset.records)
    }
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(columns * 3.3, rows * 3.15),
        constrained_layout=True,
    )
    axes_array = np.atleast_1d(axes).reshape(-1)

    for axis, row in zip(axes_array, examples, strict=True):
        dataset_index = index_by_filename[row["filename"]]
        image, _ = dataset[dataset_index]
        inputs = image.unsqueeze(0).to(device)
        predicted_index = int(row["predicted_index"])
        heatmap, logits = gradcam(inputs, class_index=predicted_index)
        probability = logits.softmax(dim=1)[0, predicted_index].item() * 100

        axis.imshow(denormalize(image))
        axis.imshow(make_overlay(heatmap))
        if errors:
            label = f"{row['true_label']} → {row['predicted_label']}"
            title_color = "#A40000"
        else:
            label = row["true_label"]
            title_color = "#006B4F"
        axis.set_title(
            f"{label} ({probability:.1f}%) · plot {row['plot_id']}",
            color=title_color,
        )
        axis.axis("off")

    colorbar = fig.colorbar(
        ScalarMappable(norm=Normalize(0, 1), cmap="turbo"),
        ax=axes_array.tolist(),
        location="right",
        fraction=0.018,
        pad=0.015,
    )
    colorbar.set_label("Grad-CAM relevance")
    fig.suptitle(title, fontsize=13)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    project_dir = args.project_dir.resolve()
    checkpoint_path = args.checkpoint.resolve()
    predictions_path = args.predictions.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)

    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    if checkpoint["model_name"] != "resnet18":
        raise ValueError("Grad-CAM target layer is configured for ResNet18 only.")

    config = checkpoint["config"]
    model = build_model(
        "resnet18",
        pretrained=False,
        freeze_backbone=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    dataset = MaizeNitrogenDataset(
        project_dir,
        "test",
        image_size=int(config["image_size"]),
    )
    predictions = read_predictions(predictions_path)
    correct_examples, error_examples = select_examples(predictions)
    gradcam = GradCAM(model, model.layer4[-1])

    try:
        plot_examples(
            correct_examples,
            dataset,
            gradcam,
            device,
            output_dir / "gradcam_correct.png",
            rows=3,
            columns=3,
            title="ResNet18 Grad-CAM: high-confidence correct predictions",
            errors=False,
        )
        plot_examples(
            error_examples,
            dataset,
            gradcam,
            device,
            output_dir / "gradcam_errors.png",
            rows=3,
            columns=4,
            title="ResNet18 Grad-CAM: high-confidence errors",
            errors=True,
        )
    finally:
        gradcam.close()

    print(f"device: {device}")
    print(f"correct examples: {len(correct_examples)}")
    print(f"error examples: {len(error_examples)}")
    for path in sorted(output_dir.glob("*.png")):
        print(path)


if __name__ == "__main__":
    main()
