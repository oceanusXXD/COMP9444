#!/usr/bin/env python3
"""Model definitions for maize nitrogen deficiency classification."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


NUM_CLASSES = 3
MODEL_NAMES = ("simple_cnn", "resnet18")


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )


class SimpleCNN(nn.Module):
    """Compact CNN trained from scratch as an experimental baseline."""

    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.35) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))


def build_resnet18(
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    dropout: float = 0.25,
) -> nn.Module:
    """Build ResNet18 and replace its ImageNet classification head."""
    # Keep downloaded weights inside the project instead of the user's home folder.
    project_dir = Path(__file__).resolve().parents[1]
    torch.hub.set_dir(str(project_dir / ".torch" / "hub"))
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)

    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False

    input_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(input_features, num_classes),
    )
    return model


def build_model(
    name: str,
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    """Build a model by its command-line name."""
    if name == "simple_cnn":
        return SimpleCNN(num_classes=num_classes)
    if name == "resnet18":
        return build_resnet18(
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )
    raise ValueError(f"Unknown model {name!r}; expected one of {MODEL_NAMES}.")


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return total and trainable parameter counts."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return total, trainable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test a classification model.")
    parser.add_argument("--model", choices=MODEL_NAMES, default="simple_cnn")
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Load ImageNet weights (may download them on first use).",
    )
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=224)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = build_model(
        args.model,
        pretrained=args.pretrained,
        freeze_backbone=args.freeze_backbone,
    )
    inputs = torch.randn(args.batch_size, 3, args.image_size, args.image_size)
    targets = torch.randint(0, NUM_CLASSES, (args.batch_size,))
    outputs = model(inputs)
    loss = nn.CrossEntropyLoss()(outputs, targets)
    loss.backward()

    total, trainable = count_parameters(model)
    print(f"model: {args.model}")
    print(f"input: {tuple(inputs.shape)}")
    print(f"output: {tuple(outputs.shape)}")
    print(f"loss: {loss.item():.4f}")
    print(f"parameters: total={total:,}, trainable={trainable:,}")
    print("backward pass: passed")


if __name__ == "__main__":
    main()
