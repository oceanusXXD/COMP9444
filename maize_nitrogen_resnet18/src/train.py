#!/usr/bin/env python3
"""Train and validate maize nitrogen classification models."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn

from dataset import LABELS, build_dataloaders
from model import MODEL_NAMES, build_model, count_parameters


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Train a maize classifier.")
    parser.add_argument("--project-dir", type=Path, default=project_dir)
    parser.add_argument("--model", choices=MODEL_NAMES, default="simple_cnn")
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use ImageNet weights for ResNet18.",
    )
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument(
        "--initial-checkpoint",
        type=Path,
        default=None,
        help="Initialize model weights from an earlier training stage.",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=9444)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: project/models/<model name>).",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--max-val-batches",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available.")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    max_batches: int | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_samples = 0
    predictions: list[int] = []
    targets: list[int] = []

    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for batch_index, (images, labels) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break

            images = images.to(device)
            labels = labels.to(device)

            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, labels)

            if training:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
            targets.extend(labels.detach().cpu().tolist())

    if total_samples == 0:
        raise RuntimeError("No batches were processed.")

    return {
        "loss": total_loss / total_samples,
        "accuracy": accuracy_score(targets, predictions),
        "macro_f1": f1_score(
            targets,
            predictions,
            labels=list(range(len(LABELS))),
            average="macro",
            zero_division=0,
        ),
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    args: argparse.Namespace,
    epoch: int,
    metrics: dict[str, float],
) -> None:
    checkpoint = {
        "model_name": args.model,
        "model_state_dict": model.state_dict(),
        "labels": LABELS,
        "epoch": epoch,
        "validation_metrics": metrics,
        "config": {
            "pretrained": args.pretrained,
            "freeze_backbone": args.freeze_backbone,
            "image_size": args.image_size,
            "seed": args.seed,
        },
    }
    torch.save(checkpoint, path)


def main() -> None:
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1.")
    if args.patience < 1:
        raise ValueError("--patience must be at least 1.")

    set_seed(args.seed)
    device = select_device(args.device)
    if args.output_dir is None:
        args.output_dir = args.project_dir / "models" / args.model
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    loaders = build_dataloaders(
        args.project_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    model = build_model(
        args.model,
        pretrained=(args.pretrained and args.initial_checkpoint is None),
        freeze_backbone=args.freeze_backbone,
    )
    if args.initial_checkpoint is not None:
        initial_checkpoint_path = args.initial_checkpoint.resolve()
        if not initial_checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Initial checkpoint not found: {initial_checkpoint_path}"
            )
        initial_checkpoint = torch.load(
            initial_checkpoint_path, map_location="cpu", weights_only=False
        )
        if initial_checkpoint["model_name"] != args.model:
            raise ValueError(
                "Initial checkpoint model does not match --model: "
                f"{initial_checkpoint['model_name']} != {args.model}"
            )
        model.load_state_dict(initial_checkpoint["model_state_dict"])
        print(f"initialized from: {initial_checkpoint_path}")
    model = model.to(device)
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )
    criterion = nn.CrossEntropyLoss()

    total_parameters, trainable_count = count_parameters(model)
    print(f"device: {device}")
    print(f"model: {args.model}")
    print(
        f"parameters: total={total_parameters:,}, trainable={trainable_count:,}"
    )
    print(
        f"samples: train={len(loaders['train'].dataset)}, "
        f"val={len(loaders['val'].dataset)}"
    )

    history: list[dict[str, float | int]] = []
    best_f1 = -1.0
    epochs_without_improvement = 0
    checkpoint_path = args.output_dir / "best_model.pt"
    start_time = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            loaders["train"],
            criterion,
            device,
            optimizer=optimizer,
            max_batches=args.max_train_batches,
        )
        val_metrics = run_epoch(
            model,
            loaders["val"],
            criterion,
            device,
            max_batches=args.max_val_batches,
        )
        scheduler.step(val_metrics["macro_f1"])

        row: dict[str, float | int] = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(row)
        print(
            f"epoch {epoch:02d}/{args.epochs:02d} | "
            f"train loss={train_metrics['loss']:.4f} "
            f"acc={train_metrics['accuracy']:.4f} "
            f"f1={train_metrics['macro_f1']:.4f} | "
            f"val loss={val_metrics['loss']:.4f} "
            f"acc={val_metrics['accuracy']:.4f} "
            f"f1={val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            epochs_without_improvement = 0
            save_checkpoint(checkpoint_path, model, args, epoch, val_metrics)
            print(f"  saved new best checkpoint: {checkpoint_path}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"early stopping after {epoch} epochs")
                break

    elapsed_seconds = time.perf_counter() - start_time
    history_path = args.output_dir / "history.json"
    with history_path.open("w", encoding="utf-8") as history_file:
        json.dump(history, history_file, indent=2)

    print(f"best validation macro-F1: {best_f1:.4f}")
    print(f"history: {history_path}")
    print(f"elapsed: {elapsed_seconds:.1f} seconds")


if __name__ == "__main__":
    main()
