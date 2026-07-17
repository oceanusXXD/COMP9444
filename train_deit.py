"""Fine-tune ImageNet-pretrained DeiT-Tiny for maize nitrogen classification.

Parameter locations
-------------------
Primary experimental parameters are declared in ``parse_args`` and should be
changed through CLI flags rather than by editing source code:
``--learning-rate``, ``--weight-decay``, ``--dropout``,
``--label-smoothing``, ``--augmentation``, and ``--seed``.

Training-budget parameters are also in ``parse_args``:
``--epochs``, ``--warmup-epochs``, ``--batch-size``, ``--patience``,
``--grad-clip``, ``--workers``, and the warm-up/minimum learning rates.

Data, model, output, and evaluation paths are controlled by ``--csv``,
``--data-root``, ``--pretrained-checkpoint``, ``--output-dir``,
``--checkpoint``, and the evaluation flags. Augmentation definitions are kept
in ``maize_data.build_transform``. Class names and fixed split validation are
kept in ``split_utils`` and are not hyperparameters.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from maize_data import (
    MaizeNitrogenDataset,
    build_transform,
)
from split_utils import (
    CLASS_NAMES,
    discover_data_root,
    load_and_validate_split_csv,
    validate_image_paths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="split.csv")
    parser.add_argument(
        "--pretrained-checkpoint",
        default="deit_tiny_patch16_224-a1311bcf.pth",
        help="Local official DeiT-Tiny ImageNet checkpoint (no network download)",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="Directory containing Images/. It is auto-discovered when omitted.",
    )
    parser.add_argument("--output-dir", default="outputs/deit_tiny_seed42")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--warmup-learning-rate", type=float, default=1e-6)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--augmentation", choices=("mild", "medium"), default="mild")
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--overfit-samples",
        type=int,
        default=0,
        help="Diagnostic mode: train/evaluate on a balanced subset with augmentation disabled",
    )
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:0, or cpu")
    parser.add_argument("--no-amp", action="store_true", help="Disable CUDA mixed precision")
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Evaluate the held-out test set after training. Use only after tuning is final.",
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Load --checkpoint and evaluate --eval-split without training.",
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--eval-split", choices=("val", "test"), default="val")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        raise ValueError("epochs and batch-size must be positive; workers cannot be negative")
    if args.warmup_epochs < 0 or args.warmup_epochs >= args.epochs:
        raise ValueError("warmup-epochs must be >= 0 and smaller than epochs")
    if args.patience < 1:
        raise ValueError("patience must be positive")
    if args.overfit_samples != 0 and args.overfit_samples < len(CLASS_NAMES):
        raise ValueError(f"overfit-samples must be 0 or at least {len(CLASS_NAMES)}")
    if args.overfit_samples and (args.evaluate_only or args.evaluate_test):
        raise ValueError("overfit-samples cannot be combined with test/evaluate-only modes")
    if args.evaluate_only and not args.checkpoint:
        raise ValueError("--evaluate-only requires --checkpoint")
    if args.warmup_learning_rate > args.learning_rate:
        raise ValueError("warmup-learning-rate cannot exceed learning-rate")
    if args.min_learning_rate > args.learning_rate:
        raise ValueError("min-learning-rate cannot exceed learning-rate")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def build_loader(dataset, batch_size: int, workers: int, shuffle: bool, device, seed: int):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def load_weights_only(path: Path):
    """Load a tensor checkpoint without enabling arbitrary pickle execution."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        # Compatibility for older PyTorch versions; requirements specify >= 2.1.
        return torch.load(path, map_location="cpu")


def resolve_pretrained_checkpoint(value: str) -> Path:
    checkpoint_path = Path(value).expanduser()
    candidates = [checkpoint_path]
    if not checkpoint_path.is_absolute():
        candidates.append(Path(__file__).resolve().parent / checkpoint_path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    searched = "\n".join(str(candidate.resolve()) for candidate in candidates)
    raise FileNotFoundError(
        "Local DeiT-Tiny pretrained checkpoint was not found. Searched:\n" + searched
    )


def create_model(dropout: float, pretrained_checkpoint: Path | None = None) -> nn.Module:
    # This is the same DeiT-Tiny architecture registered in the reference DeiT repo:
    # patch size 16, embedding size 192, 12 blocks, and 3 attention heads.
    model = timm.create_model(
        "deit_tiny_patch16_224",
        pretrained=False,
        num_classes=len(CLASS_NAMES),
        drop_rate=dropout,
    )
    if pretrained_checkpoint is None:
        return model

    checkpoint = load_weights_only(pretrained_checkpoint)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported checkpoint object in {pretrained_checkpoint}")
    state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    if not isinstance(state_dict, dict):
        raise ValueError(
            f"Checkpoint does not contain a model state dictionary: {pretrained_checkpoint}"
        )
    if state_dict and all(str(key).startswith("module.") for key in state_dict):
        state_dict = {str(key)[7:]: value for key, value in state_dict.items()}
    else:
        state_dict = dict(state_dict)

    # The official checkpoint predicts 1,000 ImageNet classes. Our model predicts
    # N0/N75/NFull, so its freshly initialized three-class head must be retained.
    state_dict.pop("head.weight", None)
    state_dict.pop("head.bias", None)
    incompatible = model.load_state_dict(state_dict, strict=False)
    allowed_missing = {"head.weight", "head.bias"}
    unexpected_missing = set(incompatible.missing_keys).difference(allowed_missing)
    if unexpected_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "Local checkpoint is incompatible with deit_tiny_patch16_224. "
            f"Missing: {sorted(unexpected_missing)}; "
            f"unexpected: {sorted(incompatible.unexpected_keys)}"
        )
    print(f"Loaded local ImageNet pretrained weights: {pretrained_checkpoint}")
    return model


def create_step_scheduler(optimizer, args, steps_per_epoch: int) -> LambdaLR:
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch
    start_factor = args.warmup_learning_rate / args.learning_rate
    min_factor = args.min_learning_rate / args.learning_rate

    def learning_rate_factor(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            progress = step / max(1, warmup_steps)
            return start_factor + progress * (1.0 - start_factor)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_factor + (1.0 - min_factor) * cosine

    return LambdaLR(optimizer, lr_lambda=learning_rate_factor)


def autocast_context(use_amp: bool):
    return torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp else nullcontext()


def create_grad_scaler(use_amp: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=use_amp)


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, device, use_amp, grad_clip):
    model.train()
    running_loss = 0.0
    correct = 0
    sample_count = 0
    progress = tqdm(loader, desc="Train", leave=False)
    for images, labels, _ in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        sample_count += batch_size
        progress.set_postfix(loss=f"{loss.item():.4f}")
    return running_loss / sample_count, correct / sample_count


@torch.inference_mode()
def evaluate(model, loader, criterion, device, use_amp):
    model.eval()
    running_loss = 0.0
    sample_count = 0
    targets = []
    predictions = []
    filepaths = []
    for images, labels, paths in tqdm(loader, desc="Evaluate", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast_context(use_amp):
            logits = model(images)
            loss = criterion(logits, labels)
        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        sample_count += batch_size
        targets.extend(labels.cpu().tolist())
        predictions.extend(logits.argmax(dim=1).cpu().tolist())
        filepaths.extend(paths)

    targets_array = np.asarray(targets, dtype=np.int64)
    predictions_array = np.asarray(predictions, dtype=np.int64)
    metrics = {
        "loss": running_loss / sample_count,
        "accuracy": accuracy_score(targets_array, predictions_array),
        "balanced_accuracy": balanced_accuracy_score(targets_array, predictions_array),
        "macro_f1": f1_score(
            targets_array,
            predictions_array,
            labels=list(range(len(CLASS_NAMES))),
            average="macro",
            zero_division=0,
        ),
        "level_mae": np.abs(targets_array - predictions_array).mean(),
    }
    return metrics, targets_array, predictions_array, filepaths


def to_builtin(value):
    if isinstance(value, dict):
        return {key: to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_evaluation(output_dir: Path, split: str, metrics, targets, predictions, paths) -> None:
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    report = classification_report(
        targets,
        predictions,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    (split_dir / "metrics.json").write_text(
        json.dumps(to_builtin({**metrics, "per_class": report}), indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(report).transpose().to_csv(split_dir / "classification_report.csv")
    pd.DataFrame(
        {
            "filepath": paths,
            "target_index": targets,
            "target": [CLASS_NAMES[index] for index in targets],
            "prediction_index": predictions,
            "prediction": [CLASS_NAMES[index] for index in predictions],
        }
    ).to_csv(split_dir / "predictions.csv", index=False)

    matrix = confusion_matrix(targets, predictions, labels=list(range(len(CLASS_NAMES))))
    row_totals = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix,
        row_totals,
        out=np.zeros_like(matrix, dtype=float),
        where=row_totals != 0,
    )
    pd.DataFrame(matrix, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        split_dir / "confusion_matrix.csv"
    )
    pd.DataFrame(normalized, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        split_dir / "confusion_matrix_normalized.csv"
    )

    fig, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(normalized, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(image, ax=axis)
    axis.set(
        xticks=range(len(CLASS_NAMES)),
        yticks=range(len(CLASS_NAMES)),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        xlabel="Predicted label",
        ylabel="True label",
        title=f"Normalized confusion matrix ({split})",
    )
    for row in range(len(CLASS_NAMES)):
        for column in range(len(CLASS_NAMES)):
            axis.text(
                column,
                row,
                f"{normalized[row, column]:.2f}",
                ha="center",
                va="center",
                color="white" if normalized[row, column] > 0.5 else "black",
            )
    fig.tight_layout()
    fig.savefig(split_dir / "confusion_matrix_normalized.png", dpi=200)
    plt.close(fig)


def save_history_plot(history: list[dict], output_dir: Path) -> None:
    history_frame = pd.DataFrame(history)
    history_frame.to_csv(output_dir / "history.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history_frame["epoch"], history_frame["train_loss"], label="train")
    axes[0].plot(history_frame["epoch"], history_frame["val_loss"], label="validation")
    axes[0].set(title="Loss", xlabel="Epoch")
    axes[0].legend()
    axes[1].plot(history_frame["epoch"], history_frame["train_accuracy"], label="train accuracy")
    axes[1].plot(history_frame["epoch"], history_frame["val_macro_f1"], label="validation Macro-F1")
    axes[1].set(title="Training progress", xlabel="Epoch")
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=200)
    plt.close(fig)


def load_checkpoint(model: nn.Module, checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    return checkpoint


def make_dataset(frame, split, data_root, args):
    return MaizeNitrogenDataset(
        frame=frame,
        split=split,
        data_root=data_root,
        transform=build_transform(split, args.augmentation),
    )


def make_balanced_overfit_frame(frame: pd.DataFrame, sample_count: int, seed: int):
    train_frame = frame.loc[frame["split"] == "train"]
    if sample_count > len(train_frame):
        raise ValueError(
            f"overfit-samples={sample_count} exceeds the {len(train_frame)} training samples"
        )
    base_count, remainder = divmod(sample_count, len(CLASS_NAMES))
    selected_groups = []
    for class_index, class_name in enumerate(CLASS_NAMES):
        class_count = base_count + int(class_index < remainder)
        class_rows = train_frame.loc[train_frame["label"] == class_name]
        if class_count > len(class_rows):
            raise ValueError(f"Not enough {class_name} samples for overfit diagnostic")
        selected_groups.append(
            class_rows.sample(n=class_count, random_state=seed + class_index)
        )
    return pd.concat(selected_groups, ignore_index=True)


def run_split_evaluation(model, frame, split, data_root, args, criterion, device, use_amp, output_dir):
    dataset = make_dataset(frame, split, data_root, args)
    loader = build_loader(dataset, args.batch_size, args.workers, False, device, args.seed)
    metrics, targets, predictions, paths = evaluate(model, loader, criterion, device, use_amp)
    save_evaluation(output_dir, split, metrics, targets, predictions, paths)
    print(
        f"{split}: accuracy={metrics['accuracy']:.4f}, "
        f"balanced_accuracy={metrics['balanced_accuracy']:.4f}, "
        f"macro_f1={metrics['macro_f1']:.4f}, level_mae={metrics['level_mae']:.4f}"
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)
    device = choose_device(args.device)
    use_amp = device.type == "cuda" and not args.no_amp
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = Path(args.csv).expanduser().resolve()
    frame = load_and_validate_split_csv(csv_path)
    data_root = discover_data_root(csv_path, frame, args.data_root)
    validate_image_paths(frame, data_root)
    split_summary = frame.groupby(["split", "label"]).size().unstack(fill_value=0)
    print(f"Device: {device} | AMP: {use_amp}")
    print(f"Data root: {data_root}")
    print(split_summary.to_string())

    configuration = vars(args).copy()
    configuration.update(
        {
            "resolved_csv": str(csv_path),
            "resolved_data_root": str(data_root),
            "class_names": list(CLASS_NAMES),
            "label_mapping": {name: index for index, name in enumerate(CLASS_NAMES)},
        }
    )
    config_filename = "evaluation_config.json" if args.evaluate_only else "config.json"
    (output_dir / config_filename).write_text(
        json.dumps(configuration, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    if args.evaluate_only:
        model = create_model(dropout=args.dropout).to(device)
        load_checkpoint(model, Path(args.checkpoint).expanduser().resolve(), device)
        run_split_evaluation(
            model,
            frame,
            args.eval_split,
            data_root,
            args,
            criterion,
            device,
            use_amp,
            output_dir,
        )
        return

    if args.overfit_samples:
        diagnostic_frame = make_balanced_overfit_frame(
            frame, args.overfit_samples, args.seed
        )
        # Both loaders intentionally use the same small subset and deterministic
        # evaluation preprocessing. This diagnoses the pipeline; it is not a result.
        diagnostic_dataset = MaizeNitrogenDataset(
            frame=diagnostic_frame,
            split="train",
            data_root=data_root,
            transform=build_transform("val", args.augmentation),
        )
        train_dataset = diagnostic_dataset
        val_dataset = diagnostic_dataset
        print(
            f"OVERFIT DIAGNOSTIC: train and validation both use the same "
            f"{len(diagnostic_dataset)} samples; test evaluation is disabled."
        )
    else:
        train_dataset = make_dataset(frame, "train", data_root, args)
        val_dataset = make_dataset(frame, "val", data_root, args)
    train_loader = build_loader(
        train_dataset, args.batch_size, args.workers, True, device, args.seed
    )
    val_loader = build_loader(
        val_dataset, args.batch_size, args.workers, False, device, args.seed
    )

    # The project design requires ImageNet pretraining because only 840 images are used to train.
    pretrained_checkpoint = resolve_pretrained_checkpoint(args.pretrained_checkpoint)
    configuration["resolved_pretrained_checkpoint"] = str(pretrained_checkpoint)
    (output_dir / "config.json").write_text(
        json.dumps(configuration, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    model = create_model(
        dropout=args.dropout, pretrained_checkpoint=pretrained_checkpoint
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    print(
        f"Model: deit_tiny_patch16_224 | parameters={parameter_count:,} "
        f"| trainable={trainable_parameter_count:,}"
    )

    optimizer = AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = create_step_scheduler(optimizer, args, len(train_loader))
    scaler = create_grad_scaler(use_amp)
    best_macro_f1 = -1.0
    epochs_without_improvement = 0
    best_checkpoint_path = output_dir / "best_checkpoint.pt"
    history = []
    training_start = time.perf_counter()

    for epoch_index in range(args.epochs):
        epoch = epoch_index + 1
        epoch_start = time.perf_counter()
        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scheduler,
            scaler,
            device,
            use_amp,
            args.grad_clip,
        )
        val_metrics, targets, predictions, paths = evaluate(
            model, val_loader, criterion, device, use_amp
        )
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_level_mae": val_metrics["level_mae"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_seconds": time.perf_counter() - epoch_start,
        }
        history.append(to_builtin(epoch_record))
        save_history_plot(history, output_dir)
        print(
            f"Epoch {epoch:03d}/{args.epochs}: train_loss={train_loss:.4f}, "
            f"train_acc={train_accuracy:.4f}, val_loss={val_metrics['loss']:.4f}, "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_macro_f1 + 1e-8:
            best_macro_f1 = val_metrics["macro_f1"]
            epochs_without_improvement = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "best_val_macro_f1": best_macro_f1,
                    "class_names": CLASS_NAMES,
                    "args": vars(args),
                    "parameter_count": parameter_count,
                },
                best_checkpoint_path,
            )
            save_evaluation(
                output_dir, "val", val_metrics, targets, predictions, paths
            )
            print(f"  Saved new best checkpoint: {best_checkpoint_path}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping after {args.patience} epochs without improvement")
                break

    training_seconds = time.perf_counter() - training_start
    (output_dir / "training_summary.json").write_text(
        json.dumps(
            {
                "best_val_macro_f1": best_macro_f1,
                "epochs_completed": len(history),
                "training_seconds": training_seconds,
                "parameter_count": parameter_count,
                "trainable_parameter_count": trainable_parameter_count,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    load_checkpoint(model, best_checkpoint_path, device)
    print(f"Best validation Macro-F1: {best_macro_f1:.4f}")
    if args.evaluate_test:
        print("Explicit test evaluation enabled; evaluating held-out test split once.")
        run_split_evaluation(
            model,
            frame,
            "test",
            data_root,
            args,
            criterion,
            device,
            use_amp,
            output_dir,
        )
    else:
        print("Test split was not evaluated. Add --evaluate-test only after tuning is final.")


if __name__ == "__main__":
    main()
