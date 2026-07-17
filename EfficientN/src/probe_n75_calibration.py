"""Fit a one-dimensional N75 logit bias on validation data and evaluate it once."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

import rmof_efficientnet as rmof


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="split_40_10_50.csv")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def collect_logits(model: torch.nn.Module, loader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    labels, logits = [], []
    model.eval()
    with torch.inference_mode():
        for images, target, _ in loader:
            output = model(images.to(device, non_blocking=device.type == "cuda"))
            labels.append(target.numpy())
            logits.append(output["logits"].float().cpu().numpy())
    return np.concatenate(labels), np.concatenate(logits)


def metrics(labels: np.ndarray, prediction: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(labels, prediction)),
        "macro_f1": float(f1_score(labels, prediction, average="macro")),
        "f1_n75": float(f1_score(labels, prediction, labels=[1], average=None)[0]),
        "confusion": confusion_matrix(labels, prediction, labels=[0, 1, 2]).tolist(),
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    loader_args = argparse.Namespace(
        csv=args.csv, data_root=".", batch_size=args.batch_size, num_workers=0,
        device=args.device, augmentation="mild", train_fraction=1.0, seed=42,
        cache_images=True, image_size=224,
    )
    _, validation, test = rmof.make_loaders(loader_args)
    config = replace(rmof.preset_config("cnn_baseline"), dropout=0.4)
    model = rmof.build_model("efficientnet_b0", config, pretrained=True).to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    val_labels, val_logits = collect_logits(model, validation, device)
    test_labels, test_logits = collect_logits(model, test, device)

    candidates = np.linspace(-1.0, 1.0, 401)
    scores = []
    for bias in candidates:
        prediction = (val_logits + np.array([0.0, bias, 0.0])).argmax(axis=1)
        scores.append(f1_score(val_labels, prediction, average="macro"))
    best_index = int(np.argmax(scores))
    best_bias = float(candidates[best_index])
    baseline_prediction = test_logits.argmax(axis=1)
    calibrated_prediction = (test_logits + np.array([0.0, best_bias, 0.0])).argmax(axis=1)
    result = {
        "selected_on": "validation_macro_f1",
        "n75_logit_bias": best_bias,
        "validation_macro_f1": float(scores[best_index]),
        "test_baseline": metrics(test_labels, baseline_prediction),
        "test_calibrated": metrics(test_labels, calibrated_prediction),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
