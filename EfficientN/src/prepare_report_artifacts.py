"""Create the report table data and one verified baseline-corrected test example."""

from __future__ import annotations

import json
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
REPORT = PROJECT_ROOT.parent / "report"
FIGURES = REPORT / "figures"

EXPERIMENTS = [
    ("efficientnet_b0_10e", "EfficientNet-B0 baseline"),
    ("stage4_residual_10e", "+ Stage-4 residual"),
    ("multiscale_regions_10e", "+ Multi-scale regions"),
    ("colour_stats_10e", "+ Explicit colour statistics"),
    ("colour_texture_10e", "+ Learned colour texture"),
    ("scalar_gate_10e", "+ Scalar fusion gate"),
    ("full_rmof_10e", "Full RMOF (end-to-end)"),
    ("frozen_full_rmof_10e", "Frozen-base full RMOF candidate"),
    ("frozen_regions_stats_64_ce_10e", "Frozen region+statistics candidate"),
    ("frozen_regions_stats_64_leaf_10e", "Frozen region+statistics (leaf augmentation)"),
    ("frozen_regions_stats_64_gate_10e", "Frozen region+statistics (global gate)"),
]
CLASS_NAMES = ["N0", "N75", "NFull"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="report10_40_10_50_outputs")
    parser.add_argument("--split-manifest", default="split_40_10_50.csv")
    parser.add_argument("--classic-experiment", default="frozen_regions_stats_64_ce_10e")
    return parser.parse_args()


def load_metrics(outputs: Path, experiment: str) -> dict:
    path = outputs / experiment / "seed_42" / "metrics.json"
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_result_table(outputs: Path) -> pd.DataFrame:
    rows = []
    for experiment, method in EXPERIMENTS:
        metrics = load_metrics(outputs, experiment)
        rows.append(
            {
                "experiment": experiment,
                "method": method,
                "deficiency_f1": metrics["f1_deficiency"],
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "f1_n0": metrics["f1_n0"],
                "f1_n75": metrics["f1_n75"],
                "f1_nfull": metrics["f1_nfull"],
                "ordinal_mae": metrics["ordinal_mae"],
                "endpoint_confusion": metrics["endpoint_confusion"],
                "trainable_parameters": metrics["parameters"],
                "latency_ms_per_image": metrics["inference_ms_per_image"],
                "best_validation_macro_f1": metrics["best_validation_macro_f1"],
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(REPORT / "report10_results.csv", index=False)
    return table


def write_classic_case(outputs: Path, classic_experiment: str, split_manifest: str) -> None:
    baseline = pd.read_csv(
        outputs / "efficientnet_b0_10e" / "seed_42" / "predictions.csv"
    ).rename(columns={"prediction": "baseline_prediction"})
    selected = pd.read_csv(
        outputs / classic_experiment / "seed_42" / "predictions.csv"
    ).rename(columns={"prediction": "selected_prediction"})
    fixed = baseline.merge(selected[["filename", "selected_prediction"]], on="filename")
    fixed = fixed.loc[
        (fixed["baseline_prediction"] != fixed["label"])
        & (fixed["selected_prediction"] == fixed["label"])
    ].copy()
    if fixed.empty:
        raise RuntimeError("The selected model did not correct a baseline test error.")

    # Prefer the ambiguous middle treatment, then use filename for determinism.
    fixed["priority"] = (fixed["label"] != 1).astype(int)
    chosen = fixed.sort_values(["priority", "filename"]).iloc[0]
    split = pd.read_csv(PROJECT_ROOT / split_manifest)
    image_path = split.loc[split["filename"] == chosen.filename, "filepath"].iloc[0]
    source = PROJECT_ROOT / image_path

    case = pd.DataFrame(
        [
            {
                "filename": chosen.filename,
                "filepath": image_path,
                "true_label": CLASS_NAMES[int(chosen.label)],
                "baseline_prediction": CLASS_NAMES[int(chosen.baseline_prediction)],
                "candidate_prediction": CLASS_NAMES[int(chosen.selected_prediction)],
                "candidate_experiment": classic_experiment,
                "selection_rule": "baseline incorrect; candidate model correct",
            }
        ]
    )
    case.to_csv(FIGURES / "classic_case.csv", index=False)

    with Image.open(source) as image:
        rgb = ImageOps.exif_transpose(image).convert("RGB").copy()
    # This camera frame is stored sideways without an EXIF orientation tag.
    rgb = rgb.transpose(Image.Transpose.ROTATE_270)
    figure, axis = plt.subplots(figsize=(4.7, 4.2))
    axis.imshow(rgb)
    axis.set_axis_off()
    axis.set_title(
        "True: N75 | EfficientNet-B0: NFull | RMOF candidate: N75",
        fontsize=10,
        pad=8,
    )
    figure.tight_layout(pad=0.2)
    figure.savefig(FIGURES / "classic_case.png", dpi=300, bbox_inches="tight")
    figure.savefig(FIGURES / "classic_case.pdf", bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    outputs = PROJECT_ROOT / args.source
    FIGURES.mkdir(parents=True, exist_ok=True)
    write_result_table(outputs)
    write_classic_case(outputs, args.classic_experiment, args.split_manifest)
    print(f"Wrote {REPORT / 'report10_results.csv'} and classic-case artifacts to {FIGURES}")


if __name__ == "__main__":
    main()
