"""Export report tables and plotting source data as plain CSV files.

The script reads only saved experiment artifacts. It does not train models or
change checkpoints. The generated files live in ../report/tables/ and make the
numbers used by report/main.tex easy to plot in Excel, Python, or MATLAB.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean, stdev
from typing import Iterable


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
REPORT = PROJECT_ROOT.parent / "report"
TABLES = REPORT / "tables"
CURRENT_OUTPUTS = PROJECT_ROOT / "report10_40_10_50_outputs"
LEGACY_SUMMARY = PROJECT_ROOT / "ablation_outputs" / "summary" / "baseline_comparison.csv"
CLASS_NAMES = ("N0", "N75", "NFull")

METHODS = {
    "efficientnet_b0_10e": "EfficientNet-B0 baseline",
    "stage4_residual_10e": "+ Stage-4 residual",
    "multiscale_regions_10e": "+ Multi-scale regions",
    "colour_stats_10e": "+ Explicit colour statistics",
    "colour_texture_10e": "+ Learned colour texture",
    "scalar_gate_10e": "+ Scalar fusion gate",
    "full_rmof_10e": "Full RMOF (end-to-end)",
    "frozen_full_rmof_10e": "Frozen-base full RMOF candidate",
    "frozen_regions_stats_64_ce_10e": "Frozen region+statistics candidate",
    "frozen_regions_stats_64_leaf_10e": "Frozen region+statistics (leaf augmentation)",
    "frozen_regions_stats_64_gate_10e": "Frozen region+statistics (global gate)",
}

END_TO_END = (
    "efficientnet_b0_10e",
    "stage4_residual_10e",
    "multiscale_regions_10e",
    "colour_stats_10e",
    "colour_texture_10e",
    "scalar_gate_10e",
    "full_rmof_10e",
)
FROZEN_FOLLOWUPS = (
    "efficientnet_b0_10e",
    "frozen_full_rmof_10e",
    "frozen_regions_stats_64_ce_10e",
    "frozen_regions_stats_64_leaf_10e",
    "frozen_regions_stats_64_gate_10e",
)

METRICS = (
    "deficiency_f1",
    "accuracy",
    "macro_f1",
    "f1_n0",
    "f1_n75",
    "f1_nfull",
    "ordinal_mae",
    "endpoint_confusion",
    "trainable_parameters",
    "latency_ms_per_image",
    "best_validation_macro_f1",
)
TABLE_FIELDS = ("experiment", "method", *METRICS)
MASTER_FIELDS = (
    "protocol",
    "experiment",
    "method",
    "n_seeds",
    *METRICS,
    *(f"{metric}_std" for metric in METRICS if metric not in {"best_validation_macro_f1"}),
)


def write_csv(path: Path, fieldnames: Iterable[str], rows: Iterable[dict]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def current_metrics(experiment: str) -> dict:
    path = CURRENT_OUTPUTS / experiment / "seed_42" / "metrics.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing required report artifact: {path}")
    with path.open(encoding="utf-8") as handle:
        metrics = json.load(handle)
    return {
        "experiment": experiment,
        "method": METHODS[experiment],
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


def read_legacy_rows() -> list[dict]:
    if not LEGACY_SUMMARY.is_file():
        raise FileNotFoundError(f"Missing legacy summary: {LEGACY_SUMMARY}")
    with LEGACY_SUMMARY.open(encoding="utf-8", newline="") as handle:
        source = {row["experiment"]: row for row in csv.DictReader(handle)}
    names = (("resnet18", "ResNet-18"), ("deit_tiny", "DeiT-Tiny"))
    rows = []
    for experiment, method in names:
        record = source[experiment]
        deficiency_scores = []
        for path in sorted((PROJECT_ROOT / "ablation_outputs" / experiment).glob("seed_*/predictions.csv")):
            with path.open(encoding="utf-8", newline="") as handle:
                predictions = list(csv.DictReader(handle))
            counts = Counter(
                (int(row["label"]) != 2, int(row["prediction"]) != 2)
                for row in predictions
            )
            true_positive = counts[(True, True)]
            false_positive = counts[(False, True)]
            false_negative = counts[(True, False)]
            denominator = 2 * true_positive + false_positive + false_negative
            deficiency_scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
        row = {
            "protocol": "legacy_3_epoch_reference",
            "experiment": experiment,
            "method": method,
            "n_seeds": record["n_seeds"],
            "deficiency_f1": mean(deficiency_scores),
            "accuracy": record["accuracy_mean"],
            "macro_f1": record["macro_f1_mean"],
            "f1_n0": record["f1_n0_mean"],
            "f1_n75": record["f1_n75_mean"],
            "f1_nfull": record["f1_nfull_mean"],
            "ordinal_mae": record["ordinal_mae_mean"],
            "endpoint_confusion": record["endpoint_confusion_mean"],
            "trainable_parameters": record["parameters_mean"],
            "latency_ms_per_image": record["inference_ms_per_image_mean"],
        }
        for metric, source_metric in (
            ("accuracy", "accuracy"),
            ("macro_f1", "macro_f1"),
            ("f1_n0", "f1_n0"),
            ("f1_n75", "f1_n75"),
            ("f1_nfull", "f1_nfull"),
            ("ordinal_mae", "ordinal_mae"),
            ("endpoint_confusion", "endpoint_confusion"),
            ("trainable_parameters", "parameters"),
            ("latency_ms_per_image", "inference_ms_per_image"),
        ):
            row[f"{metric}_std"] = record[f"{source_metric}_std"]
        row["deficiency_f1_std"] = stdev(deficiency_scores) if len(deficiency_scores) > 1 else 0.0
        rows.append(row)
    return rows


def write_learning_curves() -> None:
    rows = []
    for experiment in ("efficientnet_b0_10e", "colour_texture_10e"):
        path = CURRENT_OUTPUTS / experiment / "seed_42" / "history.csv"
        with path.open(encoding="utf-8", newline="") as handle:
            for record in csv.DictReader(handle):
                rows.append(
                    {
                        "experiment": experiment,
                        "method": METHODS[experiment],
                        "epoch": record["epoch"],
                        "train_loss": record["train_loss"],
                        "train_macro_f1": record["train_macro_f1"],
                        "validation_loss": record["val_loss"],
                        "validation_macro_f1": record["val_macro_f1"],
                    }
                )
    write_csv(
        TABLES / "learning_curves.csv",
        ("experiment", "method", "epoch", "train_loss", "train_macro_f1", "validation_loss", "validation_macro_f1"),
        rows,
    )


def write_confusion_data() -> None:
    rows = []
    for experiment in ("efficientnet_b0_10e", "colour_texture_10e"):
        path = CURRENT_OUTPUTS / experiment / "seed_42" / "predictions.csv"
        with path.open(encoding="utf-8", newline="") as handle:
            predictions = list(csv.DictReader(handle))
        counts = Counter((int(row["label"]), int(row["prediction"])) for row in predictions)
        actual_counts = Counter(int(row["label"]) for row in predictions)
        for actual, actual_name in enumerate(CLASS_NAMES):
            for predicted, predicted_name in enumerate(CLASS_NAMES):
                count = counts[(actual, predicted)]
                rows.append(
                    {
                        "experiment": experiment,
                        "method": METHODS[experiment],
                        "true_label": actual_name,
                        "predicted_label": predicted_name,
                        "count": count,
                        "row_normalized_rate": count / actual_counts[actual],
                    }
                )
    write_csv(
        TABLES / "confusion_baseline_vs_colour_texture.csv",
        ("experiment", "method", "true_label", "predicted_label", "count", "row_normalized_rate"),
        rows,
    )


def main() -> None:
    current_rows = [current_metrics(experiment) for experiment in METHODS]
    write_csv(REPORT / "data.csv", MASTER_FIELDS, [
        {"protocol": "current_40_10_50_seed42_10_epoch", "n_seeds": 1, **row} for row in current_rows
    ] + read_legacy_rows())
    write_csv(TABLES / "end_to_end_ablation.csv", TABLE_FIELDS, [current_metrics(name) for name in END_TO_END])
    write_csv(TABLES / "frozen_followups.csv", TABLE_FIELDS, [current_metrics(name) for name in FROZEN_FOLLOWUPS])
    write_csv(TABLES / "legacy_backbones.csv", MASTER_FIELDS, read_legacy_rows())
    write_learning_curves()
    write_confusion_data()
    (TABLES / "README.md").write_text(
        "# Plot Data\n\n"
        "- `end_to_end_ablation.csv`: Table I / end-to-end module ablation.\n"
        "- `frozen_followups.csv`: Table II / frozen-base follow-ups.\n"
        "- `legacy_backbones.csv`: Table III / legacy three-seed references.\n"
        "- `learning_curves.csv`: epoch-level training and validation values.\n"
        "- `confusion_baseline_vs_colour_texture.csv`: count and row-normalized confusion values.\n"
        "- `../data.csv`: one combined machine-readable result table.\n",
        encoding="utf-8",
    )
    print(f"Wrote CSV tables to {TABLES} and master data to {REPORT / 'data.csv'}")


if __name__ == "__main__":
    main()
