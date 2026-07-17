"""Aggregate seed-level outputs into the requested tables and plots."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXPERIMENTS = {
    "report10": [
        "efficientnet_b0_10e",
        "stage4_residual_10e", "multiscale_regions_10e", "colour_stats_10e",
        "colour_texture_10e", "scalar_gate_10e", "full_rmof_10e",
        "frozen_full_rmof_10e",
    ],
    "quick": ["baseline_5e", "residual_deep_5e"],
    "baseline": ["simple_cnn", "efficientnet_b0", "resnet18", "deit_tiny"],
    "main": [
        "cnn_baseline", "multiscale", "regions", "color_stats", "color_texture",
        "region_cross_attention", "ordinal_supervision",
    ],
    "fusion": [
        "fusion_concat", "fusion_gate", "fusion_cross_attention", "fusion_region_attention",
    ],
    "loss": ["loss_ce", "loss_ce_emd", "loss_ce_score", "loss_ce_emd_score"],
}
METRICS = [
    "accuracy", "f1_deficiency", "macro_f1", "f1_n0", "f1_n75", "f1_nfull", "ordinal_mae",
    "endpoint_confusion", "endpoint_confusion_rate", "parameters", "inference_ms_per_image",
]


def load_records(source: Path) -> pd.DataFrame:
    records = []
    for path in source.glob("**/metrics.json"):
        with path.open(encoding="utf-8") as handle:
            record = json.load(handle)
        for metric in METRICS:
            record.setdefault(metric, np.nan)
        record["run_dir"] = str(path.parent)
        records.append(record)
    if not records:
        raise FileNotFoundError(f"No metrics.json files found under {source}")
    return pd.DataFrame(records)


def aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["experiment", "model", "n_seeds"])
    grouped = frame.groupby(["experiment", "model"])
    result = grouped.size().reset_index(name="n_seeds")
    values = grouped[METRICS].agg(["mean", "std"])
    values.columns = [f"{metric}_{stat}" for metric, stat in values.columns]
    values = values.reset_index()
    return result.merge(values, on=["experiment", "model"]).fillna(0.0).sort_values("experiment")


def write_tables(records: pd.DataFrame, target: Path) -> None:
    for table_name, experiments in EXPERIMENTS.items():
        subset = records.loc[records["experiment"].isin(experiments)]
        if not subset.empty:
            table = aggregate(subset)
            table["experiment"] = pd.Categorical(table["experiment"], experiments, ordered=True)
            table.sort_values("experiment").to_csv(target / f"{table_name}_comparison.csv", index=False)


def plot_confusion(records: pd.DataFrame, target: Path) -> None:
    selected = records.loc[records["experiment"] == "ordinal_supervision"]
    if selected.empty:
        return
    matrix = sum((np.load(Path(run_dir) / "confusion.npy") for run_dir in selected["run_dir"]), np.zeros((3, 3)))
    figure, axis = plt.subplots(figsize=(4.6, 4.0))
    image = axis.imshow(matrix, cmap="Blues")
    for row in range(3):
        for column in range(3):
            axis.text(column, row, str(int(matrix[row, column])), ha="center", va="center")
    axis.set(xticks=range(3), yticks=range(3), xticklabels=["N0", "N75", "NFull"], yticklabels=["N0", "N75", "NFull"])
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title("Ordinal-supervision confusion matrix (all seeds)")
    figure.colorbar(image, ax=axis, shrink=0.8)
    figure.tight_layout()
    figure.savefig(target / "confusion_matrix.png", dpi=200)
    plt.close(figure)


def plot_increment_curve(records: pd.DataFrame, target: Path) -> None:
    order = EXPERIMENTS["main"]
    summary = aggregate(records.loc[records["experiment"].isin(order)]).set_index("experiment")
    available = [name for name in order if name in summary.index]
    if not available:
        return
    means = [summary.loc[name, "macro_f1_mean"] for name in available]
    stds = [summary.loc[name, "macro_f1_std"] for name in available]
    figure, axis = plt.subplots(figsize=(9, 4))
    positions = np.arange(len(available))
    axis.errorbar(positions, means, yerr=stds, marker="o", capsize=4, color="#1769aa")
    axis.set_xticks(positions, [name.replace("_", "\n") for name in available])
    axis.set_ylabel("Test macro-F1")
    axis.set_title("Incremental module ablation")
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(target / "module_increment_curve.png", dpi=200)
    plt.close(figure)


def write_interpretation(records: pd.DataFrame, target: Path) -> None:
    """State every observed comparison, including non-improvements."""
    lines = ["# Ablation interpretation", ""]
    for table_name, experiments in EXPERIMENTS.items():
        table = aggregate(records.loc[records["experiment"].isin(experiments)])
        if table.empty:
            continue
        lines.extend((f"## {table_name.title()}", ""))
        for _, row in table.iterrows():
            lines.append(
                f"- `{row.experiment}` ({int(row.n_seeds)} seeds): macro-F1 "
                f"{row.macro_f1_mean:.4f} +/- {row.macro_f1_std:.4f}, ordinal MAE "
                f"{row.ordinal_mae_mean:.4f} +/- {row.ordinal_mae_std:.4f}."
            )
        lines.append("")
    main = aggregate(records.loc[records["experiment"].isin(EXPERIMENTS["main"])]).set_index("experiment")
    ordered = [name for name in EXPERIMENTS["main"] if name in main.index]
    if len(ordered) > 1:
        lines.extend(("## Incremental Effects", ""))
        for previous, current in zip(ordered, ordered[1:]):
            delta = main.loc[current, "macro_f1_mean"] - main.loc[previous, "macro_f1_mean"]
            spread = math.hypot(main.loc[current, "macro_f1_std"], main.loc[previous, "macro_f1_std"])
            verdict = "exceeds the combined seed standard deviation" if delta > spread else "does not exceed the combined seed standard deviation"
            lines.append(f"- `{current}` versus `{previous}`: macro-F1 change {delta:+.4f}; it {verdict} ({spread:.4f}).")
        lines.append("")
    lines.append("A positive mean alone is not described as a stable improvement above; the seed-level spread is reported for every configuration.")
    (target / "interpretation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="outputs")
    args = parser.parse_args()
    source = Path(args.source)
    target = source / "summary"
    target.mkdir(parents=True, exist_ok=True)
    records = load_records(source)
    write_tables(records, target)
    plot_confusion(records, target)
    plot_increment_curve(records, target)
    write_interpretation(records, target)
    print(f"Wrote tables and figures to {target}")


if __name__ == "__main__":
    main()
