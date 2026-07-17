"""Create a deterministic plot-disjoint train/validation/test split."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="split.csv", help="Manifest with filepath, label, and plot_id columns.")
    parser.add_argument("--output", default="split_40_10_50.csv")
    parser.add_argument("--train-ratio", type=float, default=0.40)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ratios = np.array([args.train_ratio, args.val_ratio, args.test_ratio], dtype=float)
    if not np.isclose(ratios.sum(), 1.0):
        raise ValueError("Train, validation, and test ratios must sum to 1.")

    source = Path(args.source)
    frame = pd.read_csv(source)
    required = {"filepath", "filename", "label", "label_index", "plot_id"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{source} is missing required columns: {sorted(missing)}")

    by_plot = frame.groupby("plot_id")["label_index"].agg(lambda values: tuple(sorted(values)))
    if by_plot.nunique() != 1:
        raise ValueError("Each plot must have the same label composition for exact group-stratified allocation.")

    plot_ids = np.array(sorted(frame["plot_id"].unique()))
    shuffled = np.random.default_rng(args.seed).permutation(plot_ids)
    sizes = np.rint(ratios * len(shuffled)).astype(int)
    sizes[-1] = len(shuffled) - sizes[:-1].sum()
    if (sizes <= 0).any():
        raise ValueError("Each partition must receive at least one plot.")

    assignments = {}
    cursor = 0
    for name, size in zip(("train", "val", "test"), sizes):
        for plot_id in shuffled[cursor : cursor + size]:
            assignments[int(plot_id)] = name
        cursor += size
    frame = frame.copy()
    frame["split"] = frame["plot_id"].map(assignments)

    leakage = frame.groupby("plot_id")["split"].nunique().max()
    if leakage != 1:
        raise RuntimeError("A plot was assigned to more than one partition.")
    counts = pd.crosstab(frame["split"], frame["label"])
    output = Path(args.output)
    frame.to_csv(output, index=False)
    print(f"Wrote {output}")
    print(counts.to_string())
    print(f"plots: {dict(frame.groupby('split')['plot_id'].nunique())}; seed: {args.seed}")


if __name__ == "__main__":
    main()
