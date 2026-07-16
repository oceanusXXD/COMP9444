#!/usr/bin/env python3
"""Create a leakage-free train/validation/test manifest for the maize images."""

from __future__ import annotations

import argparse
import csv
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


LABELS = ("N0", "N75", "NFull")
LABEL_TO_INDEX = {label: index for index, label in enumerate(LABELS)}
FILENAME_PATTERN = re.compile(r"^(N0|N75|NFull) \((\d+)\)\.JPG$")
EXPECTED_IMAGE_COUNT = 1_200
EXPECTED_PLOT_COUNT = 400


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Split maize images by plot ID to prevent data leakage."
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=project_dir / "Images",
        help="Directory containing the JPG images.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_dir / "data" / "split.csv",
        help="Output CSV manifest.",
    )
    parser.add_argument("--seed", type=int, default=9444)
    return parser.parse_args()


def scan_images(images_dir: Path) -> list[dict[str, object]]:
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {images_dir}")

    records: list[dict[str, object]] = []
    unmatched: list[str] = []

    for path in sorted(images_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() != ".jpg":
            continue
        match = FILENAME_PATTERN.fullmatch(path.name)
        if match is None:
            unmatched.append(path.name)
            continue
        label, plot_id_text = match.groups()
        records.append(
            {
                "filepath": f"Images/{path.name}",
                "filename": path.name,
                "label": label,
                "label_index": LABEL_TO_INDEX[label],
                "plot_id": int(plot_id_text),
            }
        )

    if unmatched:
        preview = ", ".join(unmatched[:5])
        raise ValueError(f"Found JPG files with unexpected names: {preview}")
    if len(records) != EXPECTED_IMAGE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_IMAGE_COUNT} images, found {len(records)}."
        )
    return records


def validate_groups(records: list[dict[str, object]]) -> list[int]:
    labels_by_plot: dict[int, set[str]] = defaultdict(set)
    for record in records:
        labels_by_plot[int(record["plot_id"])].add(str(record["label"]))

    if len(labels_by_plot) != EXPECTED_PLOT_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_PLOT_COUNT} plot IDs, found {len(labels_by_plot)}."
        )

    expected_labels = set(LABELS)
    incomplete = {
        plot_id: sorted(expected_labels - labels)
        for plot_id, labels in labels_by_plot.items()
        if labels != expected_labels
    }
    if incomplete:
        raise ValueError(f"Plots with missing or invalid labels: {incomplete}")

    return sorted(labels_by_plot)


def assign_splits(plot_ids: list[int], seed: int) -> dict[int, str]:
    shuffled_ids = plot_ids.copy()
    random.Random(seed).shuffle(shuffled_ids)

    # 400 plots -> 280 train, 60 validation, 60 test.
    train_end = int(0.70 * len(shuffled_ids))
    val_end = train_end + int(0.15 * len(shuffled_ids))
    split_ids = {
        "train": shuffled_ids[:train_end],
        "val": shuffled_ids[train_end:val_end],
        "test": shuffled_ids[val_end:],
    }

    assignment: dict[int, str] = {}
    for split, ids in split_ids.items():
        for plot_id in ids:
            assignment[plot_id] = split
    return assignment


def write_manifest(
    records: list[dict[str, object]], assignment: dict[int, str], output: Path
) -> None:
    split_order = {"train": 0, "val": 1, "test": 2}
    for record in records:
        record["split"] = assignment[int(record["plot_id"])]

    records.sort(
        key=lambda row: (
            split_order[str(row["split"])],
            int(row["plot_id"]),
            int(row["label_index"]),
        )
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=(
                "filepath",
                "filename",
                "label",
                "label_index",
                "plot_id",
                "split",
            ),
        )
        writer.writeheader()
        writer.writerows(records)


def print_summary(records: list[dict[str, object]], output: Path, seed: int) -> None:
    counts = Counter((str(row["split"]), str(row["label"])) for row in records)
    plots = defaultdict(set)
    for row in records:
        plots[str(row["split"])].add(int(row["plot_id"]))

    print(f"Wrote: {output}")
    print(f"Seed: {seed}")
    print("split  plots  N0  N75  NFull  total")
    for split in ("train", "val", "test"):
        label_counts = [counts[(split, label)] for label in LABELS]
        print(
            f"{split:<6} {len(plots[split]):>5} "
            f"{label_counts[0]:>3} {label_counts[1]:>4} "
            f"{label_counts[2]:>6} {sum(label_counts):>6}"
        )

    assert plots["train"].isdisjoint(plots["val"])
    assert plots["train"].isdisjoint(plots["test"])
    assert plots["val"].isdisjoint(plots["test"])
    print("Leakage check: passed (plot IDs are disjoint across splits)")


def main() -> None:
    args = parse_args()
    records = scan_images(args.images_dir.resolve())
    plot_ids = validate_groups(records)
    assignment = assign_splits(plot_ids, args.seed)
    write_manifest(records, assignment, args.output.resolve())
    print_summary(records, args.output.resolve(), args.seed)


if __name__ == "__main__":
    main()
