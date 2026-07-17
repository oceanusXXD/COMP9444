"""Audit the fixed maize dataset split before model training.

Parameter locations
-------------------
Audit paths and the perceptual-hash warning threshold are declared in
``parse_args``. They can be changed with ``--csv``, ``--data-root``,
``--output``, and ``--near-duplicate-threshold``. These settings do not change
model training and should not be treated as model hyperparameters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageStat

from split_utils import (
    discover_data_root,
    load_and_validate_split_csv,
    safe_relative_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="split.csv", help="Path to the fixed split CSV")
    parser.add_argument(
        "--data-root",
        default=None,
        help="Directory containing Images/. If omitted it is discovered automatically.",
    )
    parser.add_argument("--output", default="outputs/data_audit.json")
    parser.add_argument(
        "--near-duplicate-threshold",
        type=int,
        default=3,
        help="Maximum 64-bit average-hash Hamming distance for a near-duplicate warning",
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def difference_hash(image: Image.Image) -> int:
    """Return a 64-bit perceptual hash based on adjacent horizontal gradients."""
    sample = np.asarray(image.convert("L").resize((9, 8)), dtype=np.float32)
    bits = sample[:, 1:] >= sample[:, :-1]
    value = 0
    for bit in bits.flat:
        value = (value << 1) | int(bit)
    return value


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv).expanduser().resolve()
    frame = load_and_validate_split_csv(csv_path)
    data_root = discover_data_root(csv_path, frame, args.data_root)

    corrupt = []
    suspicious = []
    records = []
    hashes: dict[str, list[int]] = defaultdict(list)

    for row_index, row in frame.iterrows():
        path = data_root / safe_relative_path(row["filepath"])
        try:
            sha256 = file_sha256(path)
            with Image.open(path) as image_file:
                image = image_file.convert("RGB")
                image.load()
            stats = ImageStat.Stat(image.convert("L"))
            mean_luminance = float(stats.mean[0])
            std_luminance = float(stats.stddev[0])
            if mean_luminance < 10 or mean_luminance > 245 or std_luminance < 5:
                suspicious.append(
                    {
                        "filepath": row["filepath"],
                        "mean_luminance": mean_luminance,
                        "std_luminance": std_luminance,
                    }
                )
            hashes[sha256].append(row_index)
            records.append(
                {
                    "row": row_index,
                    "filepath": row["filepath"],
                    "split": row["split"],
                    "label": row["label"],
                    "width": image.width,
                    "height": image.height,
                    "difference_hash": difference_hash(image),
                }
            )
        except Exception as exc:
            corrupt.append({"filepath": row["filepath"], "error": str(exc)})

    exact_duplicates = []
    for duplicate_rows in hashes.values():
        if len(duplicate_rows) > 1:
            exact_duplicates.append(
                [str(frame.iloc[index]["filepath"]) for index in duplicate_rows]
            )

    # Only cross-split near duplicates can leak evaluation information.
    near_duplicate_cross_split = []
    for left_index, left in enumerate(records):
        for right in records[left_index + 1 :]:
            if left["split"] == right["split"]:
                continue
            distance = (left["difference_hash"] ^ right["difference_hash"]).bit_count()
            if distance <= args.near_duplicate_threshold:
                near_duplicate_cross_split.append(
                    {
                        "left": left["filepath"],
                        "left_split": left["split"],
                        "right": right["filepath"],
                        "right_split": right["split"],
                        "hamming_distance": distance,
                    }
                )

    split_counts = (
        frame.groupby(["split", "label"], sort=True)
        .size()
        .rename("count")
        .reset_index()
        .to_dict(orient="records")
    )
    audit = {
        "csv": str(csv_path),
        "data_root": str(data_root),
        "sample_count": len(frame),
        "split_class_counts": split_counts,
        "corrupt_images": corrupt,
        "suspicious_luminance_images": suspicious,
        "exact_duplicate_groups": exact_duplicates,
        "near_duplicate_cross_split": near_duplicate_cross_split,
        "near_duplicate_threshold": args.near_duplicate_threshold,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Dataset root: {data_root}")
    print(f"Samples: {len(frame)} | corrupt: {len(corrupt)}")
    print(f"Exact duplicate groups: {len(exact_duplicates)}")
    print(f"Cross-split near-duplicate warnings: {len(near_duplicate_cross_split)}")
    print(f"Audit report: {output_path.resolve()}")


if __name__ == "__main__":
    main()
