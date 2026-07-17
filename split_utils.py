"""Validation and path discovery for the fixed maize split file.

Fixed configuration locations
-----------------------------
``CLASS_NAMES``, ``LABEL_TO_INDEX``, ``EXPECTED_SPLITS``, and
``REQUIRED_COLUMNS`` define the dataset contract. They are intentionally fixed
for all experiments and must not be modified as hyperparameters. Experiment
settings belong in the CLI arguments exposed by ``train_deit.py``.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Optional

import pandas as pd


CLASS_NAMES = ("N0", "N75", "NFull")
LABEL_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
EXPECTED_SPLITS = {"train", "val", "test"}
REQUIRED_COLUMNS = {
    "filepath",
    "filename",
    "label",
    "label_index",
    "plot_id",
    "split",
}


def safe_relative_path(value: object) -> Path:
    """Convert a CSV path to a safe platform-native relative path."""
    raw = str(value).strip().replace("\\", "/")
    parts = PurePosixPath(raw).parts
    if not parts or raw.startswith("/") or ".." in parts:
        raise ValueError(f"Unsafe relative image path in split CSV: {value!r}")
    if parts[0].endswith(":"):
        raise ValueError(f"Absolute image path is not allowed in split CSV: {value!r}")
    return Path(*parts)


def load_and_validate_split_csv(csv_path: Path | str) -> pd.DataFrame:
    """Load the fixed split file and fail early on label or leakage mistakes."""
    csv_path = Path(csv_path).expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"Split CSV does not exist: {csv_path}")

    frame = pd.read_csv(csv_path)
    missing_columns = REQUIRED_COLUMNS.difference(frame.columns)
    if missing_columns:
        raise ValueError(f"Split CSV is missing columns: {sorted(missing_columns)}")
    if frame.empty:
        raise ValueError("Split CSV contains no samples")

    frame = frame.copy()
    frame["filepath"] = frame["filepath"].astype(str).str.strip()
    frame["label"] = frame["label"].astype(str).str.strip()
    frame["split"] = frame["split"].astype(str).str.strip().str.lower()
    frame["label_index"] = pd.to_numeric(frame["label_index"], errors="raise").astype(int)

    unknown_labels = set(frame["label"]).difference(CLASS_NAMES)
    if unknown_labels:
        raise ValueError(f"Unknown labels in split CSV: {sorted(unknown_labels)}")
    unknown_splits = set(frame["split"]).difference(EXPECTED_SPLITS)
    if unknown_splits:
        raise ValueError(f"Unknown split names in split CSV: {sorted(unknown_splits)}")
    missing_splits = EXPECTED_SPLITS.difference(frame["split"])
    if missing_splits:
        raise ValueError(f"Split CSV is missing required splits: {sorted(missing_splits)}")

    expected_indices = frame["label"].map(LABEL_TO_INDEX)
    inconsistent = frame["label_index"] != expected_indices
    if inconsistent.any():
        examples = frame.loc[inconsistent, ["filepath", "label", "label_index"]].head()
        raise ValueError(
            "label and label_index are inconsistent. Expected "
            f"{LABEL_TO_INDEX}. Examples:\n{examples.to_string(index=False)}"
        )

    duplicated_paths = frame[frame["filepath"].duplicated(keep=False)]
    if not duplicated_paths.empty:
        examples = duplicated_paths[["filepath", "split"]].head()
        raise ValueError(
            "An image occurs more than once in the split CSV; this can leak data. "
            f"Examples:\n{examples.to_string(index=False)}"
        )

    # A plot is the grouping unit used by this fixed split, so it must not cross splits.
    plot_split_counts = frame.groupby("plot_id")["split"].nunique()
    leaking_plots = plot_split_counts[plot_split_counts > 1]
    if not leaking_plots.empty:
        raise ValueError(
            "plot_id values cross dataset splits: "
            f"{leaking_plots.index.astype(str).tolist()[:10]}"
        )

    frame["filepath"].map(safe_relative_path)
    return frame


def discover_data_root(
    csv_path: Path | str,
    frame: pd.DataFrame,
    data_root: Optional[Path | str] = None,
) -> Path:
    """Find the directory under which CSV paths such as Images/N0 (1).JPG live."""
    csv_path = Path(csv_path).expanduser().resolve()
    first_relative_path = safe_relative_path(frame.iloc[0]["filepath"])

    if data_root is not None:
        root = Path(data_root).expanduser().resolve()
        if not (root / first_relative_path).is_file():
            raise FileNotFoundError(
                f"--data-root is incorrect: {root / first_relative_path} was not found"
            )
        return root

    candidates = [csv_path.parent]
    first_directory = first_relative_path.parts[0]
    candidates.extend(
        path.parent for path in csv_path.parent.rglob(first_directory) if path.is_dir()
    )
    for candidate in candidates:
        if (candidate / first_relative_path).is_file():
            return candidate.resolve()

    raise FileNotFoundError(
        "Could not locate the image directory automatically. Pass the directory "
        "containing Images/ with --data-root."
    )


def validate_image_paths(frame: pd.DataFrame, data_root: Path) -> None:
    missing = []
    for value in frame["filepath"]:
        path = data_root / safe_relative_path(value)
        if not path.is_file():
            missing.append(str(path))
            if len(missing) == 10:
                break
    if missing:
        raise FileNotFoundError(
            "Images referenced by split.csv are missing. First examples:\n"
            + "\n".join(missing)
        )
