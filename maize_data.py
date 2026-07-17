"""Dataset and transforms shared by the maize classification experiments.

Parameter locations
-------------------
The selectable augmentation mode and image size enter through
``build_transform``. The exact mild/medium crop, rotation, affine, flip, and
brightness/contrast values are defined inside that function. Normalization is
defined by ``IMAGENET_MEAN`` and ``IMAGENET_STD`` and should remain fixed while
using ImageNet-pretrained weights. Dataset splits and labels come only from the
validated CSV and must not be tuned here.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from split_utils import EXPECTED_SPLITS, safe_relative_path


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class MaizeNitrogenDataset(Dataset):
    """CSV-backed RGB image dataset, following the EfficientNet baseline layout."""

    def __init__(self, frame: pd.DataFrame, split: str, data_root: Path, transform=None):
        split = split.lower()
        if split not in EXPECTED_SPLITS:
            raise ValueError(f"split must be one of {sorted(EXPECTED_SPLITS)}, got {split!r}")
        self.frame = frame.loc[frame["split"] == split].reset_index(drop=True)
        if self.frame.empty:
            raise ValueError(f"No samples found for split {split!r}")
        self.split = split
        self.data_root = Path(data_root)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        image_path = self.data_root / safe_relative_path(row["filepath"])
        try:
            with Image.open(image_path) as image_file:
                image = image_file.convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"Failed to read image: {image_path}") from exc
        if self.transform is not None:
            image = self.transform(image)
        return image, int(row["label_index"]), str(row["filepath"])


def build_transform(split: str, augmentation: str = "mild", image_size: int = 224):
    """Build colour-conservative transforms suitable for nitrogen classification."""
    if split != "train":
        resize_size = round(image_size / 0.875)
        return transforms.Compose(
            [
                transforms.Resize(resize_size, interpolation=InterpolationMode.BICUBIC),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    common_tail = [
        transforms.RandomHorizontalFlip(p=0.5),
        # Nitrogen level is colour-sensitive: vary brightness/contrast only slightly.
        transforms.ColorJitter(brightness=0.10, contrast=0.10),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    if augmentation == "mild":
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(0.85, 1.0),
                    ratio=(0.95, 1.05),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.RandomRotation(10, interpolation=InterpolationMode.BILINEAR),
                *common_tail,
            ]
        )
    if augmentation == "medium":
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(0.70, 1.0),
                    ratio=(0.90, 1.10),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.RandomAffine(
                    degrees=15,
                    translate=(0.08, 0.08),
                    scale=(0.90, 1.10),
                    interpolation=InterpolationMode.BILINEAR,
                ),
                *common_tail,
            ]
        )
    raise ValueError("augmentation must be 'mild' or 'medium'")
