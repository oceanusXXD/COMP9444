#!/usr/bin/env python3
"""PyTorch dataset and data loaders for the maize nitrogen project."""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter
from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


LABELS = ("N0", "N75", "NFull")
VALID_SPLITS = ("train", "val", "test")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(split: str, image_size: int = 224) -> Callable:
    """Return augmentation for training or deterministic evaluation preprocessing."""
    if split not in VALID_SPLITS:
        raise ValueError(f"Unknown split {split!r}; expected one of {VALID_SPLITS}.")

    if split == "train":
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    image_size, scale=(0.75, 1.0), ratio=(0.9, 1.1)
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                # Nitrogen stress is partly expressed through colour, so jitter is mild.
                transforms.ColorJitter(
                    brightness=0.15,
                    contrast=0.15,
                    saturation=0.10,
                    hue=0.02,
                ),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    resize_size = int(round(image_size / 0.875))
    return transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class MaizeNitrogenDataset(Dataset):
    """Load one split from the CSV manifest created by prepare_data.py."""

    def __init__(
        self,
        project_dir: str | Path,
        split: str,
        image_size: int = 224,
        transform: Callable | None = None,
    ) -> None:
        if split not in VALID_SPLITS:
            raise ValueError(f"Unknown split {split!r}; expected one of {VALID_SPLITS}.")

        self.project_dir = Path(project_dir).resolve()
        self.split = split
        manifest_path = self.project_dir / "data" / "split.csv"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Manifest not found: {manifest_path}. Run src/prepare_data.py first."
            )

        with manifest_path.open(newline="", encoding="utf-8") as csv_file:
            self.records = [
                row for row in csv.DictReader(csv_file) if row["split"] == split
            ]

        if not self.records:
            raise ValueError(f"No records found for split {split!r} in {manifest_path}.")

        self.transform = transform or build_transform(split, image_size)
        self.targets = [int(row["label_index"]) for row in self.records]
        self.class_counts = Counter(row["label"] for row in self.records)
        self._validate_records()

    def _validate_records(self) -> None:
        for row in self.records:
            label_index = int(row["label_index"])
            if row["label"] != LABELS[label_index]:
                raise ValueError(
                    f"Label/index mismatch in manifest: {row['filename']}"
                )
            image_path = self.project_dir / row["filepath"]
            if not image_path.is_file():
                raise FileNotFoundError(f"Image listed in manifest is missing: {image_path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.records[index]
        image_path = self.project_dir / row["filepath"]
        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")
            image = self.transform(image)
        return image, int(row["label_index"])


def seed_worker(worker_id: int) -> None:
    """Give each DataLoader worker a deterministic Python random seed."""
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)


def build_dataloaders(
    project_dir: str | Path,
    batch_size: int = 16,
    image_size: int = 224,
    num_workers: int = 0,
    seed: int = 9444,
) -> dict[str, DataLoader]:
    """Construct train, validation, and test loaders."""
    generator = torch.Generator().manual_seed(seed)
    datasets = {
        split: MaizeNitrogenDataset(project_dir, split, image_size=image_size)
        for split in VALID_SPLITS
    }
    return {
        split: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=seed_worker,
            generator=generator,
        )
        for split, dataset in datasets.items()
    }


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Smoke-test the maize data loaders.")
    parser.add_argument("--project-dir", type=Path, default=project_dir)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    loaders = build_dataloaders(
        args.project_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
    )
    for split, loader in loaders.items():
        images, labels = next(iter(loader))
        dataset = loader.dataset
        print(
            f"{split:<5} samples={len(dataset):>3} "
            f"classes={dict(dataset.class_counts)} "
            f"batch={tuple(images.shape)} labels={tuple(labels.shape)} "
            f"range=[{images.min().item():.3f}, {images.max().item():.3f}]"
        )


if __name__ == "__main__":
    main()
