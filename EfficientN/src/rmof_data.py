"""Dataset, image transforms, and data loaders for RMOF-Net experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


CLASS_NAMES = ("N0", "N75", "NFull")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
ASPECT_FILL = (124, 116, 104)


class AspectResizePad:
    """Resize without distortion, then pad to a fixed square canvas."""

    def __init__(
        self,
        height: int,
        width: int,
        fill: Tuple[int, int, int] = ASPECT_FILL,
    ) -> None:
        self.height = height
        self.width = width
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        scale = min(self.width / width, self.height / height)
        resized = image.resize(
            (max(1, round(width * scale)), max(1, round(height * scale))),
            Image.Resampling.BILINEAR,
        )
        canvas = Image.new("RGB", (self.width, self.height), self.fill)
        canvas.paste(
            resized,
            ((self.width - resized.width) // 2, (self.height - resized.height) // 2),
        )
        return canvas


class MaizeDataset(Dataset):
    """One split from a CSV manifest of maize images."""

    def __init__(
        self,
        csv_path: Path,
        data_root: Path,
        split: str,
        transform: transforms.Compose,
        train_fraction: float = 1.0,
        seed: int = 42,
        cache_images: bool = False,
        image_size: int = 224,
        aspect_pad: bool = False,
    ) -> None:
        frame = pd.read_csv(csv_path)
        required = {"filepath", "filename", "label_index", "split"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")
        if not 0.0 < train_fraction <= 1.0:
            raise ValueError("train_fraction must be in (0, 1]")

        self.frame = frame.loc[frame["split"] == split].reset_index(drop=True)
        if split == "train" and train_fraction < 1.0:
            if "plot_id" in self.frame:
                plot_ids = np.sort(self.frame["plot_id"].unique())
                count = max(1, round(len(plot_ids) * train_fraction))
                selected = np.random.default_rng(seed).choice(
                    plot_ids, size=count, replace=False
                )
                self.frame = self.frame.loc[
                    self.frame["plot_id"].isin(selected)
                ].reset_index(drop=True)
            else:
                self.frame = self.frame.groupby(
                    "label_index", group_keys=False
                ).sample(frac=train_fraction, random_state=seed).reset_index(drop=True)
        if self.frame.empty:
            raise ValueError(f"No samples with split={split!r} in {csv_path}")

        self.data_root = data_root
        self.transform = transform
        self.cache_images = cache_images
        self.image_size = image_size
        self.aspect_pad = aspect_pad
        self._image_cache: Dict[int, Image.Image] = {}

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, str]:
        row = self.frame.iloc[index]
        if self.cache_images and index in self._image_cache:
            image = self._image_cache[index].copy()
        else:
            image_path = self.data_root / str(row.filepath)
            if not image_path.is_file():
                raise FileNotFoundError(
                    f"Image not found: {image_path}. Set --data-root to the directory "
                    "that contains Images/."
                )
            with Image.open(image_path) as opened:
                image = opened.convert("RGB")
            if self.cache_images:
                if self.aspect_pad:
                    image = AspectResizePad(self.image_size, self.image_size)(image)
                else:
                    image = image.resize(
                        (self.image_size, self.image_size), Image.Resampling.BILINEAR
                    )
                self._image_cache[index] = image.copy()
        return self.transform(image), int(row.label_index), str(row.filename)


def build_transforms(
    image_size: int,
    augmentation: str,
    images_pre_resized: bool = False,
    aspect_pad: bool = False,
) -> Tuple[transforms.Compose, transforms.Compose]:
    """Build ImageNet-normalized train and evaluation transforms."""
    normalise = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    train_steps: list[object] = [transforms.RandomHorizontalFlip()]
    if not images_pre_resized:
        resize = AspectResizePad(image_size, image_size) if aspect_pad else transforms.Resize((image_size, image_size))
        train_steps.insert(0, resize)
    if augmentation == "strong":
        train_steps.extend(
            (transforms.RandomVerticalFlip(p=0.2), transforms.RandomRotation(15, fill=ASPECT_FILL))
        )
    elif augmentation == "leaf":
        train_steps.extend(
            (
                transforms.RandomRotation(8, fill=ASPECT_FILL),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.05, 0.05),
                    scale=(0.9, 1.1),
                    fill=ASPECT_FILL,
                ),
            )
        )
    train_steps.extend((transforms.ToTensor(), normalise))

    eval_steps: list[object] = [transforms.ToTensor(), normalise]
    if not images_pre_resized:
        resize = AspectResizePad(image_size, image_size) if aspect_pad else transforms.Resize((image_size, image_size))
        eval_steps.insert(0, resize)
    return transforms.Compose(train_steps), transforms.Compose(eval_steps)


def make_loaders(
    args: argparse.Namespace, include_test: bool = True
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """Create train, validation, and test loaders from the experiment arguments."""
    csv_path = Path(args.csv)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Split file not found: {csv_path}")

    cache_images = getattr(args, "cache_images", False)
    train_transform, eval_transform = build_transforms(
        args.image_size,
        args.augmentation,
        images_pre_resized=cache_images,
        aspect_pad=getattr(args, "aspect_pad", False),
    )
    splits = set(pd.read_csv(csv_path)["split"].unique())
    validation_split = "val" if "val" in splits else "test"
    test_split = "test" if "test" in splits else validation_split
    root = Path(args.data_root)

    train_options: dict[str, object] = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
    }
    # Evaluation runs after every epoch, so another persistent worker pool
    # only oversubscribes the host without improving throughput.
    evaluation_options: dict[str, object] = {
        "batch_size": args.batch_size,
        "num_workers": 0,
    }
    if args.device.startswith("cuda"):
        train_options["pin_memory"] = True
        evaluation_options["pin_memory"] = True
    if args.num_workers > 0:
        train_options["persistent_workers"] = True

    train_generator = torch.Generator().manual_seed(args.seed)
    train = DataLoader(
        MaizeDataset(
            csv_path,
            root,
            "train",
            train_transform,
            args.train_fraction,
            args.seed,
            cache_images,
            args.image_size,
            getattr(args, "aspect_pad", False),
        ),
        shuffle=True,
        generator=train_generator,
        **train_options,
    )
    validation = DataLoader(
        MaizeDataset(
            csv_path,
            root,
            validation_split,
            eval_transform,
            cache_images=cache_images,
            image_size=args.image_size,
            aspect_pad=getattr(args, "aspect_pad", False),
        ),
        **evaluation_options,
    )
    test = None
    if include_test:
        test = DataLoader(
            MaizeDataset(
                csv_path,
                root,
                test_split,
                eval_transform,
                cache_images=cache_images,
                image_size=args.image_size,
                aspect_pad=getattr(args, "aspect_pad", False),
            ),
            **evaluation_options,
        )
    return train, validation, test
