from __future__ import annotations

import csv
import multiprocessing
import os
import random
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import requests
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from .config import AppConfig


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
COMPETITION_CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]


class CompetitionTrainDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(
        self,
        image_dir: str | Path,
        csv_path: str | Path,
        transform: transforms.Compose | None = None,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.csv_path = Path(csv_path)
        self.transform = transform
        self.classes = COMPETITION_CLASSES.copy()
        self.samples: list[tuple[Path, int]] = []

        with self.csv_path.open("r", encoding="utf-8", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                image_path = self.image_dir / row["file_name"]
                category_id = int(row["category_id"])
                if not 1 <= category_id <= len(self.classes):
                    raise ValueError(f"Invalid category_id={category_id} for file {image_path.name}")
                self.samples.append((image_path, category_id - 1))

        if not self.samples:
            raise ValueError(f"No training samples found in {self.csv_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


@dataclass(frozen=True)
class DatasetBundle:
    classes: list[str]
    base_dataset: Dataset[tuple[torch.Tensor, int]]
    train_dataset: Dataset[tuple[torch.Tensor, int]]
    eval_dataset: Dataset[tuple[torch.Tensor, int]]
    train_indices: list[int]
    val_indices: list[int]
    test_indices: list[int]


@dataclass(frozen=True)
class LoaderBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    classes: list[str]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def _load_competition_dataset(config: AppConfig) -> tuple[Path, Path] | None:
    if config.data.train_image_dir.exists() and config.data.train_csv.exists():
        return config.data.train_image_dir, config.data.train_csv
    return None


def _load_tracked_dataset(config: AppConfig) -> Path | None:
    if config.data.processed_dir.exists():
        return config.data.processed_dir
    return None


def download_and_prepare_dataset(config: AppConfig) -> Path:
    config.data.download_dir.mkdir(parents=True, exist_ok=True)
    config.data.raw_dir.mkdir(parents=True, exist_ok=True)
    config.data.processed_dir.parent.mkdir(parents=True, exist_ok=True)

    archive_name = Path(urlparse(config.data.source_url).path).name or "dataset.zip"
    archive_path = config.data.download_dir / archive_name

    if not archive_path.exists():
        response = requests.get(config.data.source_url, timeout=120)
        response.raise_for_status()
        archive_path.write_bytes(response.content)

    extract_dir = config.data.raw_dir / "trashnet-master"
    if not extract_dir.exists():
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            zip_ref.extractall(config.data.raw_dir)

    source_dataset = extract_dir / "data" / "dataset-resized"
    nested_archive = extract_dir / "data" / "dataset-resized.zip"
    if not source_dataset.exists() and nested_archive.exists():
        with zipfile.ZipFile(nested_archive, "r") as zip_ref:
            zip_ref.extractall(extract_dir / "data")

    if not source_dataset.exists():
        raise FileNotFoundError(f"Dataset directory not found: {source_dataset}")

    if config.data.processed_dir.exists():
        shutil.rmtree(config.data.processed_dir)
    shutil.copytree(source_dataset, config.data.processed_dir)
    return config.data.processed_dir


def resolve_dataset_source(config: AppConfig) -> tuple[str, Path | tuple[Path, Path]]:
    competition_dataset = _load_competition_dataset(config)
    if competition_dataset is not None:
        return "competition_csv", competition_dataset

    tracked_dataset = _load_tracked_dataset(config)
    if tracked_dataset is not None:
        return "imagefolder", tracked_dataset

    return "imagefolder", download_and_prepare_dataset(config)


def build_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return train_transform, eval_transform


def _filtered_classes(dataset_dir: Path) -> list[str]:
    return sorted(
        [
            item.name
            for item in dataset_dir.iterdir()
            if item.is_dir() and not item.name.startswith(".")
        ]
    )


def build_dataset_bundle(config: AppConfig) -> DatasetBundle:
    set_seed(config.seed)
    source_type, source = resolve_dataset_source(config)
    train_transform, eval_transform = build_transforms(config.data.image_size)

    if source_type == "competition_csv":
        image_dir, csv_path = source
        base_dataset = CompetitionTrainDataset(image_dir, csv_path)
        train_dataset = CompetitionTrainDataset(image_dir, csv_path, transform=train_transform)
        eval_dataset = CompetitionTrainDataset(image_dir, csv_path, transform=eval_transform)
        classes = base_dataset.classes
    else:
        dataset_dir = source
        base_dataset = datasets.ImageFolder(dataset_dir)
        train_dataset = datasets.ImageFolder(dataset_dir, transform=train_transform)
        eval_dataset = datasets.ImageFolder(dataset_dir, transform=eval_transform)
        classes = _filtered_classes(dataset_dir)

    total_samples = len(base_dataset)
    train_size = int(config.data.train_split * total_samples)
    val_size = int(config.data.val_split * total_samples)
    test_size = total_samples - train_size - val_size
    if test_size < 0:
        raise ValueError("Invalid split sizes in params.yaml")

    generator = torch.Generator().manual_seed(config.seed)
    train_split, val_split, test_split = torch.utils.data.random_split(
        range(total_samples),
        [train_size, val_size, test_size],
        generator=generator,
    )

    return DatasetBundle(
        classes=classes,
        base_dataset=base_dataset,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        train_indices=list(train_split.indices),
        val_indices=list(val_split.indices),
        test_indices=list(test_split.indices),
    )


def build_dataloaders(config: AppConfig) -> LoaderBundle:
    bundle = build_dataset_bundle(config)
    num_workers = min(config.data.num_workers, multiprocessing.cpu_count())
    if os.name == "nt":
        num_workers = 0
    pin_memory = torch.cuda.is_available()

    loader_kwargs: dict[str, object] = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(
        Subset(bundle.train_dataset, bundle.train_indices),
        batch_size=config.data.batch_size,
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        Subset(bundle.eval_dataset, bundle.val_indices),
        batch_size=config.data.batch_size * config.data.eval_batch_multiplier,
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        Subset(bundle.eval_dataset, bundle.test_indices),
        batch_size=config.data.batch_size * config.data.eval_batch_multiplier,
        shuffle=False,
        **loader_kwargs,
    )
    return LoaderBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        classes=bundle.classes,
    )


def denormalize_image(image_tensor: torch.Tensor) -> torch.Tensor:
    image = image_tensor.detach().cpu().clone()
    for channel, mean, std in zip(image, IMAGENET_MEAN, IMAGENET_STD):
        channel.mul_(std).add_(mean)
    return image.clamp(0, 1)
