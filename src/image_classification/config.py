from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    source_url: str
    download_dir: Path
    raw_dir: Path
    processed_dir: Path
    image_size: int
    train_split: float
    val_split: float
    test_split: float
    batch_size: int
    eval_batch_multiplier: int
    num_workers: int


@dataclass(frozen=True)
class TrainConfig:
    architecture: str
    pretrained: bool
    learning_rate: float
    epochs: int
    patience: int
    device: str
    amp: bool
    multi_gpu: bool


@dataclass(frozen=True)
class ArtifactConfig:
    model_dir: Path
    report_dir: Path


@dataclass(frozen=True)
class AppConfig:
    seed: int
    data: DataConfig
    train: TrainConfig
    artifacts: ArtifactConfig


def _as_path(root: Path, value: str) -> Path:
    return (root / value).resolve()


def load_config(params_path: str | Path = "params.yaml") -> AppConfig:
    params_path = Path(params_path).resolve()
    root = params_path.parent
    raw: dict[str, Any] = yaml.safe_load(params_path.read_text(encoding="utf-8"))

    data = raw["data"]
    train = raw["train"]
    artifacts = raw["artifacts"]

    return AppConfig(
        seed=int(raw["seed"]),
        data=DataConfig(
            source_url=data["source_url"],
            download_dir=_as_path(root, data["download_dir"]),
            raw_dir=_as_path(root, data["raw_dir"]),
            processed_dir=_as_path(root, data["processed_dir"]),
            image_size=int(data["image_size"]),
            train_split=float(data["train_split"]),
            val_split=float(data["val_split"]),
            test_split=float(data["test_split"]),
            batch_size=int(data["batch_size"]),
            eval_batch_multiplier=int(data["eval_batch_multiplier"]),
            num_workers=int(data["num_workers"]),
        ),
        train=TrainConfig(
            architecture=train["architecture"],
            pretrained=bool(train["pretrained"]),
            learning_rate=float(train["learning_rate"]),
            epochs=int(train["epochs"]),
            patience=int(train["patience"]),
            device=train["device"],
            amp=bool(train["amp"]),
            multi_gpu=bool(train["multi_gpu"]),
        ),
        artifacts=ArtifactConfig(
            model_dir=_as_path(root, artifacts["model_dir"]),
            report_dir=_as_path(root, artifacts["report_dir"]),
        ),
    )
