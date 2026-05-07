from __future__ import annotations

import json

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

from .config import AppConfig
from .model import create_model
from .training import move_to_device, resolve_device


def _load_model(config: AppConfig, class_names: list[str]) -> torch.nn.Module:
    device = resolve_device(config.train.device)
    amp_enabled = config.train.amp and device.type == "cuda"
    model = create_model(
        num_classes=len(class_names),
        pretrained=False,
        multi_gpu=False,
        amp_enabled=amp_enabled,
    ).to(device)
    state_dict = torch.load(config.artifacts.model_dir / "best_model.pt", map_location=device)
    model.network.load_state_dict(state_dict)
    model.eval()
    return model


def evaluate_model(config: AppConfig, dataloader: DataLoader, class_names: list[str]) -> dict:
    device = resolve_device(config.train.device)
    model = _load_model(config, class_names)

    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = move_to_device(batch, device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    per_class: dict[str, dict[str, float]] = {}
    accuracies: list[float] = []
    for index, class_name in enumerate(class_names):
        class_labels = [1 if label == index else 0 for label in all_labels]
        class_preds = [1 if pred == index else 0 for pred in all_preds]
        class_accuracy = accuracy_score(class_labels, class_preds) * 100
        accuracies.append(class_accuracy)
        per_class[class_name] = {
            "accuracy": class_accuracy,
            "precision": precision_score(class_labels, class_preds, zero_division=0) * 100,
            "recall": recall_score(class_labels, class_preds, zero_division=0) * 100,
            "f1_score": f1_score(class_labels, class_preds, zero_division=0) * 100,
        }

    metrics = {
        "overall_accuracy": accuracy_score(all_labels, all_preds) * 100,
        "per_class": per_class,
    }

    config.artifacts.report_dir.mkdir(parents=True, exist_ok=True)
    (config.artifacts.report_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    confusion = confusion_matrix(all_labels, all_preds)
    confusion_percentage = confusion.astype("float") / confusion.sum(axis=1)[:, None] * 100
    fig, ax = plt.subplots(figsize=(10, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_percentage, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation="vertical", ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(config.artifacts.report_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(class_names, accuracies)
    ax.set_xlabel("Classes")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy per Class")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(config.artifacts.report_dir / "accuracy_per_class.png", dpi=150)
    plt.close(fig)

    return metrics
