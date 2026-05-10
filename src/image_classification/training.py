from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import platform

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from .config import AppConfig
from .model import ResNetClassifier, create_model


@dataclass
class TrainingResult:
    best_val_loss: float
    best_val_top1_acc: float
    best_val_top5_acc: float
    history: list[dict[str, float]]
    model: ResNetClassifier


class EarlyStopping:
    def __init__(self, patience: int = 5, verbose: bool = False, restore_best_weights: bool = True) -> None:
        self.patience = patience
        self.verbose = verbose
        self.best_loss = float("inf")
        self.counter = 0
        self.early_stop = False
        self.restore_best_weights = restore_best_weights
        self.best_model_weights: dict[str, torch.Tensor] | None = None

    def __call__(self, val_loss: float, model: torch.nn.Module) -> None:
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_model_weights = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
            if self.verbose:
                print(f"Validation loss improved: {val_loss:.4f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"Validation loss did not improve: {val_loss:.4f}")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights and self.best_model_weights is not None:
                    model.load_state_dict(self.best_model_weights)
                    if self.verbose:
                        print("Restored best model weights.")


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def print_runtime_diagnostics(config: AppConfig, device: torch.device, amp_enabled: bool) -> None:
    print(f"PyTorch version: {torch.__version__}")
    print(f"Python version: {platform.python_version()}")
    print(f"Configured device: {config.train.device}")
    print(f"Resolved device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA build: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}")
    print(f"AMP enabled: {amp_enabled}")

    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            print(f"GPU {index}: {torch.cuda.get_device_name(index)}")
    else:
        if "+cpu" in torch.__version__ or torch.version.cuda is None:
            print(
                "WARNING: Current PyTorch build is CPU-only. "
                "Install a CUDA build of torch/torchvision to train on GPU."
            )
        else:
            print(
                "WARNING: CUDA is not available in the current runtime, "
                "so training will run on CPU."
            )


def move_to_device(data: torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor], device: torch.device):
    if isinstance(data, (list, tuple)):
        return [move_to_device(item, device) for item in data]
    return data.to(device, non_blocking=True)


def topk_accuracy(outputs: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    k = min(k, outputs.size(1))
    _, topk_preds = torch.topk(outputs, k=k, dim=1)
    correct = topk_preds.eq(labels.view(-1, 1)).any(dim=1)
    return correct.float().mean().item() * 100.0


@torch.no_grad()
def evaluate_model(model: ResNetClassifier, dataloader: DataLoader, amp_enabled: bool) -> dict[str, float]:
    model.eval()
    outputs: list[dict[str, torch.Tensor]] = []
    top1_scores: list[float] = []
    top5_scores: list[float] = []
    for batch in dataloader:
        images, labels = batch
        if amp_enabled:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                validation_output = model.validating((images, labels))
                logits = model(images)
        else:
            validation_output = model.validating((images, labels))
            logits = model(images)
        outputs.append(validation_output)
        top1_scores.append(topk_accuracy(logits, labels, k=1))
        top5_scores.append(topk_accuracy(logits, labels, k=5))

    result = model.validating_epoch_final(outputs)
    result["Validation Top-1 Accuracy"] = sum(top1_scores) / max(len(top1_scores), 1)
    result["Validation Top-5 Accuracy"] = sum(top5_scores) / max(len(top5_scores), 1)
    result["Validation Accuracy"] = result["Validation Top-1 Accuracy"]
    return result


def train_model(
    config: AppConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_names: list[str],
) -> TrainingResult:
    device = resolve_device(config.train.device)
    amp_enabled = config.train.amp and device.type == "cuda"
    print_runtime_diagnostics(config, device, amp_enabled)
    model = create_model(
        num_classes=len(class_names),
        pretrained=config.train.pretrained,
        multi_gpu=config.train.multi_gpu,
        amp_enabled=amp_enabled,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), config.train.learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    early_stopping = EarlyStopping(patience=config.train.patience, verbose=True)
    history: list[dict[str, float]] = []
    best_val_top1_acc = 0.0
    best_val_top5_acc = 0.0

    config.artifacts.model_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(config.train.epochs):
        model.train()
        train_loss: list[torch.Tensor] = []

        for batch in train_loader:
            inputs, labels = batch
            batch_on_device = move_to_device((inputs, labels), device)
            loss = model.training_step((batch_on_device[0], batch_on_device[1]))
            train_loss.append(loss.detach().float().cpu())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        validation_result = evaluate_model(model, _DeviceLoader(val_loader, device), amp_enabled)
        validation_result["Train Loss"] = torch.stack(train_loss).mean().item()
        history.append(validation_result)
        best_val_top1_acc = max(best_val_top1_acc, validation_result["Validation Top-1 Accuracy"])
        best_val_top5_acc = max(best_val_top5_acc, validation_result["Validation Top-5 Accuracy"])

        print(
            (
                "Epoch [{}], Training Loss: {:.4f}, Validation Loss: {:.4f}, "
                "Validation Top-1 Accuracy: {:.4f}, Validation Top-5 Accuracy: {:.4f}"
            ).format(
                epoch + 1,
                validation_result["Train Loss"],
                validation_result["Validation Loss"],
                validation_result["Validation Top-1 Accuracy"],
                validation_result["Validation Top-5 Accuracy"],
            )
        )

        early_stopping(validation_result["Validation Loss"], model)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    model_to_save = model.network.module if isinstance(model.network, torch.nn.DataParallel) else model.network
    torch.save(model_to_save.state_dict(), config.artifacts.model_dir / "best_model.pt")
    (config.artifacts.model_dir / "class_names.json").write_text(
        json.dumps(class_names, indent=2), encoding="utf-8"
    )
    (config.artifacts.model_dir / "training_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )

    summary = {
        "device": str(device),
        "amp_enabled": amp_enabled,
        "gpu_count": torch.cuda.device_count() if device.type == "cuda" else 0,
        "epochs_ran": len(history),
        "best_val_loss": early_stopping.best_loss,
        "best_val_top1_acc": best_val_top1_acc,
        "best_val_top5_acc": best_val_top5_acc,
    }
    (config.artifacts.model_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    return TrainingResult(
        best_val_loss=early_stopping.best_loss,
        best_val_top1_acc=best_val_top1_acc,
        best_val_top5_acc=best_val_top5_acc,
        history=history,
        model=model,
    )


class _DeviceLoader:
    def __init__(self, data: DataLoader, device: torch.device) -> None:
        self.data = data
        self.device = device

    def __iter__(self):
        for batch in self.data:
            moved = move_to_device(batch, self.device)
            yield moved[0], moved[1]

    def __len__(self) -> int:
        return len(self.data)
