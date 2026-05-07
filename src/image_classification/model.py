from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class ImageClassification(nn.Module):
    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        return loss

    def validating(self, batch: tuple[torch.Tensor, torch.Tensor]) -> dict[str, torch.Tensor]:
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return {"Validation Loss": loss.detach(), "Validation Accuracy": acc}

    def validating_epoch_final(self, outputs: list[dict[str, torch.Tensor]]) -> dict[str, float]:
        batch_loss = [item["Validation Loss"] for item in outputs]
        epoch_loss = torch.stack(batch_loss).mean()
        batch_accuracy = [item["Validation Accuracy"] for item in outputs]
        epoch_accuracy = torch.stack(batch_accuracy).mean()
        return {"Validation Loss": epoch_loss.item(), "Validation Accuracy": epoch_accuracy.item()}


class ResNetClassifier(ImageClassification):
    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        multi_gpu: bool = True,
        amp_enabled: bool = False,
    ) -> None:
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        self.network = models.resnet50(weights=weights)
        features = self.network.fc.in_features
        self.network.fc = nn.Linear(features, num_classes)
        if multi_gpu and torch.cuda.is_available() and torch.cuda.device_count() > 1:
            self.network = nn.DataParallel(self.network)
        self.amp_enabled = amp_enabled

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.network(image))

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        images, labels = batch
        autocast_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self.amp_enabled
            else nullcontext()
        )
        with autocast_context:
            out = self(images)
            loss = F.cross_entropy(out, labels)
        return loss


def create_model(
    num_classes: int,
    pretrained: bool = True,
    multi_gpu: bool = True,
    amp_enabled: bool = False,
) -> ResNetClassifier:
    return ResNetClassifier(
        num_classes=num_classes,
        pretrained=pretrained,
        multi_gpu=multi_gpu,
        amp_enabled=amp_enabled,
    )
