from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


SUPPORTED_ARCHITECTURES = {
    "resnet18",
    "resnet34",
    "mobilenet_v2",
    "mobilenet_v3_small",
    "mobilenet_v3_large",
    "shufflenet_v2_x0_5",
    "shufflenet_v2_x1_0",
    "squeezenet1_0",
    "squeezenet1_1",
    "mnasnet0_5",
    "mnasnet0_75",
    "mnasnet1_0",
    "efficientnet_b0",
    "efficientnet_b1",
    "efficientnet_b2",
    "googlenet",
    "densenet121",
    "regnet_x_400mf",
    "regnet_x_800mf",
    "regnet_y_400mf",
    "regnet_y_800mf",
}


def accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


def _replace_classifier(network: nn.Module, architecture: str, num_classes: int) -> nn.Module:
    if architecture.startswith("resnet") or architecture == "googlenet":
        features = network.fc.in_features
        network.fc = nn.Linear(features, num_classes)
        return network
    if architecture.startswith("mobilenet") or architecture.startswith("efficientnet") or architecture.startswith("mnasnet"):
        features = network.classifier[-1].in_features
        network.classifier[-1] = nn.Linear(features, num_classes)
        return network
    if architecture.startswith("shufflenet"):
        features = network.fc.in_features
        network.fc = nn.Linear(features, num_classes)
        return network
    if architecture.startswith("squeezenet"):
        network.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
        network.num_classes = num_classes
        return network
    if architecture == "densenet121":
        features = network.classifier.in_features
        network.classifier = nn.Linear(features, num_classes)
        return network
    if architecture.startswith("regnet"):
        features = network.fc.in_features
        network.fc = nn.Linear(features, num_classes)
        return network
    raise ValueError(f"Unsupported architecture: {architecture}")


def _create_network(architecture: str, pretrained: bool, num_classes: int) -> nn.Module:
    if architecture not in SUPPORTED_ARCHITECTURES:
        raise ValueError(
            f"Unsupported architecture: {architecture}. Supported values: {sorted(SUPPORTED_ARCHITECTURES)}"
        )

    weights = models.get_model_weights(architecture).DEFAULT if pretrained else None
    network_builder = getattr(models, architecture)
    network = network_builder(weights=weights)
    return _replace_classifier(network, architecture, num_classes)


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


class ImageClassifier(ImageClassification):
    def __init__(
        self,
        architecture: str,
        num_classes: int,
        pretrained: bool = True,
        multi_gpu: bool = True,
        amp_enabled: bool = False,
    ) -> None:
        super().__init__()
        self.network = _create_network(architecture, pretrained, num_classes)
        if multi_gpu and torch.cuda.is_available() and torch.cuda.device_count() > 1:
            self.network = nn.DataParallel(self.network)
        self.amp_enabled = amp_enabled

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.network(image)

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
    architecture: str,
    num_classes: int,
    pretrained: bool = True,
    multi_gpu: bool = True,
    amp_enabled: bool = False,
) -> ImageClassifier:
    return ImageClassifier(
        architecture=architecture,
        num_classes=num_classes,
        pretrained=pretrained,
        multi_gpu=multi_gpu,
        amp_enabled=amp_enabled,
    )
