from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image

from .config import load_config
from .data import build_transforms
from .model import create_model
from .training import resolve_device


def load_model_artifacts(
    params_path: str | Path = "params.yaml",
) -> tuple[torch.nn.Module, list[str], torch.device, int]:
    config = load_config(params_path)
    class_names = json.loads((config.artifacts.model_dir / "class_names.json").read_text(encoding="utf-8"))
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
    return model, class_names, device, config.data.image_size


def predict_image(image: Image.Image, params_path: str | Path = "params.yaml") -> list[tuple[str, float]]:
    model, class_names, device, image_size = load_model_artifacts(params_path)
    _, eval_transform = build_transforms(image_size)
    tensor = eval_transform(image.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        scores = model(tensor)[0].cpu().tolist()

    ranked = sorted(zip(class_names, scores), key=lambda item: item[1], reverse=True)
    return ranked
