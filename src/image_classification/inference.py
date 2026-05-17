from __future__ import annotations

import csv
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


def predict_image_file(
    image_path: str | Path,
    model: torch.nn.Module,
    class_names: list[str],
    device: torch.device,
    image_size: int,
) -> int:
    _, eval_transform = build_transforms(image_size)
    image = Image.open(image_path).convert("RGB")
    tensor = eval_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        scores = model(tensor)
        predicted_index = int(torch.argmax(scores, dim=1).item())

    # Competition labels are 1-based instead of 0-based.
    return predicted_index + 1


def generate_submission(
    image_dir: str | Path,
    output_path: str | Path = "submission.csv",
    params_path: str | Path = "params.yaml",
) -> Path:
    image_dir = Path(image_dir)
    output_path = Path(output_path)
    model, class_names, device, image_size = load_model_artifacts(params_path)

    image_paths = sorted(
        [
            path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
    )
    if not image_paths:
        raise FileNotFoundError(f"No supported image files found in {image_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["file_name", "category_id"])
        for image_path in image_paths:
            category_id = predict_image_file(
                image_path=image_path,
                model=model,
                class_names=class_names,
                device=device,
                image_size=image_size,
            )
            writer.writerow([image_path.name, category_id])

    return output_path.resolve()
