from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from image_classification.config import load_config
from image_classification.data import build_dataloaders
from image_classification.training import resolve_device, train_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()

    config = load_config(args.params)
    device = resolve_device(config.train.device)
    if device.type != "cuda":
        print("WARNING: Training is not using CUDA. Expect slow training on CPU.")
    loaders = build_dataloaders(config)
    result = train_model(config, loaders.train_loader, loaders.val_loader, loaders.classes)
    print(f"Best validation top-1 accuracy: {result.best_val_top1_acc:.2f}%")
    print(f"Best validation top-5 accuracy: {result.best_val_top5_acc:.2f}%")
    print(f"Best validation loss: {result.best_val_loss:.4f}")


if __name__ == "__main__":
    main()
