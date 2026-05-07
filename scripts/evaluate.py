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
from image_classification.evaluate import evaluate_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()

    config = load_config(args.params)
    loaders = build_dataloaders(config)
    metrics = evaluate_model(config, loaders.val_loader, loaders.classes)
    print(f"Evaluation accuracy: {metrics['overall_accuracy']:.2f}%")


if __name__ == "__main__":
    main()
