from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from image_classification.config import load_config
from image_classification.data import download_and_prepare_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()

    config = load_config(args.params)
    output_dir = download_and_prepare_dataset(config)
    print(f"Prepared dataset at: {output_dir}")


if __name__ == "__main__":
    main()
