from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from image_classification.inference import generate_submission


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run inference on a folder of images and create submission.csv.",
    )
    parser.add_argument("--images", default="images", help="Folder containing private-test images.")
    parser.add_argument("--output", default="submission.csv", help="Path to the output CSV file.")
    parser.add_argument("--params", default="params.yaml", help="Path to params.yaml.")
    args = parser.parse_args()

    submission_path = generate_submission(
        image_dir=args.images,
        output_path=args.output,
        params_path=args.params,
    )
    print(f"Submission saved to: {submission_path}")


if __name__ == "__main__":
    main()
