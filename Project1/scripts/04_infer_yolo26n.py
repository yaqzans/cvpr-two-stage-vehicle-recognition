"""04 — Run the Stage-1 YOLO26n detector on a file, image folder, or video."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT / ".ultralytics"))

import torch
from ultralytics import YOLO

BEST_WEIGHTS = PROJECT_ROOT / "runs" / "stage1_yolo26n" / "train" / "weights" / "best.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RSUD20K-trained YOLO26n inference.")
    parser.add_argument("--source", required=True, help="Image, folder, video, or webcam source.")
    parser.add_argument("--conf", type=float, default=0.25, help="Minimum detection confidence.")
    args = parser.parse_args()

    if not BEST_WEIGHTS.exists():
        raise FileNotFoundError(f"No best checkpoint found: {BEST_WEIGHTS}")
    model = YOLO(str(BEST_WEIGHTS))
    model.predict(
        source=args.source,
        imgsz=640,
        conf=args.conf,
        device=0 if torch.cuda.is_available() else "cpu",
        save=True,
        project=str(PROJECT_ROOT / "runs" / "stage1_yolo26n"),
        name="predictions",
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
