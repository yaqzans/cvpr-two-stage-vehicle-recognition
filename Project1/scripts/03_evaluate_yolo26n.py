"""03 — Evaluate the best YOLO26n checkpoint on untouched RSUD20K test images."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT / ".ultralytics"))

import torch
from ultralytics import YOLO

DATA_YAML = PROJECT_ROOT / "configs" / "rsud20k_yolo.yaml"
RUNS_DIR = PROJECT_ROOT / "runs" / "stage1_yolo26n"
BEST_WEIGHTS = RUNS_DIR / "train" / "weights" / "best.pt"


def main() -> None:
    if not BEST_WEIGHTS.exists():
        raise FileNotFoundError(f"No best checkpoint found: {BEST_WEIGHTS}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch cannot access a CUDA GPU.")

    model = YOLO(str(BEST_WEIGHTS))
    metrics = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=640,
        device=0,
        batch=16,
        workers=4,
        project=str(RUNS_DIR),
        name="test",
        exist_ok=True,
        plots=True,
        verbose=True,
    )
    print("\nTest metrics")
    print(f"mAP50:    {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")


if __name__ == "__main__":
    main()
