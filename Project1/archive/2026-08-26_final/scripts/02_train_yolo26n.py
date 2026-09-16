"""02 — Train or resume the Stage-1 YOLO26n detector on RSUD20K."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT / ".ultralytics"))

import torch
from ultralytics import YOLO

RUNS_DIR = PROJECT_ROOT / "runs" / "stage1_yolo26n"
LAST_WEIGHTS = RUNS_DIR / "train" / "weights" / "last.pt"


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch cannot access a CUDA GPU.")

    if not LAST_WEIGHTS.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {LAST_WEIGHTS}")

    print(f"Resuming on CUDA device: {torch.cuda.get_device_name(0)}")
    model = YOLO(str(LAST_WEIGHTS))
    model.train(resume=True)


if __name__ == "__main__":
    main()