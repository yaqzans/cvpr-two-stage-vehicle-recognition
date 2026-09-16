"""02 — Train the Stage-1 YOLO12m detector on RSUD20K, locally, resumably.

Same recipe as scripts/02_train_yolo26s.py (rare-class oversampling +
augmentation policy — the only proven-good Stage-1 config so far), swapped
onto a bigger backbone. Auto-resumes on its own: if a previous run's
last.pt is already sitting in runs/stage1_yolo12m/train/weights/, just
re-running this exact script continues from it instead of starting over.
That's the whole crash-recovery story — ultralytics writes last.pt after
every epoch, so a power cut or a PC shutdown loses at most one in-progress
epoch, and needs no special handling here beyond checking for that file.

Run from the project root:
    python scripts/02_train_yolo12m.py

Unattended-run friendly: on an out-of-memory error during the very first
attempt, it halves the batch size and retries once before giving up, since
nobody may be around to react to a crash.
"""

from __future__ import annotations

import argparse
import os
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT / ".ultralytics"))

import torch
from ultralytics import YOLO

DATASET_ROOT = PROJECT_ROOT / "Datasets" / "rsud20k"
CONFIG_DIR = PROJECT_ROOT / "configs"
RUNS_DIR = PROJECT_ROOT / "runs" / "stage1_yolo12m"
LAST_WEIGHTS = RUNS_DIR / "train" / "weights" / "last.pt"
PRETRAINED = PROJECT_ROOT / "yolo12m.pt"

CLASS_NAMES = [
    "person", "rickshaw", "rickshaw_van", "auto_rickshaw", "truck",
    "pickup_truck", "private_car", "motorcycle", "bicycle", "bus",
    "micro_bus", "covered_van", "human_hauler",
]

OVERSAMPLE_EXTRA_COPIES = {
    "truck": 3,
    "pickup_truck": 3,
    "covered_van": 3,
    "rickshaw_van": 2,
    "bus": 2,
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_oversampled_train_list() -> Path:
    images_dir = DATASET_ROOT / "images" / "train"
    labels_dir = DATASET_ROOT / "labels" / "train"

    target_ids = {
        CLASS_NAMES.index(name): extra
        for name, extra in OVERSAMPLE_EXTRA_COPIES.items()
    }

    images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)

    lines: list[str] = []
    dup_counts: Counter = Counter()

    for image_path in images:
        lines.append(str(image_path.resolve()))

        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue

        present_ids = set()
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            try:
                present_ids.add(int(parts[0]))
            except ValueError:
                continue

        matched_extras = [target_ids[cid] for cid in present_ids if cid in target_ids]
        if matched_extras:
            extra_copies = max(matched_extras)
            for _ in range(extra_copies):
                lines.append(str(image_path.resolve()))
            for cid in present_ids:
                if cid in target_ids:
                    dup_counts[CLASS_NAMES[cid]] += extra_copies

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    list_path = CONFIG_DIR / "rsud20k_train_oversampled.txt"
    list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Base train images: {len(images)}")
    print(f"Total train entries after oversampling: {len(lines)}")
    print("Extra duplicated occurrences by target class:")
    for name in OVERSAMPLE_EXTRA_COPIES:
        print(f"  {name:15s} +{dup_counts.get(name, 0)}")

    return list_path


def write_data_yaml(train_list_path: Path) -> Path:
    val_dir = (DATASET_ROOT / "images" / "val").resolve()
    test_dir = (DATASET_ROOT / "images" / "test").resolve()

    names_block = "\n".join(f"  {i}: {name}" for i, name in enumerate(CLASS_NAMES))
    yaml_text = (
        f"train: {train_list_path.resolve().as_posix()}\n"
        f"val: {val_dir.as_posix()}\n"
        f"test: {test_dir.as_posix()}\n"
        f"names:\n{names_block}\n"
    )

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    yaml_path = CONFIG_DIR / "rsud20k_yolo12m_oversampled.yaml"
    yaml_path.write_text(yaml_text, encoding="utf-8")
    print(f"\nData yaml written to: {yaml_path}")
    return yaml_path


def train_fresh(imgsz: int, batch: int, epochs: int, patience: int) -> None:
    train_list_path = build_oversampled_train_list()
    data_yaml = write_data_yaml(train_list_path)
    model = YOLO(str(PRETRAINED))
    model.train(
        data=str(data_yaml),
        imgsz=imgsz,
        epochs=epochs,
        patience=patience,
        batch=batch,
        device=0,
        workers=2,
        optimizer="auto",
        close_mosaic=10,
        seed=42,
        deterministic=True,
        project=str(RUNS_DIR),
        name="train",
        exist_ok=True,
        fliplr=0.5,
        flipud=0.0,
        degrees=0.0,
        translate=0.10,
        scale=0.50,
        hsv_h=0.015,
        hsv_s=0.70,
        hsv_v=0.40,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Stage-1 YOLO12m on RSUD20K with oversampling.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8, help="Conservative default for 8GB VRAM at imgsz 640.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch cannot access a CUDA GPU.")

    if LAST_WEIGHTS.exists():
        print(f"Found existing checkpoint: {LAST_WEIGHTS}")
        print(f"Resuming on CUDA device: {torch.cuda.get_device_name(0)}")
        model = YOLO(str(LAST_WEIGHTS))
        model.train(resume=True)
        return

    print(f"No checkpoint found, starting fresh on CUDA device: {torch.cuda.get_device_name(0)}")
    try:
        train_fresh(args.imgsz, args.batch, args.epochs, args.patience)
    except torch.cuda.OutOfMemoryError:
        fallback_batch = max(1, args.batch // 2)
        print(f"\nOut of memory at batch={args.batch}. Retrying once at batch={fallback_batch}.")
        torch.cuda.empty_cache()
        train_fresh(args.imgsz, fallback_batch, args.epochs, args.patience)


if __name__ == "__main__":
    main()
