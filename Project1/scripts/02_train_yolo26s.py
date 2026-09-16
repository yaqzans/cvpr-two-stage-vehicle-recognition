"""02 — Train the Stage-1 YOLO26s detector on RSUD20K.

Changes vs. the YOLO26n baseline:
    1. Model upgraded: yolo26n.pt -> yolo26s.pt (more capacity to separate
       visually similar large-vehicle classes that were collapsing into
       "private_car" or "background" in the confusion matrix).
    2. imgsz 640 -> 960 (more pixels for distant/partially-visible trucks,
       buses, covered vans, which is where recall was weakest).
    3. Rare/confused-class oversampling: images containing truck,
       pickup_truck, covered_van, rickshaw_van, or bus are duplicated extra
       times in the train list so the model sees them more often per epoch,
       without touching val/test (evaluation stays on the untouched split).

Run from the project root:
    python scripts/02_train_yolo26s.py

Unattended-run friendly: everything needed to resume is written to disk
(oversampled list + generated data yaml), so if training is interrupted you
can resume with:
    python -c "from ultralytics import YOLO; YOLO('runs/stage1_yolo26s/train/weights/last.pt').train(resume=True)"
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
RUNS_DIR = PROJECT_ROOT / "runs" / "stage1_yolo26s"

CLASS_NAMES = [
    "person", "rickshaw", "rickshaw_van", "auto_rickshaw", "truck",
    "pickup_truck", "private_car", "motorcycle", "bicycle", "bus",
    "micro_bus", "covered_van", "human_hauler",
]

# Extra copies added on top of the single base occurrence, keyed by class
# name. Chosen from the confusion-matrix / crop-recall analysis: truck,
# pickup_truck, and covered_van had the worst recall (~35-45%) and get the
# strongest boost; rickshaw_van and bus were moderately weak and get a
# smaller boost.
OVERSAMPLE_EXTRA_COPIES = {
    "truck": 3,
    "pickup_truck": 3,
    "covered_van": 3,
    "rickshaw_van": 2,
    "bus": 2,
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_oversampled_train_list() -> Path:
    """Write a train image list (paths, one per line) with rare/confused
    classes duplicated. Returns the path to the list file. Val/test are
    left untouched — this only affects what the model sees during training.
    """
    images_dir = DATASET_ROOT / "images" / "train"
    labels_dir = DATASET_ROOT / "labels" / "train"

    target_ids = {
        CLASS_NAMES.index(name): extra
        for name, extra in OVERSAMPLE_EXTRA_COPIES.items()
    }

    images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)

    lines: list[str] = []
    dup_counts = Counter()

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
    """Generate a data yaml pointing train at the oversampled list while
    keeping val/test on the original, untouched directories."""
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
    yaml_path = CONFIG_DIR / "rsud20k_yolo26s_oversampled.yaml"
    yaml_path.write_text(yaml_text, encoding="utf-8")
    print(f"\nData yaml written to: {yaml_path}")
    return yaml_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Stage-1 YOLO26s on RSUD20K with oversampling.")
    parser.add_argument("--imgsz", type=int, default=960, help="Training image size.")
    parser.add_argument("--batch", type=int, default=6, help="Batch size (kept conservative for 8GB VRAM at imgsz 960).")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--no-oversample", action="store_true", help="Disable rare-class oversampling and train on the plain split.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch cannot access a CUDA GPU.")

    print(f"Training on CUDA device: {torch.cuda.get_device_name(0)}")

    if args.no_oversample:
        data_yaml = CONFIG_DIR / "rsud20k_yolo.yaml"
        if not data_yaml.exists():
            raise FileNotFoundError(f"Expected existing config at {data_yaml}")
        print("Oversampling disabled — using the original data yaml.")
    else:
        train_list_path = build_oversampled_train_list()
        data_yaml = write_data_yaml(train_list_path)

    model = YOLO("yolo26s.pt")

    model.train(
        data=str(data_yaml),
        imgsz=args.imgsz,
        epochs=args.epochs,
        patience=args.patience,
        batch=args.batch,
        device=0,
        workers=4,
        optimizer="auto",
        close_mosaic=10,
        seed=42,
        deterministic=True,
        project=str(RUNS_DIR),
        name="train",
        exist_ok=True,
        # Same augmentation policy as the YOLO26n baseline run.
        fliplr=0.5,
        flipud=0.0,
        degrees=0.0,
        translate=0.10,
        scale=0.50,
        hsv_h=0.015,
        hsv_s=0.70,
        hsv_v=0.40,
    )


if __name__ == "__main__":
    main()
