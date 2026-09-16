"""01 — Verify RSUD20K image/label pairs before Stage-1 YOLO training."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from PIL import Image, UnidentifiedImageError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "Datasets" / "rsud20k"
SPLITS = ("train", "val", "test")
CLASS_COUNT = 13
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> None:
    failures: list[str] = []
    totals = Counter()

    for split in SPLITS:
        images_dir = DATASET_ROOT / "images" / split
        labels_dir = DATASET_ROOT / "labels" / split
        images = {path.stem: path for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS}
        labels = {path.stem: path for path in labels_dir.glob("*.txt")}

        missing_labels = sorted(images.keys() - labels.keys())
        missing_images = sorted(labels.keys() - images.keys())
        malformed = 0
        invalid_boxes = 0
        class_counts = Counter()

        for stem, image_path in images.items():
            try:
                with Image.open(image_path) as image:
                    image.verify()
            except (UnidentifiedImageError, OSError):
                failures.append(f"Unreadable image: {image_path}")

            label_path = labels.get(stem)
            if label_path is None:
                continue
            for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
                parts = line.split()
                if len(parts) != 5:
                    malformed += 1
                    continue
                try:
                    class_id = int(parts[0])
                    cx, cy, width, height = map(float, parts[1:])
                except ValueError:
                    malformed += 1
                    continue
                if not (0 <= class_id < CLASS_COUNT and 0 < width <= 1 and 0 < height <= 1 and 0 <= cx <= 1 and 0 <= cy <= 1):
                    invalid_boxes += 1
                else:
                    class_counts[class_id] += 1

        print(f"{split}: {len(images)} images, {len(labels)} labels, {sum(class_counts.values())} valid boxes")
        print(f"  Missing labels: {len(missing_labels)} | Missing images: {len(missing_images)} | Malformed: {malformed} | Invalid boxes: {invalid_boxes}")
        totals.update(class_counts)
        if missing_labels or missing_images or malformed or invalid_boxes:
            failures.append(f"{split} failed integrity checks")

    print("\nTotal labels by class id:")
    for class_id in range(CLASS_COUNT):
        print(f"  {class_id:2d}: {totals[class_id]}")

    if failures:
        raise SystemExit("\nFAILED:\n" + "\n".join(failures))
    print("\nPASS: RSUD20K is ready for Stage-1 YOLO training.")


if __name__ == "__main__":
    main()
