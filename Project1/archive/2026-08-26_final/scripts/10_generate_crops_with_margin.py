"""10 - Regenerate the classification crops with a context margin.

Why
---
train_stage2_convnext_tiny_v2.py jitters the crop box during training so the
classifier sees boxes like the ones Stage 1 actually produces (~0.89 mean IoU).
But the crops in Datasets/BangladeshVehicleClassification/ were cut exactly on
the ground-truth box, so they carry no surrounding pixels. When the jitter
expands outward it can only pad with neutral grey - which is not what a
too-loose detector box looks like in reality.

This regenerates the same crops with a margin around each box, so jitter up to
that margin lands on real image content. Everything else - the split, the class
folders, the filenames - is identical to the original generator, so the two
datasets are drop-in interchangeable.

The output goes to a SEPARATE directory. The original dataset is never touched,
so the v1/v2 results stay reproducible.

    python scripts/10_generate_crops_with_margin.py                  # 25% margin
    python scripts/10_generate_crops_with_margin.py --margin 0.35
"""

from __future__ import annotations

import argparse
import csv
import time
from collections import Counter
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SPLITS = ["train", "val", "test"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MIN_CROP_SIZE = 5


def slugify(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def find_image(images_dir: Path, stem: str):
    for ext in IMAGE_EXTENSIONS:
        for candidate in (images_dir / f"{stem}{ext}", images_dir / f"{stem}{ext.upper()}"):
            if candidate.exists():
                return candidate
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate crops with a context margin.")
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "Datasets" / "rsud20k")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--margin", type=float, default=0.25,
                        help="Fraction of box width/height added on each side.")
    args = parser.parse_args()

    output_dir = args.output or (
        PROJECT_ROOT / "Datasets" / f"BangladeshVehicleClassification_margin{int(args.margin * 100)}"
    )

    classes = [line.strip() for line in
               (args.input / "classes.txt").read_text(encoding="utf-8").splitlines()
               if line.strip()]

    print("=" * 66)
    print("CROP REGENERATION WITH CONTEXT MARGIN")
    print("=" * 66)
    print(f"Input  : {args.input}")
    print(f"Output : {output_dir}")
    print(f"Margin : {args.margin:.0%} of box size on each side")
    print(f"Classes: {len(classes)}")
    print()

    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(
            f"Output directory already exists and is not empty:\n  {output_dir}\n"
            "Refusing to overwrite. Delete it or pass a different --output."
        )

    for split in SPLITS:
        for name in classes:
            (output_dir / split / slugify(name)).mkdir(parents=True, exist_ok=True)

    stats = {"crops": 0, "too_small": 0, "missing": 0, "errors": 0,
             "clipped_at_edge": 0}
    per_class = Counter()
    per_split = Counter()

    csv_path = output_dir / "dataset_statistics.csv"
    start = time.time()

    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Filename", "Class", "Split", "Source Image",
                         "X1", "Y1", "X2", "Y2", "Width", "Height",
                         "MarginApplied"])

        for split in SPLITS:
            images_dir = args.input / "images" / split
            labels_dir = args.input / "labels" / split
            label_files = sorted(labels_dir.glob("*.txt"))
            print(f"{split}: {len(label_files)} annotation files")

            for index, label_path in enumerate(label_files, start=1):
                image_path = find_image(images_dir, label_path.stem)
                if image_path is None:
                    stats["missing"] += 1
                    continue
                try:
                    with Image.open(image_path) as image:
                        image = image.convert("RGB")
                        img_w, img_h = image.size
                        counter = Counter()

                        for line in label_path.read_text(encoding="utf-8").splitlines():
                            parts = line.split()
                            if len(parts) < 5:
                                continue
                            try:
                                class_id = int(float(parts[0]))
                                cx, cy, bw, bh = (float(p) for p in parts[1:5])
                            except ValueError:
                                continue
                            if not 0 <= class_id < len(classes):
                                continue

                            box_w, box_h = bw * img_w, bh * img_h
                            centre_x, centre_y = cx * img_w, cy * img_h

                            # Expand by the margin, then clamp to the image.
                            pad_x = box_w * args.margin
                            pad_y = box_h * args.margin
                            x1 = centre_x - box_w / 2 - pad_x
                            y1 = centre_y - box_h / 2 - pad_y
                            x2 = centre_x + box_w / 2 + pad_x
                            y2 = centre_y + box_h / 2 + pad_y

                            if x1 < 0 or y1 < 0 or x2 > img_w or y2 > img_h:
                                stats["clipped_at_edge"] += 1

                            x1 = int(round(max(0, min(x1, img_w))))
                            y1 = int(round(max(0, min(y1, img_h))))
                            x2 = int(round(max(0, min(x2, img_w))))
                            y2 = int(round(max(0, min(y2, img_h))))

                            if x2 - x1 < MIN_CROP_SIZE or y2 - y1 < MIN_CROP_SIZE:
                                stats["too_small"] += 1
                                continue

                            class_name = classes[class_id]
                            counter[class_name] += 1
                            # Same filename scheme as the original generator so
                            # the two datasets line up one to one.
                            filename = f"{label_path.stem}_{counter[class_name]}.jpg"
                            out_path = output_dir / split / slugify(class_name) / filename
                            image.crop((x1, y1, x2, y2)).save(out_path, "JPEG", quality=95)

                            writer.writerow([filename, class_name, split, image_path.name,
                                             x1, y1, x2, y2, x2 - x1, y2 - y1, args.margin])
                            stats["crops"] += 1
                            per_class[class_name] += 1
                            per_split[split] += 1
                except Exception as exc:
                    stats["errors"] += 1
                    print(f"  [!] {image_path.name}: {exc}")

                if index % 2000 == 0:
                    print(f"  {index}/{len(label_files)}  ({stats['crops']:,} crops)")

    elapsed = time.time() - start

    summary = output_dir / "dataset_summary.txt"
    with open(summary, "w", encoding="utf-8") as f:
        f.write("BangladeshVehicleClassification (context margin) - Summary\n")
        f.write("=" * 58 + "\n\n")
        f.write(f"Source        : {args.input}\n")
        f.write(f"Context margin: {args.margin:.0%} of box size per side\n")
        f.write(f"Total crops   : {stats['crops']:,}\n")
        f.write(f"Too small     : {stats['too_small']}\n")
        f.write(f"Missing images: {stats['missing']}\n")
        f.write(f"Errors        : {stats['errors']}\n")
        f.write(f"Clipped at image edge: {stats['clipped_at_edge']:,} "
                f"({stats['clipped_at_edge'] / max(1, stats['crops']):.1%} of crops - "
                f"these got less margin than requested on at least one side)\n")
        f.write(f"Time          : {elapsed:.1f} s\n\n")
        f.write("Per split:\n")
        for split in SPLITS:
            f.write(f"  {split}: {per_split[split]:,}\n")
        f.write("\nPer class:\n")
        for name in classes:
            f.write(f"  {name}: {per_class[name]:,}\n")

    print()
    print("=" * 66)
    print(f"DONE  {stats['crops']:,} crops in {elapsed:.1f}s -> {output_dir}")
    print(f"Clipped at image edge: {stats['clipped_at_edge']:,} "
          f"({stats['clipped_at_edge'] / max(1, stats['crops']):.1%})")
    print("=" * 66)
    print()
    print("To train on it, point DATA_DIR in train_stage2_convnext_tiny_v2.py at")
    print("this folder and raise BOX_JITTER_FRAC toward the margin value, then")
    print("save to a NEW run dir so v2's results are preserved.")


if __name__ == "__main__":
    main()
