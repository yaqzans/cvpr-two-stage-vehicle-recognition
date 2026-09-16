#!/usr/bin/env python3
"""
generate_classification_dataset.py

Converts the RSUD20K object detection dataset (YOLO-format annotations)
into an image classification dataset by cropping every annotated object
into its own image file.

Input structure expected:

    RSUD20K/
        images/
            train/
            val/
            test/
        labels/
            train/
            val/
            test/
        classes.txt

Output structure produced:

    BangladeshVehicleClassification/
        dataset_statistics.csv
        dataset_summary.txt
        train/
            person/
            rickshaw/
            ...
        val/
            ...
        test/
            ...

Usage:
    python generate_classification_dataset.py \
        --input /path/to/RSUD20K \
        --output /path/to/BangladeshVehicleClassification

If --input / --output are omitted, the defaults below are used.
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

from PIL import Image

# --------------------------------------------------------------------------
# Defaults (edit these if you don't want to pass command-line arguments)
# --------------------------------------------------------------------------
DEFAULT_INPUT_DIR = "rsud20k"
DEFAULT_OUTPUT_DIR = "BangladeshVehicleClassification"

SPLITS = ["train", "val", "test"]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Minimum crop size (in pixels) to keep. Extremely tiny boxes are usually
# annotation noise and are not useful for classification.
MIN_CROP_SIZE = 5


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def slugify(name: str) -> str:
    """Turn a class name like 'Rickshaw Van' into 'rickshaw_van' for folder use."""
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def load_classes(classes_path: Path):
    """Read classes.txt and return a list of class names indexed by class id."""
    if not classes_path.exists():
        raise FileNotFoundError(f"classes.txt not found at: {classes_path}")

    with open(classes_path, "r", encoding="utf-8") as f:
        classes = [line.strip() for line in f if line.strip() != ""]

    if not classes:
        raise ValueError("classes.txt appears to be empty.")

    return classes


def find_image_file(images_dir: Path, stem: str):
    """Given a directory and a filename stem (no extension), find the matching image file."""
    for ext in IMAGE_EXTENSIONS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
        candidate_upper = images_dir / f"{stem}{ext.upper()}"
        if candidate_upper.exists():
            return candidate_upper
    return None


def yolo_to_pixel_box(cx, cy, w, h, img_w, img_h):
    """Convert normalized YOLO (center_x, center_y, width, height) to pixel (x1, y1, x2, y2)."""
    box_w = w * img_w
    box_h = h * img_h
    center_x = cx * img_w
    center_y = cy * img_h

    x1 = center_x - (box_w / 2)
    y1 = center_y - (box_h / 2)
    x2 = center_x + (box_w / 2)
    y2 = center_y + (box_h / 2)

    # Clamp to image boundaries
    x1 = max(0, min(x1, img_w))
    y1 = max(0, min(y1, img_h))
    x2 = max(0, min(x2, img_w))
    y2 = max(0, min(y2, img_h))

    return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))


def parse_annotation_line(line: str):
    """Parse one YOLO annotation line into (class_id, cx, cy, w, h). Returns None if malformed."""
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    try:
        class_id = int(float(parts[0]))
        cx, cy, w, h = (float(p) for p in parts[1:5])
        return class_id, cx, cy, w, h
    except ValueError:
        return None


# --------------------------------------------------------------------------
# Main conversion logic
# --------------------------------------------------------------------------
def build_output_folders(output_dir: Path, class_names):
    for split in SPLITS:
        for class_name in class_names:
            folder = output_dir / split / slugify(class_name)
            folder.mkdir(parents=True, exist_ok=True)


def process_split(input_dir: Path, output_dir: Path, split: str, class_names,
                   csv_writer, stats):
    images_dir = input_dir / "images" / split
    labels_dir = input_dir / "labels" / split

    if not images_dir.exists():
        print(f"  [!] Skipping split '{split}': images directory not found at {images_dir}")
        return

    if not labels_dir.exists():
        print(f"  [!] Skipping split '{split}': labels directory not found at {labels_dir}")
        return

    label_files = sorted(labels_dir.glob("*.txt"))
    total = len(label_files)
    print(f"  Found {total} annotation files in split '{split}'")

    for idx, label_path in enumerate(label_files, start=1):
        stem = label_path.stem
        image_path = find_image_file(images_dir, stem)

        if image_path is None:
            stats["missing_images"] += 1
            continue

        try:
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                img_w, img_h = img.size

                with open(label_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()

                # Track how many objects of each class have been cropped
                # from this particular source image, for filename numbering.
                instance_counter = {}

                for line in lines:
                    parsed = parse_annotation_line(line)
                    if parsed is None:
                        stats["malformed_annotations"] += 1
                        continue

                    class_id, cx, cy, w, h = parsed

                    if class_id < 0 or class_id >= len(class_names):
                        stats["invalid_class_ids"] += 1
                        continue

                    class_name = class_names[class_id]
                    x1, y1, x2, y2 = yolo_to_pixel_box(cx, cy, w, h, img_w, img_h)

                    crop_w = x2 - x1
                    crop_h = y2 - y1

                    if crop_w < MIN_CROP_SIZE or crop_h < MIN_CROP_SIZE:
                        stats["too_small_skipped"] += 1
                        continue

                    cropped = img.crop((x1, y1, x2, y2))

                    instance_counter[class_name] = instance_counter.get(class_name, 0) + 1
                    instance_num = instance_counter[class_name]

                    out_filename = f"{stem}_{instance_num}.jpg"
                    out_folder = output_dir / split / slugify(class_name)
                    out_path = out_folder / out_filename

                    cropped.save(out_path, "JPEG", quality=95)

                    csv_writer.writerow([
                        out_filename,
                        class_name,
                        split,
                        image_path.name,
                        x1, y1, x2, y2,
                        crop_w, crop_h,
                    ])

                    stats["total_crops"] += 1
                    stats["per_class"][class_name] = stats["per_class"].get(class_name, 0) + 1
                    stats["per_split"][split] = stats["per_split"].get(split, 0) + 1

        except Exception as e:
            stats["errors"] += 1
            print(f"    [!] Error processing {image_path.name}: {e}")

        if idx % 500 == 0 or idx == total:
            print(f"    Processed {idx}/{total} images in split '{split}'")

    stats["images_processed"] += total


def write_summary(output_dir: Path, input_dir: Path, class_names, stats, elapsed_seconds):
    summary_path = output_dir / "dataset_summary.txt"

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("BangladeshVehicleClassification - Dataset Summary\n")
        f.write("=" * 55 + "\n\n")

        f.write("Original dataset:\n")
        f.write(f"  Source directory: {input_dir}\n")
        f.write(f"  Number of classes: {len(class_names)}\n\n")

        f.write("Processing:\n")
        f.write(f"  Total source images processed: {stats['images_processed']}\n")
        f.write(f"  Total cropped images generated: {stats['total_crops']}\n")
        f.write(f"  Malformed annotation lines skipped: {stats['malformed_annotations']}\n")
        f.write(f"  Invalid class ids skipped: {stats['invalid_class_ids']}\n")
        f.write(f"  Crops skipped (too small): {stats['too_small_skipped']}\n")
        f.write(f"  Images with missing source file: {stats['missing_images']}\n")
        f.write(f"  Errors encountered: {stats['errors']}\n")
        f.write(f"  Time taken: {elapsed_seconds:.1f} seconds\n\n")

        f.write("Images per split:\n")
        for split in SPLITS:
            count = stats["per_split"].get(split, 0)
            f.write(f"  {split}: {count}\n")
        f.write("\n")

        f.write("Images per class:\n")
        for class_name in class_names:
            count = stats["per_class"].get(class_name, 0)
            f.write(f"  {class_name}: {count}\n")
        f.write("\n")

        f.write(f"Total generated dataset size: {stats['total_crops']} images\n")

    print(f"\nSummary written to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert RSUD20K into a classification dataset.")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT_DIR,
                         help="Path to the RSUD20K root directory (contains images/, labels/, classes.txt)")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_DIR,
                         help="Path to the output directory to create")
    args = parser.parse_args()

    input_dir = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()

    print(f"Input directory:  {input_dir}")
    print(f"Output directory: {output_dir}")

    if not input_dir.exists():
        print(f"[ERROR] Input directory does not exist: {input_dir}")
        sys.exit(1)

    classes_path = input_dir / "classes.txt"
    class_names = load_classes(classes_path)
    print(f"\nLoaded {len(class_names)} classes:")
    for i, name in enumerate(class_names):
        print(f"  {i} -> {name}")

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nBuilding output folder structure under: {output_dir}")
    build_output_folders(output_dir, class_names)

    stats = {
        "images_processed": 0,
        "total_crops": 0,
        "malformed_annotations": 0,
        "invalid_class_ids": 0,
        "too_small_skipped": 0,
        "missing_images": 0,
        "errors": 0,
        "per_class": {},
        "per_split": {},
    }

    csv_path = output_dir / "dataset_statistics.csv"
    start_time = time.time()

    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "Filename", "Class", "Split", "Source Image",
            "X1", "Y1", "X2", "Y2", "Width", "Height",
        ])

        for split in SPLITS:
            print(f"\nProcessing split: {split}")
            process_split(input_dir, output_dir, split, class_names, csv_writer, stats)

    elapsed = time.time() - start_time

    write_summary(output_dir, input_dir, class_names, stats, elapsed)

    print("\n" + "=" * 55)
    print("DONE")
    print(f"Total cropped images generated: {stats['total_crops']}")
    print(f"Statistics CSV: {csv_path}")
    print(f"Summary file:   {output_dir / 'dataset_summary.txt'}")
    print("=" * 55)


if __name__ == "__main__":
    main()