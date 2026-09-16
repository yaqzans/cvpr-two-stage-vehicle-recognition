"""05 - Pipeline-correct Stage-1 crop recall evaluation.

Why this exists alongside scripts04_evaluate_stage1_crops.py
------------------------------------------------------------
The two-stage design discards YOLO's class label: Stage 2 re-classifies every
crop and makes the final decision. So the Stage-1 question that matters is

    "did Stage 1 hand Stage 2 a usable box for this object?"

not

    "did Stage 1 also guess the right class?"

Script 04 required pred.class_id == gt.class_id and ran at a single conf=0.25,
which is a single-stage detector's operating point. Both choices penalise
Stage 1 for mistakes Stage 2 is designed to absorb, so its 84.63% is a floor
rather than the pipeline's real crop recall.

This script reports both definitions over a confidence sweep, so the operating
point can be chosen on evidence.

Other fixes vs. script 04:
    - Global greedy IoU matching instead of an order-dependent per-GT loop.
    - Counts false positives, so the precision cost of lowering conf is visible.
    - Crop filenames no longer collide (04 lost ~100 crops to same-name saves).

Run from the project root:
    python scripts/05_eval_stage1_crop_recall.py
    python scripts/05_eval_stage1_crop_recall.py --save-crops-at 0.10
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT / ".ultralytics"))

import cv2
import numpy as np
import torch
from ultralytics import YOLO

CLASS_NAMES = [
    "person", "rickshaw", "rickshaw_van", "auto_rickshaw", "truck",
    "pickup_truck", "private_car", "motorcycle", "bicycle", "bus",
    "micro_bus", "covered_van", "human_hauler",
]

DEFAULT_WEIGHTS = PROJECT_ROOT / "runs" / "stage1_yolo26n" / "train" / "weights" / "best.pt"
DEFAULT_SWEEP = [0.001, 0.01, 0.025, 0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 0.50]
IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png")


def load_ground_truth(label_path: Path, img_w: int, img_h: int):
    """YOLO normalized xywh -> pixel xyxy."""
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        class_id = int(parts[0])
        cx, cy, w, h = (float(p) for p in parts[1:])
        cx, w = cx * img_w, w * img_w
        cy, h = cy * img_h, h * img_h
        boxes.append({
            "class_id": class_id,
            "box": [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2],
        })
    return boxes


def iou_matrix(gt_boxes: np.ndarray, pred_boxes: np.ndarray) -> np.ndarray:
    """Vectorised pairwise IoU. gt (N,4), pred (M,4) -> (N,M)."""
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return np.zeros((len(gt_boxes), len(pred_boxes)), dtype=np.float32)

    x1 = np.maximum(gt_boxes[:, None, 0], pred_boxes[None, :, 0])
    y1 = np.maximum(gt_boxes[:, None, 1], pred_boxes[None, :, 1])
    x2 = np.minimum(gt_boxes[:, None, 2], pred_boxes[None, :, 2])
    y2 = np.minimum(gt_boxes[:, None, 3], pred_boxes[None, :, 3])

    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)

    gt_area = np.clip(gt_boxes[:, 2] - gt_boxes[:, 0], 0, None) * \
              np.clip(gt_boxes[:, 3] - gt_boxes[:, 1], 0, None)
    pred_area = np.clip(pred_boxes[:, 2] - pred_boxes[:, 0], 0, None) * \
                np.clip(pred_boxes[:, 3] - pred_boxes[:, 1], 0, None)

    union = gt_area[:, None] + pred_area[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0).astype(np.float32)


def greedy_match(ious: np.ndarray, iou_threshold: float):
    """Global greedy one-to-one matching, highest IoU first.

    Returns {gt_index: (pred_index, iou)}. Order-independent, unlike a
    per-GT first-come loop.
    """
    matches = {}
    if ious.size == 0:
        return matches

    flat = np.argsort(ious, axis=None)[::-1]
    used_gt, used_pred = set(), set()

    for flat_index in flat:
        gt_index, pred_index = np.unravel_index(flat_index, ious.shape)
        score = ious[gt_index, pred_index]
        if score < iou_threshold:
            break
        gt_index, pred_index = int(gt_index), int(pred_index)
        if gt_index in used_gt or pred_index in used_pred:
            continue
        used_gt.add(gt_index)
        used_pred.add(pred_index)
        matches[gt_index] = (pred_index, float(score))

    return matches


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline-correct Stage-1 crop recall.")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--iou", type=float, default=0.50, help="IoU for a crop to count as usable.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--save-crops-at", type=float, default=None,
                        help="Confidence at which to also write crops to disk.")
    parser.add_argument("--conf-sweep", type=float, nargs="+", default=DEFAULT_SWEEP)
    args = parser.parse_args()

    if not args.weights.exists():
        raise FileNotFoundError(f"No checkpoint found: {args.weights}")

    images_dir = PROJECT_ROOT / "Datasets" / "rsud20k" / "images" / args.split
    labels_dir = PROJECT_ROOT / "Datasets" / "rsud20k" / "labels" / args.split
    out_dir = args.weights.parents[2] / "crop_recall_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    sweep = sorted(set(args.conf_sweep))
    if args.save_crops_at is not None and args.save_crops_at not in sweep:
        sweep.append(args.save_crops_at)
        sweep.sort()

    model = YOLO(str(args.weights))
    device = 0 if torch.cuda.is_available() else "cpu"

    image_files = sorted(
        p for pattern in IMAGE_EXTENSIONS for p in images_dir.glob(pattern)
    )
    print(f"Weights : {args.weights}")
    print(f"Split   : {args.split}  ({len(image_files)} images)")
    print(f"IoU     : {args.iou}   imgsz: {args.imgsz}   max_det: {args.max_det}")
    print(f"Sweep   : {sweep}")
    print()

    # stats[conf][mode] -> per-class counters. mode is "agnostic" or "aware".
    stats = {
        conf: {
            mode: defaultdict(lambda: {"gt": 0, "matched": 0, "ious": []})
            for mode in ("agnostic", "aware")
        }
        for conf in sweep
    }
    false_positives = {conf: 0 for conf in sweep}
    total_predictions = {conf: 0 for conf in sweep}

    crop_dir = None
    crop_counts = {"matched": 0, "background": 0}
    background_records: list[dict] = []
    if args.save_crops_at is not None:
        crop_dir = out_dir / f"crops_conf{args.save_crops_at:g}"
        crop_dir.mkdir(parents=True, exist_ok=True)

    min_conf = min(sweep)

    for index, image_path in enumerate(image_files, start=1):
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"  [!] Unreadable: {image_path}")
            continue

        height, width = image.shape[:2]
        ground_truth = load_ground_truth(labels_dir / f"{image_path.stem}.txt", width, height)
        if not ground_truth:
            continue

        # One inference pass at the lowest sweep confidence; higher thresholds
        # are obtained by filtering, so the sweep is nearly free.
        result = model.predict(
            source=image,
            conf=min_conf,
            imgsz=args.imgsz,
            max_det=args.max_det,
            device=device,
            verbose=False,
        )[0]

        if result.boxes is not None and len(result.boxes):
            pred_boxes_all = result.boxes.xyxy.cpu().numpy()
            pred_conf_all = result.boxes.conf.cpu().numpy()
            pred_cls_all = result.boxes.cls.cpu().numpy().astype(int)
        else:
            pred_boxes_all = np.zeros((0, 4), dtype=np.float32)
            pred_conf_all = np.zeros((0,), dtype=np.float32)
            pred_cls_all = np.zeros((0,), dtype=int)

        gt_boxes = np.array([g["box"] for g in ground_truth], dtype=np.float32)
        gt_cls = np.array([g["class_id"] for g in ground_truth], dtype=int)

        for conf in sweep:
            keep = pred_conf_all >= conf
            pred_boxes = pred_boxes_all[keep]
            pred_cls = pred_cls_all[keep]
            total_predictions[conf] += len(pred_boxes)

            ious = iou_matrix(gt_boxes, pred_boxes)

            # Class-agnostic: the pipeline's real question.
            agnostic = greedy_match(ious, args.iou)

            # Class-aware: script 04's definition, kept for comparison.
            masked = ious.copy()
            if masked.size:
                masked[gt_cls[:, None] != pred_cls[None, :]] = 0.0
            aware = greedy_match(masked, args.iou)

            false_positives[conf] += len(pred_boxes) - len(agnostic)

            for mode, matches in (("agnostic", agnostic), ("aware", aware)):
                bucket = stats[conf][mode]
                for gt_index, gt in enumerate(ground_truth):
                    entry = bucket[gt["class_id"]]
                    entry["gt"] += 1
                    if gt_index in matches:
                        entry["matched"] += 1
                        entry["ious"].append(matches[gt_index][1])

            if crop_dir is not None and conf == args.save_crops_at:
                pred_conf = pred_conf_all[keep]

                def write_crop(target_dir: Path, name: str, box) -> bool:
                    x1, y1, x2, y2 = box
                    x1, y1 = max(0, int(x1)), max(0, int(y1))
                    x2, y2 = min(width, int(x2)), min(height, int(y2))
                    crop = image[y1:y2, x1:x2]
                    if crop.size == 0:
                        return False
                    target_dir.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(target_dir / name), crop)
                    return True

                # Matched: what Stage 2 will see for a real object. Filed
                # under the ground-truth class, because that is the label
                # Stage 2 should learn to produce for this crop.
                for gt_index, (pred_index, score) in agnostic.items():
                    class_name = CLASS_NAMES[ground_truth[gt_index]["class_id"]]
                    # gt_index in the filename makes collisions impossible.
                    name = f"{image_path.stem}_obj{gt_index:03d}_iou{score:.2f}.jpg"
                    if write_crop(crop_dir / "matched" / class_name, name, pred_boxes[pred_index]):
                        crop_counts["matched"] += 1

                # Unmatched: Stage 1 pointed at something that is not a
                # labelled object. These are the "background" examples the
                # classifier currently has no way to represent.
                matched_preds = {p for p, _ in agnostic.values()}
                for pred_index in range(len(pred_boxes)):
                    if pred_index in matched_preds:
                        continue
                    name = (f"{image_path.stem}_fp{pred_index:03d}"
                            f"_conf{pred_conf[pred_index]:.2f}"
                            f"_{CLASS_NAMES[pred_cls[pred_index]]}.jpg")
                    if write_crop(crop_dir / "background", name, pred_boxes[pred_index]):
                        crop_counts["background"] += 1
                        box = pred_boxes[pred_index]
                        background_records.append({
                            "image": image_path.name,
                            "conf": float(pred_conf[pred_index]),
                            "predicted_class": CLASS_NAMES[pred_cls[pred_index]],
                            "width": int(box[2] - box[0]),
                            "height": int(box[3] - box[1]),
                            "best_iou_with_any_gt": float(ious[:, pred_index].max()) if ious.size else 0.0,
                        })

        if index % 100 == 0 or index == len(image_files):
            print(f"  processed {index}/{len(image_files)}")

    # ---------------------------------------------------------------- report
    def totals(conf, mode):
        bucket = stats[conf][mode]
        gt = sum(v["gt"] for v in bucket.values())
        matched = sum(v["matched"] for v in bucket.values())
        ious = [i for v in bucket.values() for i in v["ious"]]
        return gt, matched, ious

    print("\n" + "=" * 78)
    print("CONFIDENCE SWEEP - usable crop recall")
    print("=" * 78)
    print(f"{'conf':>7}{'agnostic':>11}{'class-aware':>13}{'gain':>8}"
          f"{'meanIoU':>10}{'preds/img':>11}{'FP/img':>9}")
    print("-" * 78)

    n_images = max(1, len(image_files))
    for conf in sweep:
        gt, matched_ag, ious_ag = totals(conf, "agnostic")
        _, matched_aw, _ = totals(conf, "aware")
        rec_ag = matched_ag / gt if gt else 0.0
        rec_aw = matched_aw / gt if gt else 0.0
        print(f"{conf:>7.3f}{rec_ag * 100:>10.2f}%{rec_aw * 100:>12.2f}%"
              f"{(rec_ag - rec_aw) * 100:>+7.2f}%"
              f"{np.mean(ious_ag) if ious_ag else 0:>10.4f}"
              f"{total_predictions[conf] / n_images:>11.1f}"
              f"{false_positives[conf] / n_images:>9.1f}")

    with open(out_dir / f"crop_recall_sweep_{args.split}.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "conf", "gt_objects", "matched_agnostic", "recall_agnostic",
            "matched_class_aware", "recall_class_aware",
            "mean_iou", "median_iou", "predictions_per_image", "false_positives_per_image",
        ])
        for conf in sweep:
            gt, matched_ag, ious_ag = totals(conf, "agnostic")
            _, matched_aw, _ = totals(conf, "aware")
            writer.writerow([
                conf, gt, matched_ag, f"{matched_ag / gt:.6f}" if gt else 0,
                matched_aw, f"{matched_aw / gt:.6f}" if gt else 0,
                f"{np.mean(ious_ag):.6f}" if ious_ag else 0,
                f"{np.median(ious_ag):.6f}" if ious_ag else 0,
                f"{total_predictions[conf] / n_images:.4f}",
                f"{false_positives[conf] / n_images:.4f}",
            ])

    # Per-class detail at the old operating point and at low confidence.
    detail_confs = [c for c in (0.25, 0.10, 0.05) if c in sweep]
    for conf in detail_confs:
        print("\n" + "=" * 78)
        print(f"PER-CLASS at conf={conf:g}")
        print("=" * 78)
        print(f"{'class':<16}{'GT':>6}{'agnostic':>11}{'aware':>10}{'gain':>8}{'meanIoU':>10}")
        print("-" * 78)
        for class_id, name in enumerate(CLASS_NAMES):
            ag = stats[conf]["agnostic"][class_id]
            aw = stats[conf]["aware"][class_id]
            if ag["gt"] == 0:
                continue
            rec_ag = ag["matched"] / ag["gt"]
            rec_aw = aw["matched"] / aw["gt"]
            print(f"{name:<16}{ag['gt']:>6}{rec_ag * 100:>10.2f}%{rec_aw * 100:>9.2f}%"
                  f"{(rec_ag - rec_aw) * 100:>+7.2f}%"
                  f"{np.mean(ag['ious']) if ag['ious'] else 0:>10.3f}")

        with open(out_dir / f"per_class_conf{conf:g}_{args.split}.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["class", "gt", "matched_agnostic", "recall_agnostic",
                             "matched_class_aware", "recall_class_aware", "mean_iou", "median_iou"])
            for class_id, name in enumerate(CLASS_NAMES):
                ag = stats[conf]["agnostic"][class_id]
                aw = stats[conf]["aware"][class_id]
                if ag["gt"] == 0:
                    continue
                writer.writerow([
                    name, ag["gt"], ag["matched"], f"{ag['matched'] / ag['gt']:.6f}",
                    aw["matched"], f"{aw['matched'] / aw['gt']:.6f}",
                    f"{np.mean(ag['ious']):.6f}" if ag["ious"] else 0,
                    f"{np.median(ag['ious']):.6f}" if ag["ious"] else 0,
                ])

    print(f"\nResults written to: {out_dir}")

    if crop_dir is not None:
        total = crop_counts["matched"] + crop_counts["background"]
        print("\n" + "=" * 78)
        print(f"CROPS SAVED at conf={args.save_crops_at:g}")
        print("=" * 78)
        print(f"  matched (real objects) : {crop_counts['matched']:>6}")
        print(f"  background (false pos) : {crop_counts['background']:>6}"
              f"   ({crop_counts['background'] / max(1, total) * 100:.1f}% of all crops)")
        print(f"  total                  : {total:>6}")
        print(f"  -> {crop_dir}")

        if background_records:
            with open(out_dir / f"background_crops_conf{args.save_crops_at:g}_{args.split}.csv",
                      "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(background_records[0].keys()))
                writer.writeheader()
                writer.writerows(background_records)

            # What is Stage 1 hallucinating, and how confidently?
            by_class = defaultdict(int)
            for record in background_records:
                by_class[record["predicted_class"]] += 1
            print("\n  What Stage 1 called these false positives:")
            for name, count in sorted(by_class.items(), key=lambda kv: -kv[1]):
                print(f"    {name:<16}{count:>6}  ({count / len(background_records) * 100:>4.1f}%)")

            confs = np.array([r["conf"] for r in background_records])
            overlaps = np.array([r["best_iou_with_any_gt"] for r in background_records])
            sides = np.array([min(r["width"], r["height"]) for r in background_records])
            print(f"\n  Confidence  : median {np.median(confs):.2f}, "
                  f"90th pct {np.percentile(confs, 90):.2f}")
            print(f"  Shortest side: median {np.median(sides):.0f} px, "
                  f"{(sides < 32).mean() * 100:.0f}% under 32 px")
            print(f"  Partial overlap with a real object (IoU 0.1-0.5): "
                  f"{((overlaps >= 0.1) & (overlaps < 0.5)).mean() * 100:.0f}%")
            print(f"  No overlap at all (IoU < 0.1): {(overlaps < 0.1).mean() * 100:.0f}%")


if __name__ == "__main__":
    main()
