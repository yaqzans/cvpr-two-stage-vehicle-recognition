"""06 - End-to-end evaluation of the two-stage pipeline.

    road scene -> Stage 1 (YOLO) -> boxes -> crops -> Stage 2 (ConvNeXt) -> class

Until now Stage 1 and Stage 2 have only ever been scored separately, and
Stage 2 was scored on ground-truth crops it will never see at run time. This
script runs the actual pipeline and scores it as a detector, so the two-stage
system can be compared like-for-like against the single-stage baseline
(YOLO alone) that the research question asks about.

Metrics
-------
mAP@50 / mAP@50-95
    Standard 101-point interpolated AP, matching the COCO protocol. Computed
    by this file rather than by Ultralytics so the identical evaluator scores
    both the single-stage and two-stage outputs. Validated by scoring raw
    YOLO output and checking it reproduces the known Ultralytics numbers -
    see --mode single.

End-to-end recognition accuracy
    Of every ground-truth object, the fraction that is both localised
    (IoU >= 0.5) and given the correct final class. This is the number a
    reader intuitively wants: "what fraction of the road objects does the
    whole system actually get right?"

Timing
    Per-stage wall-clock with CUDA synchronisation and warmup, reported as
    ms/frame and FPS, so the real-time question can be answered.

Modes
-----
    --mode two-stage   YOLO boxes + ConvNeXt classes   (the proposal)
    --mode single      YOLO boxes + YOLO classes       (the baseline)

Examples
--------
    python scripts/06_evaluate_pipeline.py --mode single  --conf 0.001 --tag baseline_map
    python scripts/06_evaluate_pipeline.py --mode two-stage --conf 0.001 --tag twostage_map
    python scripts/06_evaluate_pipeline.py --mode two-stage --conf 0.10 --tag twostage_op
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT / ".ultralytics"))

import cv2
import numpy as np
import timm
import torch
from PIL import Image, ImageOps
from torchvision import transforms
from ultralytics import YOLO

# RSUD20K / detector class order. This is the index space every metric below
# is computed in.
DET_CLASSES = [
    "person", "rickshaw", "rickshaw_van", "auto_rickshaw", "truck",
    "pickup_truck", "private_car", "motorcycle", "bicycle", "bus",
    "micro_bus", "covered_van", "human_hauler",
]
NUM_CLASSES = len(DET_CLASSES)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
PAD_FILL = (114, 114, 114)


# ============================================================
# PREPROCESSING (must match how the classifier was trained)
# ============================================================

class PadToSquare:
    """Pad the shorter side so the crop is square, keeping the whole object."""

    def __init__(self, fill=PAD_FILL):
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        if width == height:
            return image
        side = max(width, height)
        left = (side - width) // 2
        top = (side - height) // 2
        return ImageOps.expand(
            image, border=(left, top, side - width - left, side - height - top),
            fill=self.fill,
        )


def fast_pad_resize(crop_bgr: np.ndarray, image_size: int = 224) -> np.ndarray:
    """OpenCV equivalent of PadToSquare + Resize, returning RGB uint8.

    The PIL path costs ~1.8 ms per crop, which at 8 crops/frame is a quarter
    of the whole pipeline budget. This does the same two operations in
    OpenCV and defers normalisation to the GPU, where it is free.

    INTER_AREA matches torchvision's antialiased downscale; crops are almost
    always larger than 224, so that is the branch that matters.
    """
    height, width = crop_bgr.shape[:2]
    side = max(height, width)
    if height != width:
        top = (side - height) // 2
        left = (side - width) // 2
        crop_bgr = cv2.copyMakeBorder(
            crop_bgr, top, side - height - top, left, side - width - left,
            cv2.BORDER_CONSTANT, value=PAD_FILL,
        )
    interpolation = cv2.INTER_AREA if side > image_size else cv2.INTER_LINEAR
    resized = cv2.resize(crop_bgr, (image_size, image_size), interpolation=interpolation)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)


def normalize_batch(batch_uint8: np.ndarray, device: str) -> torch.Tensor:
    """uint8 NHWC RGB -> normalised float NCHW on the target device."""
    tensor = torch.from_numpy(batch_uint8).to(device, non_blocking=True)
    tensor = tensor.permute(0, 3, 1, 2).float().div_(255.0)
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    return tensor.sub_(mean).div_(std)


def build_eval_transform(preprocessing: str, image_size: int = 224):
    """v2 checkpoints pad to square; v1 checkpoints used Resize+CenterCrop.

    The transform has to match the checkpoint or the comparison is invalid,
    so it is selected from the checkpoint's own recorded config.
    """
    if preprocessing == "pad_to_square":
        return transforms.Compose([
            PadToSquare(),
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize(int(image_size * 256 / 224), antialias=True),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# ============================================================
# DETECTION METRICS
# ============================================================

def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)
    x1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    y1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    x2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    y2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = np.clip(boxes_a[:, 2] - boxes_a[:, 0], 0, None) * \
             np.clip(boxes_a[:, 3] - boxes_a[:, 1], 0, None)
    area_b = np.clip(boxes_b[:, 2] - boxes_b[:, 0], 0, None) * \
             np.clip(boxes_b[:, 3] - boxes_b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0).astype(np.float32)


def average_precision(tp: np.ndarray, n_ground_truth: int) -> float:
    """101-point interpolated AP from a score-sorted true-positive vector."""
    if n_ground_truth == 0:
        return float("nan")
    if len(tp) == 0:
        return 0.0

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(1 - tp)
    recall = tp_cum / n_ground_truth
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)

    # Make precision monotonically decreasing, then sample at 101 recalls.
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    grid = np.linspace(0, 1, 101)
    indices = np.searchsorted(recall, grid, side="left")
    sampled = np.where(indices < len(precision), precision[np.clip(indices, 0, len(precision) - 1)], 0.0)
    return float(sampled.mean())


def evaluate_detections(predictions, ground_truths, iou_thresholds=None):
    """COCO-style mAP.

    predictions   : list of (image_id, class_id, score, box)
    ground_truths : list of (image_id, class_id, box)
    """
    if iou_thresholds is None:
        iou_thresholds = np.arange(0.5, 1.0, 0.05)

    gt_by_key = {}
    for image_id, class_id, box in ground_truths:
        gt_by_key.setdefault((image_id, class_id), []).append(box)

    gt_count = {}
    for (image_id, class_id), boxes in gt_by_key.items():
        gt_count[class_id] = gt_count.get(class_id, 0) + len(boxes)

    pred_by_class = {}
    for image_id, class_id, score, box in predictions:
        pred_by_class.setdefault(class_id, []).append((score, image_id, box))

    ap_table = np.full((NUM_CLASSES, len(iou_thresholds)), np.nan)

    for class_id in range(NUM_CLASSES):
        n_gt = gt_count.get(class_id, 0)
        if n_gt == 0:
            continue
        entries = sorted(pred_by_class.get(class_id, []), key=lambda e: -e[0])
        if not entries:
            ap_table[class_id, :] = 0.0
            continue

        boxes = np.array([e[2] for e in entries], dtype=np.float32)
        image_ids = [e[1] for e in entries]

        for t_index, threshold in enumerate(iou_thresholds):
            claimed = {}
            tp = np.zeros(len(entries), dtype=np.float32)
            for p_index, image_id in enumerate(image_ids):
                gt_boxes = gt_by_key.get((image_id, class_id))
                if not gt_boxes:
                    continue
                used = claimed.setdefault((image_id, class_id), set())
                ious = iou_matrix(boxes[p_index:p_index + 1], np.array(gt_boxes, dtype=np.float32))[0]
                order = np.argsort(-ious)
                for g_index in order:
                    if ious[g_index] < threshold:
                        break
                    if g_index in used:
                        continue
                    used.add(int(g_index))
                    tp[p_index] = 1.0
                    break
            ap_table[class_id, t_index] = average_precision(tp, n_gt)

    valid = ~np.isnan(ap_table[:, 0])
    return {
        "per_class_ap50": ap_table[:, 0],
        "per_class_ap": np.nanmean(ap_table, axis=1),
        "map50": float(np.nanmean(ap_table[valid, 0])),
        "map50_95": float(np.nanmean(ap_table[valid, :])),
    }


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end two-stage pipeline evaluation.")
    parser.add_argument("--mode", choices=["two-stage", "single"], default="two-stage")
    parser.add_argument("--det-weights", type=Path,
                        default=PROJECT_ROOT / "runs/stage1_yolo26n/train/weights/best.pt")
    parser.add_argument("--cls-checkpoint", type=Path,
                        default=PROJECT_ROOT / "runs/stage2_convnext_tiny_v2/checkpoints/best.pt")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--cls-batch", type=int, default=64)
    parser.add_argument("--tta", action="store_true", help="Horizontal-flip TTA for Stage 2.")
    parser.add_argument("--fast-crop", action="store_true",
                        help="OpenCV crop preprocessing + GPU normalisation (pad_to_square only).")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Divide classifier logits by this before softmax. "
                             "Fit on val by scripts/09_calibrate_classifier.py; "
                             "1.0 leaves the model exactly as trained.")
    parser.add_argument("--score", choices=["product", "det", "cls"], default="product",
                        help="How the final detection score is formed.")
    parser.add_argument("--limit", type=int, default=None, help="Debug: only N images.")
    parser.add_argument("--tag", default=None, help="Output subfolder name.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tag = args.tag or f"{args.mode}_conf{args.conf:g}"
    out_dir = PROJECT_ROOT / "runs" / "pipeline_eval" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    images_dir = PROJECT_ROOT / "Datasets" / "rsud20k" / "images" / args.split
    labels_dir = PROJECT_ROOT / "Datasets" / "rsud20k" / "labels" / args.split

    print("=" * 72)
    print(f"PIPELINE EVALUATION - mode={args.mode}  conf={args.conf}  split={args.split}")
    print("=" * 72)
    print(f"Detector   : {args.det_weights}")

    detector = YOLO(str(args.det_weights))

    classifier = None
    cls_transform = None
    cls_to_det = None

    if args.mode == "two-stage":
        checkpoint = torch.load(args.cls_checkpoint, map_location=device, weights_only=False)
        cls_classes = checkpoint["classes"]
        cls_config = checkpoint.get("config", {})
        preprocessing = cls_config.get("preprocessing", "resize_centercrop")
        model_name = cls_config.get("model_name", "hf_hub:timm/convnext_tiny.in12k_ft_in1k")

        print(f"Classifier : {args.cls_checkpoint}")
        print(f"             version={cls_config.get('version', 'v1')} "
              f"preprocessing={preprocessing}")

        classifier = timm.create_model(model_name, pretrained=False, num_classes=len(cls_classes))
        classifier.load_state_dict(checkpoint["model_state_dict"])
        classifier.eval().to(device)
        if device == "cuda":
            classifier = classifier.to(memory_format=torch.channels_last)

        cls_transform = build_eval_transform(preprocessing)

        # The classifier's class order is alphabetical (ImageFolder); the
        # detector's is the RSUD20K order. Everything downstream lives in the
        # detector's index space, so build the map explicitly.
        cls_to_det = np.array([DET_CLASSES.index(name) for name in cls_classes], dtype=int)
        print(f"             class map: {len(cls_classes)} classes -> detector indices")

    print()

    image_files = sorted(images_dir.glob("*.jpg"))
    if args.limit:
        image_files = image_files[:args.limit]
    print(f"Images: {len(image_files)}")

    # ---------------------------------------------------------------- warmup
    warm = cv2.imread(str(image_files[0]))
    for _ in range(3):
        result = detector.predict(source=warm, conf=args.conf, imgsz=args.imgsz,
                                  max_det=args.max_det, device=0 if device == "cuda" else "cpu",
                                  verbose=False)[0]
        if classifier is not None:
            dummy = torch.zeros(2, 3, 224, 224, device=device)
            if device == "cuda":
                dummy = dummy.to(memory_format=torch.channels_last)
            with torch.no_grad(), torch.autocast("cuda", torch.float16, enabled=device == "cuda"):
                classifier(dummy)
    if device == "cuda":
        torch.cuda.synchronize()

    # ---------------------------------------------------------------- run
    predictions, ground_truths = [], []
    timing = {"read": 0.0, "detect": 0.0, "crop": 0.0, "classify": 0.0}
    n_crops_total = 0

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()

    loop_start = time.perf_counter()

    for index, image_path in enumerate(image_files, start=1):
        image_id = image_path.stem

        t0 = time.perf_counter()
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        sync(); timing["read"] += time.perf_counter() - t0

        # ground truth
        label_path = labels_dir / f"{image_id}.txt"
        if label_path.exists():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                class_id = int(parts[0])
                cx, cy, bw, bh = (float(p) for p in parts[1:])
                cx, bw = cx * width, bw * width
                cy, bh = cy * height, bh * height
                ground_truths.append((image_id, class_id,
                                      [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2]))

        # ---- Stage 1
        t0 = time.perf_counter()
        result = detector.predict(source=image, conf=args.conf, imgsz=args.imgsz,
                                  max_det=args.max_det, device=0 if device == "cuda" else "cpu",
                                  verbose=False)[0]
        sync(); timing["detect"] += time.perf_counter() - t0

        if result.boxes is None or len(result.boxes) == 0:
            if index % 100 == 0:
                print(f"  {index}/{len(image_files)}")
            continue

        boxes = result.boxes.xyxy.cpu().numpy()
        det_conf = result.boxes.conf.cpu().numpy()
        det_cls = result.boxes.cls.cpu().numpy().astype(int)

        if args.mode == "single":
            for b, c, s in zip(boxes, det_cls, det_conf):
                predictions.append((image_id, int(c), float(s), b.tolist()))
            if index % 100 == 0:
                print(f"  {index}/{len(image_files)}")
            continue

        # ---- crop
        t0 = time.perf_counter()
        tensors, keep_indices = [], []
        for b_index, box in enumerate(boxes):
            x1, y1 = max(0, int(box[0])), max(0, int(box[1]))
            x2, y2 = min(width, int(box[2])), min(height, int(box[3]))
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            crop = image[y1:y2, x1:x2]
            if args.fast_crop:
                tensors.append(fast_pad_resize(crop))
            else:
                pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                tensors.append(cls_transform(pil))
            keep_indices.append(b_index)
        sync(); timing["crop"] += time.perf_counter() - t0

        if not tensors:
            continue
        n_crops_total += len(tensors)

        # ---- Stage 2
        t0 = time.perf_counter()
        probabilities = []
        for start in range(0, len(tensors), args.cls_batch):
            chunk = tensors[start:start + args.cls_batch]
            if args.fast_crop:
                batch = normalize_batch(np.stack(chunk), device)
            else:
                batch = torch.stack(chunk).to(device)
            if device == "cuda":
                batch = batch.to(memory_format=torch.channels_last)
            with torch.no_grad(), torch.autocast("cuda", torch.float16, enabled=device == "cuda"):
                logits = classifier(batch)
                if args.tta:
                    # Mean, not sum: summing doubles the logit scale and
                    # silently sharpens the probabilities, which would
                    # confound the temperature fitted below.
                    logits = 0.5 * (logits + classifier(torch.flip(batch, dims=[3])))
            probabilities.append(
                (logits.float() / args.temperature).softmax(1).cpu().numpy())
        probabilities = np.concatenate(probabilities, axis=0)
        sync(); timing["classify"] += time.perf_counter() - t0

        cls_index = probabilities.argmax(1)
        cls_score = probabilities.max(1)

        for local, b_index in enumerate(keep_indices):
            final_class = int(cls_to_det[cls_index[local]])
            if args.score == "product":
                score = float(det_conf[b_index] * cls_score[local])
            elif args.score == "det":
                score = float(det_conf[b_index])
            else:
                score = float(cls_score[local])
            predictions.append((image_id, final_class, score, boxes[b_index].tolist()))

        if index % 100 == 0:
            print(f"  {index}/{len(image_files)}")

    total_wall = time.perf_counter() - loop_start
    n_images = len(image_files)

    # ---------------------------------------------------------------- score
    print("\nComputing mAP...")
    metrics = evaluate_detections(predictions, ground_truths)

    # End-to-end recognition accuracy: localised AND correctly classified.
    gt_by_image = {}
    for image_id, class_id, box in ground_truths:
        gt_by_image.setdefault(image_id, []).append((class_id, box))
    pred_by_image = {}
    for image_id, class_id, score, box in predictions:
        pred_by_image.setdefault(image_id, []).append((score, class_id, box))

    localised = correct = total_gt = 0
    per_class_hit = np.zeros(NUM_CLASSES)
    per_class_loc = np.zeros(NUM_CLASSES)
    per_class_tot = np.zeros(NUM_CLASSES)

    for image_id, gt_list in gt_by_image.items():
        gt_boxes = np.array([g[1] for g in gt_list], dtype=np.float32)
        gt_cls = [g[0] for g in gt_list]
        total_gt += len(gt_list)
        for c in gt_cls:
            per_class_tot[c] += 1

        entries = sorted(pred_by_image.get(image_id, []), key=lambda e: -e[0])
        if not entries:
            continue
        pred_boxes = np.array([e[2] for e in entries], dtype=np.float32)
        pred_cls = [e[1] for e in entries]

        ious = iou_matrix(gt_boxes, pred_boxes)
        used = set()
        for g_index in range(len(gt_list)):
            order = np.argsort(-ious[g_index])
            for p_index in order:
                if ious[g_index, p_index] < 0.5:
                    break
                if p_index in used:
                    continue
                used.add(int(p_index))
                localised += 1
                per_class_loc[gt_cls[g_index]] += 1
                if pred_cls[p_index] == gt_cls[g_index]:
                    correct += 1
                    per_class_hit[gt_cls[g_index]] += 1
                break

    # ---------------------------------------------------------------- report
    ms = {k: v / n_images * 1000 for k, v in timing.items()}
    pipeline_ms = ms["read"] + ms["detect"] + ms["crop"] + ms["classify"]
    gpu_ms = ms["detect"] + ms["crop"] + ms["classify"]

    print("\n" + "=" * 72)
    print("RESULTS")
    print("=" * 72)
    print(f"mode                       : {args.mode}")
    print(f"conf                       : {args.conf}")
    print(f"predictions                : {len(predictions)}")
    print(f"ground-truth objects       : {total_gt}")
    print()
    print(f"mAP@50                     : {metrics['map50']:.4f}")
    print(f"mAP@50-95                  : {metrics['map50_95']:.4f}")
    print()
    print(f"localised (IoU>=0.5)       : {localised}/{total_gt} = {localised / max(1, total_gt):.4f}")
    print(f"END-TO-END RECOGNITION ACC : {correct}/{total_gt} = {correct / max(1, total_gt):.4f}")
    print(f"class acc | localised      : {correct / max(1, localised):.4f}")

    print("\n" + "-" * 72)
    print("TIMING (per frame, 1920x1080, batch=1)")
    print("-" * 72)
    print(f"  image read (CPU/disk)    : {ms['read']:7.2f} ms")
    print(f"  Stage 1 detect           : {ms['detect']:7.2f} ms")
    print(f"  crop + preprocess        : {ms['crop']:7.2f} ms")
    print(f"  Stage 2 classify         : {ms['classify']:7.2f} ms")
    print(f"  {'-' * 40}")
    print(f"  pipeline total           : {pipeline_ms:7.2f} ms  ->  {1000 / pipeline_ms:.1f} FPS")
    print(f"  excluding disk read      : {gpu_ms:7.2f} ms  ->  {1000 / gpu_ms:.1f} FPS")
    print(f"  crops processed          : {n_crops_total} ({n_crops_total / n_images:.1f}/frame)")

    print("\n" + "-" * 72)
    print(f"{'class':<16}{'AP50':>9}{'AP50-95':>10}{'GT':>7}{'localised':>11}{'e2e acc':>10}")
    print("-" * 72)
    for i, name in enumerate(DET_CLASSES):
        if per_class_tot[i] == 0:
            continue
        print(f"{name:<16}{metrics['per_class_ap50'][i]:>9.4f}{metrics['per_class_ap'][i]:>10.4f}"
              f"{int(per_class_tot[i]):>7}{per_class_loc[i] / per_class_tot[i]:>11.4f}"
              f"{per_class_hit[i] / per_class_tot[i]:>10.4f}")

    # ---------------------------------------------------------------- save
    summary = {
        "mode": args.mode,
        "conf": args.conf,
        "imgsz": args.imgsz,
        "split": args.split,
        "tta": args.tta,
        "score": args.score,
        "temperature": args.temperature,
        "det_weights": str(args.det_weights),
        "cls_checkpoint": str(args.cls_checkpoint) if args.mode == "two-stage" else None,
        "images": n_images,
        "predictions": len(predictions),
        "ground_truth_objects": total_gt,
        "map50": metrics["map50"],
        "map50_95": metrics["map50_95"],
        "localised": localised,
        "localisation_rate": localised / max(1, total_gt),
        "end_to_end_accuracy": correct / max(1, total_gt),
        "class_accuracy_given_localised": correct / max(1, localised),
        "timing_ms_per_frame": {
            "read": ms["read"], "detect": ms["detect"], "crop": ms["crop"],
            "classify": ms["classify"], "pipeline_total": pipeline_ms,
            "excluding_read": gpu_ms,
        },
        "fps_pipeline": 1000 / pipeline_ms,
        "fps_excluding_read": 1000 / gpu_ms,
        "crops_per_frame": n_crops_total / n_images,
        "total_wall_seconds": total_wall,
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    with open(out_dir / "per_class.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "ap50", "ap50_95", "gt", "localised",
                         "localisation_rate", "end_to_end_accuracy"])
        for i, name in enumerate(DET_CLASSES):
            if per_class_tot[i] == 0:
                continue
            writer.writerow([
                name, f"{metrics['per_class_ap50'][i]:.6f}", f"{metrics['per_class_ap'][i]:.6f}",
                int(per_class_tot[i]), int(per_class_loc[i]),
                f"{per_class_loc[i] / per_class_tot[i]:.6f}",
                f"{per_class_hit[i] / per_class_tot[i]:.6f}",
            ])

    print(f"\nSaved -> {out_dir}")


if __name__ == "__main__":
    main()
