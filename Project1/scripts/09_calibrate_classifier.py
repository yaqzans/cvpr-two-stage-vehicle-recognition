"""09 - Temperature scaling for the Stage-2 classifier.

The problem
-----------
Stage 2 is trained with label smoothing 0.1 AND class weights. Both distort
the output distribution: smoothing caps how confident the model may become,
weighting skews confidence by class frequency. The predicted class is fine -
argmax is invariant to both - but the *probability* is not calibrated.

That matters because the end-to-end detection score is

    score = detector_confidence x classifier_probability

so a squashed, badly-ordered classifier probability degrades the mAP ranking
even when the class is right.

The fix
-------
Temperature scaling (Guo et al., 2017): divide the logits by a single learned
scalar T before the softmax.

    T > 1  softens over-confident outputs
    T < 1  sharpens under-confident ones  (expected here, given smoothing)

One parameter. It cannot change any predicted class, so accuracy, macro F1
and the confusion matrix are all provably unchanged. It only re-scales the
confidence used for ranking.

Protocol
--------
T is fitted on the VALIDATION split and only ever applied to test. The crops
are real Stage-1 detections, not ground-truth boxes, because that is the
distribution the score is computed on at run time.

Nothing is overwritten: T is written to its own JSON file and the checkpoint
is not touched.

    python scripts/09_calibrate_classifier.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT / ".ultralytics"))

import cv2
import numpy as np
import timm
import torch
import torch.nn.functional as F
from ultralytics import YOLO

_spec = importlib.util.spec_from_file_location(
    "pipeline_eval", Path(__file__).with_name("06_evaluate_pipeline.py"))
_pipeline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pipeline)

DET_CLASSES = _pipeline.DET_CLASSES
iou_matrix = _pipeline.iou_matrix
fast_pad_resize = _pipeline.fast_pad_resize
normalize_batch = _pipeline.normalize_batch


def collect_logits(detector, classifier, cls_to_det, split, conf, imgsz, device, limit=None):
    """Run the real pipeline over `split` and return logits for every crop
    that matches a ground-truth object, paired with that object's class."""
    images_dir = PROJECT_ROOT / "Datasets" / "rsud20k" / "images" / split
    labels_dir = PROJECT_ROOT / "Datasets" / "rsud20k" / "labels" / split
    image_files = sorted(images_dir.glob("*.jpg"))
    if limit:
        image_files = image_files[:limit]

    all_logits, all_targets = [], []

    for index, image_path in enumerate(image_files, start=1):
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]

        gt_boxes, gt_cls = [], []
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            c = int(parts[0])
            cx, cy, bw, bh = (float(p) for p in parts[1:])
            cx, bw = cx * width, bw * width
            cy, bh = cy * height, bh * height
            gt_boxes.append([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2])
            gt_cls.append(c)
        if not gt_boxes:
            continue

        result = detector.predict(source=image, conf=conf, imgsz=imgsz,
                                  device=0 if device == "cuda" else "cpu",
                                  verbose=False)[0]
        if result.boxes is None or len(result.boxes) == 0:
            continue
        boxes = result.boxes.xyxy.cpu().numpy()

        crops, keep = [], []
        for b_index, box in enumerate(boxes):
            x1, y1 = max(0, int(box[0])), max(0, int(box[1]))
            x2, y2 = min(width, int(box[2])), min(height, int(box[3]))
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            crops.append(fast_pad_resize(image[y1:y2, x1:x2]))
            keep.append(b_index)
        if not crops:
            continue

        # Match detections to ground truth; only matched crops have a label.
        ious = iou_matrix(np.array(gt_boxes, np.float32), boxes[keep])
        pred_to_gt = {}
        if ious.size:
            order = np.argsort(ious, axis=None)[::-1]
            used_gt, used_pred = set(), set()
            for flat in order:
                g, p = np.unravel_index(flat, ious.shape)
                if ious[g, p] < 0.5:
                    break
                g, p = int(g), int(p)
                if g in used_gt or p in used_pred:
                    continue
                used_gt.add(g); used_pred.add(p)
                pred_to_gt[p] = g
        if not pred_to_gt:
            continue

        matched = sorted(pred_to_gt.keys())
        batch = normalize_batch(np.stack([crops[p] for p in matched]), device)
        if device == "cuda":
            batch = batch.to(memory_format=torch.channels_last)
        with torch.no_grad(), torch.autocast("cuda", torch.float16, enabled=device == "cuda"):
            logits = classifier(batch)
        all_logits.append(logits.float().cpu())
        all_targets.extend(gt_cls[pred_to_gt[p]] for p in matched)

        if index % 200 == 0:
            print(f"  {index}/{len(image_files)}  crops so far: {len(all_targets)}")

    return torch.cat(all_logits), torch.tensor(all_targets, dtype=torch.long)


def fit_temperature(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Minimise NLL over a single scalar with LBFGS."""
    log_t = torch.zeros(1, requires_grad=True)  # optimise log T, keeps T > 0
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_t.exp().item())


def metrics(logits: torch.Tensor, targets: torch.Tensor, temperature: float):
    scaled = logits / temperature
    probs = scaled.softmax(1)
    confidence, prediction = probs.max(1)
    correct = (prediction == targets).float()

    nll = float(F.cross_entropy(scaled, targets))
    accuracy = float(correct.mean())

    # Expected calibration error, 15 equal-width bins.
    ece = 0.0
    for lower in np.linspace(0, 1, 16)[:-1]:
        upper = lower + 1 / 15
        mask = (confidence > lower) & (confidence <= upper)
        if mask.sum() == 0:
            continue
        ece += float(mask.float().mean()) * abs(
            float(correct[mask].mean()) - float(confidence[mask].mean()))

    return {"nll": nll, "accuracy": accuracy, "ece": ece,
            "mean_confidence": float(confidence.mean())}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit Stage-2 temperature on val.")
    parser.add_argument("--det-weights", type=Path,
                        default=PROJECT_ROOT / "runs/stage1_yolo26n/train/weights/best.pt")
    parser.add_argument("--cls-checkpoint", type=Path,
                        default=PROJECT_ROOT / "runs/stage2_convnext_tiny_v2/checkpoints/best.pt")
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", type=Path,
                        default=PROJECT_ROOT / "runs/stage2_convnext_tiny_v2/calibration.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 72)
    print("TEMPERATURE CALIBRATION  (fitted on val, applied to test)")
    print("=" * 72)
    print(f"Detector   : {args.det_weights}")
    print(f"Classifier : {args.cls_checkpoint}")

    detector = YOLO(str(args.det_weights))
    checkpoint = torch.load(args.cls_checkpoint, map_location=device, weights_only=False)
    cls_classes = checkpoint["classes"]
    config = checkpoint.get("config", {})
    classifier = timm.create_model(
        config.get("model_name", "hf_hub:timm/convnext_tiny.in12k_ft_in1k"),
        pretrained=False, num_classes=len(cls_classes))
    classifier.load_state_dict(checkpoint["model_state_dict"])
    classifier.eval().to(device)
    if device == "cuda":
        classifier = classifier.to(memory_format=torch.channels_last)
    cls_to_det = np.array([DET_CLASSES.index(n) for n in cls_classes], dtype=int)

    # The classifier emits its own class order; convert its logits into the
    # detector index space so targets and logits agree.
    reorder = np.argsort(cls_to_det)

    print("\nCollecting validation logits from real Stage-1 crops...")
    logits, targets = collect_logits(detector, classifier, cls_to_det, "val",
                                     args.conf, args.imgsz, device, args.limit)
    logits = logits[:, reorder]
    print(f"Matched crops: {len(targets)}")

    before = metrics(logits, targets, 1.0)
    temperature = fit_temperature(logits, targets)
    after = metrics(logits, targets, temperature)

    print("\n" + "-" * 72)
    print(f"Fitted temperature T = {temperature:.4f}")
    print("-" * 72)
    print(f"{'':<20}{'T=1 (as trained)':>20}{'T fitted':>14}")
    for key, label in [("nll", "NLL"), ("ece", "calibration error"),
                       ("mean_confidence", "mean confidence"), ("accuracy", "accuracy")]:
        print(f"{label:<20}{before[key]:>20.4f}{after[key]:>14.4f}")
    print()
    if abs(before["accuracy"] - after["accuracy"]) < 1e-9:
        print("Accuracy identical, as expected - temperature cannot change argmax.")
    else:
        print("WARNING: accuracy changed. That should be impossible; investigate.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({
            "temperature": temperature,
            "fitted_on": "val",
            "det_weights": str(args.det_weights),
            "cls_checkpoint": str(args.cls_checkpoint),
            "conf": args.conf,
            "imgsz": args.imgsz,
            "matched_crops": int(len(targets)),
            "before": before,
            "after": after,
        }, f, indent=4)
    print(f"\nSaved -> {args.out}")
    print(f"\nApply with:  --temperature {temperature:.4f}")


if __name__ == "__main__":
    main()
