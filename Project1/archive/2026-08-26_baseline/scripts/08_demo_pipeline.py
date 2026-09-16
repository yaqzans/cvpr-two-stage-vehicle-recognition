"""08 - Qualitative demo of the two-stage pipeline.

Draws the final pipeline output on real test images, side by side with the
ground truth, so the system can be inspected visually rather than only
through metrics. Useful for the presentation and for spotting failure modes
the aggregate numbers hide.

Colour key on the prediction panel:
    green   correct class, well localised
    orange  localised but classified as the wrong class
    red     no matching ground-truth object (a false positive, or an object
            the dataset never labelled - both are common here)

    python scripts/08_demo_pipeline.py --count 24
    python scripts/08_demo_pipeline.py --count 12 --only-errors
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT / ".ultralytics"))

import cv2
import numpy as np
import timm
import torch
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "pipeline_eval", Path(__file__).with_name("06_evaluate_pipeline.py"))
_pipeline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pipeline)

DET_CLASSES = _pipeline.DET_CLASSES
build_eval_transform = _pipeline.build_eval_transform
iou_matrix = _pipeline.iou_matrix

GREEN = (80, 220, 100)
ORANGE = (0, 165, 255)
RED = (60, 60, 235)
BLUE = (230, 180, 80)


def draw_box(image, box, label, colour, thickness=2):
    x1, y1, x2, y2 = (int(v) for v in box)
    cv2.rectangle(image, (x1, y1), (x2, y2), colour, thickness)
    if not label:
        return
    scale = 0.5
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    ty = max(y1, th + 4)
    cv2.rectangle(image, (x1, ty - th - 4), (x1 + tw + 4, ty + 2), colour, -1)
    cv2.putText(image, label, (x1 + 2, ty - 2),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (20, 20, 20), 1, cv2.LINE_AA)


def banner(image, text, colour=(35, 35, 35)):
    height = 34
    strip = np.full((height, image.shape[1], 3), colour, np.uint8)
    cv2.putText(strip, text, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (240, 240, 240), 1, cv2.LINE_AA)
    return np.vstack([strip, image])


def main() -> None:
    parser = argparse.ArgumentParser(description="Qualitative two-stage pipeline demo.")
    parser.add_argument("--det-weights", type=Path,
                        default=PROJECT_ROOT / "runs/stage1_yolo26n/train/weights/best.pt")
    parser.add_argument("--cls-checkpoint", type=Path,
                        default=PROJECT_ROOT / "runs/stage2_convnext_tiny_v2/checkpoints/best.pt")
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--only-errors", action="store_true",
                        help="Only save frames containing a misclassification.")
    parser.add_argument("--out", type=Path,
                        default=PROJECT_ROOT / "runs" / "pipeline_eval" / "demo")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)

    detector = YOLO(str(args.det_weights))
    checkpoint = torch.load(args.cls_checkpoint, map_location=device, weights_only=False)
    cls_classes = checkpoint["classes"]
    config = checkpoint.get("config", {})
    classifier = timm.create_model(
        config.get("model_name", "hf_hub:timm/convnext_tiny.in12k_ft_in1k"),
        pretrained=False, num_classes=len(cls_classes))
    classifier.load_state_dict(checkpoint["model_state_dict"])
    classifier.eval().to(device)
    transform = build_eval_transform(config.get("preprocessing", "resize_centercrop"))
    cls_to_det = np.array([DET_CLASSES.index(n) for n in cls_classes], dtype=int)

    images_dir = PROJECT_ROOT / "Datasets" / "rsud20k" / "images" / "test"
    labels_dir = PROJECT_ROOT / "Datasets" / "rsud20k" / "labels" / "test"
    image_files = sorted(images_dir.glob("*.jpg"))

    saved = 0
    for image_path in image_files:
        if saved >= args.count:
            break
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]

        gt_boxes, gt_cls = [], []
        label_path = labels_dir / f"{image_path.stem}.txt"
        if label_path.exists():
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

        result = detector.predict(source=image, conf=args.conf, imgsz=args.imgsz,
                                  device=0 if device == "cuda" else "cpu", verbose=False)[0]
        if result.boxes is None or len(result.boxes) == 0:
            continue
        boxes = result.boxes.xyxy.cpu().numpy()
        det_conf = result.boxes.conf.cpu().numpy()

        tensors, keep = [], []
        for i, box in enumerate(boxes):
            x1, y1 = max(0, int(box[0])), max(0, int(box[1]))
            x2, y2 = min(width, int(box[2])), min(height, int(box[3]))
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            pil = Image.fromarray(cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))
            tensors.append(transform(pil))
            keep.append(i)
        if not tensors:
            continue

        with torch.no_grad(), torch.autocast("cuda", torch.float16, enabled=device == "cuda"):
            logits = classifier(torch.stack(tensors).to(device))
        probs = logits.float().softmax(1).cpu().numpy()
        final_cls = cls_to_det[probs.argmax(1)]
        final_score = probs.max(1)

        pred_boxes = boxes[keep]
        ious = iou_matrix(np.array(gt_boxes, np.float32), pred_boxes)

        # Assign each prediction to its best unclaimed ground-truth object.
        assignment = {}
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
                assignment[p] = g

        errors = sum(1 for p, g in assignment.items() if final_cls[p] != gt_cls[g])
        if args.only_errors and errors == 0:
            continue

        truth_panel = image.copy()
        for box, c in zip(gt_boxes, gt_cls):
            draw_box(truth_panel, box, DET_CLASSES[c], BLUE)

        pred_panel = image.copy()
        for p in range(len(pred_boxes)):
            name = DET_CLASSES[final_cls[p]]
            if p not in assignment:
                colour, label = RED, f"{name} {final_score[p]:.2f} (unmatched)"
            elif final_cls[p] == gt_cls[assignment[p]]:
                colour, label = GREEN, f"{name} {final_score[p]:.2f}"
            else:
                colour = ORANGE
                label = f"{name} != {DET_CLASSES[gt_cls[assignment[p]]]}"
            draw_box(pred_panel, pred_boxes[p], label, colour)

        truth_panel = banner(truth_panel, f"GROUND TRUTH  -  {len(gt_boxes)} labelled objects")
        pred_panel = banner(
            pred_panel,
            f"TWO-STAGE PIPELINE  -  {len(assignment)}/{len(gt_boxes)} localised, "
            f"{errors} misclassified, {len(pred_boxes) - len(assignment)} unmatched")

        separator = np.full((truth_panel.shape[0], 6, 3), 200, np.uint8)
        combined = np.hstack([truth_panel, separator, pred_panel])
        scale = 1600 / combined.shape[1]
        combined = cv2.resize(combined, (1600, int(combined.shape[0] * scale)))

        cv2.imwrite(str(args.out / f"{image_path.stem}.jpg"), combined,
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
        saved += 1

    print(f"Saved {saved} demo images -> {args.out}")


if __name__ == "__main__":
    main()
