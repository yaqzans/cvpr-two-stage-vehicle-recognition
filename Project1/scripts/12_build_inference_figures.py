"""12 - The three figures that need a forward pass.

No training. Runs the saved checkpoints over the saved data to recover
quantities that were never written to disk:

    fig19  Stage-2 precision-recall curves      (test ground-truth crops)
    fig20  Reliability diagram, before / after  (val, real Stage-1 crops)
    fig21  IoU distribution of matched boxes    (test, conf 0.05)

    python scripts/12_build_inference_figures.py
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT / ".ultralytics"))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
from PIL import Image
from ultralytics import YOLO

_spec = importlib.util.spec_from_file_location(
    "pipeline_eval", Path(__file__).with_name("06_evaluate_pipeline.py"))
_pipeline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pipeline)

DET_CLASSES = _pipeline.DET_CLASSES
iou_matrix = _pipeline.iou_matrix
fast_pad_resize = _pipeline.fast_pad_resize
normalize_batch = _pipeline.normalize_batch
build_eval_transform = _pipeline.build_eval_transform

# ---------------------------------------------------------------------------
# Minimal numpy replacements for sklearn.metrics.
#
# sklearn is deliberately NOT imported here: on this machine, importing
# ultralytics after cv2 + matplotlib + timm + torch + sklearn are all loaded
# segfaults the interpreter (an OpenMP runtime clash). These two functions are
# the only sklearn calls this script needed, so reimplementing them removes the
# dependency and the crash.
# ---------------------------------------------------------------------------

def precision_recall_curve(y_true, scores):
    """Same contract as sklearn: returns (precision, recall, thresholds)."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-scores, kind="mergesort")
    y_true, scores = y_true[order], scores[order]

    distinct = np.where(np.diff(scores))[0]
    idx = np.r_[distinct, y_true.size - 1]

    tp = np.cumsum(y_true)[idx]
    fp = np.cumsum(1 - y_true)[idx]
    n_pos = y_true.sum()

    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float),
                          where=(tp + fp) > 0)
    recall = tp / n_pos if n_pos else np.zeros_like(tp, dtype=float)

    # sklearn reverses to descending recall and appends the (P=1, R=0)
    # endpoint. It does not truncate, so neither do we: the flat tail at
    # recall=1 is real whenever many samples share the lowest score, and it
    # is exactly the part of the curve that shows the precision floor.
    return (np.r_[precision[::-1], 1.0],
            np.r_[recall[::-1], 0.0],
            scores[idx][::-1])


def average_precision_score(y_true, scores):
    """Step-wise AP, matching sklearn's sum((R_n - R_{n-1}) * P_n)."""
    precision, recall, _ = precision_recall_curve(y_true, scores)
    precision, recall = precision[::-1], recall[::-1]
    return float(np.sum(np.diff(recall) * precision[1:]))


FIG_DIR = PROJECT_ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from _figure_manifest import write_manifest as _write_manifest

S2V2 = PROJECT_ROOT / "runs/stage2_convnext_tiny_v2"
DET_WEIGHTS = PROJECT_ROOT / "runs/stage1_yolo26n/train/weights/best.pt"

TEAL = "#0f6f62"; TEAL_L = "#5fb3a5"; CLAY = "#a8562f"
OCHRE = "#b48a1e"; SLATE = "#40515c"; GREY = "#8d9a95"; INK = "#1b2321"

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 9.5,
    "axes.edgecolor": "#3a4744", "axes.linewidth": 0.9,
    "axes.grid": True, "grid.color": "#d8dedc", "grid.linewidth": 0.6,
    "axes.axisbelow": True, "legend.frameon": False, "legend.fontsize": 8,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "xtick.color": INK, "ytick.color": INK, "text.color": INK,
    "axes.labelcolor": INK, "axes.titlecolor": INK,
})

NEW_ENTRIES: list[tuple[str, str]] = []


def save(fig, name, description):
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{name}.{ext}")
    plt.close(fig)
    NEW_ENTRIES.append((name, description))
    print(f"  {name}")


def load_classifier(device):
    checkpoint = torch.load(S2V2 / "checkpoints/best.pt", map_location=device, weights_only=False)
    classes = checkpoint["classes"]
    config = checkpoint.get("config", {})
    model = timm.create_model(
        config.get("model_name", "hf_hub:timm/convnext_tiny.in12k_ft_in1k"),
        pretrained=False, num_classes=len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval().to(device)
    if device == "cuda":
        model = model.to(memory_format=torch.channels_last)
    return model, classes, build_eval_transform(config.get("preprocessing", "resize_centercrop"))


# ============================================================
# fig19  Stage-2 PR curves on ground-truth test crops
# ============================================================
def fig_pr_curves(device):
    print("\n[1/3] Stage-2 PR curves — test ground-truth crops")
    model, classes, transform = load_classifier(device)
    root = PROJECT_ROOT / "Datasets/BangladeshVehicleClassification/test"

    files, targets = [], []
    for index, name in enumerate(classes):
        for path in sorted((root / name).glob("*.jpg")):
            files.append(path); targets.append(index)
    targets = np.array(targets)
    print(f"      {len(files)} crops")

    probabilities = []
    for start in range(0, len(files), 64):
        batch = torch.stack([transform(Image.open(p).convert("RGB"))
                             for p in files[start:start + 64]]).to(device)
        if device == "cuda":
            batch = batch.to(memory_format=torch.channels_last)
        with torch.no_grad(), torch.autocast("cuda", torch.float16, enabled=device == "cuda"):
            logits = model(batch)
        probabilities.append(logits.float().softmax(1).cpu().numpy())
    probabilities = np.concatenate(probabilities)

    aps = {}
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    colours = plt.cm.viridis(np.linspace(0, .92, len(classes)))
    for index, name in enumerate(classes):
        binary = (targets == index).astype(int)
        if binary.sum() == 0:
            continue
        precision, recall, _ = precision_recall_curve(binary, probabilities[:, index])
        ap = average_precision_score(binary, probabilities[:, index])
        aps[name] = ap
        ax.plot(recall, precision, lw=1.5, color=colours[index],
                label=f"{name}  {ap:.3f}")

    macro = float(np.mean(list(aps.values())))
    ax.set_xlabel("recall"); ax.set_ylabel("precision")
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
    ax.set_title(f"Stage 2 per-class precision-recall (test, GT crops) — macro AP {macro:.3f}")
    ax.legend(ncol=2, loc="lower left", fontsize=7.2)
    save(fig, "fig19_stage2_pr_curves",
         "One-vs-rest precision-recall curve per class for the Stage-2 v2 classifier on "
         "ground-truth test crops, with average precision in the legend. truck sits clearly "
         "below every other class. Complements the confusion matrix by showing behaviour "
         f"across all decision thresholds rather than at argmax. Macro AP {macro:.3f}.")

    with open(FIG_DIR / "fig19_stage2_average_precision.json", "w", encoding="utf-8") as f:
        json.dump({"per_class_ap": aps, "macro_ap": macro}, f, indent=4)


# ============================================================
# fig20  reliability diagram on real Stage-1 crops
# ============================================================
def fig_reliability(device):
    print("\n[2/3] Reliability diagram — val, real Stage-1 crops")
    model, classes, _ = load_classifier(device)
    detector = YOLO(str(DET_WEIGHTS))
    cls_to_det = np.array([DET_CLASSES.index(n) for n in classes], dtype=int)
    reorder = np.argsort(cls_to_det)

    images_dir = PROJECT_ROOT / "Datasets/rsud20k/images/val"
    labels_dir = PROJECT_ROOT / "Datasets/rsud20k/labels/val"

    all_logits, all_targets = [], []
    files = sorted(images_dir.glob("*.jpg"))
    for index, image_path in enumerate(files, start=1):
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]

        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        gt_boxes, gt_cls = [], []
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            cx, cy, bw, bh = (float(p) for p in parts[1:])
            cx, bw = cx * width, bw * width
            cy, bh = cy * height, bh * height
            gt_boxes.append([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2])
            gt_cls.append(int(parts[0]))
        if not gt_boxes:
            continue

        result = detector.predict(source=image, conf=0.10, imgsz=640,
                                  device=0 if device == "cuda" else "cpu", verbose=False)[0]
        if result.boxes is None or len(result.boxes) == 0:
            continue
        boxes = result.boxes.xyxy.cpu().numpy()

        crops, keep = [], []
        for b_index, box in enumerate(boxes):
            x1, y1 = max(0, int(box[0])), max(0, int(box[1]))
            x2, y2 = min(width, int(box[2])), min(height, int(box[3]))
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            crops.append(fast_pad_resize(image[y1:y2, x1:x2])); keep.append(b_index)
        if not crops:
            continue

        ious = iou_matrix(np.array(gt_boxes, np.float32), boxes[keep])
        pred_to_gt = {}
        if ious.size:
            for flat in np.argsort(ious, axis=None)[::-1]:
                g, p = np.unravel_index(flat, ious.shape)
                if ious[g, p] < 0.5:
                    break
                g, p = int(g), int(p)
                if g in {v for v in pred_to_gt.values()} or p in pred_to_gt:
                    continue
                pred_to_gt[p] = g
        if not pred_to_gt:
            continue

        matched = sorted(pred_to_gt)
        batch = normalize_batch(np.stack([crops[p] for p in matched]), device)
        if device == "cuda":
            batch = batch.to(memory_format=torch.channels_last)
        with torch.no_grad(), torch.autocast("cuda", torch.float16, enabled=device == "cuda"):
            logits = model(batch)
        all_logits.append(logits.float().cpu())
        all_targets.extend(gt_cls[pred_to_gt[p]] for p in matched)

        if index % 300 == 0:
            print(f"      {index}/{len(files)}")

    logits = torch.cat(all_logits)[:, reorder]
    targets = torch.tensor(all_targets, dtype=torch.long)
    print(f"      {len(targets)} matched crops")

    with open(S2V2 / "calibration.json", encoding="utf-8") as f:
        temperature = json.load(f)["temperature"]

    def bins(t):
        probabilities = (logits / t).softmax(1)
        confidence, prediction = probabilities.max(1)
        correct = (prediction == targets).float().numpy()
        confidence = confidence.numpy()
        edges = np.linspace(0, 1, 16)
        centres, accuracies, weights = [], [], []
        for lower, upper in zip(edges[:-1], edges[1:]):
            mask = (confidence > lower) & (confidence <= upper)
            # Skip only empty bins, matching 09_calibrate_classifier.py, so the
            # ECE printed here is identical to the one in calibration.json.
            if mask.sum() == 0:
                continue
            # Plot and score at the MEAN confidence inside the bin, not the bin
            # centre. This is the standard ECE definition and is what
            # 09_calibrate_classifier.py uses, so the two agree exactly.
            centres.append(confidence[mask].mean())
            accuracies.append(correct[mask].mean())
            weights.append(mask.mean())
        ece = sum(w * abs(a - c) for c, a, w in zip(centres, accuracies, weights))
        return np.array(centres), np.array(accuracies), np.array(weights), ece, confidence

    c1, a1, w1, ece1, conf1 = bins(1.0)
    c2, a2, w2, ece2, conf2 = bins(temperature)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.3),
                                  gridspec_kw={"width_ratios": [1.15, 1]})

    ax.plot([0, 1], [0, 1], ls="--", color=GREY, lw=1.2, label="perfect calibration")
    ax.plot(c1, a1, "s-", color=CLAY, lw=1.8, ms=5, label=f"T = 1.0   ECE {ece1:.4f}")
    ax.plot(c2, a2, "o-", color=TEAL, lw=1.8, ms=5,
            label=f"T = {temperature:.3f}   ECE {ece2:.4f}")
    ax.set_xlabel("predicted confidence"); ax.set_ylabel("observed accuracy")
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
    ax.set_title("Reliability diagram — the uncalibrated curve sits above the diagonal")
    ax.legend(loc="upper left")

    ax2.hist(conf1, bins=30, range=(0, 1), color=CLAY, alpha=.62, label="T = 1.0")
    ax2.hist(conf2, bins=30, range=(0, 1), color=TEAL, alpha=.62,
             label=f"T = {temperature:.3f}")
    ax2.set_xlabel("predicted confidence"); ax2.set_ylabel("crops")
    ax2.set_title("Confidence distribution"); ax2.legend()

    fig.suptitle("Stage-2 calibration on real detector crops (val split)",
                 fontsize=11.5, fontweight="bold")
    fig.tight_layout()

    save(fig, "fig20_reliability_diagram",
         "Reliability diagram for the Stage-2 classifier on real Stage-1 crops from val, "
         "before and after temperature scaling. The uncalibrated curve sits above the "
         f"diagonal - label smoothing left the model under-confident. Fitting T={temperature:.4f} "
         f"cuts expected calibration error from {ece1:.4f} to {ece2:.4f} without changing any "
         "prediction. Right panel shows the confidence distribution shifting toward 1.0.")


# ============================================================
# fig21  IoU distribution of matched boxes
# ============================================================
def fig_iou_distribution(device):
    print("\n[3/3] IoU distribution — test, conf 0.05")
    detector = YOLO(str(DET_WEIGHTS))
    images_dir = PROJECT_ROOT / "Datasets/rsud20k/images/test"
    labels_dir = PROJECT_ROOT / "Datasets/rsud20k/labels/test"

    per_class = {i: [] for i in range(len(DET_CLASSES))}
    every = []

    for image_path in sorted(images_dir.glob("*.jpg")):
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        gt_boxes, gt_cls = [], []
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            cx, cy, bw, bh = (float(p) for p in parts[1:])
            cx, bw = cx * width, bw * width
            cy, bh = cy * height, bh * height
            gt_boxes.append([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2])
            gt_cls.append(int(parts[0]))
        if not gt_boxes:
            continue

        result = detector.predict(source=image, conf=0.05, imgsz=640,
                                  device=0 if device == "cuda" else "cpu", verbose=False)[0]
        if result.boxes is None or len(result.boxes) == 0:
            continue
        boxes = result.boxes.xyxy.cpu().numpy()

        ious = iou_matrix(np.array(gt_boxes, np.float32), boxes)
        used_gt, used_pred = set(), set()
        for flat in np.argsort(ious, axis=None)[::-1]:
            g, p = np.unravel_index(flat, ious.shape)
            if ious[g, p] < 0.5:
                break
            g, p = int(g), int(p)
            if g in used_gt or p in used_pred:
                continue
            used_gt.add(g); used_pred.add(p)
            every.append(float(ious[g, p]))
            per_class[gt_cls[g]].append(float(ious[g, p]))

    every = np.array(every)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.2),
                                  gridspec_kw={"width_ratios": [1, 1.2]})

    ax.hist(every, bins=40, range=(0.5, 1.0), color=TEAL, alpha=.85)
    ax.axvline(every.mean(), color=CLAY, lw=1.6,
               label=f"mean {every.mean():.3f}")
    ax.axvline(np.median(every), color=OCHRE, lw=1.6, ls="--",
               label=f"median {np.median(every):.3f}")
    ax.set_xlabel("IoU with ground truth"); ax.set_ylabel("matched boxes")
    ax.set_title(f"Localisation quality, {len(every):,} matched boxes")
    ax.legend()

    order = sorted(range(len(DET_CLASSES)),
                   key=lambda i: np.median(per_class[i]) if per_class[i] else 0)
    data = [per_class[i] for i in order if per_class[i]]
    names = [DET_CLASSES[i] for i in order if per_class[i]]
    parts = ax2.boxplot(data, vert=False, patch_artist=True, showfliers=False, widths=.62)
    for patch in parts["boxes"]:
        patch.set_facecolor(TEAL_L); patch.set_alpha(.7); patch.set_edgecolor(TEAL)
    for key in ("whiskers", "caps"):
        for line in parts[key]:
            line.set_color(TEAL)
    for line in parts["medians"]:
        line.set_color(CLAY); line.set_linewidth(1.7)
    ax2.set_yticklabels(names, fontsize=8)
    ax2.set_xlabel("IoU with ground truth")
    ax2.set_title("By class")
    ax2.grid(axis="y", visible=False)

    fig.suptitle("Stage-1 box quality when a detection is made (test, conf 0.05)",
                 fontsize=11.5, fontweight="bold")
    fig.tight_layout()

    save(fig, "fig21_iou_distribution",
         "Distribution of IoU between matched Stage-1 boxes and ground truth at conf 0.05. "
         f"Mean {every.mean():.3f}, median {np.median(every):.3f} over {len(every):,} matched "
         "boxes. Supports the central Stage-1 finding: when the detector finds an object its "
         "box is excellent, so the weakness is recall and confidence ranking, not "
         "localisation. rickshaw_van and person have the loosest boxes.")

    np.save(FIG_DIR / "fig21_matched_ious.npy", every)


def _wrap(text: str, width: int = 76):
    words = text.split(); lines, current = [], ""
    for word in words:
        if len(current) + len(word) + 1 > width:
            lines.append(current); current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    fig_pr_curves(device)
    fig_reliability(device)
    fig_iou_distribution(device)

    total = _write_manifest(FIG_DIR, NEW_ENTRIES)
    print(f"\n{len(NEW_ENTRIES)} figures written this run, {total} in the manifest")


if __name__ == "__main__":
    main()
