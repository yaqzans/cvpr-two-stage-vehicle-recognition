"""11 - Build every paper/poster figure that can be made from saved results.

Reads only CSVs and images already on disk. No inference, no training.
Companion script 12_build_inference_figures.py covers the three figures that
need a forward pass (PR curves, reliability diagram, IoU histogram).

    python scripts/11_build_figures.py

Everything lands in figures/ as 300 dpi PNG plus PDF for LaTeX.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT_ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from _figure_manifest import write_manifest as _write_manifest


RUNS = PROJECT_ROOT / "runs"
S1 = RUNS / "stage1_yolo26n"
S2V1 = RUNS / "stage2_convnext_tiny"
S2V2 = RUNS / "stage2_convnext_tiny_v2"
PE = RUNS / "pipeline_eval"

DET_CLASSES = [
    "person", "rickshaw", "rickshaw_van", "auto_rickshaw", "truck",
    "pickup_truck", "private_car", "motorcycle", "bicycle", "bus",
    "micro_bus", "covered_van", "human_hauler",
]

# ---- house style -----------------------------------------------------------
TEAL = "#0f6f62"
TEAL_L = "#5fb3a5"
CLAY = "#a8562f"
OCHRE = "#b48a1e"
SLATE = "#40515c"
GREY = "#8d9a95"
INK = "#1b2321"

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 9.5,
    "axes.edgecolor": "#3a4744",
    "axes.linewidth": 0.9,
    "axes.grid": True,
    "grid.color": "#d8dedc",
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "legend.frameon": False,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "xtick.color": INK, "ytick.color": INK, "text.color": INK,
    "axes.labelcolor": INK, "axes.titlecolor": INK,
})

MANIFEST: list[tuple[str, str]] = []


def save(fig, name: str, description: str):
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{name}.{ext}")
    plt.close(fig)
    MANIFEST.append((name, description))
    print(f"  {name}")


def read_csv(path: Path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def prf_from_confusion(path: Path):
    """Derive per-class precision/recall/F1 from a saved confusion matrix.

    Rows are true classes, columns predicted. Deriving rather than reading a
    metrics file keeps v1 and v2 provably consistent.
    """
    rows = read_csv(path)
    classes = [c for c in rows[0].keys() if c != "true/pred"]
    matrix = np.array([[float(r[c]) for c in classes] for r in rows])
    tp = np.diag(matrix)
    support = matrix.sum(1)
    predicted = matrix.sum(0)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    denominator = precision + recall
    f1 = np.divide(2 * precision * recall, denominator,
                   out=np.zeros_like(tp), where=denominator > 0)
    return classes, precision, recall, f1, support, matrix


# ============================================================
# 01  dataset class distribution
# ============================================================
def fig_dataset_distribution():
    counts = {}
    for split in ("train", "val", "test"):
        counter = Counter()
        for label_file in (PROJECT_ROOT / "Datasets/rsud20k/labels" / split).glob("*.txt"):
            for line in label_file.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) == 5:
                    counter[int(parts[0])] += 1
        counts[split] = counter

    order = np.argsort([-counts["train"][i] for i in range(13)])
    names = [DET_CLASSES[i] for i in order]
    x = np.arange(13)
    width = 0.27

    fig, ax = plt.subplots(figsize=(9, 4.1))
    for offset, (split, colour) in zip((-width, 0, width),
                                       (("train", TEAL), ("val", SLATE), ("test", CLAY))):
        values = [counts[split][i] for i in order]
        ax.bar(x + offset, values, width, label=split, color=colour, edgecolor="none")

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=38, ha="right")
    ax.set_ylabel("annotated objects (log scale)")
    ax.set_title("RSUD20K class distribution — 72:1 imbalance between person and human_hauler")
    ax.legend(ncol=3, loc="upper right")
    ax.grid(axis="x", visible=False)

    top = counts["train"][order[0]]
    bottom = counts["train"][order[-1]]
    ax.annotate(f"{top:,}", (0 - width, top), textcoords="offset points",
                xytext=(0, 4), ha="center", fontsize=8, color=TEAL, fontweight="bold")
    ax.annotate(f"{bottom:,}", (12 - width, bottom), textcoords="offset points",
                xytext=(0, 4), ha="center", fontsize=8, color=TEAL, fontweight="bold")

    save(fig, "fig01_dataset_class_distribution",
         "Object counts per class for each split, log scale. Shows the 72:1 class "
         "imbalance (person 32,884 vs human_hauler 455) that motivates class-weighted "
         "loss in Stage 2 and makes macro-averaged metrics unreliable for rare classes.")


# ============================================================
# 02  object size distribution
# ============================================================
def fig_object_sizes():
    areas = defaultdict(list)
    for label_file in (PROJECT_ROOT / "Datasets/rsud20k/labels/train").glob("*.txt"):
        for line in label_file.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) == 5:
                areas[int(parts[0])].append(float(parts[3]) * float(parts[4]))

    medians = {i: np.median(areas[i]) * 100 for i in range(13)}
    order = sorted(range(13), key=lambda i: medians[i])
    data = [np.array(areas[i]) * 100 for i in order]

    fig, ax = plt.subplots(figsize=(9, 4.3))
    parts = ax.boxplot(data, vert=True, patch_artist=True, showfliers=False, widths=.6)
    for patch in parts["boxes"]:
        patch.set_facecolor(TEAL_L); patch.set_alpha(.65); patch.set_edgecolor(TEAL)
    for key in ("whiskers", "caps"):
        for line in parts[key]:
            line.set_color(TEAL)
    for line in parts["medians"]:
        line.set_color(CLAY); line.set_linewidth(1.8)

    ax.set_yscale("log")
    ax.set_xticklabels([DET_CLASSES[i] for i in order], rotation=38, ha="right")
    ax.set_ylabel("box area, % of image (log scale)")
    ax.set_title("Object scale by class (train split) — person and motorcycle are the smallest targets")
    ax.grid(axis="x", visible=False)

    save(fig, "fig02_object_size_distribution",
         "Box-area distribution per class as a percentage of image area, log scale. "
         "Explains why person/motorcycle/bicycle are hardest for the detector at 640 px "
         "(3x downscale from 1920x1080) and why val is harder than test.")


# ============================================================
# 03  Stage 1 training curves
# ============================================================
def fig_stage1_training():
    rows = read_csv(S1 / "train/results.csv")
    epoch = [int(r["epoch"]) for r in rows]

    def col(key):
        return [float(r[key]) for r in rows]

    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.2))

    ax = axes[0, 0]
    ax.plot(epoch, col("train/box_loss"), color=TEAL, label="train box")
    ax.plot(epoch, col("val/box_loss"), color=TEAL, ls="--", label="val box")
    ax.plot(epoch, col("train/cls_loss"), color=CLAY, label="train cls")
    ax.plot(epoch, col("val/cls_loss"), color=CLAY, ls="--", label="val cls")
    ax.set_title("Losses"); ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.legend(ncol=2)

    ax = axes[0, 1]
    ax.plot(epoch, col("metrics/mAP50(B)"), color=TEAL, lw=1.8, label="mAP@50")
    ax.plot(epoch, col("metrics/mAP50-95(B)"), color=SLATE, lw=1.8, label="mAP@50-95")
    best = int(np.argmax(col("metrics/mAP50-95(B)"))) + 1
    ax.axvline(best, color=CLAY, ls=":", lw=1.4)
    ax.annotate(f"best epoch {best}", (best, .33), rotation=90, fontsize=8,
                color=CLAY, ha="right", va="bottom")
    ax.set_title("Detection accuracy"); ax.set_xlabel("epoch"); ax.set_ylabel("mAP"); ax.legend()

    ax = axes[1, 0]
    ax.plot(epoch, col("metrics/precision(B)"), color=TEAL, label="precision")
    ax.plot(epoch, col("metrics/recall(B)"), color=CLAY, label="recall")
    ax.set_title("Precision and recall"); ax.set_xlabel("epoch"); ax.set_ylabel("value"); ax.legend()

    ax = axes[1, 1]
    ax.plot(epoch, col("lr/pg0"), color=SLATE)
    ax.set_title("Learning rate (linear decay, cos_lr=false)")
    ax.set_xlabel("epoch"); ax.set_ylabel("lr")

    fig.suptitle("Stage 1 — YOLO26n training, 64 epochs (patience 20 from epoch 44)",
                 fontsize=12, fontweight="bold", y=1.0)
    fig.tight_layout()

    save(fig, "fig03_stage1_training_curves",
         "Four-panel Stage-1 training summary: box/cls losses, mAP@50 and mAP@50-95, "
         "precision/recall, and the learning-rate schedule. Shows the hard plateau after "
         "epoch ~37 that triggered early stopping at 64.")


# ============================================================
# 04  Stage 1 per-class AP
# ============================================================
def fig_stage1_per_class():
    rows = read_csv(PE / "01_single_stage_baseline/per_class.csv")
    rows.sort(key=lambda r: -float(r["ap50"]))
    names = [r["class"] for r in rows]
    ap50 = [float(r["ap50"]) for r in rows]
    ap = [float(r["ap50_95"]) for r in rows]
    gt = [int(r["gt"]) for r in rows]

    x = np.arange(len(names)); width = .38
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.bar(x - width / 2, ap50, width, label="AP@50", color=TEAL)
    ax.bar(x + width / 2, ap, width, label="AP@50-95", color=SLATE)
    for i, n in enumerate(gt):
        ax.annotate(f"n={n}", (i, max(ap50[i], ap[i]) + .015), ha="center",
                    fontsize=6.8, color=GREY)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=38, ha="right")
    ax.set_ylabel("average precision"); ax.set_ylim(0, 1.05)
    ax.set_title("Stage 1 per-class detection performance (test split)")
    ax.legend(); ax.grid(axis="x", visible=False)

    save(fig, "fig04_stage1_per_class_ap",
         "Per-class AP@50 and AP@50-95 for the Stage-1 detector on the test split, "
         "sorted best to worst, with ground-truth counts annotated. truck and covered_van "
         "are the weakest classes; note their very small supports.")


# ============================================================
# 05  crop recall confidence sweep
# ============================================================
def fig_crop_recall_sweep():
    rows = read_csv(S1 / "crop_recall_eval/crop_recall_sweep_test.csv")
    conf = [float(r["conf"]) for r in rows]
    agnostic = [float(r["recall_agnostic"]) for r in rows]
    aware = [float(r["recall_class_aware"]) for r in rows]
    iou = [float(r["mean_iou"]) for r in rows]
    fp = [float(r["false_positives_per_image"]) for r in rows]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.9))

    ax.plot(conf, agnostic, "o-", color=TEAL, lw=2, ms=4.5, label="class-agnostic (pipeline)")
    ax.plot(conf, aware, "s--", color=CLAY, lw=1.6, ms=4, label="class-aware (original)")
    ax.plot(conf, iou, "^:", color=GREY, lw=1.3, ms=4, label="mean IoU of matches")
    ax.set_xscale("log"); ax.set_xlabel("detector confidence threshold")
    ax.set_ylabel("usable crop recall"); ax.set_ylim(.7, 1.02)
    ax.axvline(.05, color=OCHRE, ls=":", lw=1.4)
    ax.annotate("operating\npoint 0.05", (.05, .74), fontsize=7.5, color=OCHRE, ha="center")
    ax.set_title("Recall vs confidence"); ax.legend(loc="lower left")

    ax2.plot(conf, fp, "o-", color=SLATE, lw=2, ms=4.5)
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.set_xlabel("detector confidence threshold")
    ax2.set_ylabel("unmatched boxes per image")
    ax2.axvline(.05, color=OCHRE, ls=":", lw=1.4)
    ax2.set_title("Cost: unmatched boxes")

    fig.suptitle("Stage 1 usable crop recall — localisation ceiling is 99.1%",
                 fontsize=11.5, fontweight="bold")
    fig.tight_layout()

    save(fig, "fig05_stage1_crop_recall_sweep",
         "Left: usable crop recall vs detector confidence, class-agnostic (the correct "
         "metric for this architecture) against the original class-aware definition, with "
         "mean IoU flat at ~0.89-0.91 throughout. Right: unmatched boxes per image, the "
         "cost of lowering the threshold. Key figure for justifying conf=0.05.")


# ============================================================
# 06  per-class crop recall
# ============================================================
def fig_per_class_crop_recall():
    data = {}
    for conf in ("0.25", "0.1", "0.05"):
        for r in read_csv(S1 / f"crop_recall_eval/per_class_conf{conf}_test.csv"):
            data.setdefault(r["class"], {})[conf] = float(r["recall_agnostic"])

    names = sorted(data, key=lambda c: -data[c]["0.05"])
    x = np.arange(len(names)); width = .27
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for offset, (conf, colour, lab) in zip(
            (-width, 0, width),
            (("0.25", GREY, "conf 0.25"), ("0.1", TEAL_L, "conf 0.10"), ("0.05", TEAL, "conf 0.05"))):
        ax.bar(x + offset, [data[n][conf] for n in names], width, label=lab, color=colour)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=38, ha="right")
    ax.set_ylabel("class-agnostic crop recall"); ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color=GREY, lw=.8, ls=":")
    ax.set_title("Stage 1 per-class crop recall at three operating points (test split)")
    ax.legend(ncol=3); ax.grid(axis="x", visible=False)

    save(fig, "fig06_stage1_per_class_crop_recall",
         "Per-class usable crop recall at conf 0.25 / 0.10 / 0.05. Every class improves as "
         "the threshold falls except truck, which plateaus at 0.759 - the one genuine "
         "localisation failure in the system.")


# ============================================================
# 07  Stage 2 training curves, v1 vs v2
# ============================================================
def fig_stage2_training():
    v1 = read_csv(S2V1 / "history.csv")
    v2 = read_csv(S2V2 / "history.csv")

    def series(rows, key):
        return [int(r["epoch"]) for r in rows], [float(r[key]) for r in rows]

    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.2))

    ax = axes[0, 0]
    ax.plot(*series(v1, "val_macro_f1"), color=CLAY, lw=1.9, label="v1 val")
    ax.plot(*series(v2, "val_macro_f1"), color=TEAL, lw=1.9, label="v2 val")
    for rows, colour in ((v1, CLAY), (v2, TEAL)):
        e, f = series(rows, "val_macro_f1")
        best = int(np.argmax(f))
        ax.plot(e[best], f[best], "o", color=colour, ms=7, mfc="white", mew=1.8)
        ax.annotate(f"{f[best]:.4f}", (e[best], f[best]), textcoords="offset points",
                    xytext=(6, -10), fontsize=8, color=colour, fontweight="bold")
    ax.set_title("Validation macro F1  (checkpoint selection metric)")
    ax.set_xlabel("epoch"); ax.set_ylabel("macro F1"); ax.legend()

    ax = axes[0, 1]
    ax.plot(*series(v1, "train_accuracy"), color=CLAY, ls="--", lw=1.3, label="v1 train")
    ax.plot(*series(v1, "val_accuracy"), color=CLAY, lw=1.9, label="v1 val")
    ax.plot(*series(v2, "train_accuracy"), color=TEAL, ls="--", lw=1.3, label="v2 train")
    ax.plot(*series(v2, "val_accuracy"), color=TEAL, lw=1.9, label="v2 val")
    ax.set_title("Accuracy — the train/val gap is overfitting")
    ax.set_xlabel("epoch"); ax.set_ylabel("accuracy"); ax.legend(ncol=2, fontsize=7.5)

    ax = axes[1, 0]
    ax.plot(*series(v1, "train_loss"), color=CLAY, ls="--", lw=1.3, label="v1 train")
    ax.plot(*series(v1, "val_loss"), color=CLAY, lw=1.9, label="v1 val")
    ax.plot(*series(v2, "train_loss"), color=TEAL, ls="--", lw=1.3, label="v2 train")
    ax.plot(*series(v2, "val_loss"), color=TEAL, lw=1.9, label="v2 val")
    ax.set_title("Loss"); ax.set_xlabel("epoch"); ax.set_ylabel("loss")
    ax.legend(ncol=2, fontsize=7.5)

    ax = axes[1, 1]
    ax.plot(*series(v1, "learning_rate"), color=CLAY, lw=1.9, label="v1  T_max=100, stopped at 28")
    ax.plot(*series(v2, "learning_rate"), color=TEAL, lw=1.9, label="v2  warmup + T_max=30")
    ax.set_yscale("log")
    ax.set_title("Learning-rate schedule")
    ax.set_xlabel("epoch"); ax.set_ylabel("lr (log)"); ax.legend(fontsize=7.5)

    fig.suptitle("Stage 2 — ConvNeXt-Tiny, v1 vs v2", fontsize=12, fontweight="bold", y=1.0)
    fig.tight_layout()

    save(fig, "fig07_stage2_training_curves",
         "Stage-2 training, v1 against v2, four panels. Top-left is the key one: v2 exceeds "
         "v1's all-time best validation macro F1 (0.7912) on its first epoch and peaks at "
         "0.8552. Bottom-right shows why v1 underperformed - its cosine schedule barely "
         "decayed before early stopping.")


# ============================================================
# 08 / 09  confusion matrices
# ============================================================
def fig_confusion(version: str, path: Path, number: str):
    classes, _, _, _, _, matrix = prf_from_confusion(path)
    normalised = matrix / np.maximum(matrix.sum(1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(7.4, 6.4))
    im = ax.imshow(normalised, cmap="BuGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=7.6)
    ax.set_yticklabels(classes, fontsize=7.6)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    for i in range(len(classes)):
        for j in range(len(classes)):
            value = normalised[i, j]
            if value >= .005:
                ax.text(j, i, f"{value:.2f}".lstrip("0"), ha="center", va="center",
                        fontsize=6.4, color="white" if value > .55 else INK)
    ax.set_title(f"Stage 2 {version} — row-normalised confusion matrix (test)")
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=.78, label="fraction of true class")
    fig.tight_layout()

    save(fig, f"fig{number}_stage2_confusion_{version}",
         f"Row-normalised confusion matrix for Stage 2 {version} on the test split. "
         f"Diagonal is per-class recall. The truck row is the notable failure: much of it "
         f"leaks into covered_van and pickup_truck.")


# ============================================================
# 10  per-class F1, v1 vs v2
# ============================================================
def fig_stage2_per_class_f1():
    classes, _, _, f1_v1, support, _ = prf_from_confusion(S2V1 / "test_confusion_matrix.csv")
    _, _, _, f1_v2, _, _ = prf_from_confusion(S2V2 / "test_confusion_matrix.csv")

    order = np.argsort(f1_v2)
    names = [classes[i] for i in order]
    a = f1_v1[order]; b = f1_v2[order]; n = support[order]

    y = np.arange(len(names)); height = .38
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    ax.barh(y - height / 2, a, height, label="v1", color=CLAY)
    ax.barh(y + height / 2, b, height, label="v2", color=TEAL)
    for i in range(len(names)):
        delta = b[i] - a[i]
        colour = TEAL if delta >= 0 else CLAY
        ax.annotate(f"{delta:+.3f}", (max(a[i], b[i]) + .012, y[i]), va="center",
                    fontsize=7.4, color=colour, fontweight="bold")
        ax.annotate(f"n={int(n[i])}", (.012, y[i]), va="center", fontsize=6.8, color="white")
    ax.set_yticks(y); ax.set_yticklabels(names)
    ax.set_xlabel("F1 score"); ax.set_xlim(0, 1.14)
    ax.set_title("Stage 2 per-class F1 — v1 vs v2 (test split, ground-truth crops)")
    ax.legend(loc="lower right"); ax.grid(axis="y", visible=False)

    save(fig, "fig10_stage2_per_class_f1",
         "Per-class F1 for Stage 2 v1 against v2, sorted worst to best, with deltas and "
         "supports. 11 of 13 classes improved. The largest gains (bus, rickshaw_van, truck, "
         "covered_van) are exactly the wide/tall classes the v1 centre-crop was mutilating.")


# ============================================================
# 11  preprocessing comparison (the v1 defect, visually)
# ============================================================
def fig_preprocessing():
    from PIL import Image, ImageOps
    base = PROJECT_ROOT / "Datasets/BangladeshVehicleClassification/test"
    picks = [("person", "tall crop"), ("truck", "wide-ish crop"), ("rickshaw", "tall crop")]

    fig, axes = plt.subplots(3, 3, figsize=(7.2, 7.4))
    for row, (cls, note) in enumerate(picks):
        files = sorted((base / cls).glob("*.jpg"))
        image = Image.open(files[len(files) // 3]).convert("RGB")
        w, h = image.size

        short = 256
        scale = short / min(w, h)
        v1 = image.resize((max(1, int(w * scale)), max(1, int(h * scale))))
        left = (v1.width - 224) // 2; top = (v1.height - 224) // 2
        v1 = v1.crop((left, top, left + 224, top + 224))

        side = max(w, h)
        pad = ImageOps.expand(image, border=((side - w) // 2, (side - h) // 2,
                                             side - w - (side - w) // 2,
                                             side - h - (side - h) // 2), fill=(114, 114, 114))
        v2 = pad.resize((224, 224))

        kept = 224 / (256 * (max(w, h) / min(w, h)))
        for col, (img, title) in enumerate((
                (image, f"original\n{w}x{h} px"),
                (v1, f"v1: Resize+CenterCrop\nkeeps {kept*100:.0f}% of long side"),
                (v2, "v2: PadToSquare+Resize\nkeeps 100%"))):
            ax = axes[row, col]
            ax.imshow(img); ax.axis("off")
            if row == 0:
                ax.set_title(title, fontsize=8.6, pad=6)
            else:
                ax.set_title(title.split("\n")[-1] if col else f"{w}x{h} px", fontsize=7.6, pad=4)
        axes[row, 0].set_ylabel(cls)
        axes[row, 0].text(-.08, .5, cls, transform=axes[row, 0].transAxes,
                          rotation=90, va="center", ha="center", fontsize=9, fontweight="bold")

    fig.suptitle("The Stage-2 preprocessing defect and its fix", fontsize=12, fontweight="bold")
    fig.tight_layout()

    save(fig, "fig11_preprocessing_comparison",
         "Visual proof of the v1 preprocessing defect. Left: the original crop. Middle: what "
         "v1 actually fed the network - Resize(256)+CenterCrop(224) discards everything "
         "outside a centred square, removing the head and legs of a person and the front and "
         "rear of a vehicle. Right: v2 pads to square instead, keeping the whole object.")


# ============================================================
# 12  speed / accuracy trade-off
# ============================================================
def fig_speed_accuracy():
    runs = {}
    for metrics_path in PE.glob("*/metrics.json"):
        with open(metrics_path, encoding="utf-8") as f:
            runs[metrics_path.parent.name] = json.load(f)

    two = [("15_fastcrop_conf0001", 0.001), ("14_fastcrop_conf005", 0.05),
           ("13_fastcrop_conf010", 0.10), ("14_fastcrop_conf025", 0.25)]
    two = [(runs[t]["fps_pipeline"], runs[t]["end_to_end_accuracy"], c)
           for t, c in two if t in runs]
    single = [("01_single_stage_baseline", 0.001), ("26_test_single_conf005", 0.05),
              ("08_single_conf025", 0.25)]
    single = [(runs[t]["fps_pipeline"], runs[t]["end_to_end_accuracy"], c)
              for t, c in single if t in runs]

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for series, colour, label, marker in ((two, TEAL, "two-stage", "o"),
                                          (single, CLAY, "single-stage", "s")):
        xs = [p[0] for p in series]; ys = [p[1] for p in series]
        ax.plot(xs, ys, marker + "-", color=colour, lw=1.9, ms=7, label=label)
        for x, y, c in series:
            ax.annotate(f"conf {c:g}", (x, y), textcoords="offset points",
                        xytext=(6, -11), fontsize=7.2, color=colour)

    ax.axvspan(25, 31, color=OCHRE, alpha=.10)
    # Axes-fraction y so the label cannot fall outside the data range.
    ax.annotate("real-time\n25-30 FPS", (28, 0.055),
                xycoords=("data", "axes fraction"),
                ha="center", fontsize=7.8, color=OCHRE, fontweight="bold")
    ax.set_xlabel("frames per second  (1920x1080, batch 1, RTX 4060 Ti)")
    ax.set_ylabel("end-to-end recognition accuracy")
    ax.set_title("Speed / accuracy trade-off — the project's core engineering claim")
    ax.legend(loc="lower left")

    save(fig, "fig12_speed_accuracy_tradeoff",
         "End-to-end recognition accuracy against throughput for both systems across "
         "confidence thresholds, with the real-time band marked. Two-stage costs roughly 42% "
         "of throughput and buys accuracy; the gap between the curves widens as confidence "
         "falls. The single most useful figure for a poster.")


# ============================================================
# 13  single vs two-stage end-to-end
# ============================================================
def fig_single_vs_two():
    runs = {}
    for metrics_path in PE.glob("*/metrics.json"):
        with open(metrics_path, encoding="utf-8") as f:
            runs[metrics_path.parent.name] = json.load(f)

    groups = [("test\nconf 0.001", "01_single_stage_baseline", "02_two_stage_v2"),
              ("test\nconf 0.05", "26_test_single_conf005", "14_fastcrop_conf005"),
              ("val\nconf 0.001", "14_val_single", "15_val_uncalibrated"),
              ("val\nconf 0.05", "25_val_single_conf005", "24_val_op_conf005")]
    groups = [(lab, s, t) for lab, s, t in groups if s in runs and t in runs]

    labels = [g[0] for g in groups]
    single = [runs[g[1]]["end_to_end_accuracy"] for g in groups]
    two = [runs[g[2]]["end_to_end_accuracy"] for g in groups]

    x = np.arange(len(labels)); width = .34
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.bar(x - width / 2, single, width, label="single-stage (YOLO labels)", color=CLAY)
    ax.bar(x + width / 2, two, width, label="two-stage (ConvNeXt labels)", color=TEAL)
    for i in range(len(labels)):
        ax.annotate(f"{single[i]:.3f}", (x[i] - width / 2, single[i] + .006),
                    ha="center", fontsize=7.6, color=CLAY)
        ax.annotate(f"{two[i]:.3f}", (x[i] + width / 2, two[i] + .006),
                    ha="center", fontsize=7.6, color=TEAL, fontweight="bold")
        ax.annotate(f"+{two[i]-single[i]:.3f}", (x[i], max(single[i], two[i]) + .028),
                    ha="center", fontsize=8, color=INK, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("end-to-end recognition accuracy"); ax.set_ylim(.6, 1.03)
    ax.set_title("Two-stage vs single-stage — identical detector, identical boxes, different labeller")
    ax.legend(loc="lower right"); ax.grid(axis="x", visible=False)

    save(fig, "fig13_single_vs_two_stage",
         "End-to-end recognition accuracy, both systems, both splits, both operating points. "
         "The advantage is positive everywhere but shrinks at higher confidence: +4.3/+4.4 at "
         "conf 0.001 against +2.4/+1.4 at conf 0.05. Quote the operating point with the number.")


# ============================================================
# 14  mAP stability (the retraction)
# ============================================================
def fig_map_stability():
    rows = read_csv(PE / "METRIC_STABILITY.csv")
    summary = {}
    for split in ("val", "test"):
        for populated in ("True", "False"):
            deltas = [float(r["delta_ap50"]) for r in rows
                      if r["split"] == split and r["well_populated"] == populated]
            summary[(split, populated)] = np.mean(deltas)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.2),
                                  gridspec_kw={"width_ratios": [1, 1.35]})

    labels = ["well-populated\n(5 classes, >=250 obj)", "small\n(8 classes, 23-241 obj)"]
    val = [summary[("val", "True")], summary[("val", "False")]]
    test = [summary[("test", "True")], summary[("test", "False")]]
    x = np.arange(2); width = .34
    ax.bar(x - width / 2, val, width, label="val", color=SLATE)
    ax.bar(x + width / 2, test, width, label="test", color=TEAL)
    ax.axhline(0, color=INK, lw=.9)
    for i in range(2):
        ax.annotate(f"{val[i]:+.4f}", (x[i] - width / 2, val[i]),
                    ha="center", va="bottom" if val[i] > 0 else "top",
                    fontsize=7.6, color=SLATE)
        ax.annotate(f"{test[i]:+.4f}", (x[i] + width / 2, test[i]),
                    ha="center", va="bottom" if test[i] > 0 else "top",
                    fontsize=7.6, color=TEAL)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("mean AP@50 change (two-stage - single)")
    ax.set_title("Small classes flip sign between splits")
    ax.legend(); ax.grid(axis="x", visible=False)

    for split, colour, marker in (("val", SLATE, "s"), ("test", TEAL, "o")):
        subset = [r for r in rows if r["split"] == split]
        gt = [int(r["gt"]) for r in subset]
        delta = [float(r["delta_ap50"]) for r in subset]
        ax2.scatter(gt, delta, s=42, color=colour, marker=marker, label=split,
                    alpha=.85, edgecolor="white", linewidth=.6)
    ax2.axhline(0, color=INK, lw=.9)
    ax2.axvline(250, color=OCHRE, ls=":", lw=1.4)
    ax2.annotate("support = 250", (250, .19), rotation=90, fontsize=7.4,
                 color=OCHRE, ha="right")
    ax2.set_xscale("log")
    ax2.set_xlabel("ground-truth objects in class (log)")
    ax2.set_ylabel("AP@50 change")
    ax2.set_title("Variance is entirely a small-support effect")
    ax2.legend()

    fig.suptitle("Why the mAP comparison was retracted", fontsize=12, fontweight="bold")
    fig.tight_layout()

    save(fig, "fig14_map_metric_stability",
         "Evidence for the retracted mAP claim. Left: mean AP@50 change split by class "
         "support - well-populated classes barely move on either split, small classes swing "
         "hard and reverse sign. Right: the same per class against support, showing the "
         "variance collapses as support grows. An effect that flips sign between splits is "
         "sampling noise, not a real difference.")


# ============================================================
# 15  ablations
# ============================================================
def fig_ablations():
    runs = {}
    for metrics_path in PE.glob("*/metrics.json"):
        with open(metrics_path, encoding="utf-8") as f:
            runs[metrics_path.parent.name] = json.load(f)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.0))

    fusion = [("product\n(used)", "02_two_stage_v2"), ("det_conf\nonly", "17_test_scoredet"),
              ("cls_prob\nonly", "19_test_scorecls")]
    fusion = [(lab, runs[t]["map50"]) for lab, t in fusion if t in runs]
    colours = [TEAL] + [GREY] * (len(fusion) - 1)
    ax.bar([f[0] for f in fusion], [f[1] for f in fusion], color=colours, width=.55)
    for i, (_, v) in enumerate(fusion):
        ax.annotate(f"{v:.4f}", (i, v + .012), ha="center", fontsize=8, fontweight="bold")
    ax.set_ylabel("mAP@50 (test)"); ax.set_ylim(0, .88)
    ax.set_title("Detection-score fusion"); ax.grid(axis="x", visible=False)

    res = [("640\n(trained)", "13_fastcrop_conf010"), ("960", "10_imgsz960"),
           ("1280", "11_imgsz1280")]
    res = [(lab, runs[t]["map50"]) for lab, t in res if t in runs]
    ax2.bar([r[0] for r in res], [r[1] for r in res],
            color=[TEAL] + [GREY] * (len(res) - 1), width=.55)
    for i, (_, v) in enumerate(res):
        ax2.annotate(f"{v:.4f}", (i, v + .008), ha="center", fontsize=8, fontweight="bold")
    ax2.set_ylabel("mAP@50 (test)"); ax2.set_ylim(.55, .78)
    ax2.set_xlabel("inference image size (px)")
    ax2.set_title("Inference resolution"); ax2.grid(axis="x", visible=False)

    fig.suptitle("Negative results — both interventions make things worse",
                 fontsize=11.5, fontweight="bold")
    fig.tight_layout()

    save(fig, "fig15_ablations_score_and_resolution",
         "Two negative-result ablations. Left: the product of detector confidence and "
         "classifier probability beats either alone; classifier probability by itself "
         "collapses because it carries no objectness signal. Right: raising inference "
         "resolution above the 640 px the detector was trained at consistently hurts, so any "
         "resolution gain must come from training at the higher size.")


# ============================================================
# 16 / 17  qualitative grids
# ============================================================
def fig_qualitative(source: Path, name: str, title: str, description: str, count: int = 4):
    from PIL import Image
    files = sorted(source.glob("*.jpg"))[:count]
    if not files:
        print(f"  [skip] {name}: no images in {source}")
        return
    fig, axes = plt.subplots(len(files), 1, figsize=(10, 2.15 * len(files)))
    axes = np.atleast_1d(axes)
    for ax, path in zip(axes, files):
        ax.imshow(Image.open(path)); ax.axis("off")
        ax.set_title(path.stem, fontsize=7.5, color=GREY, loc="left", pad=2)
    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.0)
    fig.tight_layout()
    save(fig, name, description)


# ============================================================
# 18  architecture diagram
# ============================================================
def fig_architecture():
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 3.2); ax.axis("off")

    def box(x, y, w, h, title, subtitle, colour, fill):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                                    linewidth=1.6, edgecolor=colour, facecolor=fill))
        ax.text(x + w / 2, y + h * .64, title, ha="center", va="center",
                fontsize=10, fontweight="bold", color=INK)
        ax.text(x + w / 2, y + h * .28, subtitle, ha="center", va="center",
                fontsize=7.8, color=SLATE)

    def arrow(x1, x2, y, label):
        ax.add_patch(FancyArrowPatch((x1, y), (x2, y), arrowstyle="-|>",
                                     mutation_scale=13, linewidth=1.4, color=SLATE))
        ax.text((x1 + x2) / 2, y + .17, label, ha="center", fontsize=7.4, color=SLATE)

    box(0.1, 1.1, 1.85, 1.1, "road scene", "1920 x 1080", GREY, "#eef1f0")
    arrow(2.05, 2.75, 1.65, "")
    box(2.85, 1.1, 2.25, 1.1, "Stage 1", "YOLO26n  ·  640 px\n2.38 M params", TEAL, "#dceeea")
    arrow(5.2, 5.9, 1.65, "boxes only")
    box(6.0, 1.1, 1.5, 1.1, "crop", "full-res", GREY, "#eef1f0")
    arrow(7.6, 8.25, 1.65, "")
    box(8.35, 1.1, 1.55, 1.1, "Stage 2", "ConvNeXt-T\n224 px", CLAY, "#f7e6de")

    ax.text(4.0, .72, "class label discarded", ha="center", fontsize=7.6,
            color=CLAY, style="italic")
    ax.annotate("", xy=(4.0, 1.05), xytext=(4.0, .92),
                arrowprops=dict(arrowstyle="-", color=CLAY, lw=1.0, ls=":"))
    ax.text(9.12, .72, "final class", ha="center", fontsize=7.6,
            color=TEAL, fontweight="bold")

    ax.text(2.85, 2.42, "WHERE  +  how confident", fontsize=8.4, color=TEAL, fontweight="bold")
    ax.text(8.35, 2.42, "WHAT", fontsize=8.4, color=CLAY, fontweight="bold")

    ax.text(5.0, .15, "22.8 FPS end-to-end  ·  91.8% recognition accuracy (test, conf 0.05)",
            ha="center", fontsize=8.4, color=INK)

    save(fig, "fig16_architecture_diagram",
         "Two-stage pipeline architecture. Stage 1 answers where an object is and how "
         "confident the detector is; its class prediction is discarded. Stage 2 re-classifies "
         "each crop and makes the final decision. Redraw of the concept in slides 7-8 of the "
         "existing deck.")


# ============================================================
def main():
    print(f"Building figures -> {FIG_DIR}\n")
    fig_dataset_distribution()
    fig_object_sizes()
    fig_stage1_training()
    fig_stage1_per_class()
    fig_crop_recall_sweep()
    fig_per_class_crop_recall()
    fig_stage2_training()
    fig_confusion("v1", S2V1 / "test_confusion_matrix.csv", "08")
    fig_confusion("v2", S2V2 / "test_confusion_matrix.csv", "09")
    fig_stage2_per_class_f1()
    fig_preprocessing()
    fig_speed_accuracy()
    fig_single_vs_two()
    fig_map_stability()
    fig_ablations()
    fig_architecture()
    fig_qualitative(PE / "demo", "fig17_qualitative_results",
                    "Qualitative results — ground truth (left) vs two-stage pipeline (right)",
                    "Four test frames, ground truth on the left and the full pipeline output on "
                    "the right. Green = correct class, orange = localised but misclassified, "
                    "red = no matching labelled object. Many red boxes are real vehicles the "
                    "dataset never labelled - see fig18.")
    fig_qualitative(PE / "demo_errors", "fig18_failure_cases",
                    "Failure cases — frames containing at least one misclassification",
                    "Frames selected specifically because they contain a misclassification. "
                    "Use for the limitations section. Also shows the annotation-gap problem: "
                    "red 'unmatched' boxes are frequently correct detections of unlabelled "
                    "objects.")

    total = _write_manifest(FIG_DIR, MANIFEST)
    print(f"\n{len(MANIFEST)} figures written this run, {total} in the manifest")
    print(f"Manifest -> {FIG_DIR / 'FIGURES.txt'}")


def _wrap(text: str, width: int = 76):
    words = text.split()
    lines, current = [], ""
    for word in words:
        if len(current) + len(word) + 1 > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines


if __name__ == "__main__":
    main()
