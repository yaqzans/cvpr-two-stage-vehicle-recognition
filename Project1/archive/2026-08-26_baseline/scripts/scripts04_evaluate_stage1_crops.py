from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np
from collections import defaultdict

# =========================
# CONFIG
# =========================

# Project root = folder containing this script's parent folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "runs" / "stage1_yolo26n" / "train" / "weights" / "best.pt"



TEST_IMAGES = PROJECT_ROOT / "Datasets" / "rsud20k" / "images" / "test"
TEST_LABELS = PROJECT_ROOT / "Datasets" / "rsud20k" / "labels" / "test"

OUTPUT_DIR = PROJECT_ROOT / "runs" / "stage1_yolo26n" / "crop_evaluation"

CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.50

# =========================
# LOAD MODEL
# =========================

model = YOLO(MODEL_PATH)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# CLASS NAMES
# =========================

names = [
    "person",
    "rickshaw",
    "rickshaw_van",
    "auto_rickshaw",
    "truck",
    "pickup_truck",
    "private_car",
    "motorcycle",
    "bicycle",
    "bus",
    "micro_bus",
    "covered_van",
    "human_hauler"
]

# =========================
# HELPERS
# =========================

def box_iou(box1, box2):
    """
    Calculate IoU between two boxes.
    Boxes are [x1, y1, x2, y2].
    """

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])

    union = area1 + area2 - intersection

    if union == 0:
        return 0

    return intersection / union


def load_labels(label_path, img_width, img_height):
    """
    Read YOLO-format labels and convert them
    from normalized xywh to pixel xyxy.
    """

    labels = []

    if not label_path.exists():
        return labels

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) != 5:
                continue

            class_id = int(parts[0])

            x_center = float(parts[1]) * img_width
            y_center = float(parts[2]) * img_height
            width = float(parts[3]) * img_width
            height = float(parts[4]) * img_height

            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2

            labels.append({
                "class_id": class_id,
                "box": [x1, y1, x2, y2]
            })

    return labels


# =========================
# STATISTICS
# =========================

total_gt = 0
matched_gt = 0

ious = []

class_stats = defaultdict(lambda: {
    "gt": 0,
    "matched": 0,
    "ious": []
})

# =========================
# PROCESS TEST SET
# =========================

image_files = sorted(
    list(TEST_IMAGES.glob("*.jpg")) +
    list(TEST_IMAGES.glob("*.jpeg")) +
    list(TEST_IMAGES.glob("*.png"))
)

print(f"Found {len(image_files)} test images.")

for image_path in image_files:

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Could not read: {image_path}")
        continue

    height, width = image.shape[:2]

    label_path = TEST_LABELS / f"{image_path.stem}.txt"

    ground_truth = load_labels(
        label_path,
        width,
        height
    )

    if not ground_truth:
        continue

    # -------------------------
    # YOLO prediction
    # -------------------------

    result = model.predict(
        source=image,
        conf=CONF_THRESHOLD,
        verbose=False
    )[0]

    predictions = []

    if result.boxes is not None:

        for box in result.boxes:

            xyxy = box.xyxy[0].cpu().numpy()

            predictions.append({
                "class_id": int(box.cls[0].item()),
                "confidence": float(box.conf[0].item()),
                "box": xyxy.tolist()
            })

    # -------------------------
    # Match GT objects
    # -------------------------

    used_predictions = set()

    for gt in ground_truth:

        total_gt += 1

        class_id = gt["class_id"]

        class_stats[class_id]["gt"] += 1

        best_iou = 0
        best_prediction = None
        best_index = -1

        for i, pred in enumerate(predictions):

            if i in used_predictions:
                continue

            # Require same class
            if pred["class_id"] != class_id:
                continue

            iou = box_iou(
                gt["box"],
                pred["box"]
            )

            if iou > best_iou:
                best_iou = iou
                best_prediction = pred
                best_index = i

        # -------------------------
        # Successful crop
        # -------------------------

        if best_prediction is not None and best_iou >= IOU_THRESHOLD:

            matched_gt += 1

            used_predictions.add(best_index)

            ious.append(best_iou)

            class_stats[class_id]["matched"] += 1
            class_stats[class_id]["ious"].append(best_iou)

            # Save crop
            x1, y1, x2, y2 = map(
                int,
                best_prediction["box"]
            )

            # Keep coordinates inside image
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, x2)
            y2 = min(height, y2)

            crop = image[y1:y2, x1:x2]

            if crop.size > 0:

                class_name = names[class_id]

                class_dir = OUTPUT_DIR / class_name
                class_dir.mkdir(
                    parents=True,
                    exist_ok=True
                )

                crop_name = (
                    f"{image_path.stem}"
                    f"_gt_{class_id}"
                    f"_iou_{best_iou:.2f}.jpg"
                )

                cv2.imwrite(
                    str(class_dir / crop_name),
                    crop
                )


# =========================
# RESULTS
# =========================

print("\n" + "=" * 60)
print("STAGE 1 CROP QUALITY RESULTS")
print("=" * 60)

print(f"Total ground-truth objects : {total_gt}")
print(f"Matched objects            : {matched_gt}")

if total_gt > 0:

    crop_recall = matched_gt / total_gt

    print(
        f"Usable crop recall        : "
        f"{crop_recall:.4f} "
        f"({crop_recall * 100:.2f}%)"
    )

if ious:

    print(
        f"Average IoU of matched crops: "
        f"{np.mean(ious):.4f}"
    )

    print(
        f"Median IoU of matched crops : "
        f"{np.median(ious):.4f}"
    )

print("\nPER-CLASS RESULTS")
print("-" * 60)

for class_id in sorted(class_stats):

    stats = class_stats[class_id]

    gt = stats["gt"]
    matched = stats["matched"]

    recall = matched / gt if gt else 0

    avg_iou = (
        np.mean(stats["ious"])
        if stats["ious"]
        else 0
    )

    print(
        f"{names[class_id]:20s} "
        f"GT={gt:5d} "
        f"Matched={matched:5d} "
        f"Recall={recall * 100:6.2f}% "
        f"AvgIoU={avg_iou:.3f}"
    )

print("\nCrops saved to:")
print(OUTPUT_DIR)