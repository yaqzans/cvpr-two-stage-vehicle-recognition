import cv2
import os

# ----------------------------
# Files
# ----------------------------
image_path = "train4.jpg"
label_path = "train4.txt"
classes_path = "classes.txt"

output_folder = "output"

# ----------------------------
# Read classes
# ----------------------------
with open(classes_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# ----------------------------
# Load image
# ----------------------------
image = cv2.imread(image_path)

height, width = image.shape[:2]

# ----------------------------
# Read labels
# ----------------------------
with open(label_path, "r") as f:
    labels = f.readlines()

# Counter for filenames
counter = {}

# ----------------------------
# Crop each object
# ----------------------------
for label in labels:

    class_id, x_center, y_center, box_w, box_h = map(float, label.split())

    class_id = int(class_id)
    class_name = classes[class_id]

    # Create class folder
    class_folder = os.path.join(output_folder, class_name)
    os.makedirs(class_folder, exist_ok=True)

    # Convert normalized YOLO coordinates to pixels
    x_center *= width
    y_center *= height
    box_w *= width
    box_h *= height

    x1 = int(x_center - box_w / 2)
    y1 = int(y_center - box_h / 2)
    x2 = int(x_center + box_w / 2)
    y2 = int(y_center + box_h / 2)

    # Prevent going outside image
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width, x2)
    y2 = min(height, y2)

    # Crop image
    crop = image[y1:y2, x1:x2]

    # Skip empty crops
    if crop.size == 0:
        continue

    # Filename counter
    counter[class_name] = counter.get(class_name, 0) + 1

    filename = f"{class_name.replace(' ', '_')}_{counter[class_name]}.jpg"

    save_path = os.path.join(class_folder, filename)

    cv2.imwrite(save_path, crop)

    print(f"Saved: {save_path}")

print("\nDone!")