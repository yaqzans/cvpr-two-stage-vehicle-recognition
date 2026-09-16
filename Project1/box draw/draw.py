import cv2
import os

# ----------------------------
# File names
# ----------------------------
image_path = "train4.jpg"
label_path = "train4.txt"
classes_path = "classes.txt"

# ----------------------------
# Load class names
# ----------------------------
with open(classes_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# ----------------------------
# Load image
# ----------------------------
image = cv2.imread(image_path)
height, width = image.shape[:2]

# ----------------------------
# Read YOLO labels
# ----------------------------
with open(label_path, "r") as f:
    labels = f.readlines()

# ----------------------------
# Draw boxes
# ----------------------------
for label in labels:

    class_id, x_center, y_center, box_w, box_h = map(float, label.split())

    class_id = int(class_id)

    # Convert YOLO normalized coordinates to pixels
    x_center *= width
    y_center *= height
    box_w *= width
    box_h *= height

    x1 = int(x_center - box_w / 2)
    y1 = int(y_center - box_h / 2)
    x2 = int(x_center + box_w / 2)
    y2 = int(y_center + box_h / 2)

    # Draw rectangle
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Draw class name
    class_name = classes[class_id]

    cv2.putText(
        image,
        class_name,
        (x1, max(y1 - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

# ----------------------------
# Save result
# ----------------------------
cv2.imwrite("output.jpg", image)

print("Done! Saved as output.jpg")

# Optional: Show image
cv2.imshow("YOLO Bounding Boxes", image)
cv2.waitKey(0)
cv2.destroyAllWindows()