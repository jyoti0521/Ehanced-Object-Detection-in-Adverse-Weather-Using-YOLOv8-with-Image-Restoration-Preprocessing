import cv2
import numpy as np
import torch
from ultralytics import YOLO
import os

# Load YOLOv8 model (choose yolov8n.pt for speed, or yolov8m.pt for better accuracy)
yolo_model = YOLO("yolov8n.pt")

# Load sample image
img_path = "test_images/sample_snow.jpg"
original = cv2.imread(img_path)

# Resize for consistency
image = cv2.resize(original, (512, 512))

# === Step 1: Preprocessing (Fake Desnow for now) ===
# NOTE: Replace with actual desnow model here
def fake_desnow(img):
    print("[INFO] Running fake desnow filter (blur)...")
    return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

restored = fake_desnow(image)

# Save or view restored
cv2.imwrite("results/restored.jpg", restored)

# === Step 2: YOLOv8 Detection ===
results = yolo_model(restored)

# Visualize detections
results[0].save(filename="results/detection.jpg")
print("[INFO] Detection completed and saved to results/detection.jpg")
