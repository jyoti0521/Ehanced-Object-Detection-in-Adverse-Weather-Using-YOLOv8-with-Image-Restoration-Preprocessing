from ultralytics import YOLO
import cv2
from restoration import remove_fog, remove_rain, remove_snow
import os

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # or yolov8s.pt for small version

# Choose your weather restoration
def preprocess(image_path):
    image = cv2.imread(image_path)
    
    # Choose weather type manually or auto-classify
    restored = remove_fog(image)       # or remove_rain / remove_snow
    return restored

def detect(image):
    results = model(image)
    results.save(filename="output.jpg")
    results.show()

if __name__ == "__main__":
    input_path = "sample_images/foggy_road.jpg"
    restored_img = preprocess(input_path)
    detect(restored_img)
