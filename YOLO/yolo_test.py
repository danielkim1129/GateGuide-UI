from ultralytics import YOLO
from roboflow import Roboflow
import cv2

model = YOLO("airport_signage.v8-dataset-without-transformations.yolov8/")
results = model("test_image.jpg")

for result in results:
    boxes = result.boxes  # Bounding boxes
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]  # Extract coordinates
        conf = box.conf[0]  # Confidence
        cls = int(box.cls[0])  # Class index
        print(f"Detected class {cls} with confidence {conf}")
