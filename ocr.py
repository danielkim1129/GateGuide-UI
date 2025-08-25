import cv2
import easyocr
from ultralytics import YOLO
import numpy as np

def sign_bounding(boxes, frame_width, frame_height, left_expansion=200, right_expansion=400, top_expansion=100, bottom_expansion=300):
    if not boxes:
        return []

    x1 = min(box[0] for box in boxes)
    y1 = min(box[1] for box in boxes)
    x2 = max(box[2] for box in boxes)
    y2 = max(box[3] for box in boxes)

    expanded_x1 = max(0, x1 - left_expansion)
    expanded_y1 = max(0, y1 - top_expansion)
    expanded_x2 = min(frame_width, x2 + right_expansion)
    expanded_y2 = min(frame_height, y2 + bottom_expansion)

    return [(expanded_x1, expanded_y1, expanded_x2, expanded_y2)]


model = YOLO("best.pt")

image_path = "image6.jpg"
image = cv2.imread(image_path)


print("Stuff detected in the green:")
detected_boxes = []

results = model(image)
for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = box.conf[0].item()
        if confidence > 0.3:
            detected_boxes.append((x1, y1, x2, y2))
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2) # green box

merged_regions = sign_bounding(detected_boxes, 1920, 1080)
for x1, y1, x2, y2 in merged_regions:
    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 2)  # purple box




grouped_texts = []


horizontal_threshold = 250  # Adjust for horizontal proximity
vertical_threshold = 50  # Allow slight vertical alignment


for group in grouped_texts:
    x1s, y1s, x2s, y2s = [], [], [], []
    for _, (x1, y1, x2, y2), _ in group:
        x1s.append(x1)
        y1s.append(y1)
        x2s.append(x2)
        y2s.append(y2)

    gx1, gy1, gx2, gy2 = int(min(x1s)), int(min(y1s)), int(max(x2s)), int(max(y2s))
    cv2.rectangle(image, (gx1, gy1), (gx2, gy2), (0, 0, 255), 2)  # Red box for grouped text

    grouped_text = " ".join([word[0] for word in group])
    print(f"Grouped Text: {grouped_text}")



cv2.imshow("Detected Texts", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
