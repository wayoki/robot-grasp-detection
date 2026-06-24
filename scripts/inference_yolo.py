import cv2
import torch
from ultralytics import YOLO
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
TEST_IMAGES_DIR = ROOT_DIR / "dataset" / "test" / "images"

device = 0 if torch.cuda.is_available() else "cpu"
model = YOLO(str(ROOT_DIR / "models" / "yolo_model.pt"))
images = list(TEST_IMAGES_DIR.glob("*.jpg"))

for image in images:
    results = model.predict(source = str(image), conf = 0.5, device = device)
    for result in results:
        boxes = result.boxes
        img = cv2.imread(str(image))
        for index, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            h, w, _ = img.shape
            pad = int(max(x2 - x1, y2 - y1) * 0.08)
            pad = max(10, min(pad, 30))
            roi = img[max(0, y1 - pad):min(h, y2 + pad), max(0, x1 - pad):min(w, x2 + pad)]
            if class_name == "cell":
                cv2.imwrite(str(ROOT_DIR / "outputs" / "roi" / "cell" /  f"{image.stem}_{index}.jpg"), roi)
            elif class_name == "sensor":
                cv2.imwrite(str(ROOT_DIR / "outputs" / "roi" / "sensor" / f"{image.stem}_{index}.jpg"), roi)
            else:
                print("Неизвестный класс")

