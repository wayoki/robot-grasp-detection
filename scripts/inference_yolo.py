import cv2
from ultralytics import YOLO
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
TEST_IMAGES_DIR = ROOT_DIR / "dataset" / "test" / "images"

model = YOLO(str(ROOT_DIR / "models" / "yolo_model.pt"))
images = list(TEST_IMAGES_DIR.glob("*.jpg"))

for image in images:
    results = model.predict(source = str(image), conf = 0.5, device = 0)
    for result in results:
        boxes = result.boxes
        img = cv2.imread(str(image))
        for index, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            roi = img[y1:y2, x1:x2]
            cv2.imwrite(f"{ROOT_DIR}/outputs/roi/{image.stem}_{index}.jpg", roi)

