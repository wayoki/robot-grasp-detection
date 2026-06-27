from ultralytics import YOLO
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "yolo_model.pt"
DATA_PATH = Path(__file__).resolve().parents[1] / "configs" / "data.yaml"

model = YOLO(str(MODEL_PATH))

# для удачного экспорта модели в onnx(int8 и fp32), нужно сначала экспортировать в int8(int8=True), после в fp32(int8=False). иначе модель int8 заменяет собой fp32.
model.export(format="onnx", imgsz=640, int8=False, data=str(DATA_PATH)) 

