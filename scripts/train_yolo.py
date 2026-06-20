import yaml
from ultralytics import YOLO
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parents[1] / "dataset" / "configs"

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

config_yolo = load_config(CONFIG_DIR / "yolo.yaml")

model = YOLO(config_yolo["model"])
model.train(
    data = str(config_yolo["data"]),
    imgsz = config_yolo["imgsz"],
    epochs = config_yolo["epochs"],
    batch = config_yolo["batch"],
    device = config_yolo["device"],
)

