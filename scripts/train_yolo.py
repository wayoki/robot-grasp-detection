import yaml
import torch
from ultralytics import YOLO
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

config_yolo = load_config(CONFIG_DIR / "yolo.yaml")
device = 0 if torch.cuda.is_available() else "cpu"

model = YOLO(config_yolo["model"])
model.train(
    data = str(config_yolo["data"]),
    imgsz = config_yolo["imgsz"],
    epochs = config_yolo["epochs"],
    batch = config_yolo["batch"],
    device = device,
)

