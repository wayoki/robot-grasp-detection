import csv
import time
import torch
from pathlib import Path
import onnxruntime as ort
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "configs" / "data.yaml"
TEST_IMAGES_DIR = ROOT_DIR / "dataset" / "test" / "images"
OUTPUT_PATH = ROOT_DIR / "outputs" / "benchmark.csv"

MODELS = {
    "PT FP32": ROOT_DIR / "models" / "yolo_model.pt",
    "ONNX FP32": ROOT_DIR / "models" / "yolo_model.onnx",
    "ONNX INT8": ROOT_DIR / "models" / "yolo_model_int8.onnx",
}


def get_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def get_test_images() -> list[Path]:
    images = []

    for pattern in ("*.jpg", "*.jpeg", "*.png"):
        images.extend(TEST_IMAGES_DIR.glob(pattern))

    images = sorted(images)

    if not images:
        raise FileNotFoundError(f"No test images found in {TEST_IMAGES_DIR}")

    return images


def measure_inference_time(model: YOLO, images: list[Path]) -> float:
    for image_path in images[:5]:
        model.predict(
            source=str(image_path),
            imgsz=640,
            conf=0.5,
            device=0,
            verbose=False,
        )

    times = []

    for image_path in images:
        torch.cuda.synchronize()

        start = time.perf_counter()

        model.predict(
            source=str(image_path),
            imgsz=640,
            conf=0.5,
            device=0,
            verbose=False,
        )

        torch.cuda.synchronize()

        end = time.perf_counter()
        times.append((end - start) * 1000)

    return sum(times) / len(times)


def validate_model(model: YOLO):
    metrics = model.val(
        data=str(DATA_PATH),
        split="test",
        imgsz=640,
        device=0,
        verbose=False,
    )

    return metrics.box.map50, metrics.box.map


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("CUDAExecutionProvider is not available")

    images = get_test_images()
    rows = []

    for model_name, model_path in MODELS.items():
        model = YOLO(str(model_path), task="detect")

        avg_ms = measure_inference_time(model, images)
        map50, map5095 = validate_model(model)

        rows.append({
            "model": model_name,
            "path": str(model_path.relative_to(ROOT_DIR)),
            "size_mb": round(get_size_mb(model_path), 2),
            "avg_inference_ms": round(avg_ms, 2),
            "fps": round(1000 / avg_ms, 2),
            "mAP50": round(float(map50), 4),
            "mAP50-95": round(float(map5095), 4),
        })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "model",
                "path",
                "size_mb",
                "avg_inference_ms",
                "fps",
                "mAP50",
                "mAP50-95",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
