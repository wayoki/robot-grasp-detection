import shutil
import os
import random
from pathlib import Path

DATASET_DIR = Path.home() / "Documents" / "Repositories" / "robot-grasp-detection" / "data" / "processed" / "dataset"
VALID_IMAGES_DIR = DATASET_DIR / "valid" / "images"
VALID_LABELS_DIR = DATASET_DIR / "valid" / "labels"
TEST_IMAGES_DIR = DATASET_DIR / "test" / "images"
TEST_LABELS_DIR = DATASET_DIR / "test" / "labels"

if os.listdir(TEST_IMAGES_DIR) or os.listdir(TEST_LABELS_DIR):
    raise ValueError("Папки не пусты.")

images = sorted(list(VALID_IMAGES_DIR.glob("*.jpg")))

test_percent = 0.25
random.seed(0)
random.shuffle(images)

test_images = images[:(int(len(images) * test_percent))]

def move_files(test_images, TEST_IMAGES_DIR, TEST_LABELS_DIR):
    for image in test_images:
        label = VALID_LABELS_DIR / (image.stem + ".txt")
        if label.exists():
            shutil.move(str(image), str(TEST_IMAGES_DIR / image.name))
            shutil.move(str(label), str(TEST_LABELS_DIR / label.name))
        else:
            raise ValueError(f"Label для {image.name} не найдена.")

move_files(test_images, TEST_IMAGES_DIR, TEST_LABELS_DIR)
