import cv2
from pathlib import Path
import numpy as np

def show_image(image, title = None):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyWindow(title)

ROOT_DIR = Path(__file__).resolve().parents[1]
ROI_DIR = ROOT_DIR / "outputs" / "roi"

cell = list((ROI_DIR / "cell").glob("*.jpg"))
sensor = list((ROI_DIR / "sensor").glob("*.jpg"))

for image in cell:
    image = cv2.imread(str(image))
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB) 
    l, a, b = cv2.split(image_lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    clahe_image = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    image_gray = l
    gauss_filter = [(7, 7)]
    gauss_blur = cv2.GaussianBlur(image_gray, gauss_filter[0], 0)
    image_bin = cv2.adaptiveThreshold(gauss_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 3)
    kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    image_morph = cv2.morphologyEx(image_bin, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
    image_morph = cv2.morphologyEx(image_morph, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8))
    image_morph = cv2.morphologyEx(image_morph, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
    cont, _ = cv2.findContours(image_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    conts = []
    roi_h, roi_w = image_morph.shape
    for c in cont:
        s = cv2.contourArea(c)
        if s >= roi_h * roi_w * 0.05:
            conts.append(c)
    best_cont = max(conts, key=cv2.contourArea)
    image_cont = cv2.drawContours(image.copy(), [best_cont], -1, (0, 255, 0), 2)
    show_image(np.hstack((image_cont, image)))

for image in sensor:
    image = cv2.imread(str(image))
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB) 
    l, a, b = cv2.split(image_lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    clahe_image = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    image_gray = l
    gauss_filter = [(15, 15)]
    gauss_blur = cv2.GaussianBlur(image_gray, gauss_filter[0], 0)
    _, image_bin = cv2.threshold(gauss_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel_ellipse_3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_ellipse_5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    image_morph = cv2.morphologyEx(image_bin, cv2.MORPH_OPEN, kernel_ellipse_3)
    image_morph = cv2.morphologyEx(image_morph, cv2.MORPH_CLOSE, kernel_ellipse_5)
    image_morph = cv2.morphologyEx(image_morph, cv2.MORPH_OPEN, kernel_ellipse_3)
    cont, _ = cv2.findContours(image_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    conts = []
    roi_h, roi_w = image_morph.shape
    for c in cont:
        s = cv2.contourArea(c)
        if s >= roi_h * roi_w * 0.05:
            conts.append(c)
    best_cont = max(conts, key=cv2.contourArea)
    image_cont = cv2.drawContours(image.copy(), [best_cont], -1, (0, 255, 0), 2)
    show_image(np.hstack((image_cont, image)))

