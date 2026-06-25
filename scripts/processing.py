import cv2
import csv
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
parameters = []
k = 1.15

for image_class in cell + sensor:
    image = cv2.imread(str(image_class))
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB) 
    l, a, b = cv2.split(image_lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    clahe_image = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    image_gray = l
    kernel_ellipse_3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_ellipse_5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    if image_class in cell:
        gauss_filter = [(7, 7)]
        gauss_blur = cv2.GaussianBlur(image_gray, gauss_filter[0], 0)
        image_bin = cv2.adaptiveThreshold(gauss_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 3)
        image_morph = cv2.morphologyEx(image_bin, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
        image_morph = cv2.morphologyEx(image_morph, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8))
        image_morph = cv2.morphologyEx(image_morph, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
        folder = "cell"
    elif image_class in sensor:
        gauss_filter = [(15, 15)]
        gauss_blur = cv2.GaussianBlur(image_gray, gauss_filter[0], 0)
        _, image_bin = cv2.threshold(gauss_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        image_morph = cv2.morphologyEx(image_bin, cv2.MORPH_OPEN, kernel_ellipse_3)
        image_morph = cv2.morphologyEx(image_morph, cv2.MORPH_CLOSE, kernel_ellipse_5)
        image_morph = cv2.morphologyEx(image_morph, cv2.MORPH_OPEN, kernel_ellipse_3)
        folder = "sensor"

    cont, _ = cv2.findContours(image_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    conts = []
    roi_h, roi_w = image_morph.shape

    for c in cont:
        s = cv2.contourArea(c)
        if s >= roi_h * roi_w * 0.05:
            conts.append(c)
    
    if not conts:
        print(f"Контур не найден: {image_class.name}")
        continue

    best_cont = max(conts, key=cv2.contourArea)
    s_cont = cv2.contourArea(best_cont)
    # image_cont = cv2.drawContours(image.copy(), [best_cont], -1, (0, 255, 0), 2)
    approx = cv2.approxPolyDP(best_cont, 0.01 * cv2.arcLength(best_cont, True), True)
    hull = cv2.convexHull(approx)
    rect = cv2.minAreaRect(hull)
    (c_x, c_y), (w, h), ang = rect
    long_side = max(w, h)
    short_side = min(w, h)
    grasp_ang = (ang if w < h else ang + 90) % 180
    grasp_width = short_side * k
    rect_box = cv2.boxPoints(rect).astype(np.int32)
    angle_rad = np.deg2rad(grasp_ang)
    x1 = int(c_x - grasp_width / 2 * np.cos(angle_rad))
    y1 = int(c_y - grasp_width / 2 * np.sin(angle_rad))
    x2 = int(c_x + grasp_width / 2 * np.cos(angle_rad))
    y2 = int(c_y + grasp_width / 2 * np.sin(angle_rad))

    drawn_image = image.copy()
    cv2.drawContours(drawn_image, [rect_box], -1, (0, 0, 255), 1)
    cv2.drawContours(drawn_image, [hull], -1, (0, 255, 0), 1)
    cv2.circle(drawn_image, (int(c_x), int(c_y)), 3, (255, 0, 0), -1)
    cv2.line(drawn_image, (x1, y1), (x2, y2), (255, 255, 0), 2)

    # show_image(drawn_image)
    cv2.imwrite(f"{ROOT_DIR}/outputs/processed/{folder}/{image_class.stem}_processed.jpg", drawn_image)

    parameters.append({
        "filename": image_class.name,
        "class": folder,
        "s_contour": round(s_cont, 2),
        "center_x": round(c_x, 2),
        "center_y": round(c_y, 2),
        "width": round(w, 2),
        "height": round(h, 2),
        "grasp_angle": round(grasp_ang, 2),
        "long_side": round(long_side, 2),
        "short_side": round(short_side, 2),
        "k": round(k, 2),
        "grasp_width": round(grasp_width, 2),
    })
with open(f"{ROOT_DIR}/outputs/processed/parameters.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=parameters[0].keys())
    writer.writeheader()
    writer.writerows(parameters)