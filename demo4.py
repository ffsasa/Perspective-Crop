import cv2
import numpy as np
import os


# =========================
# Utility: sắp xếp 4 điểm
# =========================
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]      # top-left
    rect[2] = pts[np.argmax(s)]      # bottom-right
    rect[1] = pts[np.argmin(diff)]   # top-right
    rect[3] = pts[np.argmax(diff)]   # bottom-left

    return rect


# =========================
# Perspective transform
# =========================
def four_point_transform(image, pts):

    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    width = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    height = int(max(heightA, heightB))

    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (width, height))

    return warped


# =========================
# DEBUG PIPELINE
# =========================
def debug_pipeline(image_path):

    image = cv2.imread(image_path)
    if image is None:
        raise Exception("Không đọc được ảnh")

    # =========================
    # 1️⃣ Resize để ổn định
    # =========================
    scale_height = 1200
    ratio = image.shape[0] / scale_height
    image = cv2.resize(image, (int(image.shape[1] / ratio), scale_height))

    mask = np.zeros(image.shape[:2], np.uint8)

    bgModel = np.zeros((1,65), np.float64)
    fgModel = np.zeros((1,65), np.float64)

    # Giả sử giấy nằm trung tâm ảnh
    rect = (50, 50, image.shape[1]-100, image.shape[0]-100)

    cv2.grabCut(image, mask, rect, bgModel, fgModel, 5, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask==2)|(mask==0), 0, 255).astype('uint8')

    # =========================
    # 2️⃣ Morphology làm sạch
    # =========================
    kernel = np.ones((7,7), np.uint8)
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel, iterations=2)

    # =========================
    # 3️⃣ Find contour
    # =========================
    contours, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise Exception("Không tìm thấy contour")

    contour = max(contours, key=cv2.contourArea)

    # =========================
    # 4️⃣ Fit rotated rectangle
    # =========================
    # 1. Convex hull
    hull = cv2.convexHull(contour)

    # 2. Rotated bounding box
    rect = cv2.minAreaRect(hull)
    box = cv2.boxPoints(rect)

    quad = order_points(np.array(box, dtype="float32"))

    # Scale quad về ảnh gốc
    quad *= ratio

    # =========================
    # 5️⃣ Perspective-Crop
    # =========================
    warped = four_point_transform(cv2.imread(image_path), quad)

    cv2.imwrite("output.jpg", warped)

    print("Detect + Perspective-Crop thành công")


# =========================
# RUN
# =========================
if __name__ == "__main__":
    debug_pipeline("input.jpg")
