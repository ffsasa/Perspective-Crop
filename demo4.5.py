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

    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

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

    os.makedirs("debug", exist_ok=True)
    cv2.imwrite("debug/0_input.jpg", image)

    # =========================
    # 1️⃣ Resize
    # =========================
    scale_height = 1200
    ratio = image.shape[0] / scale_height
    image = cv2.resize(image, (int(image.shape[1] / ratio), scale_height))

    # Blur nhẹ để GrabCut ổn định hơn
    image_blur = cv2.GaussianBlur(image, (5,5), 0)

    # =========================
    # 2️⃣ GrabCut an toàn hơn
    # =========================
    mask = np.full(image.shape[:2], cv2.GC_PR_BGD, np.uint8)

    h, w = image.shape[:2]

    # Rect nhỏ hơn để tránh ăn mép
    rect = (int(w*0.05), int(h*0.05),
            int(w*0.9), int(h*0.9))

    bgModel = np.zeros((1,65), np.float64)
    fgModel = np.zeros((1,65), np.float64)

    cv2.grabCut(image_blur, mask, rect, bgModel, fgModel, 5, cv2.GC_INIT_WITH_RECT)

    # Chỉ giữ foreground
    mask2 = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
        255, 0).astype('uint8')

    cv2.imwrite("debug/1_grabcut_mask.jpg", mask2)

    # =========================
    # 3️⃣ Morphology OPEN (không CLOSE)
    # =========================
    kernel = np.ones((5,5), np.uint8)
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel, iterations=1)

    cv2.imwrite("debug/2_mask_clean.jpg", mask2)

    # =========================
    # 4️⃣ Find contour lớn nhất
    # =========================
    contours, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise Exception("Không tìm thấy contour")

    contour = max(contours, key=cv2.contourArea)

    # Loại bỏ contour quá nhỏ
    if cv2.contourArea(contour) < 0.1 * (h*w):
        raise Exception("Contour quá nhỏ, segmentation lỗi")

    contour_img = image.copy()
    cv2.drawContours(contour_img, [contour], -1, (0,255,0), 3)
    cv2.imwrite("debug/3_largest_contour.jpg", contour_img)

    # =========================
    # 5️⃣ Ép thành tứ giác thật sự
    # =========================
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

    if len(approx) == 4:
        quad = order_points(approx.reshape(4,2))
    else:
        # fallback nếu không ra 4 điểm
        hull = cv2.convexHull(contour)
        rect = cv2.minAreaRect(hull)
        box = cv2.boxPoints(rect)
        quad = order_points(np.array(box, dtype="float32"))

    # =========================
    # 6️⃣ Vẽ 4 góc
    # =========================
    result = image.copy()

    for point in quad:
        x, y = int(point[0]), int(point[1])
        cv2.circle(result, (x,y), 15, (0,0,255), -1)

    cv2.imwrite("debug/4_detected_corners.jpg", result)

    print("Detect thành công (phiên bản ổn định)")

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
