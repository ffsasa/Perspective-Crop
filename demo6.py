import cv2
import numpy as np
import os


# =========================
# Utility: sort 4 points
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
# Main pipeline - VERSION 8
# =========================
def process(image_path):

    os.makedirs("debug", exist_ok=True)

    original = cv2.imread(image_path)
    if original is None:
        raise Exception("Không đọc được ảnh")

    # Resize
    scale_height = 1200
    ratio = original.shape[0] / scale_height
    image = cv2.resize(
        original,
        (int(original.shape[1] / ratio), scale_height)
    )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("debug/1_gray.jpg", gray)

    # Texture energy
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    energy = np.abs(lap)

    energy_blur = cv2.GaussianBlur(energy, (31, 31), 0)
    energy_blur = cv2.normalize(
        energy_blur, None, 0, 255, cv2.NORM_MINMAX
    ).astype("uint8")

    cv2.imwrite("debug/2_energy_blur.jpg", energy_blur)

    # Adaptive threshold
    mask = cv2.adaptiveThreshold(
        energy_blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        51,
        -10
    )

    # Morph close
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (15, 20)
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    cv2.imwrite("debug/3_mask.jpg", mask)

    # Find contours
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    contour_img = image.copy()
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
    cv2.imwrite("debug/4_all_contours.jpg", contour_img)

    if not contours:
        raise Exception("Không tìm thấy contour")

    # =========================
    # Geometry + Texture scoring
    # =========================
    candidates = []

    for c in contours:
        hull = cv2.convexHull(c)
        area = cv2.contourArea(hull)

        if area < 2000:
            continue

        # Texture density inside contour
        mask_single = np.zeros_like(gray)
        cv2.drawContours(mask_single, [hull], -1, 255, -1)

        mean_texture = cv2.mean(energy_blur, mask=mask_single)[0]

        # approx to check if near quadrilateral
        peri = cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, 0.02 * peri, True)

        # If 4 edges → boost score
        if len(approx) == 4:
            score = area * mean_texture * 1.5
        else:
            score = area * mean_texture

        candidates.append((score, hull))

    if not candidates:
        raise Exception("Không tìm được contour phù hợp")

    # Select best score
    candidates = sorted(candidates, key=lambda x: x[0], reverse=True)
    best_hull = candidates[0][1]

    best_debug = image.copy()
    cv2.drawContours(best_debug, [best_hull], -1, (0, 0, 255), 3)
    cv2.imwrite("debug/5_best_hull.jpg", best_debug)

    # =========================
    # Final quad extraction
    # =========================
    peri = cv2.arcLength(best_hull, True)
    approx = cv2.approxPolyDP(best_hull, 0.02 * peri, True)

    if len(approx) == 4:
        quad = order_points(approx.reshape(4, 2))
        print("Using approx 4-point polygon")
    else:
        print("Fallback to minAreaRect")
        rect = cv2.minAreaRect(best_hull)
        box = cv2.boxPoints(rect)
        quad = order_points(np.array(box, dtype="float32"))

    quad_debug = image.copy()
    for p in quad:
        cv2.circle(quad_debug, (int(p[0]), int(p[1])), 12, (0, 0, 255), -1)

    cv2.imwrite("debug/6_quad.jpg", quad_debug)

    # Warp
    quad *= ratio
    warped = four_point_transform(original, quad)

    cv2.imwrite("output.jpg", warped)

    print("Version 8 hoàn thành – kiểm tra thư mục debug/")


# RUN
if __name__ == "__main__":
    process("input.jpg")