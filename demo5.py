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
# Border filter
# =========================
def near_border(contour, h, w, margin=10):
    for p in contour:
        x, y = p[0]
        if x < margin or x > w - margin:
            return True
        if y < margin or y > h - margin:
            return True
    return False


# =========================
# Main pipeline
# =========================
def process(image_path):

    os.makedirs("debug", exist_ok=True)

    original = cv2.imread(image_path)
    if original is None:
        raise Exception("Không đọc được ảnh")

    # =========================
    # Resize
    # =========================
    scale_height = 1200
    ratio = original.shape[0] / scale_height
    image = cv2.resize(
        original,
        (int(original.shape[1] / ratio), scale_height)
    )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("debug/1_gray.jpg", gray)

    # =========================
    # Texture energy (Laplacian)
    # =========================
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    energy = np.abs(lap)

    energy_blur = cv2.GaussianBlur(energy, (31, 31), 0)
    energy_blur = cv2.normalize(
        energy_blur, None, 0, 255, cv2.NORM_MINMAX
    ).astype("uint8")

    cv2.imwrite("debug/2_energy_blur.jpg", energy_blur)

    # =========================
    # Adaptive threshold
    # =========================
    mask_energy = cv2.adaptiveThreshold(
        energy_blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        51,
        -10
    )

    cv2.imwrite("debug/3_mask_energy.jpg", mask_energy)

    # =========================
    # White mask (HSV)
    # =========================
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # White
    white_mask = cv2.inRange(
        hsv,
        (0, 0, 150),
        (180, 80, 255)
    )

    # Red range 1
    red_mask1 = cv2.inRange(
        hsv,
        (0, 40, 80),
        (15, 255, 255)
    )

    # Red range 2
    red_mask2 = cv2.inRange(
        hsv,
        (165, 40, 80),
        (180, 255, 255)
    )

    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    # Combine white + red
    color_mask = cv2.bitwise_or(white_mask, red_mask)

    cv2.imwrite("debug/4_color_mask.jpg", color_mask)

    # =========================
    # Combine masks
    # =========================
    mask = cv2.bitwise_and(mask_energy, white_mask)
    cv2.imwrite("debug/5_combined.jpg", mask)

    # =========================
    # Morph close
    # =========================
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (15,20)
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite("debug/6_after_close.jpg", mask)

    # =========================
    # Find contours
    # =========================
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    contour_img = image.copy()
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
    cv2.imwrite("debug/7_all_contours.jpg", contour_img)

    if not contours:
        raise Exception("Không tìm thấy contour")

    image_area = image.shape[0] * image.shape[1]

    valid_contours = [
        c for c in contours
        if cv2.contourArea(c) > 0.01 * image_area
        and not near_border(c, image.shape[0], image.shape[1])
    ]

    if not valid_contours:
        raise Exception("Không có contour hợp lệ")

    # Lấy merge contour
    merged = np.vstack(valid_contours)
    contour = cv2.convexHull(merged)

    contour_big = image.copy()
    cv2.drawContours(contour_big, [contour], -1, (0, 0, 255), 3)
    cv2.imwrite("debug/8_largest_contour.jpg", contour_big)

    # =========================
    # Approx polygon
    # =========================
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

    if len(approx) == 4:
        quad = order_points(approx.reshape(4, 2))
    else:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        quad = order_points(np.array(box, dtype="float32"))

    quad_debug = image.copy()
    for p in quad:
        cv2.circle(quad_debug, (int(p[0]), int(p[1])), 10, (0, 0, 255), -1)

    cv2.imwrite("debug/9_quad.jpg", quad_debug)

    # =========================
    # Warp
    # =========================
    quad *= ratio

    # Expand 3% outward
    center = np.mean(quad, axis=0)
    expanded = []

    for p in quad:
        direction = p - center
        expanded.append(p + 0.03 * direction)

    quad = np.array(expanded, dtype="float32")

    warped = four_point_transform(original, quad)

    cv2.imwrite("output.jpg", warped)

    print("v5.1 hoàn thành – kiểm tra thư mục debug/")


# =========================
# RUN
# =========================
if __name__ == "__main__":
    process("input.jpg")
