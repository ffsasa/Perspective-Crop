import cv2
import numpy as np


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
    width = (widthA + widthB) / 2

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    height = (heightA + heightB) / 2

    width = int(width)
    height = int(height)

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
# Lọc contour đúng là giấy
# =========================
def find_document_contour(contours, image_shape):

    img_h, img_w = image_shape[:2]
    img_area = img_h * img_w

    best = None
    best_score = 0

    for c in contours:
        area = cv2.contourArea(c)

        if area < 0.25 * img_area:  # giảm từ 0.3 xuống 0.25
            continue

        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)

        if hull_area == 0:
            continue

        solidity = area / hull_area

        if solidity < 0.85:
            continue

        score = area * solidity

        if score > best_score:
            best = c
            best_score = score

    return best


# =========================
# Tìm quad nội tiếp (an toàn)
# =========================
def find_inscribed_quad(contour, inset=10):

    hull = cv2.convexHull(contour)
    pts = hull.reshape(-1, 2)

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    quad = np.array([tl, tr, br, bl], dtype=np.float32)

    center = np.mean(quad, axis=0)

    new_quad = []
    for p in quad:
        direction = center - p
        norm = np.linalg.norm(direction)
        if norm == 0:
            new_quad.append(p)
            continue
        direction = direction / norm
        new_p = p + direction * inset
        new_quad.append(new_p)

    return np.array(new_quad, dtype=np.float32)


# =========================
# Main pipeline
# =========================
def perspective_crop_safe(image_path, output_path):

    image = cv2.imread(image_path)

    if image is None:
        raise Exception("Không đọc được ảnh")

    orig = image.copy()

    # ======= THAY CANNY BẰNG SEGMENT VÙNG SÁNG =======
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold tách giấy (giấy sáng hơn nền)
    _, th = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

    kernel = np.ones((7,7), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)
    th = cv2.dilate(th, kernel, iterations=1)

    # DEBUG nếu cần:
    # cv2.imshow("mask", th)
    # cv2.waitKey(0)

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise Exception("Không tìm thấy contour")

    contour = find_document_contour(contours, image.shape)

    if contour is None:
        raise Exception("Không tìm được giấy phù hợp")

    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

    if len(approx) == 4:
        quad = approx.reshape(4, 2).astype(np.float32)
    else:
        quad = find_inscribed_quad(contour, inset=15)

    warped = four_point_transform(orig, quad)

    cv2.imwrite(output_path, warped)
    print("Done:", output_path)


# =========================
# RUN
# =========================
if __name__ == "__main__":
    perspective_crop_safe("input.jpg", "output.jpg")
