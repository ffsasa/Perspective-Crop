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
# Lọc contour đúng là giấy (bắt tờ ở trên)
# =========================
def find_document_contour(contours, image_shape):

    img_h, img_w = image_shape[:2]
    img_area = img_h * img_w

    candidates = []

    for c in contours:
        area = cv2.contourArea(c)

        if area < 0.1 * img_area:   # giảm ngưỡng để không loại tờ trên
            continue

        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)

        if hull_area == 0:
            continue

        solidity = area / hull_area

        # approx để xem có gần hình chữ nhật không
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # cần ít nhất 4 cạnh
        if len(approx) < 4:
            continue

        # score ưu tiên:
        # 1️⃣ solidity cao (ít bị che)
        # 2️⃣ diện tích đủ lớn
        score = solidity * 2 + (area / img_area)

        candidates.append((c, score))

    if not candidates:
        return None

    # chọn contour có score cao nhất
    best = sorted(candidates, key=lambda x: x[1], reverse=True)[0][0]

    return best


# =========================
# DEBUG PIPELINE
# =========================
def debug_pipeline(image_path):

    image = cv2.imread(image_path)
    if image is None:
        raise Exception("Không đọc được ảnh")

    os.makedirs("debug", exist_ok=True)

    # 1️⃣ Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("debug/1_gray.jpg", gray)

    # 2️⃣ Edge detection 
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    edges = cv2.Canny(L, 30, 120)
    cv2.imwrite("debug/2_edges.jpg", edges)

    # 3️⃣ Hough Line detection
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=100,
        minLineLength=0.2 * max(image.shape),
        maxLineGap=30
    )

    line_img = image.copy()

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_img, (x1, y1), (x2, y2), (0,255,0), 3)

    cv2.imwrite("debug/3_hough_lines.jpg", line_img)

    # morphology
    kernel = np.ones((7,7), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    cv2.imwrite("debug/3.1_morphology.jpg", edges)

    # 4️⃣ Tìm contour
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    all_contours_img = image.copy()
    cv2.drawContours(all_contours_img, contours, -1, (0,255,0), 3)
    cv2.imwrite("debug/4_all_contours.jpg", all_contours_img)

    if not contours:
        raise Exception("Không tìm thấy contour")

    # 5️⃣ Chọn contour giấy
    contour = find_document_contour(contours, image.shape)

    if contour is None:
        raise Exception("Không tìm được giấy phù hợp")

    selected_contour_img = image.copy()
    cv2.drawContours(selected_contour_img, [contour], -1, (255,0,0), 5)
    cv2.imwrite("debug/5_selected_contour.jpg", selected_contour_img)

    # 6️⃣ Approx polygon
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

    approx_img = image.copy()
    cv2.drawContours(approx_img, [approx], -1, (0,255,255), 5)
    cv2.imwrite("debug/6_approx_polygon.jpg", approx_img)

    # 7️⃣ Lấy 4 góc
    if len(approx) >= 4:
        pts = approx.reshape(-1, 2)

        if len(pts) > 4:
            hull = cv2.convexHull(contour)
            pts = hull.reshape(-1, 2)

            s = pts.sum(axis=1)
            diff = np.diff(pts, axis=1).reshape(-1)

            tl = pts[np.argmin(s)]
            br = pts[np.argmax(s)]
            tr = pts[np.argmin(diff)]
            bl = pts[np.argmax(diff)]

            quad = np.array([tl, tr, br, bl], dtype=np.float32)
        else:
            quad = approx.reshape(4, 2).astype(np.float32)
    else:
        raise Exception("Không đủ điểm để tạo quad")

    quad = order_points(quad)

    # 8️⃣ Vẽ 4 góc đỏ
    corner_img = image.copy()
    for point in quad:
        x, y = int(point[0]), int(point[1])
        cv2.circle(corner_img, (x, y), 15, (0,0,255), -1)

    cv2.imwrite("debug/7_corners.jpg", corner_img)

    print("Đã xuất toàn bộ ảnh debug trong thư mục /debug")


# =========================
# RUN
# =========================
if __name__ == "__main__":
    debug_pipeline("input.jpg")
