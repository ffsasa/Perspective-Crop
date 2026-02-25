import cv2
import numpy as np


# =========================
# Utility
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
# Color label: TK (white) / XX (pink)
# =========================
def classify_center_color(bgr_img):
    """
    Trả về "XX" nếu thiên hồng, ngược lại "TK".
    Dùng HSV để ổn định hơn so với RGB.
    """
    h, w = bgr_img.shape[:2]
    cx1, cx2 = int(w * 0.40), int(w * 0.60)
    cy1, cy2 = int(h * 0.40), int(h * 0.60)
    patch = bgr_img[cy1:cy2, cx1:cx2]

    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    H = hsv[..., 0].astype(np.float32)  # 0..179
    S = hsv[..., 1].astype(np.float32)  # 0..255

    # Pink thường có hue ~ 150..179 hoặc ~0..10 (đỏ/hồng) với saturation đủ cao
    pink_mask = ((H >= 150) | (H <= 10)) & (S >= 35)

    pink_ratio = float(np.mean(pink_mask))
    return ("XX" if pink_ratio > 0.12 else "TK")


# =========================
# Main: GrabCut-lite + trim tail
# =========================
def detect_document(image_path, debug=True, grabcut_iters=2):
    original = cv2.imread(image_path)
    if original is None:
        raise Exception("Không đọc được ảnh")

    # ---- Downscale để GrabCut chạy nhanh
    target_h = 700
    ratio = original.shape[0] / target_h
    small = cv2.resize(original, (int(original.shape[1] / ratio), target_h))
    h, w = small.shape[:2]

    # ---- (A) Classify màu để đặt tên file (không ảnh hưởng detect)
    label = classify_center_color(small)

    # ---- (B) GrabCut init bằng rect
    mask = np.zeros((h, w), np.uint8)
    mx = int(w * 0.05)
    my = int(h * 0.05)
    rect = (mx, my, w - 2 * mx, h - 2 * my)

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    cv2.grabCut(small, mask, rect, bgdModel, fgdModel, grabcut_iters, cv2.GC_INIT_WITH_RECT)

    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

    # ---- (C) Morph close để nối mép, nhưng đừng quá mạnh
    k = max(9, (min(h, w) // 90) | 1)
    kernel = np.ones((k, k), np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=2)

    # ---- (D) Giữ component chứa tâm (fallback largest)
    num_labels, lab_cc, stats, _ = cv2.connectedComponentsWithStats(fg)
    if num_labels <= 1:
        raise Exception("GrabCut không tách được foreground")

    center_label = lab_cc[h // 2, w // 2]
    if center_label == 0:
        center_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

    comp = (lab_cc == center_label).astype(np.uint8) * 255

    # =========================
    # (E) TRIM TAIL: cắt phần đáy thừa bằng shape filter
    # =========================
    contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise Exception("Không tìm thấy contour")

    cnt = max(contours, key=cv2.contourArea)

    # 1) Fit minAreaRect của hull (hình chữ nhật giấy)
    hull = cv2.convexHull(cnt)
    rect = cv2.minAreaRect(hull)
    box = cv2.boxPoints(rect).astype(np.int32)

    # 2) Tạo mask hình chữ nhật, intersect để loại đuôi rơi xuống (tail)
    rect_mask = np.zeros_like(comp)
    cv2.fillConvexPoly(rect_mask, box, 255)

    comp2 = cv2.bitwise_and(comp, rect_mask)

    # 3) Sau intersect, lấy lại contour lớn nhất để ổn định
    contours2, _ = cv2.findContours(comp2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours2:
        comp2 = comp  # fallback
        contours2 = contours

    cnt2 = max(contours2, key=cv2.contourArea)
    hull2 = cv2.convexHull(cnt2)
    rect2 = cv2.minAreaRect(hull2)
    box2 = cv2.boxPoints(rect2).astype(np.float32)

    # ---- Debug
    if debug:
        dbg = small.copy()
        cv2.drawContours(dbg, [box2.astype(int)], -1, (0, 255, 0), 3)
        cv2.imwrite("debug_mask_best.jpg", fg)
        cv2.imwrite("debug_component_best.jpg", comp2)
        cv2.imwrite("debug_detected_box.jpg", dbg)
        with open("debug_info.txt", "w", encoding="utf-8") as f:
            f.write(f"label={label}\n")
            f.write(f"grabcut_iters={grabcut_iters}\n")
            f.write(f"kernel_size={k}\n")

    # ---- Scale box về ảnh gốc và warp
    box2 *= ratio
    warped = four_point_transform(original, box2)

    return warped, label


# =========================
# RUN
# =========================
if __name__ == "__main__":
    warped, label = detect_document("input5.jpg", debug=True, grabcut_iters=2)

    # đặt tên theo màu: XX.jpg nếu hồng, TK.jpg nếu trắng
    out_name = f"{label}.jpg"
    cv2.imwrite(out_name, warped)
    print("Detect + Perspective thành công ->", out_name)