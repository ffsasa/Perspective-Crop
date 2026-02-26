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
    return cv2.warpPerspective(image, M, (width, height))


# =========================
# Color label: TK (white) / XX (pink)  (chỉ dùng để đặt tên)
# =========================
def classify_center_color(bgr_img):
    h, w = bgr_img.shape[:2]
    cx1, cx2 = int(w * 0.40), int(w * 0.60)
    cy1, cy2 = int(h * 0.40), int(h * 0.60)
    patch = bgr_img[cy1:cy2, cx1:cx2]

    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    H = hsv[..., 0].astype(np.float32)
    S = hsv[..., 1].astype(np.float32)

    pink_mask = ((H >= 150) | (H <= 10)) & (S >= 25)  # giảm S để bắt hồng nhạt
    pink_ratio = float(np.mean(pink_mask))
    return ("XX" if pink_ratio > 0.10 else "TK")


# =========================
# GrabCut helper (FIXED)
# =========================
def run_grabcut(small_bgr, iters=1, hard_bg_bottom=False, debug=False):
    h, w = small_bgr.shape[:2]

    mx = int(w * 0.05)
    my = int(h * 0.05)
    rect = (mx, my, w - 2 * mx, h - 2 * my)

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    mask = np.zeros((h, w), np.uint8)
    cv2.grabCut(small_bgr, mask, rect, bgdModel, fgdModel, 1, cv2.GC_INIT_WITH_RECT)

    # --- AUTO: chỉ xử lý đáy nếu thật sự có "tail" bám đáy
    if hard_bg_bottom:
        fg0 = ((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)).astype(np.uint8)
        bottom_band = fg0[int(h*0.92):, :]   # 8% đáy
        bottom_fill = float(bottom_band.mean())  # tỉ lệ foreground ở đáy

        # nếu foreground ở đáy quá nhiều => đang dính nền => ép đáy mạnh hơn
        if bottom_fill > 0.25:
            y0 = int(h * 0.86)               # ép nhiều hơn (14% đáy)
            mask[y0:, :] = cv2.GC_BGD        # mạnh hẳn (BG cứng)
            cv2.grabCut(small_bgr, mask, None, bgdModel, fgdModel, 1, cv2.GC_INIT_WITH_MASK)
        else:
            # bình thường: ép rất ít và mềm (giữ chữ ký nếu có)
            y0 = int(h * 0.92)
            mask[y0:, :] = cv2.GC_PR_BGD
            cv2.grabCut(small_bgr, mask, None, bgdModel, fgdModel, 1, cv2.GC_INIT_WITH_MASK)

    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

    k = max(9, (min(h, w) // 90) | 1)
    kernel = np.ones((k, k), np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=2)

    if debug:
        cv2.imwrite("debug_fg_mask.jpg", fg)

    return fg


def component_center_or_largest(binary_mask):
    h, w = binary_mask.shape[:2]
    num_labels, lab_cc, stats, _ = cv2.connectedComponentsWithStats(binary_mask)
    if num_labels <= 1:
        return None, 0.0

    center_label = lab_cc[h // 2, w // 2]
    if center_label == 0:
        center_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

    comp = (lab_cc == center_label).astype(np.uint8) * 255
    area = stats[center_label, cv2.CC_STAT_AREA]
    area_ratio = area / float(h * w)
    return comp, area_ratio


def minarea_box_from_mask(comp_mask):
    contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(cnt)
    rect = cv2.minAreaRect(hull)
    box = cv2.boxPoints(rect).astype(np.float32)
    return box


def trim_tail_by_rectmask(comp_mask):
    # intersect comp với rect_mask (cắt đuôi)
    box = minarea_box_from_mask(comp_mask)
    if box is None:
        return comp_mask, box

    rect_mask = np.zeros_like(comp_mask)
    cv2.fillConvexPoly(rect_mask, box.astype(np.int32), 255)
    comp2 = cv2.bitwise_and(comp_mask, rect_mask)

    # lấy lại comp lớn nhất sau trim
    contours, _ = cv2.findContours(comp2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return comp_mask, box
    cnt2 = max(contours, key=cv2.contourArea)
    hull2 = cv2.convexHull(cnt2)
    rect2 = cv2.minAreaRect(hull2)
    box2 = cv2.boxPoints(rect2).astype(np.float32)
    return comp2, box2


def pad_bottom(box, pad_ratio=0.03, max_px=40):
    box = order_points(box.astype(np.float32))
    tl, tr, br, bl = box

    # chiều cao ước lượng
    h1 = np.linalg.norm(bl - tl)
    h2 = np.linalg.norm(br - tr)
    h = max(h1, h2)

    pad = min(max_px, pad_ratio * h)

    # hướng xuống
    v_left = bl - tl
    v_right = br - tr
    v_left = v_left / (np.linalg.norm(v_left) + 1e-6)
    v_right = v_right / (np.linalg.norm(v_right) + 1e-6)

    bl2 = bl + v_left * pad
    br2 = br + v_right * pad

    return np.array([tl, tr, br2, bl2], dtype=np.float32)

# =========================
# Main
# =========================
def detect_document(image_path, debug=True):
    original = cv2.imread(image_path)
    if original is None:
        raise Exception("Không đọc được ảnh")

    # Downscale
    target_h = 700
    ratio = original.shape[0] / target_h
    small = cv2.resize(original, (int(original.shape[1] / ratio), target_h))

    label = classify_center_color(small)

    # --- Pass 1: GrabCut thường
    fg1 = run_grabcut(small, iters=1, hard_bg_bottom=False, debug=False)
    comp1, area_ratio1 = component_center_or_largest(fg1)
    if comp1 is None:
        raise Exception("GrabCut không tách được foreground")

    # Quyết định nhánh theo diện tích giấy (KHÔNG theo màu)
    LARGE_THRESH = 0.5
    is_large = area_ratio1 >= LARGE_THRESH

    if is_large:
        # ===== Nhánh A: giấy lớn - KHÔNG trim rectmask để khỏi mất rìa
        box = minarea_box_from_mask(comp1)
        comp_final = comp1
        box_final = box
        method = "LARGE_NO_TRIM"
    else:
        # ===== Nhánh B: giấy nhỏ - hard BG ở đáy + trim tail
        fg2 = run_grabcut(small, iters=1, hard_bg_bottom=True, debug=False)
        comp2, area_ratio2 = component_center_or_largest(fg2)
        if comp2 is None:
            comp2 = comp1

        comp_trim, box_trim = trim_tail_by_rectmask(comp2)
        comp_final = comp_trim
        box_final = box_trim
        method = "SMALL_BG_BOTTOM_TRIM"

    if box_final is None:
        raise Exception("Không tìm được box")

    # Debug
    if debug:
        dbg = small.copy()
        cv2.drawContours(dbg, [box_final.astype(int)], -1, (0, 255, 0), 3)
        cv2.imwrite("debug_component_best.jpg", comp_final)
        cv2.imwrite("debug_detected_box.jpg", dbg)
        with open("debug_info.txt", "w", encoding="utf-8") as f:
            f.write(f"label={label}\n")
            f.write(f"area_ratio_pass1={area_ratio1:.4f}\n")
            f.write(f"method={method}\n")

    # Warp back to original
    box_final *= ratio
    box_final = pad_bottom(box_final, pad_ratio=0.05, max_px=150)
    warped = four_point_transform(original, box_final)

    return warped, label

if __name__ == "__main__":
    warped, label = detect_document("input12.jpg", debug=True)
    out_name = f"{label}.jpg"  # XX.jpg / TK.jpg
    cv2.imwrite(out_name, warped)
    print("Detect + Perspective thành công ->", out_name)