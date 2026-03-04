import cv2
import numpy as np
from ultralytics import YOLO

# =========================
# CONFIG
# =========================
MODEL_PATH = r"C:\Users\Admin\Downloads\PerspectiveCrop.v1i.yolov8\runs\segment\train2\weights\best.pt"  
DEVICE = 0  # 0 = GPU, "cpu" nếu muốn CPU
IMGSZ = 640
CONF = 0.25

# =========================
# Utility (GIỮ NGUYÊN)
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
# Color label: TK (white) / XX (pink)  (GIỮ NGUYÊN)
# =========================
def classify_center_color(bgr_img):
    h, w = bgr_img.shape[:2]
    cx1, cx2 = int(w * 0.40), int(w * 0.60)
    cy1, cy2 = int(h * 0.40), int(h * 0.60)
    patch = bgr_img[cy1:cy2, cx1:cx2]

    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    H = hsv[..., 0].astype(np.float32)
    S = hsv[..., 1].astype(np.float32)

    pink_mask = ((H >= 150) | (H <= 10)) & (S >= 25)
    pink_ratio = float(np.mean(pink_mask))
    return ("XX" if pink_ratio > 0.10 else "TK")


def keep_largest_component(binary_mask):
    num_labels, lab, stats, _ = cv2.connectedComponentsWithStats((binary_mask > 0).astype(np.uint8))
    if num_labels <= 1:
        return binary_mask
    idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    out = (lab == idx).astype(np.uint8) * 255
    return out


# =========================
# Quad scoring (GIỮ NGUYÊN từ demo10.py)
# =========================
def _quad_angles_deg(q):
    q = order_points(q.astype(np.float32))
    angs = []
    for i in range(4):
        p0 = q[i]
        p1 = q[(i - 1) % 4]
        p2 = q[(i + 1) % 4]
        v1 = p1 - p0
        v2 = p2 - p0
        n1 = np.linalg.norm(v1) + 1e-6
        n2 = np.linalg.norm(v2) + 1e-6
        cs = float(np.dot(v1, v2) / (n1 * n2))
        cs = max(-1.0, min(1.0, cs))
        angs.append(float(np.degrees(np.arccos(cs))))
    return angs


def _quad_side_lengths(q):
    q = order_points(q.astype(np.float32))
    lens = []
    for i in range(4):
        lens.append(float(np.linalg.norm(q[(i + 1) % 4] - q[i])))
    return lens


def _quad_mask_metrics(comp_mask, quad):
    h, w = comp_mask.shape[:2]
    quad = order_points(quad.astype(np.float32))

    qm = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(qm, quad.astype(np.int32), 255)

    m = (comp_mask > 0).astype(np.uint8)
    q = (qm > 0).astype(np.uint8)

    inter = int((m & q).sum())
    am = int(m.sum())
    aq = int(q.sum())

    cover = inter / (am + 1e-6)
    spill = (aq - inter) / (aq + 1e-6)
    return float(cover), float(spill)


def score_quad(comp_mask, quad,
               min_cover=0.90, max_spill=0.10,
               min_angle=25.0, max_angle=155.0,
               max_side_ratio=3.5,
               min_area_ratio=0.15):
    if quad is None:
        return -1e9, "None"

    quad = order_points(quad.astype(np.float32))
    h, w = comp_mask.shape[:2]

    area = float(cv2.contourArea(quad.reshape(-1, 1, 2)))
    if area < min_area_ratio * (h * w):
        return -1e9, "too_small"

    angs = _quad_angles_deg(quad)
    if min(angs) < min_angle or max(angs) > max_angle:
        return -1e9, f"bad_angles min={min(angs):.1f} max={max(angs):.1f}"

    lens = _quad_side_lengths(quad)
    r = max(lens) / (min(lens) + 1e-6)
    if r > max_side_ratio:
        return -1e9, f"bad_side_ratio {r:.2f}"

    cover, spill = _quad_mask_metrics(comp_mask, quad)
    if cover < min_cover:
        return -1e9, f"low_cover {cover:.3f}"
    if spill > max_spill:
        return -1e9, f"high_spill {spill:.3f}"

    score = 0.0
    score += 3.0 * cover
    score += 2.0 * (1.0 - spill)
    score += 0.3 * min(1.0, area / (h * w))
    return score, f"cover={cover:.3f} spill={spill:.3f} sideR={r:.2f}"


def quad_candidates_from_mask(comp_mask,
                              eps_list=(0.010, 0.020, 0.030),
                              min_area_ratio=0.20):
    h, w = comp_mask.shape[:2]
    contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    if area < min_area_ratio * (h * w):
        return []

    hull = cv2.convexHull(cnt)
    peri = cv2.arcLength(hull, True)

    cands = []
    for eps in eps_list:
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) != 4:
            continue
        if not cv2.isContourConvex(approx):
            continue
        a = cv2.contourArea(approx)
        if a < min_area_ratio * (h * w):
            continue
        cands.append(approx.reshape(4, 2).astype(np.float32))

    return cands


def minarea_box_from_mask(comp_mask):
    contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(cnt)
    rect = cv2.minAreaRect(hull)
    box = cv2.boxPoints(rect).astype(np.float32)
    return box


def pad_box(box,
            pad_top=0.05, pad_bottom=0.05,
            pad_left=0.025, pad_right=0.025,
            max_px_tb=150,
            max_px_lr=100):

    box = order_points(box.astype(np.float32))
    tl, tr, br, bl = box

    v_left = bl - tl
    v_right = br - tr

    h1 = np.linalg.norm(v_left)
    h2 = np.linalg.norm(v_right)
    h = max(h1, h2)

    v_left /= (h1 + 1e-6)
    v_right /= (h2 + 1e-6)

    v_top = tr - tl
    v_bottom = br - bl

    w1 = np.linalg.norm(v_top)
    w2 = np.linalg.norm(v_bottom)
    w = max(w1, w2)

    v_top /= (w1 + 1e-6)
    v_bottom /= (w2 + 1e-6)

    pt = min(max_px_tb, pad_top * h)
    pb = min(max_px_tb, pad_bottom * h)
    pl = min(max_px_lr, pad_left * w)
    pr = min(max_px_lr, pad_right * w)

    tl2 = tl - v_left * pt - v_top * pl
    tr2 = tr - v_right * pt + v_top * pr
    br2 = br + v_right * pb + v_bottom * pr
    bl2 = bl + v_left * pb - v_bottom * pl

    return np.array([tl2, tr2, br2, bl2], dtype=np.float32)


# =========================
# YOLO helpers
# =========================
_yolo_model = None

def get_model():
    global _yolo_model
    if _yolo_model is None:
        _yolo_model = YOLO(MODEL_PATH)
    return _yolo_model


def pick_best_instance(masks_np, confs_np, H, W):
    """
    Chọn giấy chính: ưu tiên lớn + gần tâm + confidence
    """
    cx, cy = W / 2.0, H / 2.0
    best_i, best_score = None, -1e18

    for i in range(len(masks_np)):
        m = (masks_np[i] > 0.5).astype(np.uint8)
        area = int(m.sum())
        if area < 2000:
            continue

        ys, xs = np.where(m > 0)
        if len(xs) < 10:
            continue
        mx, my = float(xs.mean()), float(ys.mean())
        center_dist = np.hypot(mx - cx, my - cy) / (np.hypot(cx, cy) + 1e-6)

        conf = float(confs_np[i]) if confs_np is not None else 0.0
        score = (area / float(H * W)) * 2.0 + conf * 1.0 - center_dist * 0.8

        if score > best_score:
            best_score = score
            best_i = i

    return best_i


def mask_postprocess(mask_u8):
    # mask_u8: 0/255
    mask_u8 = keep_largest_component(mask_u8)
    k = max(7, (min(mask_u8.shape[:2]) // 120) | 1)
    kernel = np.ones((k, k), np.uint8)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_u8 = keep_largest_component(mask_u8)
    return mask_u8


# =========================
# Main (YOLO version)
# =========================
def detect_document(image_path, debug=True):
    original = cv2.imread(image_path)
    if original is None:
        raise Exception("Không đọc được ảnh")

    # giữ logic resize như demo10.py để ổn định + nhanh
    target_h = 640
    ratio = original.shape[0] / target_h
    small = cv2.resize(original, (int(original.shape[1] / ratio), target_h))

    label = classify_center_color(small)

    model = get_model()
    res = model.predict(small, imgsz=IMGSZ, conf=CONF, device=DEVICE, verbose=False)[0]

    if res.masks is None or res.boxes is None:
        raise Exception("YOLO không trả masks/boxes (không phát hiện giấy)")

    masks = res.masks.data.detach().cpu().numpy()   # (N,H,W)
    confs = res.boxes.conf.detach().cpu().numpy()   # (N,)

    H, W = small.shape[:2]
    best_i = pick_best_instance(masks, confs, H, W)
    if best_i is None:
        raise Exception("Không chọn được instance giấy hợp lệ")

    best_mask = (masks[best_i] > 0.5).astype(np.uint8) * 255

    best_mask = mask_postprocess(best_mask)
    
    # ===== DEBUG: overlay mask ngay sau YOLO =====
    if debug:
        overlay = small.copy()
        color = np.zeros_like(overlay)
        color[:, :] = (255, 0, 0)  # màu xanh dương (BGR) cho dễ thấy

        m = (best_mask > 0).astype(np.uint8)
        overlay = cv2.addWeighted(overlay, 1.0, color, 0.35, 0, dst=overlay)
        overlay[m == 0] = small[m == 0]  # chỉ tô vùng mask

        # vẽ thêm bbox YOLO cho chắc
        if res.boxes is not None:
            xyxy = res.boxes.xyxy.detach().cpu().numpy()[best_i].astype(int)
            cv2.rectangle(overlay, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 255), 2)
            cv2.putText(overlay, f"AI conf={float(confs[best_i]):.3f}", (xyxy[0], max(30, xyxy[1]-10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imwrite("debug_yolo_overlay.jpg", overlay)

    # ====== chọn quad tốt nhất bằng scoring (reuse logic bạn đã có) ======
    cands = []
    cands += quad_candidates_from_mask(best_mask)

    q_rect = minarea_box_from_mask(best_mask)
    if q_rect is not None:
        cands.append(q_rect)

    best_q, best_s, best_reason = None, -1e18, "n/a"
    for q in cands:
        s, reason = score_quad(best_mask, q)
        if s > best_s:
            best_s, best_q, best_reason = s, q, reason

    if best_q is None:
        raise Exception("YOLO có mask nhưng không suy được quad hợp lệ")

    if debug:
        dbg = small.copy()
        q = order_points(best_q.astype(np.float32)).astype(int).reshape(-1, 1, 2)
        cv2.polylines(dbg, [q], True, (0, 255, 0), 3)

        cv2.imwrite("debug_component_best.jpg", best_mask)
        cv2.imwrite("debug_detected_box.jpg", dbg)
        with open("debug_info.txt", "w", encoding="utf-8") as f:
            f.write(f"label={label}\n")
            f.write(f"best_reason={best_reason}\n")
            f.write(f"conf={float(confs[best_i]):.4f}\n")

    # scale quad về original (giống demo10.py)
    box_final = order_points(best_q.astype(np.float32))
    box_final *= ratio

    warped = four_point_transform(original, box_final)
    return warped, label


if __name__ == "__main__":
    warped, label = detect_document("input20.jpg", debug=False)
    out_name = f"{label}.jpg"
    cv2.imwrite(out_name, warped)
    print("Detect + Perspective thành công ->", out_name)