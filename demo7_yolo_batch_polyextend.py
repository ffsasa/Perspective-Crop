import os
import cv2
import numpy as np
from ultralytics import YOLO

# =========================
# CONFIG
# =========================
MODEL_PATH = r"C:\Users\Admin\Downloads\PerspectiveCrop.v1i.yolov8\runs\segment\train2\weights\best.pt"
DEVICE = 0          # 0 = GPU, "cpu" nếu muốn CPU
IMGSZ = 640
CONF = 0.25

# Resize ảnh trước khi YOLO để ổn định + nhanh (bám theo bản YOLO của bạn)
TARGET_H = 640

# Pad nhẹ để tránh cắt sát mép (bám theo demo7)
PAD_CFG = dict(
    pad_top=0.01,
    pad_bottom=0.01,
    pad_left=0.01,
    pad_right=0.01,
    max_px_tb=10,
    max_px_lr=10
)

# =========================
# Unicode-safe IO (GIỮ từ demo7)
# =========================
def imread_unicode(path):
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img

def imwrite_unicode(path, img):
    ext = os.path.splitext(path)[1].lower()
    if ext == "":
        ext = ".jpg"
        path = path + ext
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise Exception(f"Không thể encode ảnh để ghi: {path}")
    buf.tofile(path)

# =========================
# Utility (GIỮ)
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
    rect = order_points(pts.astype(np.float32))
    (tl, tr, br, bl) = rect

    # Tính vector cạnh
    top_vec    = tr - tl
    bottom_vec = br - bl
    left_vec   = bl - tl
    right_vec  = br - tr

    # Lấy độ dài trung bình cho mỗi cặp cạnh song song
    width  = (np.linalg.norm(top_vec) + np.linalg.norm(bottom_vec)) / 2.0
    height = (np.linalg.norm(left_vec) + np.linalg.norm(right_vec)) / 2.0

    # 🔥 Chuẩn hóa hình chữ nhật: ép 2 cạnh song song có cùng độ dài
    width  = int(width)
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
# Color label: TK (white) / XX (pink)  (GIỮ)
# =========================
def classify_center_color(bgr_img):
    h, w = bgr_img.shape[:2]

    cx1, cx2 = int(w * 0.40), int(w * 0.60)
    cy1, cy2 = int(h * 0.40), int(h * 0.60)
    patch = bgr_img[cy1:cy2, cx1:cx2]

    # Chuyển sang float
    patch = patch.astype("float32")

    B = patch[..., 0]
    G = patch[..., 1]
    R = patch[..., 2]

    # Đo độ lệch màu (distance giữa các kênh)
    color_variation = np.mean(
        np.abs(R - G) + np.abs(R - B) + np.abs(G - B)
    )

    # Nếu gần như không lệch màu → trắng → TK
    return "TK" if color_variation < 25 else "XX"

# =========================
# Mask helpers (từ YOLO bản của bạn)
# =========================
def keep_largest_component(binary_mask):
    num_labels, lab, stats, _ = cv2.connectedComponentsWithStats((binary_mask > 0).astype(np.uint8))
    if num_labels <= 1:
        return binary_mask
    idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    out = (lab == idx).astype(np.uint8) * 255
    return out

def mask_postprocess(mask_u8):
    mask_u8 = keep_largest_component(mask_u8)
    k = max(7, (min(mask_u8.shape[:2]) // 120) | 1)
    kernel = np.ones((k, k), np.uint8)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_u8 = keep_largest_component(mask_u8)
    return mask_u8



def get_main_contour(mask_u8):
    m = (mask_u8 > 0).astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)
# =========================
# Quad from mask (NEW): poly 4/5 -> recover missing corner by extending true edges
# =========================

# Geometry helpers (from demo10)
def line_from_two_points(p1, p2):
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    a = y2 - y1
    b = x1 - x2
    c = x2 * y1 - x1 * y2
    n = (a*a + b*b) ** 0.5 + 1e-6
    return (a/n, b/n, c/n)

def intersect_lines(l1, l2):
    a1, b1, c1 = l1
    a2, b2, c2 = l2
    det = a1*b2 - a2*b1
    if abs(det) < 1e-6:
        return None
    x = (b1*c2 - b2*c1) / det
    y = (a2*c1 - a1*c2) / det
    return np.array([x, y], dtype=np.float32)

def draw_line_infinite(img, p1, p2, color=(0, 255, 255), thickness=2):
    """Vẽ đường thẳng qua p1-p2 kéo dài cắt biên ảnh (để debug dễ nhìn)."""
    h, w = img.shape[:2]
    a, b, c = line_from_two_points(p1, p2)
    pts = []

    for x in [0, w - 1]:
        if abs(b) > 1e-6:
            y = -(a * x + c) / b
            if 0 <= y <= h - 1:
                pts.append((int(x), int(y)))

    for y in [0, h - 1]:
        if abs(a) > 1e-6:
            x = -(b * y + c) / a
            if 0 <= x <= w - 1:
                pts.append((int(x), int(y)))

    if len(pts) >= 2:
        cv2.line(img, pts[0], pts[1], color, thickness)

def interior_angle_deg(prev_pt, cur_pt, next_pt):
    v1 = prev_pt - cur_pt
    v2 = next_pt - cur_pt
    n1 = np.linalg.norm(v1) + 1e-6
    n2 = np.linalg.norm(v2) + 1e-6
    cs = float(np.dot(v1, v2) / (n1 * n2))
    cs = max(-1.0, min(1.0, cs))
    return float(np.degrees(np.arccos(cs)))

def are_adjacent(i, j, n):
    return (abs(i - j) == 1) or (abs(i - j) == n - 1)

def quad_from_mask_poly_extend(mask_u8, eps_ratio=0.01, max_poly=12):
    """
    - contour -> hull -> approx poly
    - poly 4: dùng luôn
    - poly 5: coi như 1 góc bị chém (chamfer), suy ra góc bị mất bằng kéo dài 2 cạnh thật kề chamfer
    Trả về: quad, poly, dbg_dict, reason
    """
    cnt = get_main_contour(mask_u8)
    if cnt is None or cv2.contourArea(cnt) < 800:
        return None, None, {}, "no_contour"

    hull = cv2.convexHull(cnt)
    peri = cv2.arcLength(hull, True)

    poly = cv2.approxPolyDP(hull, eps_ratio * peri, True).reshape(-1, 2).astype(np.float32)
    if len(poly) > max_poly:
        poly = cv2.approxPolyDP(hull, (eps_ratio * 2.0) * peri, True).reshape(-1, 2).astype(np.float32)

    n = len(poly)
    dbg = {"angles": None, "chamfer_idx": None, "missing_corner": None, "edge_lines": None}

    if n == 4:
        return order_points(poly), poly, dbg, "poly4_direct"

    if n != 5:
        rect = cv2.boxPoints(cv2.minAreaRect(hull)).astype(np.float32)
        return order_points(rect), poly, dbg, f"fallback_minAreaRect_poly_n={n}"

    # --- n == 5 ---
    angles = []
    for i in range(n):
        prev_pt = poly[(i - 1) % n]
        cur_pt  = poly[i]
        next_pt = poly[(i + 1) % n]
        angles.append(interior_angle_deg(prev_pt, cur_pt, next_pt))
    dbg["angles"] = angles

    idx_sorted = sorted(range(n), key=lambda i: angles[i], reverse=True)
    cand = idx_sorted[:3]

    best_pair = None
    best_score = -1e9
    for i in range(len(cand)):
        for j in range(i + 1, len(cand)):
            a, b = cand[i], cand[j]
            if not are_adjacent(a, b, n):
                continue
            edge_len = float(np.linalg.norm(poly[a] - poly[b]))
            score = angles[a] + angles[b] - 0.15 * edge_len  # ưu tiên góc lớn + cạnh chamfer ngắn
            if score > best_score:
                best_score = score
                best_pair = (a, b)

    if best_pair is None:
        for a in range(n):
            b = (a + 1) % n
            edge_len = float(np.linalg.norm(poly[a] - poly[b]))
            score = angles[a] + angles[b] - 0.15 * edge_len
            if score > best_score:
                best_score = score
                best_pair = (a, b)

    i1, i2 = best_pair
    dbg["chamfer_idx"] = (int(i1), int(i2))

    # neighbor thật của i1
    n1a = (i1 - 1) % n
    n1b = (i1 + 1) % n
    other1 = n1a if n1b == i2 else n1b

    # neighbor thật của i2
    n2a = (i2 - 1) % n
    n2b = (i2 + 1) % n
    other2 = n2a if n2b == i1 else n2b

    pA1, pA2 = poly[other1], poly[i1]
    pB1, pB2 = poly[i2], poly[other2]
    dbg["edge_lines"] = (pA1, pA2, pB1, pB2)

    inter = intersect_lines(line_from_two_points(pA1, pA2), line_from_two_points(pB1, pB2))
    if inter is None:
        rect = cv2.boxPoints(cv2.minAreaRect(hull)).astype(np.float32)
        return order_points(rect), poly, dbg, "poly5_extend_failed_parallel_fallback_minAreaRect"

    dbg["missing_corner"] = (float(inter[0]), float(inter[1]))

    corner_pts = [poly[k] for k in range(n) if k not in (i1, i2)]
    quad = np.vstack([np.array(corner_pts, dtype=np.float32), inter.reshape(1, 2)]).astype(np.float32)
    quad = order_points(quad)
    return quad, poly, dbg, "poly5_extend_recover_missing_corner"

# =========================
# Quad scoring & candidates
# (GIỮ theo bản YOLO của bạn)
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

def pca_box_from_mask(comp_mask):
    ys, xs = np.where(comp_mask > 0)
    if len(xs) < 200:
        return None
    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    mean = pts.mean(axis=0)
    centered = pts - mean
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eig(cov)
    order = np.argsort(-eigvals)
    v1 = eigvecs[:, order[0]].astype(np.float32)
    v2 = eigvecs[:, order[1]].astype(np.float32)
    v1 /= (np.linalg.norm(v1) + 1e-6)
    v2 /= (np.linalg.norm(v2) + 1e-6)

    proj1 = centered @ v1
    proj2 = centered @ v2
    min1, max1 = np.percentile(proj1, [1, 99])
    min2, max2 = np.percentile(proj2, [1, 99])

    # 4 góc của OBB theo PCA (fallback affine)
    tl = mean + v1 * min1 + v2 * min2
    tr = mean + v1 * max1 + v2 * min2
    br = mean + v1 * max1 + v2 * max2
    bl = mean + v1 * min1 + v2 * max2
    return np.array([tl, tr, br, bl], dtype=np.float32)


def quad_from_hull_corner_peaks(comp_mask, simplify_eps=0.01):
    """
    Tìm 4 'góc sắc' trên convex hull (projective-friendly).
    Không dùng Canny; chỉ dùng hull từ mask YOLO.
    """
    contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 1000:
        return None

    hull = cv2.convexHull(cnt)
    peri = cv2.arcLength(hull, True)
    poly = cv2.approxPolyDP(hull, simplify_eps * peri, True)  # giảm số điểm hull
    pts = poly.reshape(-1, 2).astype(np.float32)
    if len(pts) < 6:
        # quá ít điểm -> khó lấy "góc"
        return None

    # tính "độ sắc" theo góc (cos nhỏ -> góc nhọn hơn)
    sharp = []
    n = len(pts)
    for i in range(n):
        p0 = pts[i]
        p1 = pts[(i - 1) % n]
        p2 = pts[(i + 1) % n]
        v1 = p1 - p0
        v2 = p2 - p0
        n1 = np.linalg.norm(v1) + 1e-6
        n2 = np.linalg.norm(v2) + 1e-6
        cs = float(np.dot(v1, v2) / (n1 * n2))
        cs = max(-1.0, min(1.0, cs))
        # cs gần -1 => gần 180deg, cs gần 1 => gần 0deg
        # góc nhọn => cs lớn (gần 1). Ta lấy điểm có cs lớn nhất.
        sharp.append((cs, i))

    sharp.sort(reverse=True, key=lambda x: x[0])

    chosen = []
    min_sep = max(1, n // 6)  # tránh chọn 4 điểm sát nhau
    for _, idx in sharp:
        if all(min((idx - j) % n, (j - idx) % n) >= min_sep for j in chosen):
            chosen.append(idx)
        if len(chosen) == 4:
            break

    if len(chosen) < 4:
        return None

    quad = pts[chosen].astype(np.float32)
    return quad

def pad_box(box,
            pad_top=0.01, pad_bottom=0.01,
            pad_left=0.01, pad_right=0.01,
            max_px_tb=10, max_px_lr=10):
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
# YOLO model singleton
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

def clamp_quad_to_image(quad, w, h, margin=1):
    q = quad.copy()
    q[:, 0] = np.clip(q[:, 0], margin, w - 1 - margin)
    q[:, 1] = np.clip(q[:, 1], margin, h - 1 - margin)
    return q

# =========================
# Main detect (YOLO) - thay GrabCut của demo7
# =========================
def detect_document(image_path, debug=False):
    # Unicode-safe read
    original = imread_unicode(image_path)
    if original is None:
        raise Exception(f"Không đọc được ảnh: {image_path}")

    # Downscale để YOLO chạy ổn định
    target_h = TARGET_H
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
    # ---- quad from poly extend (4/5 corners) ----
    best_q, poly, dbgpoly, best_reason = quad_from_mask_poly_extend(best_mask, eps_ratio=0.01, max_poly=12)
    if best_q is None:
        raise Exception("YOLO có mask nhưng không suy được quad hợp lệ (poly extend fail)")


    # Optional debug output (ghi ra file cạnh nơi chạy script)
    if debug:
        overlay = small.copy()
        color = np.zeros_like(overlay)
        color[:, :] = (255, 0, 0)  # BGR
        m = (best_mask > 0).astype(np.uint8)
        overlay = cv2.addWeighted(overlay, 1.0, color, 0.35, 0, dst=overlay)
        overlay[m == 0] = small[m == 0]
        cv2.imwrite("debug_yolo_overlay.jpg", overlay)

        dbg = small.copy()
        q = order_points(best_q.astype(np.float32)).astype(int).reshape(-1, 1, 2)
        cv2.polylines(dbg, [q], True, (0, 255, 0), 3)
        cv2.imwrite("debug_detected_box.jpg", dbg)
        cv2.imwrite("debug_component_best.jpg", best_mask)

    # debug_fit_edges: vẽ poly + 2 đường kéo dài + giao điểm missing + 4 góc
    dbg2 = small.copy()
    if poly is not None and len(poly) >= 3:
        cv2.polylines(dbg2, [poly.astype(int).reshape(-1, 1, 2)], True, (255, 0, 0), 2)

    if dbgpoly is not None and isinstance(dbgpoly, dict) and dbgpoly.get("edge_lines", None) is not None:
        pA1, pA2, pB1, pB2 = dbgpoly["edge_lines"]
        draw_line_infinite(dbg2, pA1, pA2, (0, 255, 255), 2)
        draw_line_infinite(dbg2, pB1, pB2, (0, 255, 255), 2)

    if dbgpoly is not None and isinstance(dbgpoly, dict) and dbgpoly.get("missing_corner", None) is not None:
        mx, my = dbgpoly["missing_corner"]
        cv2.circle(dbg2, (int(mx), int(my)), 8, (0, 165, 255), -1)  # cam

    for p in order_points(best_q.astype(np.float32)):
        cv2.circle(dbg2, (int(p[0]), int(p[1])), 6, (0, 0, 255), -1)

    cv2.imwrite("debug_fit_edges.jpg", dbg2)

    with open("debug_info.txt", "w", encoding="utf-8") as f:
        f.write(f"label={label}\\n")
        f.write(f"best_reason={best_reason}\\n")
        f.write(f"conf={float(confs[best_i]):.4f}\\n")
        f.write(f"poly_n={(len(poly) if poly is not None else -1)}\\n")
        if dbgpoly is not None and isinstance(dbgpoly, dict):
            if dbgpoly.get("angles", None) is not None:
                f.write(f"poly_angles_deg={dbgpoly['angles']}\\n")
            if dbgpoly.get("chamfer_idx", None) is not None:
                f.write(f"chamfer_idx={dbgpoly['chamfer_idx']}\\n")
            if dbgpoly.get("missing_corner", None) is not None:
                f.write(f"missing_corner_xy={dbgpoly['missing_corner']}\\n")
    # Scale quad back to original + pad + warp
    box_final = order_points(best_q.astype(np.float32))
    box_final *= ratio
    box_final = pad_box(box_final, **PAD_CFG)

    h0, w0 = original.shape[:2]
    box_final = clamp_quad_to_image(box_final, w0, h0, margin=2)
    warped = four_point_transform(original, box_final)
    return warped, label

# =========================
# Batch folder IO (GIỮ từ demo7)
# =========================
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def is_image_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in VALID_EXTS

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def process_folder(input_root: str, output_root: str):
    input_root = os.path.abspath(input_root)
    output_root = os.path.abspath(output_root)
    ensure_dir(output_root)

    ok = 0
    err = 0
    errors = []

    for dirpath, _, filenames in os.walk(input_root):
        rel = os.path.relpath(dirpath, input_root)
        out_dir = output_root if rel == "." else os.path.join(output_root, rel)
        ensure_dir(out_dir)

        image_files = [
            os.path.join(dirpath, fn)
            for fn in filenames
            if is_image_file(os.path.join(dirpath, fn))
        ]
        if not image_files:
            continue

        tk_count = 0
        xx_count = 0

        for in_path in image_files:
            try:
                warped, label = detect_document(in_path, debug=False)

                if label == "TK":
                    tk_count += 1
                    if tk_count == 1:
                        out_name = "TK.jpg"
                    elif tk_count == 2:
                        out_name = "TK2.jpg"
                    else:
                        raise Exception("Folder có nhiều hơn 2 ảnh TK")

                elif label == "XX":
                    xx_count += 1
                    if xx_count == 1:
                        out_name = "XX.jpg"
                    else:
                        raise Exception("Folder có nhiều hơn 1 ảnh XX")
                else:
                    raise Exception(f"Nhãn lạ: {label}")

                out_path = os.path.join(out_dir, out_name)
                if os.path.exists(out_path):
                    raise FileExistsError(f"Đã tồn tại {out_path}")

                imwrite_unicode(out_path, warped)

                ok += 1
                print(f"[OK] {in_path} -> {out_path}")

            except Exception as e:
                err += 1
                errors.append((in_path, str(e)))
                print(f"[ERR] {in_path} :: {e}")

    print("\n===== TỔNG KẾT =====")
    print(f"OK: {ok}")
    print(f"Lỗi: {err}")
    if errors:
        print("\nDanh sách lỗi:")
        for p, msg in errors:
            print(f"- {p}: {msg}")

if __name__ == "__main__":
    input_folder = input("Nhập đường dẫn folder input: ").strip().strip('"').strip("'")
    output_folder = input("Nhập đường dẫn folder output: ").strip().strip('"').strip("'")

    if not os.path.isdir(input_folder):
        print("Folder input không tồn tại.")
        raise SystemExit(1)

    process_folder(input_folder, output_folder)
