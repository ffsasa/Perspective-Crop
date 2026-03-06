import cv2
import numpy as np
from ultralytics import YOLO

# =========================
# CONFIG
# =========================
MODEL_PATH = r"C:\Users\Admin\Downloads\PerspectiveCrop.v1i.yolov8\runs\segment\train2\weights\best.pt"
DEVICE = 0
IMGSZ = 640
CONF = 0.25
TARGET_H = 640

PAD_CFG = dict(
    pad_top=0.05,
    pad_bottom=0.05,
    pad_left=0.025,
    pad_right=0.025,
    max_px_tb=50,
    max_px_lr=50
)

# =========================
# IO
# =========================
def imread_unicode(path):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

# =========================
# Utility
# =========================
def order_points(pts):
    pts = pts.astype(np.float32)
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]      # TL
    rect[2] = pts[np.argmax(s)]      # BR
    rect[1] = pts[np.argmin(diff)]   # TR
    rect[3] = pts[np.argmax(diff)]   # BL
    return rect

def clamp_quad_to_image(quad, w, h, margin=1):
    q = quad.copy()
    q[:, 0] = np.clip(q[:, 0], margin, w - 1 - margin)
    q[:, 1] = np.clip(q[:, 1], margin, h - 1 - margin)
    return q

def pad_box(box,
            pad_top=0.01, pad_bottom=0.01,
            pad_left=0.01, pad_right=0.01,
            max_px_tb=30, max_px_lr=30):
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

def four_point_transform(image, pts):
    rect = order_points(pts.astype(np.float32))
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
# Label
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
    return ("XX" if float(np.mean(pink_mask)) > 0.10 else "TK")

# =========================
# Mask helpers
# =========================
def keep_largest_component(binary_mask):
    num_labels, lab, stats, _ = cv2.connectedComponentsWithStats((binary_mask > 0).astype(np.uint8))
    if num_labels <= 1:
        return binary_mask
    idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (lab == idx).astype(np.uint8) * 255

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
# Geometry helpers
# =========================
def line_from_two_points(p1, p2):
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    a = y2 - y1
    b = x1 - x2
    c = x2 * y1 - x1 * y2
    n = (a*a + b*b) ** 0.5 + 1e-6
    return (a/n, b/n, c/n)

def intersect_abc(l1, l2):
    a1, b1, c1 = l1
    a2, b2, c2 = l2
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-6:
        return None
    x = (b1 * c2 - b2 * c1) / det
    y = (a2 * c1 - a1 * c2) / det
    return np.array([x, y], dtype=np.float32)

def draw_line_infinite(img, p1, p2, color=(0, 255, 255), thickness=2):
    """Vẽ đường thẳng qua p1-p2 kéo dài cắt biên ảnh (để debug dễ nhìn)."""
    h, w = img.shape[:2]
    l = line_from_two_points(p1, p2)
    a, b, c = l
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
    """Góc trong tại cur_pt theo polygon order."""
    v1 = prev_pt - cur_pt
    v2 = next_pt - cur_pt
    n1 = np.linalg.norm(v1) + 1e-6
    n2 = np.linalg.norm(v2) + 1e-6
    cs = float(np.dot(v1, v2) / (n1 * n2))
    cs = max(-1.0, min(1.0, cs))
    return float(np.degrees(np.arccos(cs)))

def are_adjacent(i, j, n):
    return (abs(i - j) == 1) or (abs(i - j) == n - 1)

# =========================
# Core: 4 corners from convex poly (4 or 5 vertices)
# =========================
def quad_from_mask_poly_extend(mask_u8,
                              eps_ratio=0.01,
                              max_poly=10):
    """
    - Lấy contour -> hull -> approx poly.
    - Nếu poly 4 đỉnh: dùng trực tiếp.
    - Nếu poly 5 đỉnh: coi 1 góc bị "chém" (chamfer):
        + 2 đỉnh chamfer endpoints thường có góc trong lớn nhất (~135)
        + lấy 2 cạnh "thật" kề chamfer, kéo dài và giao nhau -> góc bị mất
        + quad = 3 góc còn lại + góc suy ra
    Trả về: quad, poly, debug_dict, reason
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
        # fallback: nếu polygon phức tạp (nhiều hơn 5), dùng minAreaRect để không crash,
        # nhưng vẫn ghi rõ reason để bạn biết.
        rect = cv2.boxPoints(cv2.minAreaRect(hull)).astype(np.float32)
        return order_points(rect), poly, dbg, f"fallback_minAreaRect_poly_n={n}"

    # --- n == 5: tìm 2 đỉnh chamfer endpoints bằng góc trong lớn nhất ---
    angles = []
    for i in range(n):
        prev_pt = poly[(i - 1) % n]
        cur_pt = poly[i]
        next_pt = poly[(i + 1) % n]
        angles.append(interior_angle_deg(prev_pt, cur_pt, next_pt))

    dbg["angles"] = angles

    # lấy top indices theo angle giảm dần
    idx_sorted = sorted(range(n), key=lambda i: angles[i], reverse=True)

    # chọn 1 cặp adjacent tốt nhất trong top3 (để chắc chắn là 2 đầu mút chamfer)
    cand = idx_sorted[:3]
    best_pair = None
    best_score = -1e9
    for i in range(len(cand)):
        for j in range(i + 1, len(cand)):
            a, b = cand[i], cand[j]
            if not are_adjacent(a, b, n):
                continue
            # ưu tiên tổng angle lớn, và edge giữa chúng ngắn (thường chamfer edge ngắn)
            edge_len = float(np.linalg.norm(poly[a] - poly[b]))
            score = angles[a] + angles[b] - 0.15 * edge_len
            if score > best_score:
                best_score = score
                best_pair = (a, b)

    if best_pair is None:
        # nếu không tìm được cặp adjacent trong top3, thử toàn bộ cặp adjacent
        for a in range(n):
            b = (a + 1) % n
            edge_len = float(np.linalg.norm(poly[a] - poly[b]))
            score = angles[a] + angles[b] - 0.15 * edge_len
            if score > best_score:
                best_score = score
                best_pair = (a, b)

    i1, i2 = best_pair
    dbg["chamfer_idx"] = (int(i1), int(i2))

    # xác định neighbor "không phải chamfer" cho mỗi endpoint
    # i1 neighbors: (i1-1) và (i1+1). Một trong hai là i2 (chamfer edge), còn lại là neighbor thật.
    n1a = (i1 - 1) % n
    n1b = (i1 + 1) % n
    other1 = n1a if n1b == i2 else n1b

    n2a = (i2 - 1) % n
    n2b = (i2 + 1) % n
    other2 = n2a if n2b == i1 else n2b

    # 2 cạnh thật để kéo dài:
    # line1 qua poly[other1] -> poly[i1]
    # line2 qua poly[i2] -> poly[other2]
    pA1, pA2 = poly[other1], poly[i1]
    pB1, pB2 = poly[i2], poly[other2]

    L1 = line_from_two_points(pA1, pA2)
    L2 = line_from_two_points(pB1, pB2)
    inter = intersect_abc(L1, L2)

    dbg["edge_lines"] = (pA1, pA2, pB1, pB2)

    if inter is None:
        rect = cv2.boxPoints(cv2.minAreaRect(hull)).astype(np.float32)
        return order_points(rect), poly, dbg, "poly5_extend_failed_parallel_fallback_minAreaRect"

    dbg["missing_corner"] = (float(inter[0]), float(inter[1]))

    # 3 góc còn lại là các đỉnh không thuộc chamfer endpoints
    corner_pts = [poly[k] for k in range(n) if k not in (i1, i2)]
    # quad = 3 corners + missing
    quad = np.vstack([np.array(corner_pts, dtype=np.float32), inter.reshape(1, 2)]).astype(np.float32)
    quad = order_points(quad)
    return quad, poly, dbg, "poly5_extend_recover_missing_corner"

# =========================
# YOLO model
# =========================
_yolo_model = None
def get_model():
    global _yolo_model
    if _yolo_model is None:
        _yolo_model = YOLO(MODEL_PATH)
    return _yolo_model

def pick_best_instance(masks_np, confs_np, H, W):
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

# =========================
# Main
# =========================
def detect_document(image_path, debug=True):
    original = imread_unicode(image_path)
    if original is None:
        raise Exception(f"Không đọc được ảnh: {image_path}")

    ratio = original.shape[0] / TARGET_H
    small = cv2.resize(original, (int(original.shape[1] / ratio), TARGET_H))
    label = classify_center_color(small)

    model = get_model()
    res = model.predict(small, imgsz=IMGSZ, conf=CONF, device=DEVICE, verbose=False)[0]
    if res.masks is None or res.boxes is None:
        raise Exception("YOLO không trả masks/boxes (không phát hiện giấy)")

    masks = res.masks.data.detach().cpu().numpy()
    confs = res.boxes.conf.detach().cpu().numpy()

    H, W = small.shape[:2]
    best_i = pick_best_instance(masks, confs, H, W)
    if best_i is None:
        raise Exception("Không chọn được instance giấy hợp lệ")

    best_mask = (masks[best_i] > 0.5).astype(np.uint8) * 255
    best_mask = mask_postprocess(best_mask)

    # ---- quad from poly extend (4/5 corners) ----
    quad, poly, dbgpoly, reason = quad_from_mask_poly_extend(best_mask, eps_ratio=0.01, max_poly=12)
    if quad is None:
        raise Exception("Không tìm được quad hợp lệ từ mask")

    # ==== Debug ====
    if debug:
        # debug_yolo_overlay + debug_component_best
        overlay = small.copy()
        color = np.zeros_like(overlay); color[:, :] = (255, 0, 0)
        m = (best_mask > 0).astype(np.uint8)
        overlay = cv2.addWeighted(overlay, 1.0, color, 0.35, 0, dst=overlay)
        overlay[m == 0] = small[m == 0]
        xyxy = res.boxes.xyxy.detach().cpu().numpy()[best_i].astype(int)
        cv2.rectangle(overlay, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 255), 2)
        cv2.imwrite("debug_yolo_overlay.jpg", overlay)
        cv2.imwrite("debug_component_best.jpg", best_mask)

        # debug_detected_box: quad xanh lá
        dbg_box = small.copy()
        qd = order_points(quad).astype(int).reshape(-1, 1, 2)
        cv2.polylines(dbg_box, [qd], True, (0, 255, 0), 3)
        cv2.imwrite("debug_detected_box.jpg", dbg_box)

        # debug_fit_edges: vẽ poly + 2 đường kéo dài + giao điểm missing + 4 góc
        dbg2 = small.copy()
        if poly is not None and len(poly) >= 3:
            cv2.polylines(dbg2, [poly.astype(int).reshape(-1, 1, 2)], True, (255, 0, 0), 2)

        # nếu poly_n=5 và có edge_lines thì vẽ đường kéo dài
        if dbgpoly.get("edge_lines", None) is not None:
            pA1, pA2, pB1, pB2 = dbgpoly["edge_lines"]
            draw_line_infinite(dbg2, pA1, pA2, (0, 255, 255), 2)
            draw_line_infinite(dbg2, pB1, pB2, (0, 255, 255), 2)

        # vẽ missing corner (nếu có)
        if dbgpoly.get("missing_corner", None) is not None:
            mx, my = dbgpoly["missing_corner"]
            cv2.circle(dbg2, (int(mx), int(my)), 8, (0, 165, 255), -1)  # cam

        # vẽ 4 góc quad (đỏ)
        for p in order_points(quad):
            cv2.circle(dbg2, (int(p[0]), int(p[1])), 6, (0, 0, 255), -1)

        cv2.imwrite("debug_fit_edges.jpg", dbg2)

        with open("debug_info.txt", "w", encoding="utf-8") as f:
            f.write(f"label={label}\n")
            f.write(f"reason={reason}\n")
            f.write(f"conf={float(confs[best_i]):.4f}\n")
            f.write(f"poly_n={(len(poly) if poly is not None else -1)}\n")
            if dbgpoly.get("angles", None) is not None:
                f.write(f"poly_angles_deg={dbgpoly['angles']}\n")
            if dbgpoly.get("chamfer_idx", None) is not None:
                f.write(f"chamfer_idx={dbgpoly['chamfer_idx']}\n")
            if dbgpoly.get("missing_corner", None) is not None:
                f.write(f"missing_corner_xy={dbgpoly['missing_corner']}\n")
            f.write(f"quad_angles={_quad_angles_deg(quad)}\n")
            f.write(f"quad={order_points(quad).tolist()}\n")

    # ==== Scale + pad + clamp + warp ====
    box_final = order_points(quad) * ratio
    # box_final = pad_box(box_final, **PAD_CFG)
    h0, w0 = original.shape[:2]
    box_final = clamp_quad_to_image(box_final, w0, h0, margin=2)

    warped = four_point_transform(original, box_final)
    return warped, label


if __name__ == "__main__":
    warped, label = detect_document("input6.jpg", debug=True)
    out_name = f"{label}.jpg"
    cv2.imwrite(out_name, warped)
    print("Detect + Perspective thành công ->", out_name)