import cv2
import numpy as np
import os


# =========================
# Unicode-safe IO (giống demo4.6)
# =========================
def imread_unicode(path):
    stream = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(stream, cv2.IMREAD_COLOR)
    return img


def imwrite_unicode(path, image):
    ext = os.path.splitext(path)[1]
    if ext == "":
        ext = ".jpg"
        path = path + ext
    ok, encoded = cv2.imencode(ext, image)
    if ok:
        encoded.tofile(path)


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
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array(
        [[0, 0],
         [maxWidth - 1, 0],
         [maxWidth - 1, maxHeight - 1],
         [0, maxHeight - 1]], dtype="float32"
    )

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


# =========================
# Original demo8 logic (KHÔNG SỬA LOGIC)
# =========================
def detect_document(image_path, debug=False):
    """
    Logic xử lý ảnh lấy nguyên từ demo8.
    Chỉ đổi IO đọc ảnh sang imread_unicode để hỗ trợ path Unicode.
    """
    image = imread_unicode(image_path)
    if image is None:
        raise ValueError(f"Không đọc được ảnh: {image_path}")

    ori = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edged = cv2.Canny(gray, 75, 200)

    if debug:
        cv2.imshow("Edged", edged)
        cv2.waitKey(0)

    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    screenCnt = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            screenCnt = approx
            break

    if screenCnt is None:
        raise ValueError("Không tìm thấy contour 4 góc (screenCnt=None)")

    if debug:
        tmp = ori.copy()
        cv2.drawContours(tmp, [screenCnt], -1, (0, 255, 0), 2)
        cv2.imshow("Outline", tmp)
        cv2.waitKey(0)

    warped = four_point_transform(ori, screenCnt.reshape(4, 2))

    # =========================
    # --- Demo8 segment + classify logic (giữ nguyên) ---
    # =========================
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, 0])
    upper = np.array([180, 255, 200])
    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    if debug:
        cv2.imshow("Mask", mask)
        cv2.waitKey(0)

    # GrabCut refinement
    grab_mask = np.where(mask > 0, cv2.GC_PR_FGD, cv2.GC_BGD).astype("uint8")
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    rect = (1, 1, warped.shape[1] - 2, warped.shape[0] - 2)
    cv2.grabCut(warped, grab_mask, rect, bgdModel, fgdModel, 3, cv2.GC_INIT_WITH_RECT)

    resultMask = np.where(
        (grab_mask == cv2.GC_FGD) | (grab_mask == cv2.GC_PR_FGD),
        255, 0
    ).astype("uint8")

    if debug:
        cv2.imshow("GrabCutMask", resultMask)
        cv2.waitKey(0)

    # largest contour on mask
    cnts2, _ = cv2.findContours(resultMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts2:
        raise ValueError("Không tìm thấy contour trên mask sau GrabCut")

    c2 = max(cnts2, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c2)
    cropped = warped[y:y + h, x:x + w].copy()

    if debug:
        cv2.imshow("Cropped", cropped)
        cv2.waitKey(0)

    # Heuristic classify
    H, W = cropped.shape[:2]
    ratio = W / max(1, H)

    # Determine card type: "TK" or "XX"
    # (Giữ nguyên như demo8)
    if ratio > 1.2:
        card_type = "TK"   # LARGE-ish
    else:
        card_type = "XX"   # SMALL-ish

    # Resize / pad logic (giữ nguyên)
    if card_type == "TK":
        target_w, target_h = 1600, 1000
    else:
        target_w, target_h = 1000, 1600

    # Fit into target with padding
    scale = min(target_w / W, target_h / H)
    newW, newH = int(W * scale), int(H * scale)
    resized = cv2.resize(cropped, (newW, newH), interpolation=cv2.INTER_AREA)

    canvas = np.ones((target_h, target_w, 3), dtype=np.uint8) * 255
    offsetX = (target_w - newW) // 2
    offsetY = (target_h - newH) // 2
    canvas[offsetY:offsetY + newH, offsetX:offsetX + newW] = resized

    if debug:
        cv2.imshow("Final", canvas)
        cv2.waitKey(0)

    return card_type, canvas


# =========================
# Batch IO giống demo4.6 (chỉ thêm IO, không sửa logic xử lý ảnh)
# =========================
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _is_image_file(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in VALID_EXTS


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _check_duplicate_outputs(out_dir: str, card_type: str):
    """
    Demo4.6 có rule: không được có 2 TK hoặc 2 XX trong cùng folder output.
    Nếu tồn tại file output tương ứng thì báo lỗi.
    """
    out_path = os.path.join(out_dir, f"{card_type}.jpg")
    if os.path.exists(out_path):
        raise FileExistsError(f"Đã tồn tại {out_path} (trùng loại {card_type} trong cùng folder output)")


def process_image(input_path: str, output_dir: str, debug: bool = False):
    """
    Xử lý 1 ảnh -> xuất {TK|XX}.jpg trong output_dir
    """
    _ensure_dir(output_dir)
    card_type, out_img = detect_document(input_path, debug=debug)

    # rule: không overwrite 2 TK / 2 XX trong cùng folder output
    _check_duplicate_outputs(output_dir, card_type)

    out_path = os.path.join(output_dir, f"{card_type}.jpg")
    imwrite_unicode(out_path, out_img)
    return card_type, out_path


def process_folder(input_root: str, output_root: str, debug: bool = False):
    """
    Duyệt toàn bộ ảnh trong input_root (kể cả subfolder),
    giữ nguyên cấu trúc thư mục ở output_root (giống demo4.6).
    """
    input_root = os.path.abspath(input_root)
    output_root = os.path.abspath(output_root)
    _ensure_dir(output_root)

    results = []
    errors = []

    for dirpath, _, filenames in os.walk(input_root):
        rel = os.path.relpath(dirpath, input_root)
        out_dir = os.path.join(output_root, rel) if rel != "." else output_root

        for fn in filenames:
            in_path = os.path.join(dirpath, fn)
            if not _is_image_file(in_path):
                continue

            try:
                card_type, out_path = process_image(in_path, out_dir, debug=debug)
                results.append((in_path, card_type, out_path))
                print(f"[OK] {in_path} -> {out_path}")
            except Exception as e:
                errors.append((in_path, str(e)))
                print(f"[ERR] {in_path} :: {e}")

    return results, errors


# =========================
# CLI giống demo4.6: input folder -> output folder
# =========================
if __name__ == "__main__":
    input_folder = input("Nhập đường dẫn folder input: ").strip().strip('"').strip("'")
    output_folder = input("Nhập đường dẫn folder output: ").strip().strip('"').strip("'")
    dbg = input("Debug? (y/n, mặc định n): ").strip().lower()
    debug = (dbg == "y" or dbg == "yes")

    if not os.path.isdir(input_folder):
        print("Folder input không tồn tại.")
        raise SystemExit(1)

    _ensure_dir(output_folder)

    results, errors = process_folder(input_folder, output_folder, debug=debug)

    print("\n===== TỔNG KẾT =====")
    print(f"OK: {len(results)}")
    print(f"Lỗi: {len(errors)}")

    if errors:
        print("\nDanh sách lỗi:")
        for p, msg in errors:
            print(f"- {p}: {msg}")