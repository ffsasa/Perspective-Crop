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
# Unicode-safe IO
# =========================
def imread_unicode(path):
    stream = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(stream, cv2.IMREAD_COLOR)
    return img


def imwrite_unicode(path, image):
    ext = os.path.splitext(path)[1]
    result, encoded = cv2.imencode(ext, image)
    if result:
        encoded.tofile(path)


# =========================
# Phân loại trắng (TK) / hồng (XX)
# =========================
def is_white_page(image):
    """
    Phân biệt dựa trên hue + saturation.
    Trang hồng có hue đỏ và saturation cao hơn rõ rệt.
    """

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]

    # vùng màu đỏ / hồng
    pink_mask = ((h < 15) | (h > 160)) & (s > 40)

    pink_ratio = np.sum(pink_mask) / pink_mask.size

    # nếu nhiều pixel hồng -> là XX
    if pink_ratio > 0.1:
        return False  # XX

    return True  # TK


# =========================
# Core xử lý 1 ảnh
# =========================
def process_image(input_path, output_dir):

    original = imread_unicode(input_path)
    if original is None:
        print(f"Không đọc được ảnh: {input_path}")
        return

    # ===== GIỮ NGUYÊN LOGIC PERSPECTIVE =====
    scale_height = 1200
    ratio = original.shape[0] / scale_height
    image = cv2.resize(original, (int(original.shape[1] / ratio), scale_height))
    image_blur = cv2.GaussianBlur(image, (5, 5), 0)

    mask = np.full(image.shape[:2], cv2.GC_PR_BGD, np.uint8)
    h, w = image.shape[:2]

    rect = (int(w * 0.05), int(h * 0.05),
            int(w * 0.9), int(h * 0.9))

    bgModel = np.zeros((1, 65), np.float64)
    fgModel = np.zeros((1, 65), np.float64)

    cv2.grabCut(image_blur, mask, rect, bgModel, fgModel, 5, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
        255, 0).astype('uint8')

    kernel = np.ones((5, 5), np.uint8)
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print(f"Không tìm thấy contour: {input_path}")
        return

    contour = max(contours, key=cv2.contourArea)

    if cv2.contourArea(contour) < 0.1 * (h * w):
        print(f"Contour quá nhỏ: {input_path}")
        return

    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

    if len(approx) == 4:
        quad = order_points(approx.reshape(4, 2))
    else:
        hull = cv2.convexHull(contour)
        rect = cv2.minAreaRect(hull)
        box = cv2.boxPoints(rect)
        quad = order_points(np.array(box, dtype="float32"))

    quad *= ratio

    warped = four_point_transform(original, quad)
    # ===== KẾT THÚC LOGIC GỐC =====

    # =========================
    # Đổi tên file theo nghiệp vụ
    # =========================
    if is_white_page(warped):
        filename = "TK.jpg"
    else:
        filename = "XX.jpg"

    final_path = os.path.join(output_dir, filename)

    # không được có 2 TK hoặc 2 XX
    if os.path.exists(final_path):
        print(f"LỖI LOGIC: {filename} đã tồn tại trong {output_dir}")
        print(f"Ảnh gây lỗi: {input_path}")
        return

    imwrite_unicode(final_path, warped)

    print(f"Đã xử lý: {input_path} -> {filename}")


# =========================
# Batch folder processor
# =========================
def process_folder(input_root, output_root):

    for root, dirs, files in os.walk(input_root):

        relative_path = os.path.relpath(root, input_root)
        output_dir = os.path.join(output_root, relative_path)

        os.makedirs(output_dir, exist_ok=True)

        image_files = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        for file in image_files:
            input_path = os.path.join(root, file)

            try:
                process_image(input_path, output_dir)
            except Exception as e:
                print(f"Lỗi với {input_path}: {e}")


# =========================
# RUN
# =========================
if __name__ == "__main__":

    input_folder = input("Nhập đường dẫn folder input: ").strip()
    output_folder = input("Nhập đường dẫn folder output: ").strip()

    process_folder(input_folder, output_folder)

    print("Hoàn thành toàn bộ batch processing.")