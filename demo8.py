import cv2
import numpy as np
import os

# Utility: sắp xếp 4 điểm
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

# Perspective transform
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

# Main pipeline - Optimized GrabCut version
def process(image_path, output_path="output.jpg"):
    original = cv2.imread(image_path)
    if original is None:
        raise Exception("Không đọc được ảnh")

    os.makedirs("debug", exist_ok=True)

    # 1️⃣ Resize nhỏ hơn để nhanh (600 thay 800)
    scale_height = 600
    ratio = original.shape[0] / scale_height
    image = cv2.resize(original, (int(original.shape[1] / ratio), scale_height))
    cv2.imwrite("debug/0_resized.jpg", image)

    # 2️⃣ GrabCut optimized: iter=2 (thay 5), init rect sát hơn
    mask = np.zeros(image.shape[:2], np.uint8)
    bgModel = np.zeros((1, 65), np.float64)
    fgModel = np.zeros((1, 65), np.float64)

    # Rect init: sát trung tâm 90% ảnh (giảm computation)
    h, w = image.shape[:2]
    rect = (int(w*0.05), int(h*0.05), int(w*0.9), int(h*0.9))

    cv2.grabCut(image, mask, rect, bgModel, fgModel, 2, cv2.GC_INIT_WITH_RECT)  # Giảm iter=2

    grab_mask = np.where((mask == 2) | (mask == 0), 0, 255).astype('uint8')
    cv2.imwrite("debug/1_grabcut_mask.jpg", grab_mask)

    # 3️⃣ Morph nhẹ để nhanh
    kernel = np.ones((3, 3), np.uint8)  # Kernel nhỏ
    grab_mask = cv2.morphologyEx(grab_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    grab_mask = cv2.morphologyEx(grab_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    cv2.imwrite("debug/2_mask_clean.jpg", grab_mask)

    # 4️⃣ Contour largest
    contours, _ = cv2.findContours(grab_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("Không tìm thấy contour, fallback to full image")
        warped = original
    else:
        contour = max(contours, key=cv2.contourArea)
        area_ratio = cv2.contourArea(contour) / (h * w)
        print(f"Contour area ratio: {area_ratio}")
        if area_ratio < 0.05:
            print("Contour nhỏ, fallback to full image")
            warped = original
        else:
            contour_debug = image.copy()
            cv2.drawContours(contour_debug, [contour], -1, (0, 255, 0), 3)
            cv2.imwrite("debug/3_contour.jpg", contour_debug)

            # 5️⃣ Quad: approx với epsilon cao hơn để ép sát
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.03 * peri, True)  # Epsilon 3% để sát
            if len(approx) == 4:
                quad = order_points(approx.reshape(4, 2))
            else:
                hull = cv2.convexHull(contour)
                rect = cv2.minAreaRect(hull)
                box = cv2.boxPoints(rect)
                quad = order_points(np.array(box, dtype="float32"))

            # Shrink inward 1% để loại thừa
            center = np.mean(quad, axis=0)
            quad = quad - 0.01 * (quad - center)

            # Debug quad
            quad_debug = image.copy()
            for p in quad:
                cv2.circle(quad_debug, (int(p[0]), int(p[1])), 10, (0, 0, 255), -1)
            cv2.imwrite("debug/4_quad.jpg", quad_debug)

            # 6️⃣ Scale + warp
            quad *= ratio
            warped = four_point_transform(original, quad)

    cv2.imwrite(output_path, warped)
    print("Cropped thành công:", output_path)

# RUN
if __name__ == "__main__":
    process("input.jpg")