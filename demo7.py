import cv2
import numpy as np
from sklearn.cluster import KMeans


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


def line_angle(line):
    x1, y1, x2, y2 = line
    return np.arctan2((y2 - y1), (x2 - x1))


def line_length(line):
    x1, y1, x2, y2 = line
    return np.hypot(x2 - x1, y2 - y1)


def line_to_abc(line):
    x1,y1,x2,y2 = line
    a = y2 - y1
    b = x1 - x2
    c = a*x1 + b*y1
    return a,b,c


def line_distance(l1, l2):
    a1,b1,c1 = line_to_abc(l1)
    a2,b2,c2 = line_to_abc(l2)
    return abs(c1 - c2) / np.sqrt(a1*a1 + b1*b1)


def intersection(l1, l2):
    x1,y1,x2,y2 = l1
    x3,y3,x4,y4 = l2

    A1 = y2 - y1
    B1 = x1 - x2
    C1 = A1*x1 + B1*y1

    A2 = y4 - y3
    B2 = x3 - x4
    C2 = A2*x3 + B2*y3

    det = A1*B2 - A2*B1
    if abs(det) < 1e-6:
        return None

    x = (B2*C1 - B1*C2) / det
    y = (A1*C2 - A2*C1) / det
    return [x, y]


def process(image_path):

    original = cv2.imread(image_path)
    if original is None:
        raise Exception("Không đọc được ảnh")

    scale_height = 1000
    ratio = original.shape[0] / scale_height
    image = cv2.resize(original,
                       (int(original.shape[1] / ratio), scale_height))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)

    edges = cv2.Canny(gray, 50, 150)
    cv2.imwrite("debug_edges.jpg", edges)

    lines = cv2.HoughLinesP(edges,
                            rho=1,
                            theta=np.pi/180,
                            threshold=120,
                            minLineLength=200,
                            maxLineGap=30)

    if lines is None:
        raise Exception("Không tìm thấy line")

    lines = lines[:,0]

    # chỉ giữ line dài
    lines = [l for l in lines if line_length(l) > 300]

    if len(lines) < 4:
        raise Exception("Không đủ line dài")

    # debug
    debug = image.copy()
    for l in lines:
        x1,y1,x2,y2 = map(int,l)
        cv2.line(debug,(x1,y1),(x2,y2),(0,255,0),2)
    cv2.imwrite("debug_filtered_lines.jpg", debug)

    # cluster theo góc
    angles = np.array([line_angle(l) for l in lines])
    angles = np.mod(angles, np.pi)

    kmeans = KMeans(n_clusters=2, random_state=0).fit(angles.reshape(-1,1))
    labels = kmeans.labels_

    group1 = [lines[i] for i in range(len(lines)) if labels[i]==0]
    group2 = [lines[i] for i in range(len(lines)) if labels[i]==1]

    def extreme_pair(group):
        max_dist = 0
        best = None
        for i in range(len(group)):
            for j in range(i+1, len(group)):
                d = line_distance(group[i], group[j])
                if d > max_dist:
                    max_dist = d
                    best = (group[i], group[j])
        return best

    pair1 = extreme_pair(group1)
    pair2 = extreme_pair(group2)

    if pair1 is None or pair2 is None:
        raise Exception("Không tìm được 2 cặp line")

    l1,l2 = pair1
    l3,l4 = pair2

    pts = []
    for a in [l1,l2]:
        for b in [l3,l4]:
            pt = intersection(a,b)
            if pt is not None:
                pts.append(pt)

    if len(pts) != 4:
        raise Exception("Không đủ 4 giao điểm")

    pts = np.array(pts, dtype="float32")

    quad_debug = image.copy()
    for p in pts:
        cv2.circle(quad_debug, (int(p[0]),int(p[1])), 8, (0,0,255), -1)
    cv2.imwrite("debug_quad.jpg", quad_debug)

    pts *= ratio
    warped = four_point_transform(original, pts)
    cv2.imwrite("output.jpg", warped)

    print("Version 12 FIXED hoàn thành")


if __name__ == "__main__":
    process("input.jpg")