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
# Gaussian model helpers 
# ========================= 
def fit_gaussian(X: np.ndarray, eps: float = 1e-3): 
    """ 
    X: (N, D) float32 
    returns mean (D,), inv_cov (D,D), log_det_cov (float) 
    """ 
    mu = X.mean(axis=0) 
    Xc = X - mu 
    cov = (Xc.T @ Xc) / max(1, (X.shape[0] - 1)) 
    cov = cov + np.eye(cov.shape[0], dtype=np.float32) * eps 

    sign, logdet = np.linalg.slogdet(cov.astype(np.float64)) 
    if sign <= 0:
        # fallback: thêm regularization mạnh hơn
        cov = cov + np.eye(cov.shape[0], dtype=np.float32) * (eps * 10) 
        sign, logdet = np.linalg.slogdet(cov.astype(np.float64)) 

    inv_cov = np.linalg.inv(cov.astype(np.float64)).astype(np.float32) 
    return mu.astype(np.float32), inv_cov, float(logdet) 

def gaussian_logpdf(X: np.ndarray, mu: np.ndarray, inv_cov: np.ndarray, logdet: float): 
    """ 
    X: (N, D) 
    """ 
    D = X.shape[1] 
    Xc = X - mu 
    # mahalanobis: (x-mu)^T inv_cov (x-mu) 
    m = np.sum((Xc @ inv_cov) * Xc, axis=1) 
    return -0.5 * (m + logdet + D * np.log(2.0 * np.pi)) 

# ========================= 
# Region-based pipeline (NO EDGE, NO THRESHOLD) 
# ========================= 
def detect_document(image_path, debug=True): 
    original = cv2.imread(image_path) 
    if original is None: 
        raise Exception("Không đọc được ảnh") 
    
    # ---- Downscale để GrabCut chạy nhanh 
    target_h = 700 
    # bạn có thể thử 600/800 tùy máy 
    ratio = original.shape[0] / target_h 
    small = cv2.resize(original, (int(original.shape[1] / ratio), target_h)) 
    
    h, w = small.shape[:2] 
    
    # ---- Init mask cho GrabCut
    mask = np.zeros((h, w), np.uint8) 
    
    # ---- Rect gần full ảnh nhưng chừa margin 
    mx = int(w * 0.05) 
    my = int(h * 0.05) 
    rect = (mx, my, w - 2 * mx, h - 2 * my) 
    
    # Models required by grabCut 
    bgdModel = np.zeros((1, 65), np.float64) 
    fgdModel = np.zeros((1, 65), np.float64) 
    
    # ---- GrabCut-lite: 1 iteration (rất nhanh) 
    # mode = INIT_WITH_RECT 
    cv2.grabCut(small, mask, rect, bgdModel, fgdModel, 1, cv2.GC_INIT_WITH_RECT) 
    
    # ---- foreground mask (probable/definite FG) 
    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8) 
    
    # ---- Morph close để nối đáy/viền bị đứt
    k = max(9, (min(h, w) // 80) | 1) 
    kernel = np.ones((k, k), np.uint8) 
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=2) 
    
    # ---- Giữ component lớn nhất (hoặc component chứa tâm) 
    num_labels, lab_cc, stats, _ = cv2.connectedComponentsWithStats(fg) 
    if num_labels <= 1: 
        raise Exception("GrabCut không tách được foreground") 
    
    center_label = lab_cc[h // 2, w // 2] 
    if center_label == 0: 
        # fallback: lấy largest non-zero 
        center_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]) 
        
    comp = (lab_cc == center_label).astype(np.uint8) * 255 
    
    # ---- Contour 
    contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    if not contours: 
        raise Exception("Không tìm thấy contour") 
    
    cnt = max(contours, key=cv2.contourArea) 
    hull = cv2.convexHull(cnt) 
    
    rect = cv2.minAreaRect(hull) 
    box = cv2.boxPoints(rect).astype(np.float32) 
    
    # ---- Debug 
    if debug: 
        dbg = small.copy() 
        cv2.drawContours(dbg, [box.astype(int)], -1, (0, 255, 0), 3) 
        cv2.imwrite("debug_mask_best.jpg", fg) 
        cv2.imwrite("debug_component_best.jpg", comp) 
        cv2.imwrite("debug_detected_box.jpg", dbg) 
        
    # ---- Scale box về ảnh gốc và warp 
    box *= ratio 
    warped = four_point_transform(original, box) 
    
    return warped 

# ========================= 
# RUN 
# ========================= 
if __name__ == "__main__": 
    output = detect_document("input3.jpg", debug=True) 
    cv2.imwrite("output.jpg", output) 
    print("Detect + Perspective thành công")