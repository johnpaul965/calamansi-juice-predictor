import cv2
import numpy as np

# ─────────────────────────────────────────
# FEATURE EXTRACTION MODULE
# Calamansi Juice Yield Prediction System
# ─────────────────────────────────────────

PIXELS_PER_CM   = 41.0
MIN_CIRCULARITY = 0.25
MIN_DIAMETER_CM = 1.5
MAX_DIAMETER_CM = 5.5

FEATURE_COLS = [
    'area_cm2', 'diameter_cm', 'perimeter_cm', 'circularity',
    'estimated_volume_cm3', 'mean_hue', 'mean_saturation', 'mean_value'
]


def _square_crop(image_array):
    h, w = image_array.shape[:2]
    s  = min(h, w)
    y0 = (h - s) // 2
    x0 = (w - s) // 2
    return image_array[y0:y0+s, x0:x0+s]


def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    return cv2.GaussianBlur(
        cv2.resize(_square_crop(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), (512,512)),
        (5,5), 0)


def preprocess_array(image_array):
    return cv2.GaussianBlur(
        cv2.resize(_square_crop(image_array), (512,512)),
        (5,5), 0)


def segment_fruit(img_blur):
    hsv   = cv2.cvtColor(img_blur, cv2.COLOR_RGB2HSV)
    mask1 = cv2.inRange(hsv, np.array([25,15,40]),  np.array([95,255,210]))
    mask2 = cv2.inRange(hsv, np.array([10,30,60]),  np.array([40,255,255]))
    mask  = cv2.bitwise_or(mask1, mask2)
    k     = np.ones((3,3), np.uint8)
    mask  = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask  = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
    return mask, hsv


def detect_basket(img_blur):
    gray  = cv2.cvtColor(img_blur, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray,(7,7),0), 30, 100)
    edges = cv2.dilate(edges, np.ones((5,5),np.uint8), iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_area = img_blur.shape[0] * img_blur.shape[1]
    best, best_area = None, 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < img_area * 0.05 or area > img_area * 0.95:
            continue
        if area > best_area:
            best_area = area; best = cnt
    return best, best_area


def is_calamansi(cnt, ppc):
    area = cv2.contourArea(cnt)
    if area < 50: return False
    peri = cv2.arcLength(cnt, True)
    if peri == 0: return False
    if (4*np.pi*area)/(peri**2) < MIN_CIRCULARITY: return False
    (_, _), r = cv2.minEnclosingCircle(cnt)
    d = (2*r)/ppc
    return MIN_DIAMETER_CM <= d <= MAX_DIAMETER_CM


def compute_features(cnt, mask, hsv, ppc):
    area_px  = cv2.contourArea(cnt)
    peri_px  = cv2.arcLength(cnt, True)
    (_, _), r = cv2.minEnclosingCircle(cnt)
    single   = np.zeros_like(mask)
    cv2.drawContours(single, [cnt], -1, 255, -1)
    diam     = (2*r)/ppc
    px       = hsv[single > 0]
    if len(px) == 0: return None
    return {
        'area_cm2':             area_px/(ppc**2),
        'diameter_cm':          diam,
        'perimeter_cm':         peri_px/ppc,
        'circularity':          (4*np.pi*area_px)/(peri_px**2+1e-5),
        'estimated_volume_cm3': (4/3)*np.pi*(diam/2)**3,
        'mean_hue':             np.mean(px[:,0]),
        'mean_saturation':      np.mean(px[:,1]),
        'mean_value':           np.mean(px[:,2]),
    }


def get_fruit_features(img_blur, mask, hsv, ppc):
    """Watershed segmentation to get individual fruit features."""
    dist      = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    dist_norm = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, sure_fg = cv2.threshold(dist_norm, 0.4*dist_norm.max(), 255, 0)
    sure_fg    = np.uint8(sure_fg)
    sure_bg    = cv2.dilate(mask, np.ones((3,3),np.uint8), iterations=3)
    unknown    = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers    = markers+1; markers[unknown==255]=0
    markers    = cv2.watershed(cv2.cvtColor(img_blur, cv2.COLOR_RGB2BGR), markers)

    feats, cnts = [], []
    for label in np.unique(markers):
        if label <= 1: continue
        seg = np.zeros(mask.shape, dtype=np.uint8)
        seg[markers==label] = 255
        seg = cv2.bitwise_and(seg, mask)
        cs, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cs: continue
        cnt = max(cs, key=cv2.contourArea)
        if not is_calamansi(cnt, ppc): continue
        f = compute_features(cnt, mask, hsv, ppc)
        if f: feats.append(f); cnts.append(cnt)

    # Fallback
    if not feats:
        cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cs:
            if not is_calamansi(cnt, ppc): continue
            f = compute_features(cnt, mask, hsv, ppc)
            if f: feats.append(f); cnts.append(cnt)

    return feats, cnts


def calibrate_ppc(hough_circles):
    """Estimate pixels/cm from visible fruit size (2.5cm avg diameter)."""
    if not hough_circles:
        return PIXELS_PER_CM
    avg_r_px = np.mean([r for _,_,r in hough_circles])
    return max(avg_r_px / 1.25, 5.0)


def count_hough(img_blur, mask):
    """Quick Hough circle count for calibration."""
    gray = cv2.cvtColor(img_blur, cv2.COLOR_RGB2GRAY)
    mean_v = float(gray.mean())
    bright_thresh = min(155, int(mean_v + 10))
    hsv  = cv2.cvtColor(img_blur, cv2.COLOR_RGB2HSV)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2,
        minDist=20, param1=50, param2=28, minRadius=8, maxRadius=45)
    if circles is None:
        return []
    circles = np.round(circles[0,:]).astype("int")
    valid = []
    for x,y,r in circles:
        if not (0<=y<gray.shape[0] and 0<=x<gray.shape[1]): continue
        if gray[y,x] > bright_thresh: continue
        h_val = int(hsv[y,x,0]); s_val = int(hsv[y,x,1])
        if not (10<=h_val<=95 and s_val>20): continue
        valid.append((x,y,r))
    return valid


# ─────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────

def extract_features_from_path(image_path):
    """Used by train_model.py — single fruit per training image."""
    img_blur = preprocess_image(image_path)
    if img_blur is None: return None
    mask, hsv   = segment_fruit(img_blur)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [c for c in contours if is_calamansi(c, PIXELS_PER_CM)]
    if not valid:
        valid = [c for c in contours if cv2.contourArea(c) >= 50]
    if not valid: return None
    return compute_features(max(valid, key=cv2.contourArea), mask, hsv, PIXELS_PER_CM)


def process_video_frames(frames_rgb):
    """
    Main pipeline for basket video.
    1. Picks best frame (most fruits visible)
    2. Detects each individual calamansi using watershed
    3. Returns features per fruit for juice prediction
    """
    best_frame  = None
    best_count  = 0

    for frame_rgb in frames_rgb:
        img_blur  = preprocess_array(frame_rgb)
        mask, hsv = segment_fruit(img_blur)
        circles   = count_hough(img_blur, mask)
        if len(circles) > best_count:
            best_count = len(circles)
            best_frame = {
                'img_blur':  img_blur,
                'mask':      mask,
                'hsv':       hsv,
                'circles':   circles,
            }

    if best_frame is None:
        return None

    img_blur = best_frame['img_blur']
    mask     = best_frame['mask']
    hsv      = best_frame['hsv']
    circles  = best_frame['circles']

    # Calibrate ppc from fruit size
    ppc = calibrate_ppc(circles)

    # Detect basket boundary
    basket_contour, _ = detect_basket(img_blur)

    # Get per-fruit features via watershed
    all_features, valid_cnts = get_fruit_features(img_blur, mask, hsv, ppc)

    return {
        'features':       all_features,
        'contours':       valid_cnts,
        'img_blur':       img_blur,
        'ppc':            ppc,
        'basket_contour': basket_contour,
        'hough_circles':  circles,
        'fruit_count':    len(all_features),
    }