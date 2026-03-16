import cv2
import numpy as np

# ─────────────────────────────────────────
# FEATURE EXTRACTION MODULE
# Development of an Image-Based System for Predicting
# Calamansi (Citrus microcarpa) Juice Yield Using Linear Regression
# ─────────────────────────────────────────

PIXELS_PER_CM   = 41.0
AVG_CALAMANSI_D = 2.5    # average calamansi diameter in cm
MIN_CIRCULARITY = 0.25
MIN_DIAMETER_CM = 1.5
MAX_DIAMETER_CM = 5.5

FEATURE_COLS = [
    'area_cm2', 'diameter_cm', 'perimeter_cm', 'circularity',
    'estimated_volume_cm3', 'mean_hue', 'mean_saturation', 'mean_value'
]


# ─────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────

def _square_crop(image_array):
    """Center-square crop to handle portrait/landscape uniformly."""
    h, w = image_array.shape[:2]
    s  = min(h, w)
    y0 = (h - s) // 2
    x0 = (w - s) // 2
    return image_array[y0:y0+s, x0:x0+s]

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return cv2.GaussianBlur(cv2.resize(_square_crop(img_rgb), (512,512)), (5,5), 0)

def preprocess_array(image_array):
    return cv2.GaussianBlur(cv2.resize(_square_crop(image_array), (512,512)), (5,5), 0)


# ─────────────────────────────────────────
# VIEW CLASSIFICATION
# top view  = fruits fill upper portion of image
# side view = fruits appear as a horizontal band in the middle
# ─────────────────────────────────────────

def classify_view(img_blur):
    hsv  = cv2.cvtColor(img_blur, cv2.COLOR_RGB2HSV)
    mask = cv2.bitwise_or(
        cv2.inRange(hsv, np.array([25,15,40]),  np.array([95,255,210])),
        cv2.inRange(hsv, np.array([10,30,60]),  np.array([40,255,255]))
    )
    rows  = np.sum(mask, axis=1)
    top   = float(rows[:256].sum())
    bot   = float(rows[256:].sum())
    ratio = top / (bot + 1)
    return 'top' if ratio > 1.2 else 'side'


# ─────────────────────────────────────────
# FRUIT SEGMENTATION
# ─────────────────────────────────────────

def segment_fruit(img_blur):
    hsv   = cv2.cvtColor(img_blur, cv2.COLOR_RGB2HSV)
    mask1 = cv2.inRange(hsv, np.array([25,15,40]),  np.array([95,255,210]))
    mask2 = cv2.inRange(hsv, np.array([10,30,60]),  np.array([40,255,255]))
    mask  = cv2.bitwise_or(mask1, mask2)
    k     = np.ones((3,3), np.uint8)
    mask  = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask  = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
    return mask, hsv


# ─────────────────────────────────────────
# BASKET DETECTION
# ─────────────────────────────────────────

def detect_basket(img_blur):
    gray  = cv2.cvtColor(img_blur, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray,(7,7),0), 30, 100)
    edges = cv2.dilate(edges, np.ones((5,5),np.uint8), iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_area    = img_blur.shape[0] * img_blur.shape[1]
    best, best_area = None, 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < img_area * 0.05 or area > img_area * 0.95:
            continue
        if area > best_area:
            best_area = area; best = cnt
    return best, best_area


# ─────────────────────────────────────────
# HOUGH CIRCLE COUNTING
# Counts visible fruits in one layer
# ─────────────────────────────────────────

def count_fruits_hough(img_blur, mask, avg_r_px=None):
    gray_m = cv2.bitwise_and(cv2.cvtColor(img_blur, cv2.COLOR_RGB2GRAY), mask)
    if avg_r_px:
        min_r = max(5,  int(avg_r_px * 0.6))
        max_r = min(80, int(avg_r_px * 1.4))
    else:
        min_r, max_r = 8, 45
    minD = max(min_r * 2, 10)
    circles = cv2.HoughCircles(
        gray_m, cv2.HOUGH_GRADIENT, dp=1.2,
        minDist=minD, param1=50, param2=30,
        minRadius=min_r, maxRadius=max_r
    )
    if circles is None:
        return 0, []
    circles = np.round(circles[0,:]).astype("int")
    valid   = [(x,y,r) for x,y,r in circles
               if 0<=y<mask.shape[0] and 0<=x<mask.shape[1] and mask[y,x]>0]
    return len(valid), valid


# ─────────────────────────────────────────
# LAYER DETECTION FROM SIDE VIEW
# Measures how many fruit-diameters tall the stack is
# ─────────────────────────────────────────

def measure_layers_from_side(img_blur, avg_fruit_diameter_px):
    """
    From a side-view frame, measure how many layers of fruits are stacked.
    Finds the fruit-colored vertical span and divides by avg fruit diameter.
    """
    hsv  = cv2.cvtColor(img_blur, cv2.COLOR_RGB2HSV)
    mask = cv2.bitwise_or(
        cv2.inRange(hsv, np.array([25,15,40]),  np.array([95,255,210])),
        cv2.inRange(hsv, np.array([10,30,60]),  np.array([40,255,255]))
    )
    rows      = np.sum(mask, axis=1)
    threshold = rows.max() * 0.1
    fruit_rows = np.where(rows > threshold)[0]

    if len(fruit_rows) == 0 or avg_fruit_diameter_px <= 0:
        return 1

    height_px = fruit_rows[-1] - fruit_rows[0]
    layers    = round(height_px / avg_fruit_diameter_px)
    return max(1, layers)


# ─────────────────────────────────────────
# FEATURE COMPUTATION
# ─────────────────────────────────────────

def compute_features(cnt, mask, hsv, ppc):
    area_px  = cv2.contourArea(cnt)
    peri_px  = cv2.arcLength(cnt, True)
    (_, _), r = cv2.minEnclosingCircle(cnt)
    single   = np.zeros_like(mask)
    cv2.drawContours(single, [cnt], -1, 255, -1)
    diam     = (2 * r) / ppc
    px       = hsv[single > 0]
    if len(px) == 0:
        return None
    return {
        'area_cm2':             area_px / (ppc**2),
        'diameter_cm':          diam,
        'perimeter_cm':         peri_px / ppc,
        'circularity':          (4*np.pi*area_px) / (peri_px**2 + 1e-5),
        'estimated_volume_cm3': (4/3) * np.pi * (diam/2)**3,
        'mean_hue':             np.mean(px[:,0]),
        'mean_saturation':      np.mean(px[:,1]),
        'mean_value':           np.mean(px[:,2]),
    }

def is_calamansi(cnt, ppc):
    area = cv2.contourArea(cnt)
    if area < 50: return False
    peri = cv2.arcLength(cnt, True)
    if peri == 0: return False
    if (4*np.pi*area)/(peri**2) < MIN_CIRCULARITY: return False
    (_, _), r = cv2.minEnclosingCircle(cnt)
    d = (2*r)/ppc
    return MIN_DIAMETER_CM <= d <= MAX_DIAMETER_CM


# ─────────────────────────────────────────
# PER-FRUIT FEATURES (watershed)
# ─────────────────────────────────────────

def get_fruit_features(img_blur, mask, hsv, ppc):
    dist       = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    dist_norm  = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, sure_fg = cv2.threshold(dist_norm, 0.4*dist_norm.max(), 255, 0)
    sure_fg    = np.uint8(sure_fg)
    sure_bg    = cv2.dilate(mask, np.ones((3,3),np.uint8), iterations=3)
    unknown    = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers    = markers + 1; markers[unknown==255] = 0
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
    if not feats:
        cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cs:
            if not is_calamansi(cnt, ppc): continue
            f = compute_features(cnt, mask, hsv, ppc)
            if f: feats.append(f); cnts.append(cnt)
    return feats, cnts


# ─────────────────────────────────────────
# CALIBRATE PPC FROM VISIBLE FRUITS
# No coin needed — uses visible fruit size as reference
# (Representative sampling: visible fruits = same size as hidden fruits)
# ─────────────────────────────────────────

def calibrate_ppc(hough_circles):
    if not hough_circles:
        return PIXELS_PER_CM
    avg_r_px = np.mean([r for _,_,r in hough_circles])
    # avg calamansi radius = 1.25cm
    return max(avg_r_px / 1.25, 5.0)


# ─────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────

def extract_features_from_path(image_path):
    """Used by train_model.py."""
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
    Main pipeline. Accepts list of video frames (RGB).

    Automatically:
    1. Separates top-view and side-view frames
    2. From top frames: counts fruits per layer using Hough circles
    3. From side frames: measures stack height → calculates layers
    4. Total count = per_layer × layers
    5. Extracts per-fruit features for juice prediction

    Returns result dict or None.
    """
    top_frames  = []
    side_frames = []

    for frame_rgb in frames_rgb:
        img_blur = preprocess_array(frame_rgb)
        view     = classify_view(img_blur)
        mask, hsv = segment_fruit(img_blur)
        hough_count, hough_circles = count_fruits_hough(img_blur, mask)

        data = {
            'img_blur':      img_blur,
            'mask':          mask,
            'hsv':           hsv,
            'hough_count':   hough_count,
            'hough_circles': hough_circles,
        }

        if view == 'top':
            top_frames.append(data)
        else:
            side_frames.append(data)

    if not top_frames and not side_frames:
        return None

    # Use all available frames if no top frames detected
    if not top_frames:
        top_frames = side_frames

    # Best top frame = most fruits visible
    best_top = max(top_frames, key=lambda x: x['hough_count'])

    img_blur       = best_top['img_blur']
    mask           = best_top['mask']
    hsv            = best_top['hsv']
    hough_circles  = best_top['hough_circles']
    hough_count    = best_top['hough_count']

    # Calibrate pixels/cm from visible fruit size
    ppc = calibrate_ppc(hough_circles)

    # Average fruit diameter in pixels
    avg_r_px   = np.mean([r for _,_,r in hough_circles]) if hough_circles else 15
    avg_d_px   = avg_r_px * 2
    avg_d_cm   = avg_d_px / ppc

    # Detect basket
    basket_contour, basket_area_px = detect_basket(img_blur)

    # Per-layer count from Hough
    per_layer = hough_count

    # Layer count from side frames
    if side_frames:
        layer_counts = []
        for sf in side_frames:
            layers = measure_layers_from_side(sf['img_blur'], avg_d_px)
            if layers > 0:
                layer_counts.append(layers)
        layers = int(np.median(layer_counts)) if layer_counts else 1
        method = f"Auto-detected ({len(side_frames)} side frames)"
    else:
        layers = 1
        method = "Top view only — tilt camera to side for better count"

    total_count = per_layer * layers

    # Per-fruit features for juice prediction
    all_features, valid_cnts = get_fruit_features(img_blur, mask, hsv, ppc)

    return {
        'features':        all_features,
        'contours':        valid_cnts,
        'img_blur':        img_blur,
        'mask':            mask,
        'ppc':             ppc,
        'basket_contour':  basket_contour,
        'hough_count':     hough_count,
        'hough_circles':   hough_circles,
        'per_layer':       per_layer,
        'layers':          layers,
        'total_count':     total_count,
        'avg_diameter_cm': avg_d_cm,
        'top_frames':      len(top_frames),
        'side_frames':     len(side_frames),
        'method':          method,
    }