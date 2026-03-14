import cv2
import numpy as np

# ─────────────────────────────────────────
# FEATURE EXTRACTION MODULE
# ─────────────────────────────────────────

PIXELS_PER_CM    = 41.0      # fallback if no coin detected
COIN_DIAMETER_CM = 2.3       # 1 peso coin diameter in cm
MIN_FRUIT_AREA_PX = 50
MIN_CIRCULARITY   = 0.25
MIN_DIAMETER_CM   = 1.5      # smallest valid calamansi
MAX_DIAMETER_CM   = 5.5      # largest valid calamansi
PACKING_FACTOR    = 0.75     # fruits don't pack perfectly in basket

FEATURE_COLS = [
    'area_cm2', 'diameter_cm', 'perimeter_cm', 'circularity',
    'estimated_volume_cm3', 'mean_hue', 'mean_saturation', 'mean_value'
]


# ─────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return cv2.GaussianBlur(cv2.resize(img_rgb, (512, 512)), (5, 5), 0)

def preprocess_array(image_array):
    return cv2.GaussianBlur(cv2.resize(image_array, (512, 512)), (5, 5), 0)


# ─────────────────────────────────────────
# COIN DETECTION
# ─────────────────────────────────────────

def detect_coin(img_blur):
    """Detect 1-peso coin for pixel calibration. Returns (pixels_per_cm, contour)."""
    hsv = cv2.cvtColor(img_blur, cv2.COLOR_RGB2HSV)
    # Silver coin: low saturation, medium-high brightness
    mask = cv2.inRange(hsv, np.array([0, 0, 100]), np.array([180, 60, 230]))
    k    = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best, best_circ = None, 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 300 or area > 60000:
            continue
        peri = cv2.arcLength(cnt, True)
        if peri == 0:
            continue
        circ = (4 * np.pi * area) / (peri ** 2)
        if circ > 0.55 and circ > best_circ:
            best_circ = circ
            best = cnt

    if best is not None:
        (_, _), r = cv2.minEnclosingCircle(best)
        return (2 * r) / COIN_DIAMETER_CM, best
    return PIXELS_PER_CM, None


# ─────────────────────────────────────────
# BASKET DETECTION
# ─────────────────────────────────────────

def detect_basket(img_blur, coin_contour=None):
    """Detect basket boundary. Returns (contour, area_px)."""
    gray  = cv2.cvtColor(img_blur, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (7, 7), 0), 30, 100)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_area  = img_blur.shape[0] * img_blur.shape[1]
    coin_area = cv2.contourArea(coin_contour) if coin_contour is not None else 0

    best, best_area = None, 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < img_area * 0.05 or area > img_area * 0.95:
            continue
        if abs(area - coin_area) < 500:
            continue
        if area > best_area:
            best_area = area
            best = cnt
    return best, best_area


# ─────────────────────────────────────────
# FRUIT COLOR SEGMENTATION
# ─────────────────────────────────────────

def segment_fruit(img_blur):
    hsv   = cv2.cvtColor(img_blur, cv2.COLOR_RGB2HSV)
    mask1 = cv2.inRange(hsv, np.array([25, 15, 40]),  np.array([95, 255, 210]))  # green
    mask2 = cv2.inRange(hsv, np.array([10, 30, 60]),  np.array([40, 255, 255]))  # yellow-orange
    mask  = cv2.bitwise_or(mask1, mask2)
    k     = np.ones((3, 3), np.uint8)
    mask  = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask  = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
    return mask, hsv


# ─────────────────────────────────────────
# WATERSHED — separate touching fruits
# ─────────────────────────────────────────

def apply_watershed(mask, img_blur):
    dist      = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    dist_norm = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, sure_fg = cv2.threshold(dist_norm, 0.4 * dist_norm.max(), 255, 0)
    sure_fg    = np.uint8(sure_fg)
    sure_bg    = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=3)
    unknown    = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers    = markers + 1
    markers[unknown == 255] = 0
    return cv2.watershed(cv2.cvtColor(img_blur, cv2.COLOR_RGB2BGR), markers)


# ─────────────────────────────────────────
# CALAMANSI VALIDATION — color + size + shape
# ─────────────────────────────────────────

def is_calamansi(cnt, pixels_per_cm):
    """
    Filter by 3 properties specific to calamansi:
    1. Shape  — must be round (circularity ≥ 0.25)
    2. Size   — diameter must be 1.5–5.5 cm (calamansi-specific range)
    3. Area   — minimum pixel area
    """
    area = cv2.contourArea(cnt)
    if area < MIN_FRUIT_AREA_PX:
        return False
    peri = cv2.arcLength(cnt, True)
    if peri == 0:
        return False
    if (4 * np.pi * area) / (peri ** 2) < MIN_CIRCULARITY:
        return False
    (_, _), r = cv2.minEnclosingCircle(cnt)
    diameter_cm = (2 * r) / pixels_per_cm
    return MIN_DIAMETER_CM <= diameter_cm <= MAX_DIAMETER_CM


# ─────────────────────────────────────────
# FEATURE COMPUTATION
# ─────────────────────────────────────────

def compute_features(cnt, mask, hsv, pixels_per_cm):
    area_px  = cv2.contourArea(cnt)
    peri_px  = cv2.arcLength(cnt, True)
    (_, _), r = cv2.minEnclosingCircle(cnt)

    single = np.zeros_like(mask)
    cv2.drawContours(single, [cnt], -1, 255, -1)

    diam_cm = (2 * r) / pixels_per_cm
    pixels  = hsv[single > 0]
    if len(pixels) == 0:
        return None

    return {
        'area_cm2':             area_px / (pixels_per_cm ** 2),
        'diameter_cm':          diam_cm,
        'perimeter_cm':         peri_px / pixels_per_cm,
        'circularity':          (4 * np.pi * area_px) / (peri_px ** 2 + 1e-5),
        'estimated_volume_cm3': (4/3) * np.pi * (diam_cm / 2) ** 3,
        'mean_hue':             np.mean(pixels[:, 0]),
        'mean_saturation':      np.mean(pixels[:, 1]),
        'mean_value':           np.mean(pixels[:, 2]),
    }


# ─────────────────────────────────────────
# TOTAL COUNT ESTIMATION
# ─────────────────────────────────────────

def estimate_total_count(basket_area_px, visible_contours, pixels_per_cm):
    """
    Estimate total fruit count including hidden fruits.
    Formula: (basket area × packing factor) ÷ average fruit area
    """
    if not visible_contours or basket_area_px <= 0:
        return len(visible_contours)
    avg_fruit_area = np.mean([cv2.contourArea(c) for c in visible_contours])
    if avg_fruit_area <= 0:
        return len(visible_contours)
    estimated = round((basket_area_px * PACKING_FACTOR) / avg_fruit_area)
    return max(estimated, len(visible_contours))


# ─────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────

def extract_features_from_path(image_path):
    """Used by train_model.py — single fruit per training image."""
    img_blur = preprocess_image(image_path)
    if img_blur is None:
        return None
    mask, hsv   = segment_fruit(img_blur)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # For training images: relax size filter slightly
    valid = [c for c in contours if is_calamansi(c, PIXELS_PER_CM)]
    if not valid:
        valid = [c for c in contours if cv2.contourArea(c) >= MIN_FRUIT_AREA_PX]
    if not valid:
        return None
    return compute_features(max(valid, key=cv2.contourArea), mask, hsv, PIXELS_PER_CM)


def extract_features_from_array(image_array):
    """
    Used by app.py — basket mode with coin + watershed.
    Returns:
      all_features     list of feature dicts (visible fruits)
      contours         contours of visible fruits
      mask             color mask
      img_blur         preprocessed image
      pixels_per_cm    calibrated scale
      coin_contour     coin contour or None
      basket_contour   basket contour or None
      estimated_total  estimated total count including hidden fruits
    """
    img_blur  = preprocess_array(image_array)
    mask, hsv = segment_fruit(img_blur)

    # Step 1: Calibrate with coin
    pixels_per_cm, coin_contour = detect_coin(img_blur)

    # Step 2: Detect basket boundary
    basket_contour, basket_area_px = detect_basket(img_blur, coin_contour)

    # Step 3: Separate touching fruits with watershed
    markers = apply_watershed(mask, img_blur)
    all_features, valid_contours = [], []

    for label in np.unique(markers):
        if label <= 1:
            continue
        seg = np.zeros(mask.shape, dtype=np.uint8)
        seg[markers == label] = 255
        seg = cv2.bitwise_and(seg, mask)
        cnts, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)
        if not is_calamansi(cnt, pixels_per_cm):
            continue
        feat = compute_features(cnt, mask, hsv, pixels_per_cm)
        if feat:
            all_features.append(feat)
            valid_contours.append(cnt)

    # Fallback: simple contour detection
    if not all_features:
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            if not is_calamansi(cnt, pixels_per_cm):
                continue
            feat = compute_features(cnt, mask, hsv, pixels_per_cm)
            if feat:
                all_features.append(feat)
                valid_contours.append(cnt)

    # Step 4: Estimate total count
    estimated_total = estimate_total_count(basket_area_px, valid_contours, pixels_per_cm)

    return all_features, valid_contours, mask, img_blur, pixels_per_cm, \
           coin_contour, basket_contour, estimated_total