import cv2
import numpy as np

# ─────────────────────────────────────────
# FEATURE EXTRACTION MODULE
# ─────────────────────────────────────────

PIXELS_PER_CM    = 41.0
COIN_DIAMETER_CM = 2.3
DETECT_SIZE      = 512
MIN_CIRCULARITY  = 0.25
MIN_DIAMETER_CM  = 1.5
MAX_DIAMETER_CM  = 5.5

FEATURE_COLS = [
    'area_cm2','diameter_cm','perimeter_cm','circularity',
    'estimated_volume_cm3','mean_hue','mean_saturation','mean_value'
]


def _square_crop(image_array):
    """Center-square crop so portrait/landscape images scale uniformly."""
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
    sq  = _square_crop(img_rgb)
    return cv2.GaussianBlur(cv2.resize(sq, (DETECT_SIZE, DETECT_SIZE)), (5,5), 0)


def preprocess_array(image_array):
    sq = _square_crop(image_array)
    return cv2.GaussianBlur(cv2.resize(sq, (DETECT_SIZE, DETECT_SIZE)), (5,5), 0)


# ─────────────────────────────────────────
# COIN DETECTION
# ─────────────────────────────────────────

def detect_coin(img_blur):
    """
    Detect 1-peso coin using Hough circles in the lower portion of the image
    (user places coin below or beside the basket).
    Falls back to full-image search if not found in lower half.
    """
    gray = cv2.cvtColor(img_blur, cv2.COLOR_RGB2GRAY)
    H    = img_blur.shape[0]

    # Strategy 1: Search lower 60% of image (coin placed below basket)
    for y_start_frac in [0.4, 0.0]:
        y_start  = int(H * y_start_frac)
        region   = gray[y_start:, :]
        circles  = cv2.HoughCircles(
            region, cv2.HOUGH_GRADIENT, dp=1.2,
            minDist=20, param1=50, param2=18,
            minRadius=8, maxRadius=35
        )
        if circles is not None:
            circles = np.round(circles[0,:]).astype("int")
            # Pick the most isolated circle (likely the coin, not a fruit)
            hsv   = cv2.cvtColor(img_blur, cv2.COLOR_RGB2HSV)
            fmask = cv2.bitwise_or(
                cv2.inRange(hsv, np.array([25,15,40]),  np.array([95,255,210])),
                cv2.inRange(hsv, np.array([10,30,60]),  np.array([40,255,255]))
            )
            best_coin = None
            best_r    = 0
            for (x, y_local, r) in circles:
                y_full = y_local + y_start
                # Skip if center is inside fruit area
                if 0<=y_full<H and 0<=x<img_blur.shape[1]:
                    if fmask[y_full, x] > 0:
                        continue
                # Pick reasonably sized coin (not too small artifact)
                if r > best_r:
                    best_r    = r
                    best_coin = (x, y_full, r)

            if best_coin is not None:
                x, y_full, r = best_coin
                ppc = (2*r) / COIN_DIAMETER_CM
                # Build circular contour for drawing
                pts = []
                for angle in range(0, 360, 5):
                    px = int(x + r * np.cos(np.radians(angle)))
                    py = int(y_full + r * np.sin(np.radians(angle)))
                    pts.append([[px, py]])
                coin_cnt = np.array(pts, dtype=np.int32)
                return ppc, coin_cnt

    return PIXELS_PER_CM, None


# ─────────────────────────────────────────
# BASKET DETECTION
# ─────────────────────────────────────────

def detect_basket(img_blur, coin_contour=None):
    gray  = cv2.cvtColor(img_blur, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray,(7,7),0), 30, 100)
    edges = cv2.dilate(edges, np.ones((5,5),np.uint8), iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_area  = img_blur.shape[0] * img_blur.shape[1]
    coin_area = cv2.contourArea(coin_contour) if coin_contour is not None else 0
    best, best_area = None, 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < img_area*0.05 or area > img_area*0.95:
            continue
        if abs(area - coin_area) < 500:
            continue
        if area > best_area:
            best_area = area; best = cnt
    return best, best_area


# ─────────────────────────────────────────
# FRUIT SEGMENTATION
# ─────────────────────────────────────────

def segment_fruit(img_blur):
    hsv   = cv2.cvtColor(img_blur, cv2.COLOR_RGB2HSV)
    mask1 = cv2.inRange(hsv, np.array([25,15,40]),  np.array([95,255,210]))
    mask2 = cv2.inRange(hsv, np.array([10,30,60]),  np.array([40,255,255]))
    mask  = cv2.bitwise_or(mask1, mask2)
    k     = np.ones((3,3),np.uint8)
    mask  = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask  = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
    return mask, hsv


# ─────────────────────────────────────────
# HOUGH CIRCLE COUNT — count visible fruits
# ─────────────────────────────────────────

def count_fruits_hough(img_blur, mask, pixels_per_cm):
    """Count visible fruits using Hough circles on fruit-masked image."""
    gray_m = cv2.bitwise_and(cv2.cvtColor(img_blur, cv2.COLOR_RGB2GRAY), mask)
    min_r  = max(5,  int((MIN_DIAMETER_CM/2) * pixels_per_cm))
    max_r  = min(80, int((4.5/2) * pixels_per_cm))
    minD   = max(min_r*2, 10)

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
# TOTAL ESTIMATION
# ─────────────────────────────────────────

def estimate_total(basket_area_px, visible_contours, hough_count, pixels_per_cm):
    """
    Best estimate of total fruits including hidden ones.
    Uses whichever is larger: Hough count or basket-area method.
    """
    hough = max(hough_count, len(visible_contours))

    # Basket area method
    if visible_contours and basket_area_px > 0:
        avg_fruit_area = np.mean([cv2.contourArea(c) for c in visible_contours])
        if avg_fruit_area > 0:
            basket_est = round((basket_area_px * 0.75) / avg_fruit_area)
            return max(hough, basket_est, len(visible_contours))

    return hough


# ─────────────────────────────────────────
# FEATURE COMPUTATION
# ─────────────────────────────────────────

def compute_features(cnt, mask, hsv, ppc):
    area_px = cv2.contourArea(cnt)
    peri_px = cv2.arcLength(cnt, True)
    (_, _), r = cv2.minEnclosingCircle(cnt)
    single = np.zeros_like(mask)
    cv2.drawContours(single, [cnt], -1, 255, -1)
    diam = (2*r)/ppc
    px   = hsv[single > 0]
    if len(px) == 0:
        return None
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
# GET FRUIT FEATURES (watershed)
# ─────────────────────────────────────────

def get_fruit_features(img_blur, mask, hsv, ppc):
    """Extract per-fruit features for juice prediction via watershed."""
    dist       = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    dist_norm  = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
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

    if not feats:
        cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cs:
            if not is_calamansi(cnt, ppc): continue
            f = compute_features(cnt, mask, hsv, ppc)
            if f: feats.append(f); cnts.append(cnt)

    return feats, cnts


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


def extract_features_from_array(image_array):
    """Used by app.py — full basket detection pipeline."""
    img_blur  = preprocess_array(image_array)
    mask, hsv = segment_fruit(img_blur)

    ppc, coin_contour          = detect_coin(img_blur)
    basket_contour, basket_area = detect_basket(img_blur, coin_contour)
    hough_count, _             = count_fruits_hough(img_blur, mask, ppc)
    all_features, valid_cnts   = get_fruit_features(img_blur, mask, hsv, ppc)
    est_total = estimate_total(basket_area, valid_cnts, hough_count, ppc)

    return all_features, valid_cnts, mask, img_blur, ppc, \
           coin_contour, basket_contour, est_total