import cv2
import numpy as np

# ─────────────────────────────────────────
# FEATURE EXTRACTION MODULE
# 3D-aware: uses top + side frames for volume estimation
# ─────────────────────────────────────────

PIXELS_PER_CM    = 41.0
COIN_DIAMETER_CM = 2.3
DETECT_SIZE      = 512
MIN_CIRCULARITY  = 0.25
MIN_DIAMETER_CM  = 1.5
MAX_DIAMETER_CM  = 5.5
PACKING_FACTOR   = 0.64   # sphere packing efficiency

FEATURE_COLS = [
    'area_cm2','diameter_cm','perimeter_cm','circularity',
    'estimated_volume_cm3','mean_hue','mean_saturation','mean_value'
]


# ─────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────

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
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    sq = _square_crop(img_rgb)
    return cv2.GaussianBlur(cv2.resize(sq,(DETECT_SIZE,DETECT_SIZE)),(5,5),0)

def preprocess_array(image_array):
    sq = _square_crop(image_array)
    return cv2.GaussianBlur(cv2.resize(sq,(DETECT_SIZE,DETECT_SIZE)),(5,5),0)


# ─────────────────────────────────────────
# VIEW CLASSIFICATION
# ─────────────────────────────────────────

def classify_view(img_blur):
    """
    Returns 'top' or 'side' based on where fruits appear in the frame.
    Top view:  fruits in upper/center portion → high top:bottom ratio
    Side view: fruits fill middle band → low ratio
    """
    hsv  = cv2.cvtColor(img_blur, cv2.COLOR_RGB2HSV)
    mask = cv2.bitwise_or(
        cv2.inRange(hsv, np.array([25,15,40]),  np.array([95,255,210])),
        cv2.inRange(hsv, np.array([10,30,60]),  np.array([40,255,255]))
    )
    rows      = np.sum(mask, axis=1)
    top_sum   = float(rows[:DETECT_SIZE//2].sum())
    bot_sum   = float(rows[DETECT_SIZE//2:].sum())
    ratio     = top_sum / (bot_sum + 1)
    # Side view also has fruit mask spread evenly + visible basket wall at bottom
    return 'top' if ratio > 1.2 else 'side'


# ─────────────────────────────────────────
# COIN DETECTION
# ─────────────────────────────────────────

def detect_coin(img_blur):
    """Detect 1-peso coin using Hough circles in lower half of image."""
    gray = cv2.cvtColor(img_blur, cv2.COLOR_RGB2GRAY)
    H    = img_blur.shape[0]
    hsv  = cv2.cvtColor(img_blur, cv2.COLOR_RGB2HSV)
    fmask = cv2.bitwise_or(
        cv2.inRange(hsv, np.array([25,15,40]),  np.array([95,255,210])),
        cv2.inRange(hsv, np.array([10,30,60]),  np.array([40,255,255]))
    )

    for y_start_frac in [0.4, 0.0]:
        y_start = int(H * y_start_frac)
        region  = gray[y_start:, :]
        circles = cv2.HoughCircles(
            region, cv2.HOUGH_GRADIENT, dp=1.2,
            minDist=20, param1=50, param2=18,
            minRadius=8, maxRadius=35
        )
        if circles is not None:
            circles   = np.round(circles[0,:]).astype("int")
            best, best_r = None, 0
            for (x, y_local, r) in circles:
                y_full = y_local + y_start
                if 0<=y_full<H and 0<=x<img_blur.shape[1]:
                    if fmask[y_full, x] > 0:
                        continue
                if r > best_r:
                    best_r = r; best = (x, y_full, r)
            if best is not None:
                x, y_full, r = best
                ppc = (2*r) / COIN_DIAMETER_CM
                pts = []
                for angle in range(0, 360, 5):
                    px = int(x + r * np.cos(np.radians(angle)))
                    py = int(y_full + r * np.sin(np.radians(angle)))
                    pts.append([[px, py]])
                return ppc, np.array(pts, dtype=np.int32)

    return PIXELS_PER_CM, None


# ─────────────────────────────────────────
# BASKET DETECTION
# ─────────────────────────────────────────

def detect_basket(img_blur, coin_contour=None):
    """Detect basket boundary from top view."""
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


def measure_basket_top(basket_contour, ppc):
    """
    Measure basket width and length from top view.
    Returns (width_cm, length_cm).
    """
    if basket_contour is None:
        return 0.0, 0.0
    rect     = cv2.minAreaRect(basket_contour)
    (w_px, h_px) = rect[1]
    width_cm  = min(w_px, h_px) / ppc
    length_cm = max(w_px, h_px) / ppc
    return round(width_cm, 1), round(length_cm, 1)


def measure_basket_side(img_blur, ppc):
    """
    Measure basket fill height from side view.
    Looks for the basket wall (dark rectangle) and fruit fill height.
    Returns fill_height_cm.
    """
    hsv  = cv2.cvtColor(img_blur, cv2.COLOR_RGB2HSV)
    # Fruit mask
    fmask = cv2.bitwise_or(
        cv2.inRange(hsv, np.array([25,15,40]),  np.array([95,255,210])),
        cv2.inRange(hsv, np.array([10,30,60]),  np.array([40,255,255]))
    )
    rows = np.sum(fmask, axis=1)

    # Find top of fruits (first row with significant fruit pixels)
    # and bottom of fruits (last row with significant fruit pixels)
    threshold = rows.max() * 0.1
    fruit_rows = np.where(rows > threshold)[0]
    if len(fruit_rows) == 0:
        return 0.0

    top_row    = fruit_rows[0]
    bottom_row = fruit_rows[-1]
    height_px  = bottom_row - top_row
    height_cm  = height_px / ppc
    return round(max(height_cm, 0.0), 1)


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
# HOUGH CIRCLE COUNT
# ─────────────────────────────────────────

def count_fruits_hough(img_blur, mask, ppc):
    """Count visible fruits using Hough circles on fruit-masked image."""
    gray_m = cv2.bitwise_and(cv2.cvtColor(img_blur, cv2.COLOR_RGB2GRAY), mask)
    min_r  = max(5,  int((MIN_DIAMETER_CM/2) * ppc))
    max_r  = min(80, int((4.5/2) * ppc))
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
# 3D VOLUME ESTIMATION
# ─────────────────────────────────────────

def estimate_count_3d(width_cm, length_cm, height_cm, avg_fruit_diameter_cm):
    """
    Estimate total fruit count using basket 3D volume.
    basket_volume = width × length × height
    fruit_volume  = (4/3)π(r)³
    total_count   = basket_volume × packing_factor / fruit_volume
    """
    if width_cm <= 0 or length_cm <= 0 or height_cm <= 0 or avg_fruit_diameter_cm <= 0:
        return 0
    basket_vol = width_cm * length_cm * height_cm
    fruit_vol  = (4/3) * np.pi * (avg_fruit_diameter_cm/2)**3
    count      = round((basket_vol * PACKING_FACTOR) / fruit_vol)
    return max(1, count)


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

def is_calamansi(cnt, ppc):
    area = cv2.contourArea(cnt)
    if area < 50: return False
    peri = cv2.arcLength(cnt, True)
    if peri == 0: return False
    if (4*np.pi*area)/(peri**2) < MIN_CIRCULARITY: return False
    (_, _), r = cv2.minEnclosingCircle(cnt)
    d = (2*r)/ppc
    return MIN_DIAMETER_CM <= d <= MAX_DIAMETER_CM

def get_fruit_features(img_blur, mask, hsv, ppc):
    """Per-fruit features via watershed for juice prediction."""
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
    """Single frame extraction — used for per-frame processing."""
    img_blur  = preprocess_array(image_array)
    mask, hsv = segment_fruit(img_blur)
    ppc, coin_contour          = detect_coin(img_blur)
    basket_contour, basket_area = detect_basket(img_blur, coin_contour)
    hough_count, _             = count_fruits_hough(img_blur, mask, ppc)
    all_features, valid_cnts   = get_fruit_features(img_blur, mask, hsv, ppc)
    view                       = classify_view(img_blur)
    width_cm, length_cm        = measure_basket_top(basket_contour, ppc)
    height_cm                  = measure_basket_side(img_blur, ppc) if view == 'side' else 0.0
    est_total                  = max(hough_count, len(all_features))
    return all_features, valid_cnts, mask, img_blur, ppc, \
           coin_contour, basket_contour, est_total, view, \
           width_cm, length_cm, height_cm


def process_video_frames(frames_rgb):
    """
    Process list of video frames.
    Combines top + side view measurements for 3D volume estimation.
    Returns best result dict.
    """
    top_frames  = []
    side_frames = []

    for frame_rgb in frames_rgb:
        img_blur = preprocess_array(frame_rgb)
        view     = classify_view(img_blur)
        ppc, coin_contour          = detect_coin(img_blur)
        mask, hsv                  = segment_fruit(img_blur)
        basket_contour, basket_area = detect_basket(img_blur, coin_contour)
        hough_count, _             = count_fruits_hough(img_blur, mask, ppc)
        all_features, valid_cnts   = get_fruit_features(img_blur, mask, hsv, ppc)
        width_cm, length_cm        = measure_basket_top(basket_contour, ppc)
        height_cm                  = measure_basket_side(img_blur, ppc) if view == 'side' else 0.0

        data = {
            'view':           view,
            'features':       all_features,
            'contours':       valid_cnts,
            'img_blur':       img_blur,
            'mask':           mask,
            'ppc':            ppc,
            'coin_contour':   coin_contour,
            'basket_contour': basket_contour,
            'hough_count':    hough_count,
            'width_cm':       width_cm,
            'length_cm':      length_cm,
            'height_cm':      height_cm,
        }

        if view == 'top' and all_features:
            top_frames.append(data)
        elif view == 'side':
            side_frames.append(data)

    if not top_frames and not side_frames:
        return None

    # Best top frame = most fruits visible
    all_frames = top_frames + side_frames
    best_top   = max(top_frames, key=lambda x: len(x['features'])) if top_frames else (
                 max(all_frames, key=lambda x: len(x['features'])))

    # Basket dimensions from best top frame
    best_width  = np.median([f['width_cm']  for f in (top_frames or all_frames) if f['width_cm']  > 0]) if any(f['width_cm']>0 for f in all_frames) else 0
    best_length = np.median([f['length_cm'] for f in (top_frames or all_frames) if f['length_cm'] > 0]) if any(f['length_cm']>0 for f in all_frames) else 0
    best_height = np.median([f['height_cm'] for f in side_frames if f['height_cm'] > 0]) if side_frames else 0

    # Average fruit diameter from all top frames
    all_diameters = []
    for f in top_frames:
        for feat in f['features']:
            all_diameters.append(feat['diameter_cm'])
    avg_diameter = np.mean(all_diameters) if all_diameters else 2.5

    # 3D count if we have height from side view
    if best_height > 0 and best_width > 0 and best_length > 0:
        count_3d = estimate_count_3d(best_width, best_length, best_height, avg_diameter)
        method   = '3D Volume'
    else:
        # Fallback: Hough count from best top frame
        count_3d = max(best_top['hough_count'], len(best_top['features']))
        method   = 'Hough (top view only)'

    # Average predictions from top frames
    best_ppc = best_top['ppc']

    return {
        'features':       best_top['features'],
        'contours':       best_top['contours'],
        'img_blur':       best_top['img_blur'],
        'ppc':            best_ppc,
        'coin_contour':   best_top['coin_contour'],
        'basket_contour': best_top['basket_contour'],
        'estimated_total':count_3d,
        'method':         method,
        'width_cm':       best_width,
        'length_cm':      best_length,
        'height_cm':      best_height,
        'avg_diameter':   avg_diameter,
        'top_frames':     len(top_frames),
        'side_frames':    len(side_frames),
    }