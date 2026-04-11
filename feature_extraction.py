import cv2
import numpy as np

# ─────────────────────────────────────────
# FEATURE EXTRACTION MODULE
# Calamansi Juice Yield Prediction System
# ─────────────────────────────────────────

PIXELS_PER_CM   = 41.0
MIN_CIRCULARITY = 0.40   # relaxed: camera angle makes calamansi appear slightly oval
MIN_DIAMETER_CM = 1.5    # relaxed: catch smaller fruits in frame
MAX_DIAMETER_CM = 4.5    # calamansi rarely exceeds 4.5cm

FEATURE_COLS = [
    'area_cm2', 'diameter_cm', 'perimeter_cm', 'circularity',
    'estimated_volume_cm3', 'mean_hue', 'mean_saturation', 'mean_value'
]

# ─────────────────────────────────────────
# HSV COLOR RANGES FOR CALAMANSI
# ─────────────────────────────────────────
# Green calamansi (unripe)
HSV_GREEN_LO = np.array([25, 30, 15])
HSV_GREEN_HI = np.array([90, 255, 220])
# Yellow-orange calamansi (ripe)
HSV_RIPE_LO  = np.array([10, 35, 40])
HSV_RIPE_HI  = np.array([40, 255, 255])

MIN_AVG_SAT  = 25
MIN_SAT_VAL  = 25
MIN_COVERAGE = 0.01

# FIX #2: Raised overlap threshold to reduce ghost detections
MIN_OVERLAP  = 0.50   # was 0.35


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
        cv2.resize(_square_crop(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), (512, 512)),
        (5, 5), 0)


def preprocess_array(image_array):
    return cv2.GaussianBlur(
        cv2.resize(_square_crop(image_array), (512, 512)),
        (5, 5), 0)


def segment_fruit(img_blur):
    hsv   = cv2.cvtColor(img_blur, cv2.COLOR_RGB2HSV)
    mask1 = cv2.inRange(hsv, HSV_GREEN_LO, HSV_GREEN_HI)
    mask2 = cv2.inRange(hsv, HSV_RIPE_LO,  HSV_RIPE_HI)
    mask  = cv2.bitwise_or(mask1, mask2)
    k     = np.ones((3, 3), np.uint8)
    mask  = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask  = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
    return mask, hsv


def detect_basket(img_blur):
    """Detect basket/container boundary. Falls back to largest contour."""
    gray  = cv2.cvtColor(img_blur, cv2.COLOR_RGB2GRAY)
    H, W  = img_blur.shape[:2]

    edges = cv2.Canny(cv2.GaussianBlur(gray, (7, 7), 0), 20, 80)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_area = H * W
    best, best_area = None, 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < img_area * 0.05 or area > img_area * 0.95:
            continue
        if area > best_area:
            best_area = area
            best = cnt

    if best is None:
        margin = int(min(H, W) * 0.05)
        pts = np.array([
            [[margin, margin]],
            [[W - margin, margin]],
            [[W - margin, H - margin]],
            [[margin, H - margin]]
        ], dtype=np.int32)
        best      = pts
        best_area = (W - 2 * margin) * (H - 2 * margin)

    return best, best_area


def is_calamansi(cnt, ppc):
    """Strict shape + size check for a contour to be a calamansi."""
    area = cv2.contourArea(cnt)
    if area < 100:
        return False
    peri = cv2.arcLength(cnt, True)
    if peri == 0:
        return False
    circularity = (4 * np.pi * area) / (peri ** 2)
    if circularity < MIN_CIRCULARITY:
        return False
    (_, _), r = cv2.minEnclosingCircle(cnt)
    d = (2 * r) / ppc
    return MIN_DIAMETER_CM <= d <= MAX_DIAMETER_CM


def _has_fruit_color(hsv, mask_region):
    """
    Validate that a detected region actually has calamansi color.
    Returns False if the region looks like a plain background or bright glare.
    """
    px = hsv[mask_region > 0]
    if len(px) == 0:
        return False
    avg_sat = np.mean(px[:, 1])
    avg_hue = np.mean(px[:, 0])
    avg_val = np.mean(px[:, 2])

    if avg_sat < MIN_AVG_SAT:
        return False
    if not (10 <= avg_hue <= 90):
        return False
    if avg_val < 15 or avg_val > 235:
        return False
    return True


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
        'area_cm2':             area_px / (ppc ** 2),
        'diameter_cm':          diam,
        'perimeter_cm':         peri_px / ppc,
        'circularity':          (4 * np.pi * area_px) / (peri_px ** 2 + 1e-5),
        'estimated_volume_cm3': (4 / 3) * np.pi * (diam / 2) ** 3,
        'mean_hue':             np.mean(px[:, 0]),
        'mean_saturation':      np.mean(px[:, 1]),
        'mean_value':           np.mean(px[:, 2]),
    }


def get_fruit_features(img_blur, mask, hsv, ppc):
    """Watershed segmentation to get individual fruit features."""
    dist      = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    dist_norm = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # FIX #3: Raised threshold from 0.4 → 0.55 to prevent watershed
    # from splitting a single fruit into two regions when fruits are touching
    _, sure_fg = cv2.threshold(dist_norm, 0.55 * dist_norm.max(), 255, 0)

    sure_fg    = np.uint8(sure_fg)
    sure_bg    = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=3)
    unknown    = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers    = markers + 1
    markers[unknown == 255] = 0
    markers    = cv2.watershed(cv2.cvtColor(img_blur, cv2.COLOR_RGB2BGR), markers)

    feats, cnts = [], []
    for label in np.unique(markers):
        if label <= 1:
            continue
        seg = np.zeros(mask.shape, dtype=np.uint8)
        seg[markers == label] = 255
        seg = cv2.bitwise_and(seg, mask)
        cs, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cs:
            continue
        cnt = max(cs, key=cv2.contourArea)
        if not is_calamansi(cnt, ppc):
            continue
        if not _has_fruit_color(hsv, seg):
            continue
        f = compute_features(cnt, mask, hsv, ppc)
        if f:
            feats.append(f)
            cnts.append(cnt)

    # Fallback: direct contour detection
    if not feats:
        cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cs:
            if not is_calamansi(cnt, ppc):
                continue
            single = np.zeros(mask.shape, dtype=np.uint8)
            cv2.drawContours(single, [cnt], -1, 255, -1)
            if not _has_fruit_color(hsv, single):
                continue
            f = compute_features(cnt, mask, hsv, ppc)
            if f:
                feats.append(f)
                cnts.append(cnt)

    return feats, cnts


def calibrate_ppc(hough_circles):
    """Estimate pixels/cm from visible fruit size (avg calamansi diameter = 2.5cm)."""
    if not hough_circles:
        return PIXELS_PER_CM
    avg_r_px = np.mean([r for _, _, r in hough_circles])
    return max(avg_r_px / 1.25, 5.0)


def get_features_from_hough(img_blur, circles, ppc):
    """
    Extract features directly from Hough circles.
    Each detected circle is validated for color before accepting.
    """
    hsv = cv2.cvtColor(img_blur, cv2.COLOR_RGB2HSV)
    feats, cnts = [], []

    for x, y, r in circles:
        diam_cm = (2 * r) / ppc
        if not (MIN_DIAMETER_CM <= diam_cm <= MAX_DIAMETER_CM):
            continue

        mask_c = np.zeros(img_blur.shape[:2], dtype=np.uint8)
        cv2.circle(mask_c, (x, y), r, 255, -1)
        if not _has_fruit_color(hsv, mask_c):
            continue

        area_px  = np.pi * r ** 2
        peri_px  = 2 * np.pi * r
        area_cm2 = area_px / (ppc ** 2)
        peri_cm  = peri_px / ppc
        vol      = (4 / 3) * np.pi * (diam_cm / 2) ** 3

        px = hsv[mask_c > 0]
        if len(px) == 0:
            continue

        feats.append({
            'area_cm2':             area_cm2,
            'diameter_cm':          diam_cm,
            'perimeter_cm':         peri_cm,
            'circularity':          1.0,
            'estimated_volume_cm3': vol,
            'mean_hue':             np.mean(px[:, 0]),
            'mean_saturation':      np.mean(px[:, 1]),
            'mean_value':           np.mean(px[:, 2]),
        })

        theta = np.linspace(0, 2 * np.pi, 20)
        pts   = np.array([[[int(x + r * np.cos(t)), int(y + r * np.sin(t))]]
                          for t in theta], dtype=np.int32)
        cnts.append(pts)

    return feats, cnts


def count_hough(img_blur, mask):
    """
    Hough circle detection with strict false-positive prevention.
    """
    gray = cv2.cvtColor(img_blur, cv2.COLOR_RGB2GRAY)
    hsv  = cv2.cvtColor(img_blur, cv2.COLOR_RGB2HSV)

    # Guard 1: Enough fruit-colored pixels?
    total_px = mask.shape[0] * mask.shape[1]
    fruit_px = np.sum(mask > 0)
    coverage = fruit_px / total_px
    if coverage < MIN_COVERAGE:
        return []

    # Guard 2: Average saturation of masked region must be high enough
    if fruit_px > 0:
        avg_sat = np.mean(hsv[:, :, 1][mask > 0])
        if avg_sat < MIN_AVG_SAT:
            return []

    mean_v = float(gray.mean())
    bright_thresh = min(155, int(mean_v + 10))

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        # FIX #1: Raised minDist from 28 → 45 to prevent placing two
        # circle centers on the same fruit when fruits are close together.
        # At ~41 px/cm and avg diameter 2.5cm, one fruit ≈ 103px wide,
        # so centers should be at least ~45px apart to be distinct fruits.
        minDist=45,
        param1=45,
        # FIX #4: Raised param2 from 18 → 22 to reduce weak/fake circles
        # detected on fruit edges, shadows, and glare spots.
        param2=22,
        minRadius=8,
        maxRadius=42
    )
    if circles is None:
        return []

    circles = np.round(circles[0, :]).astype("int")
    valid = []
    for x, y, r in circles:
        if not (0 <= y < gray.shape[0] and 0 <= x < gray.shape[1]):
            continue
        y1, y2 = max(0, y-2), min(gray.shape[0], y+3)
        x1, x2 = max(0, x-2), min(gray.shape[1], x+3)
        center_gray = float(np.mean(gray[y1:y2, x1:x2]))
        if center_gray > bright_thresh:
            continue

        h_val = int(np.mean(hsv[y1:y2, x1:x2, 0]))
        s_val = int(np.mean(hsv[y1:y2, x1:x2, 1]))

        if not (10 <= h_val <= 90 and s_val > MIN_SAT_VAL):
            continue

        circle_mask = np.zeros(mask.shape, dtype=np.uint8)
        cv2.circle(circle_mask, (x, y), r, 255, -1)
        overlap     = np.sum(cv2.bitwise_and(mask, circle_mask) > 0)
        circle_area = np.pi * r ** 2
        if overlap / circle_area < MIN_OVERLAP:
            continue

        if not _has_fruit_color(hsv, circle_mask):
            continue

        valid.append((x, y, r))

    return valid


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
    valid = [c for c in contours if is_calamansi(c, PIXELS_PER_CM)]
    if not valid:
        valid = [c for c in contours if cv2.contourArea(c) >= 100]
    if not valid:
        return None
    return compute_features(max(valid, key=cv2.contourArea), mask, hsv, PIXELS_PER_CM)


def process_video_frames(frames_rgb):
    """
    Main pipeline for basket video.
    1. Picks best frame (most fruits visible)
    2. Detects each individual calamansi using watershed
    3. Returns features per fruit for juice prediction
    """
    best_frame = None
    best_count = 0

    for frame_rgb in frames_rgb:
        img_blur  = preprocess_array(frame_rgb)
        mask, hsv = segment_fruit(img_blur)
        circles   = count_hough(img_blur, mask)
        if len(circles) > best_count:
            best_count = len(circles)
            best_frame = {
                'img_blur': img_blur,
                'mask':     mask,
                'hsv':      hsv,
                'circles':  circles,
            }

    if best_frame is None:
        return None

    img_blur = best_frame['img_blur']
    mask     = best_frame['mask']
    hsv      = best_frame['hsv']
    circles  = best_frame['circles']

    ppc = calibrate_ppc(circles)
    basket_contour, _ = detect_basket(img_blur)
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