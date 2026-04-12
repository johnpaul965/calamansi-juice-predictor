import cv2
import numpy as np

# ─────────────────────────────────────────
# FEATURE EXTRACTION MODULE
# Calamansi Juice Yield Prediction System
# ─────────────────────────────────────────

PIXELS_PER_CM   = 41.0
MIN_CIRCULARITY = 0.40
MIN_DIAMETER_CM = 1.5
MAX_DIAMETER_CM = 7.0

FEATURE_COLS = [
    'area_cm2', 'diameter_cm', 'perimeter_cm', 'circularity',
    'estimated_volume_cm3', 'mean_hue', 'mean_saturation', 'mean_value'
]

HSV_GREEN_LO = np.array([25, 30, 15])
HSV_GREEN_HI = np.array([100, 255, 210])
HSV_RIPE_LO  = np.array([10, 35, 35])
HSV_RIPE_HI  = np.array([40, 255, 255])

MIN_AVG_SAT  = 25
MIN_SAT_VAL  = 25
MIN_COVERAGE = 0.003
MIN_OVERLAP  = 0.50

MAX_FRUIT_COVERAGE = 0.55
MIN_BG_MEAN        = 60
MAX_BG_STD         = 75


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
    cropped = _square_crop(image_array)
    h, w = cropped.shape[:2]
    cropped = cropped[:, int(w * 0.10):]
    return cv2.GaussianBlur(
        cv2.resize(cropped, (512, 512)),
        (5, 5), 0)


def _brightness_contrast_mask(img_blur):
    """
    Fallback segmentation using brightness contrast.
    Large closing kernel bridges notebook-line gaps.
    """
    gray    = cv2.cvtColor(img_blur, cv2.COLOR_RGB2GRAY)
    bg_mean = float(np.mean(gray))
    bg_std  = float(np.std(gray))

    if bg_mean < 100 or bg_std > 60:
        return None

    dark_thresh = max(0, int(bg_mean - max(bg_std * 1.5, 25)))
    _, mask = cv2.threshold(gray, dark_thresh, 255, cv2.THRESH_BINARY_INV)

    k_close = np.ones((15, 15), np.uint8)
    k_open  = np.ones((5,  5),  np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k_open)

    # Shadow suppression
    mask = cv2.erode(mask,  np.ones((7, 7), np.uint8), iterations=2)
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=2)

    return mask


def _estimate_distance_scale(img_blur, mask):
    baseline_area = np.pi * (PIXELS_PER_CM * 1.25) ** 2
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas   = [cv2.contourArea(c) for c in cnts if cv2.contourArea(c) > 200]
    if not areas:
        return 1.0
    median_area = float(np.median(areas))
    scale       = np.sqrt(median_area / baseline_area)
    return float(np.clip(scale, 0.4, 3.5))


def _merge_overlapping_mask_blobs(mask):
    """
    Groups nearby mask fragments (caused by notebook lines or shadows)
    and redraws them as single convex-hull blobs.
    PROXIMITY controls how far apart two fragments can be and still merge.
    """
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) <= 1:
        return mask

    rects  = [cv2.boundingRect(c) for c in cnts]
    parent = list(range(len(rects)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[find(i)] = find(j)

    # Increased proximity: merge blobs up to 30px apart
    PROXIMITY = 30

    for i in range(len(rects)):
        xi, yi, wi, hi = rects[i]
        for j in range(i + 1, len(rects)):
            xj, yj, wj, hj = rects[j]
            gap_x = max(0, max(xi, xj) - min(xi + wi, xj + wj))
            gap_y = max(0, max(yi, yj) - min(yi + hi, yj + hj))
            if gap_x <= PROXIMITY and gap_y <= PROXIMITY:
                union(i, j)

    merged = np.zeros_like(mask)
    groups = {}
    for i, cnt in enumerate(cnts):
        g = find(i)
        groups.setdefault(g, []).append(cnt)

    for g, group_cnts in groups.items():
        all_pts = np.vstack(group_cnts)
        hull    = cv2.convexHull(all_pts)
        cv2.fillPoly(merged, [hull], 255)

    return merged


def segment_fruit(img_blur, scale=1.0):
    hsv   = cv2.cvtColor(img_blur, cv2.COLOR_RGB2HSV)
    mask1 = cv2.inRange(hsv, HSV_GREEN_LO, HSV_GREEN_HI)
    mask2 = cv2.inRange(hsv, HSV_RIPE_LO,  HSV_RIPE_HI)
    hsv_mask = cv2.bitwise_or(mask1, mask2)
    k        = np.ones((3, 3), np.uint8)
    hsv_mask = cv2.morphologyEx(hsv_mask, cv2.MORPH_CLOSE, k)
    hsv_mask = cv2.morphologyEx(hsv_mask, cv2.MORPH_OPEN,  k)

    bc_mask = _brightness_contrast_mask(img_blur)
    if bc_mask is not None:
        mask = cv2.bitwise_or(hsv_mask, bc_mask)
    else:
        mask = hsv_mask

    # Merge nearby fragments before watershed
    mask = _merge_overlapping_mask_blobs(mask)

    max_fruit_area_px = np.pi * (MAX_DIAMETER_CM / 2 * PIXELS_PER_CM) ** 2
    size_ceiling      = max_fruit_area_px * 3 * (scale ** 2)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in cnts:
        if cv2.contourArea(cnt) > size_ceiling:
            cv2.drawContours(mask, [cnt], -1, 0, -1)

    return mask, hsv


def _is_valid_scan_scene(img_blur, mask):
    H, W     = img_blur.shape[:2]
    total_px = H * W
    fruit_px = int(np.sum(mask > 0))

    if fruit_px / total_px > MAX_FRUIT_COVERAGE:
        return False

    bg_mask   = cv2.bitwise_not(mask)
    gray      = cv2.cvtColor(img_blur, cv2.COLOR_RGB2GRAY)
    bg_pixels = gray[bg_mask > 0]

    if len(bg_pixels) > 0:
        bg_mean = float(np.mean(bg_pixels))
        bg_std  = float(np.std(bg_pixels))
        if bg_mean < MIN_BG_MEAN:
            return False
        if bg_std > MAX_BG_STD:
            return False

    return True


def detect_basket(img_blur):
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
    px = hsv[mask_region > 0]
    if len(px) == 0:
        return False
    avg_sat = np.mean(px[:, 1])
    avg_val = np.mean(px[:, 2])
    if avg_sat < MIN_AVG_SAT:
        return False
    if avg_val > 240:
        return False
    return True


def compute_features(cnt, mask, hsv, ppc):
    area_px   = cv2.contourArea(cnt)
    peri_px   = cv2.arcLength(cnt, True)
    (_, _), r = cv2.minEnclosingCircle(cnt)
    single    = np.zeros_like(mask)
    cv2.drawContours(single, [cnt], -1, 255, -1)
    diam = (2 * r) / ppc
    px   = hsv[single > 0]
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


def _nms_circles(circles, iou_thresh=0.5):
    """Non-Maximum Suppression for Hough circles."""
    if not circles:
        return []
    circles = sorted(circles, key=lambda c: c[2], reverse=True)
    kept = []
    for cx, cy, cr in circles:
        duplicate = False
        for kx, ky, kr in kept:
            dist = np.sqrt((cx - kx) ** 2 + (cy - ky) ** 2)
            if dist < (cr + kr) * iou_thresh:
                duplicate = True
                break
        if not duplicate:
            kept.append((cx, cy, cr))
    return kept


def _dedup_contours(feats, cnts):
    """
    Aggressive deduplication of contours.

    Two contours are considered the SAME fruit if either:
      (a) their enclosing-circle centers are within 1 radius of each other, OR
      (b) the smaller circle's center falls INSIDE the larger circle.

    Only the contour with the largest area is kept per group.
    This handles the fruit+shadow split seen in the screenshots.
    """
    if len(cnts) <= 1:
        return feats, cnts

    # Build (cx, cy, r, area) for each contour
    info = []
    for cnt in cnts:
        (cx, cy), r = cv2.minEnclosingCircle(cnt)
        area = cv2.contourArea(cnt)
        info.append((cx, cy, r, area))

    parent = list(range(len(cnts)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[find(i)] = find(j)

    for i in range(len(info)):
        cx_i, cy_i, r_i, _ = info[i]
        for j in range(i + 1, len(info)):
            cx_j, cy_j, r_j, _ = info[j]
            dist = np.sqrt((cx_i - cx_j) ** 2 + (cy_i - cy_j) ** 2)
            max_r = max(r_i, r_j)
            # Merge if centers are within 1x the larger radius of each other
            # (very aggressive — catches any overlapping pair on the same fruit)
            if dist < max_r * 1.2:
                union(i, j)

    groups = {}
    for i in range(len(cnts)):
        g = find(i)
        groups.setdefault(g, []).append(i)

    kept_feats, kept_cnts = [], []
    for g, idxs in groups.items():
        # Keep the one with the largest contour area
        best_idx = max(idxs, key=lambda i: info[i][3])
        kept_feats.append(feats[best_idx])
        kept_cnts.append(cnts[best_idx])

    return kept_feats, kept_cnts


def get_fruit_features(img_blur, mask, hsv, ppc):
    """
    Watershed segmentation with high threshold + aggressive post-dedup.
    """
    dist      = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    dist_norm = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # High threshold: one seed per fruit, not per fragment
    _, sure_fg = cv2.threshold(dist_norm, 0.72 * dist_norm.max(), 255, 0)
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

    # Aggressive dedup — merges any contours on the same fruit
    feats, cnts = _dedup_contours(feats, cnts)

    return feats, cnts


def calibrate_ppc(hough_circles, scale=1.0):
    if not hough_circles:
        return PIXELS_PER_CM * scale
    avg_r_px = np.mean([r for _, _, r in hough_circles])
    return max(avg_r_px / 1.25, 5.0)


def get_features_from_hough(img_blur, circles, ppc):
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
        px       = hsv[mask_c > 0]
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


def count_hough(img_blur, mask, scale=1.0):
    gray = cv2.cvtColor(img_blur, cv2.COLOR_RGB2GRAY)
    hsv  = cv2.cvtColor(img_blur, cv2.COLOR_RGB2HSV)

    total_px = mask.shape[0] * mask.shape[1]
    fruit_px = np.sum(mask > 0)
    if fruit_px / total_px < MIN_COVERAGE:
        return []

    if not _is_valid_scan_scene(img_blur, mask):
        return []

    mean_v        = float(gray.mean())
    bright_thresh = min(int(mean_v - 15), 200)

    min_r = max(5,   int(8  * scale))
    max_r = min(150, int(80 * scale))
    min_d = max(15,  int(40 * scale))

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min_d,
        param1=35,
        param2=18,
        minRadius=min_r,
        maxRadius=max_r,
    )
    if circles is None:
        return []

    circles = np.round(circles[0, :]).astype("int")
    valid   = []
    for x, y, r in circles:
        if not (0 <= y < gray.shape[0] and 0 <= x < gray.shape[1]):
            continue
        y1, y2 = max(0, y-3), min(gray.shape[0], y+4)
        x1, x2 = max(0, x-3), min(gray.shape[1], x+4)

        center_gray = float(np.mean(gray[y1:y2, x1:x2]))
        if center_gray > bright_thresh:
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

    return _nms_circles(valid, iou_thresh=0.5)


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
    1. Estimates distance scale from blob sizes
    2. Picks best frame (most fruits visible)
    3. Detects each individual calamansi using watershed + aggressive dedup
    4. Returns features per fruit for juice prediction
    """
    best_frame = None
    best_count = 0

    for frame_rgb in frames_rgb:
        img_blur    = preprocess_array(frame_rgb)
        raw_mask, _ = segment_fruit(img_blur, scale=1.0)
        scale       = _estimate_distance_scale(img_blur, raw_mask)
        mask, hsv   = segment_fruit(img_blur, scale=scale)
        circles     = count_hough(img_blur, mask, scale=scale)
        if len(circles) > best_count:
            best_count = len(circles)
            best_frame = {
                'img_blur': img_blur,
                'mask':     mask,
                'hsv':      hsv,
                'circles':  circles,
                'scale':    scale,
            }

    if best_frame is None:
        return None

    img_blur = best_frame['img_blur']
    mask     = best_frame['mask']
    hsv      = best_frame['hsv']
    circles  = best_frame['circles']
    scale    = best_frame['scale']

    ppc = calibrate_ppc(circles, scale=scale)
    basket_contour, _        = detect_basket(img_blur)
    all_features, valid_cnts = get_fruit_features(img_blur, mask, hsv, ppc)

    return {
        'features':       all_features,
        'contours':       valid_cnts,
        'img_blur':       img_blur,
        'ppc':            ppc,
        'scale':          scale,
        'basket_contour': basket_contour,
        'hough_circles':  circles,
        'fruit_count':    len(all_features),
    }