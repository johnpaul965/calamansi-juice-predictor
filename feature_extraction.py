import cv2
import numpy as np

# ─────────────────────────────────────────
# FEATURE EXTRACTION MODULE
# Calamansi Juice Yield Prediction System
# VERSION: v6-single-fruit-fix
# ─────────────────────────────────────────
VERSION = "v7-multi-fruit-fix"
print(f"[feature_extraction] Loaded {VERSION}")

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
    mask = cv2.erode(mask,  np.ones((11, 11), np.uint8), iterations=2)
    mask = cv2.dilate(mask, np.ones((7,  7),  np.uint8), iterations=2)

    # Remove blobs whose mean brightness is too high (shadows are lighter)
    cnts_s, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    gray_s = cv2.cvtColor(img_blur, cv2.COLOR_RGB2GRAY)
    for cnt_s in cnts_s:
        tmp = np.zeros_like(mask)
        cv2.drawContours(tmp, [cnt_s], -1, 255, -1)
        mean_bright = float(np.mean(gray_s[tmp > 0]))
        if mean_bright > 100:
            cv2.drawContours(mask, [cnt_s], -1, 0, -1)

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


def _merge_overlapping_mask_blobs(mask, img_blur=None):
    """
    Groups nearby mask fragments and redraws as convex-hull blobs.
    FIX v6: PROXIMITY now uses ABSOLUTE pixel value (60px) as a floor
    so tiny fragments near a fruit always get merged regardless of their
    own radius.
    """
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) <= 1:
        return mask

    # Erase shadow/highlight blobs
    if img_blur is not None:
        gray     = cv2.cvtColor(img_blur, cv2.COLOR_RGB2GRAY)
        areas    = [cv2.contourArea(c) for c in cnts]
        max_area = max(areas)

        largest_cnt  = cnts[int(np.argmax(areas))]
        largest_mask = np.zeros_like(mask)
        cv2.drawContours(largest_mask, [largest_cnt], -1, 255, -1)
        fruit_brightness = float(np.mean(gray[largest_mask > 0]))

        clean = np.zeros_like(mask)
        for cnt, area in zip(cnts, areas):
            tmp = np.zeros_like(mask)
            cv2.drawContours(tmp, [cnt], -1, 255, -1)
            blob_brightness = float(np.mean(gray[tmp > 0]))
            if area >= max_area * 0.40 or blob_brightness <= fruit_brightness * 1.15:
                cv2.drawContours(clean, [cnt], -1, 255, -1)

        mask = clean
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(cnts) <= 1:
            return mask

    rects = [cv2.boundingRect(c) for c in cnts]
    parent = list(range(len(rects)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[find(i)] = find(j)

    # v7 FIX: use per-pair average radius with a small gap allowance.
    # Merges shadow/highlight fragments of the SAME fruit (gap < 0.5*r)
    # but does NOT merge two separate fruits sitting next to each other.
    circles_r = [cv2.minEnclosingCircle(c)[1] for c in cnts]

    for i in range(len(rects)):
        xi, yi, wi, hi = rects[i]
        for j in range(i + 1, len(rects)):
            xj, yj, wj, hj = rects[j]
            gap_x     = max(0, max(xi, xj) - min(xi + wi, xj + wj))
            gap_y     = max(0, max(yi, yj) - min(yi + hi, yj + hj))
            pair_r    = min(circles_r[i], circles_r[j])
            PROXIMITY = max(8, int(pair_r * 0.5))
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

    # Merge nearby fragments before further processing
    mask = _merge_overlapping_mask_blobs(mask, img_blur)

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


def get_fruit_features(img_blur, mask, hsv, ppc):
    """
    Detects individual calamansi fruits from the segmentation mask.

    v6 KEY FIX: After all blob filtering, we do one final aggressive
    spatial cluster check — if all surviving blobs fit within a circle
    of radius = largest_blob_radius * 2.5, they are ALL the same fruit.
    Merge them into a single convex hull and return 1 result.

    This directly fixes the "1 fruit detected as 4" bug seen when the
    fruit mask fragments into several small blobs due to shadows/highlights.
    """
    gray = cv2.cvtColor(img_blur, cv2.COLOR_RGB2GRAY)

    # ── 1. Raw blobs ──────────────────────────────────────────────────────
    raw_cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    raw_cnts = [c for c in raw_cnts if cv2.contourArea(c) > 100]
    if not raw_cnts:
        return [], []

    # ── 2. Score each blob by fruit color ────────────────────────────────
    blob_info = []
    for cnt in raw_cnts:
        tmp = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(tmp, [cnt], -1, 255, -1)
        px_gray = gray[tmp > 0]
        px_hsv  = hsv[tmp > 0]
        if len(px_gray) == 0:
            continue
        mean_bright = float(np.mean(px_gray))
        area        = cv2.contourArea(cnt)
        (cx, cy), r = cv2.minEnclosingCircle(cnt)
        mean_hue    = float(np.mean(px_hsv[:, 0]))
        mean_sat    = float(np.mean(px_hsv[:, 1]))
        hue_ok      = 1.0 if 20 <= mean_hue <= 105 else 0.0
        sat_score   = float(np.clip(mean_sat / 80.0, 0.0, 1.0))
        fruit_score = hue_ok * 0.6 + sat_score * 0.4
        blob_info.append({'cnt': cnt, 'area': area, 'bright': mean_bright,
                          'cx': cx, 'cy': cy, 'r': r,
                          'fruit_score': fruit_score,
                          'mean_hue': mean_hue, 'mean_sat': mean_sat})

    if not blob_info:
        return [], []

    max_area        = max(b['area']        for b in blob_info)
    max_fruit_score = max(b['fruit_score'] for b in blob_info)

    keep = []
    for b in blob_info:
        is_largest     = b['area'] >= max_area * 0.80
        good_color     = b['fruit_score'] >= max(0.3, max_fruit_score * 0.50)
        is_large       = b['area'] >= max_area * 0.25
        is_blue_shadow = b['mean_hue'] > 105 or b['mean_sat'] < 20
        if is_largest or good_color or (is_large and not is_blue_shadow):
            keep.append(b)

    if not keep:
        keep = [max(blob_info, key=lambda b: b['area'])]

    # ── v6 PRE-MERGE: if all kept blobs are spatially close, treat as ONE fruit ──
    # This is the primary fix for the "1 fruit → 4 detections" bug.
    if len(keep) > 1:
        max_r_keep = max(b['r'] for b in keep)
        xs = [b['cx'] for b in keep]
        ys = [b['cy'] for b in keep]
        centroid_x = float(np.mean(xs))
        centroid_y = float(np.mean(ys))
        # Max distance from centroid to any blob center
        max_spread = max(
            np.sqrt((b['cx'] - centroid_x)**2 + (b['cy'] - centroid_y)**2)
            for b in keep
        )
        # v7 FIX: tightened from 2.5x to 1.2x.
        # 2.5x was merging 5 separate clustered fruits into 1.
        # 1.2x only merges genuine shadow/highlight fragments of the same fruit.
        if max_spread < max_r_keep * 1.2:
            all_pts = np.vstack([b['cnt'] for b in keep])
            hull    = cv2.convexHull(all_pts)
            single  = np.zeros(mask.shape, dtype=np.uint8)
            cv2.drawContours(single, [hull], -1, 255, -1)
            if _has_fruit_color(hsv, single):
                f = compute_features(hull, mask, hsv, ppc)
                if f:
                    return [f], [hull]
            # fallback: just use the largest blob
            best_b = max(keep, key=lambda b: b['area'])
            f = compute_features(best_b['cnt'], mask, hsv, ppc)
            if f:
                return [f], [best_b['cnt']]
            return [], []
    # ─────────────────────────────────────────────────────────────────────

    # ── 3. Merge nearby blobs (same fruit, fragmented) ────────────────────
    parent = list(range(len(keep)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[find(i)] = find(j)

    # v7 FIX: tightened from 3.0x to 1.2x.
    # 3.0x was merging separate nearby fruits into one group.
    # 1.2x only catches genuine tiny fragments of the same fruit.
    global_max_r = max(b['r'] for b in keep)
    MERGE_RADIUS = max(global_max_r * 1.2, 12.0)

    for i in range(len(keep)):
        for j in range(i + 1, len(keep)):
            bi, bj = keep[i], keep[j]
            dist   = np.sqrt((bi['cx'] - bj['cx'])**2 + (bi['cy'] - bj['cy'])**2)
            if dist < MERGE_RADIUS:
                union(i, j)

    groups = {}
    for i in range(len(keep)):
        g = find(i)
        groups.setdefault(g, []).append(i)

    # ── 4. One result per group ───────────────────────────────────────────
    feats, result_cnts = [], []
    for g, idxs in groups.items():
        best = max(idxs, key=lambda i: keep[i]['area'])
        cnt  = keep[best]['cnt']

        if not is_calamansi(cnt, ppc):
            all_pts = np.vstack([keep[i]['cnt'] for i in idxs])
            hull    = cv2.convexHull(all_pts)
            if not is_calamansi(hull, ppc):
                continue
            cnt = hull

        single = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(single, [cnt], -1, 255, -1)
        if not _has_fruit_color(hsv, single):
            continue

        f = compute_features(cnt, mask, hsv, ppc)
        if f:
            feats.append(f)
            result_cnts.append(cnt)

    # ── 5. Final NMS on result circles ────────────────────────────────────
    if len(result_cnts) > 1:
        circles_info = []
        for cnt in result_cnts:
            (cx, cy), r = cv2.minEnclosingCircle(cnt)
            circles_info.append((cx, cy, r))

        dominated = set()
        for i in range(len(circles_info)):
            if i in dominated:
                continue
            for j in range(i + 1, len(circles_info)):
                if j in dominated:
                    continue
                cx_i, cy_i, r_i = circles_info[i]
                cx_j, cy_j, r_j = circles_info[j]
                dist = np.sqrt((cx_i - cx_j)**2 + (cy_i - cy_j)**2)
                if dist < (r_i + r_j) * 0.70:
                    area_i = cv2.contourArea(result_cnts[i])
                    area_j = cv2.contourArea(result_cnts[j])
                    dominated.add(j if area_i >= area_j else i)

        feats       = [f for i, f in enumerate(feats)       if i not in dominated]
        result_cnts = [c for i, c in enumerate(result_cnts) if i not in dominated]

    # ── 6. FINAL SAFETY: if all results overlap heavily → keep only largest ──
    if len(result_cnts) > 1:
        centers = []
        for cnt in result_cnts:
            (cx, cy), r = cv2.minEnclosingCircle(cnt)
            centers.append((cx, cy, r))

        xs = [c[0] for c in centers]
        ys = [c[1] for c in centers]
        spread   = np.sqrt((max(xs) - min(xs))**2 + (max(ys) - min(ys))**2)
        max_r_f  = max(c[2] for c in centers)

        if spread < max_r_f * 2.0:
            best_i     = max(range(len(result_cnts)),
                             key=lambda i: cv2.contourArea(result_cnts[i]))
            feats       = [feats[best_i]]
            result_cnts = [result_cnts[best_i]]

    return feats, result_cnts


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
    3. Detects each individual calamansi
    4. Returns features per fruit for juice prediction
    """
    print(f"[process_video_frames] Running {VERSION}")
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

    # Sanity-check count against mask area
    if all_features:
        total_mask_area = float(np.sum(mask > 0))
        avg_fruit_area  = float(np.mean([f['area_cm2'] for f in all_features]))
        ppc_sq          = ppc ** 2
        max_possible = max(1, int(total_mask_area / (avg_fruit_area * ppc_sq * 0.6)) + 1)

        if len(all_features) > max_possible:
            paired       = sorted(zip(all_features, valid_cnts),
                                  key=lambda x: x[0]['area_cm2'], reverse=True)
            all_features = [p[0] for p in paired[:max_possible]]
            valid_cnts   = [p[1] for p in paired[:max_possible]]

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