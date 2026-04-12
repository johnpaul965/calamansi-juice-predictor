import cv2
import numpy as np

# ─────────────────────────────────────────
# FEATURE EXTRACTION MODULE
# Calamansi Juice Yield Prediction System
# FIXED VERSION - Better single-fruit detection
# ─────────────────────────────────────────

PIXELS_PER_CM   = 41.0
MIN_CIRCULARITY = 0.40
MIN_DIAMETER_CM = 1.5
MAX_DIAMETER_CM = 4.5

FEATURE_COLS = [
    'area_cm2', 'diameter_cm', 'perimeter_cm', 'circularity',
    'estimated_volume_cm3', 'mean_hue', 'mean_saturation', 'mean_value'
]

# ─────────────────────────────────────────
# HSV COLOR RANGES FOR CALAMANSI
# Tightened to exclude dark wood/shelf backgrounds.
# ─────────────────────────────────────────
# Green calamansi (unripe)
HSV_GREEN_LO = np.array([28, 45, 35])
HSV_GREEN_HI = np.array([88, 255, 210])
# Yellow-orange calamansi (ripe)
HSV_RIPE_LO  = np.array([12, 50, 50])
HSV_RIPE_HI  = np.array([38, 255, 255])

MIN_AVG_SAT  = 40
MIN_SAT_VAL  = 40
MIN_COVERAGE = 0.01
MIN_OVERLAP  = 0.50

# ─────────────────────────────────────────
# SCENE VALIDATION THRESHOLDS
# RELAXED for close-up shots and different angles
# ─────────────────────────────────────────
MAX_FRUIT_COVERAGE = 0.70   # Increased from 0.40 to allow close-up shots
MIN_BG_MEAN        = 80     # Reduced from 100 to accept more lighting conditions
MAX_BG_STD         = 70     # Increased from 55 to accept more varied backgrounds


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
    # Trim 10% from left edge to remove shelf/background bleed-in on the left.
    # Remove this line if your camera framing no longer shows the shelf.
    h, w = cropped.shape[:2]
    cropped = cropped[:, int(w * 0.10):]
    return cv2.GaussianBlur(
        cv2.resize(cropped, (512, 512)),
        (5, 5), 0)


def segment_fruit(img_blur):
    hsv   = cv2.cvtColor(img_blur, cv2.COLOR_RGB2HSV)
    mask1 = cv2.inRange(hsv, HSV_GREEN_LO, HSV_GREEN_HI)
    mask2 = cv2.inRange(hsv, HSV_RIPE_LO,  HSV_RIPE_HI)
    mask  = cv2.bitwise_or(mask1, mask2)
    k     = np.ones((3, 3), np.uint8)
    mask  = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask  = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)

    # Remove blobs too large to be individual fruits.
    # Shelves/tables register as one giant connected blob.
    max_fruit_area_px = np.pi * (MAX_DIAMETER_CM / 2 * PIXELS_PER_CM) ** 2
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in cnts:
        if cv2.contourArea(cnt) > max_fruit_area_px * 3:
            cv2.drawContours(mask, [cnt], -1, 0, -1)

    return mask, hsv


def _is_valid_scan_scene(img_blur, mask):
    """
    Reject frames that don't look like a proper fruit scan.

    A valid scan has:
      - Fruit mask covering only a small-to-moderate portion of frame
        (not flooding 70%+ which means background noise was segmented)
      - A mostly plain, bright background (white paper / table surface)
        with low pixel variance — cluttered rooms score high std

    Returns True only if the scene passes all checks.
    """
    H, W     = img_blur.shape[:2]
    total_px = H * W
    fruit_px = int(np.sum(mask > 0))

    # Check 1: mask coverage
    # If fruit mask floods more than 70% of the frame it's almost
    # certainly background objects being misidentified, not real fruit.
    coverage = fruit_px / total_px
    if coverage > MAX_FRUIT_COVERAGE:
        return False

    # Check 2: background must be plain and bright
    # Analyse the pixels NOT covered by the fruit mask.
    bg_mask  = cv2.bitwise_not(mask)
    gray     = cv2.cvtColor(img_blur, cv2.COLOR_RGB2GRAY)
    bg_pixels = gray[bg_mask > 0]

    if len(bg_pixels) > 0:
        bg_mean = float(np.mean(bg_pixels))
        bg_std  = float(np.std(bg_pixels))

        # Dark or cluttered background (desk, room, cat, cables …)
        if bg_mean < MIN_BG_MEAN:
            return False
        if bg_std > MAX_BG_STD:
            return False

    return True


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
    Returns False if the region looks like background, wood, or glare.
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
    if avg_val < 30 or avg_val > 235:
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

    # Raised threshold 0.4→0.55 to prevent watershed splitting
    # a single touching fruit into two separate regions
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


def non_max_suppression_circles(circles, overlap_thresh=0.3):
    """
    Apply Non-Maximum Suppression to remove overlapping circle detections.
    
    Args:
        circles: List of (x, y, r) tuples
        overlap_thresh: Maximum allowed IoU overlap (0.3 = 30% overlap allowed)
    
    Returns:
        List of non-overlapping circles
    """
    if len(circles) == 0:
        return []
    
    # Convert to numpy array for easier manipulation
    circles = np.array(circles)
    
    # Calculate areas
    areas = np.pi * circles[:, 2] ** 2
    
    # Sort by radius (larger circles first - they're more likely to be correct)
    idxs = np.argsort(circles[:, 2])[::-1]
    
    keep = []
    while len(idxs) > 0:
        # Take the circle with largest radius
        i = idxs[0]
        keep.append(i)
        
        # Calculate IoU with remaining circles
        xx1 = circles[i, 0]
        yy1 = circles[i, 1]
        rr1 = circles[i, 2]
        
        remaining_idxs = idxs[1:]
        if len(remaining_idxs) == 0:
            break
            
        # Calculate distance between circle centers
        xx2 = circles[remaining_idxs, 0]
        yy2 = circles[remaining_idxs, 1]
        rr2 = circles[remaining_idxs, 2]
        
        dist = np.sqrt((xx1 - xx2) ** 2 + (yy1 - yy2) ** 2)
        
        # Calculate IoU approximation for circles
        # Two circles overlap if distance < r1 + r2
        overlap_mask = dist < (rr1 + rr2)
        
        # For overlapping circles, calculate approximate IoU
        iou = np.zeros(len(remaining_idxs))
        for j, idx in enumerate(remaining_idxs):
            if overlap_mask[j]:
                d = dist[j]
                r1, r2 = rr1, rr2[j]
                
                # If one circle is inside the other
                if d <= abs(r1 - r2):
                    smaller_area = np.pi * min(r1, r2) ** 2
                    larger_area = np.pi * max(r1, r2) ** 2
                    iou[j] = smaller_area / larger_area
                # If circles overlap
                elif d < r1 + r2:
                    # Approximation of circle intersection area
                    part1 = r1 ** 2 * np.arccos((d ** 2 + r1 ** 2 - r2 ** 2) / (2 * d * r1))
                    part2 = r2 ** 2 * np.arccos((d ** 2 + r2 ** 2 - r1 ** 2) / (2 * d * r2))
                    part3 = 0.5 * np.sqrt((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2))
                    intersection = part1 + part2 - part3
                    union = areas[i] + areas[idx] - intersection
                    iou[j] = intersection / union if union > 0 else 0
        
        # Keep only circles with IoU less than threshold
        idxs = remaining_idxs[iou <= overlap_thresh]
    
    return circles[keep].tolist()


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
    NOW WITH NON-MAXIMUM SUPPRESSION to eliminate duplicate detections.
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

    # Guard 3: Scene must look like a proper fruit scan setup.
    # Rejects cluttered rooms, desks, pets, cables, etc.
    if not _is_valid_scan_scene(img_blur, mask):
        return []

    mean_v = float(gray.mean())
    bright_thresh = min(155, int(mean_v + 10))

    # IMPROVED HOUGH PARAMETERS for better detection
    # Increased minDist to prevent multiple detections on same fruit
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=60,              # Increased from 45 to 60 - prevents overlapping detections
        param1=45,
        param2=18,               # Reduced from 22 to 18 - more sensitive detection for close-ups
        minRadius=8,
        maxRadius=60             # Increased from 42 to 60 - allows larger fruits in close-up
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

    # APPLY NON-MAXIMUM SUPPRESSION
    # This eliminates duplicate/overlapping detections on the same fruit
    valid = non_max_suppression_circles(valid, overlap_thresh=0.3)

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

    Returns None if no valid scan scene is found across all frames.
    The caller (Streamlit app) should check for None and show a warning:

        result = process_video_frames(frames)
        if result is None or result['fruit_count'] == 0:
            st.warning("⚠️ No calamansi detected. Point the camera at "
                       "fruits on a plain white surface.")
        elif result['fruit_count'] > 30:
            st.warning("⚠️ Too many detections — ensure only fruits are "
                       "in frame on a plain background.")
        else:
            st.success(f"✅ {result['fruit_count']} calamansi detected!")
    """
    best_frame = None
    best_count = 0

    for frame_rgb in frames_rgb:
        img_blur  = preprocess_array(frame_rgb)
        mask, hsv = segment_fruit(img_blur)
        circles   = count_hough(img_blur, mask)  # returns [] if scene invalid
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