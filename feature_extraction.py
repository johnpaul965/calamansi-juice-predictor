import cv2
import numpy as np

# ─────────────────────────────────────────
# FEATURE EXTRACTION MODULE
# Used by: train_model.py, evaluate_model.py, app.py
# ─────────────────────────────────────────

PIXELS_PER_CM     = 41.0
MIN_FRUIT_AREA_PX = 400
MIN_CIRCULARITY   = 0.3   # fruits are roughly circular

FEATURE_COLS = [
    'area_cm2',
    'diameter_cm',
    'perimeter_cm',
    'circularity',
    'estimated_volume_cm3',
    'mean_hue',
    'mean_saturation',
    'mean_value'
]


def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    img_rgb     = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (512, 512))
    img_blur    = cv2.GaussianBlur(img_resized, (5, 5), 0)
    return img_blur


def preprocess_array(image_array):
    img_resized = cv2.resize(image_array, (512, 512))
    img_blur    = cv2.GaussianBlur(img_resized, (5, 5), 0)
    return img_blur


def segment_fruit(img_blur):
    """Segment calamansi using HSV color masks with small morphology kernel."""
    hsv = cv2.cvtColor(img_blur, cv2.COLOR_RGB2HSV)

    # Mask 1: green (unripe)
    mask1 = cv2.inRange(hsv, np.array([25, 15, 40]), np.array([95, 255, 210]))
    # Mask 2: yellow-green to orange (ripe)
    mask2 = cv2.inRange(hsv, np.array([10, 30, 60]), np.array([40, 255, 255]))
    mask  = cv2.bitwise_or(mask1, mask2)

    # Use small kernel so tiny fruits are not erased
    kernel = np.ones((3, 3), np.uint8)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    return mask, hsv


def is_fruit_contour(cnt, img_shape=(512, 512)):
    """Check if contour is likely a calamansi fruit."""
    area = cv2.contourArea(cnt)
    if area < MIN_FRUIT_AREA_PX or area > 40000:
        return False
    peri = cv2.arcLength(cnt, True)
    if peri == 0:
        return False
    circularity = (4 * np.pi * area) / (peri ** 2)
    if circularity < MIN_CIRCULARITY:
        return False
    # Exclude contours touching the image border (likely background/table edges)
    H, W = img_shape
    x, y, w, h = cv2.boundingRect(cnt)
    if x <= 2 or y <= 2 or x + w >= W - 2 or y + h >= H - 2:
        return False
    return True


def compute_features_single(cnt, mask, hsv, pixels_per_cm=PIXELS_PER_CM):
    features = {}

    area_px      = cv2.contourArea(cnt)
    perimeter_px = cv2.arcLength(cnt, True)
    (cx, cy), radius_px = cv2.minEnclosingCircle(cnt)

    single_mask = np.zeros_like(mask)
    cv2.drawContours(single_mask, [cnt], -1, 255, -1)

    area_cm2    = area_px / (pixels_per_cm ** 2)
    diameter_cm = (2 * radius_px) / pixels_per_cm

    features['area_cm2']             = area_cm2
    features['diameter_cm']          = diameter_cm
    features['perimeter_cm']         = perimeter_px / pixels_per_cm
    features['circularity']          = (4 * np.pi * area_px) / (perimeter_px ** 2 + 1e-5)
    features['estimated_volume_cm3'] = (4/3) * np.pi * (diameter_cm / 2) ** 3

    fruit_pixels = hsv[single_mask > 0]
    if len(fruit_pixels) == 0:
        return None

    features['mean_hue']        = np.mean(fruit_pixels[:, 0])
    features['mean_saturation'] = np.mean(fruit_pixels[:, 1])
    features['mean_value']      = np.mean(fruit_pixels[:, 2])

    return features


def extract_all_fruits(img_blur, mask, hsv, pixels_per_cm=PIXELS_PER_CM):
    """Detect ALL fruits — returns list of feature dicts and contours."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fruit_contours = [c for c in contours if is_fruit_contour(c, mask.shape[:2])]

    all_features   = []
    valid_contours = []

    for cnt in fruit_contours:
        features = compute_features_single(cnt, mask, hsv, pixels_per_cm)
        if features is not None:
            all_features.append(features)
            valid_contours.append(cnt)

    return all_features, valid_contours


def extract_features_from_path(image_path, pixels_per_cm=PIXELS_PER_CM):
    """Used by train_model.py — single fruit per training image."""
    img_blur = preprocess_image(image_path)
    if img_blur is None:
        print(f"Could not read: {image_path}")
        return None
    mask, hsv      = segment_fruit(img_blur)
    contours, _    = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fruit_contours = [c for c in contours if is_fruit_contour(c, mask.shape[:2])]
    if not fruit_contours:
        return None
    cnt      = max(fruit_contours, key=cv2.contourArea)
    features = compute_features_single(cnt, mask, hsv, pixels_per_cm)
    return features


def extract_features_from_array(image_array, pixels_per_cm=PIXELS_PER_CM):
    """Used by app.py — detects ALL fruits in the uploaded image."""
    img_blur               = preprocess_array(image_array)
    mask, hsv              = segment_fruit(img_blur)
    all_features, contours = extract_all_fruits(img_blur, mask, hsv, pixels_per_cm)
    return all_features, contours, mask, img_blur