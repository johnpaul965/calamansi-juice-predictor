import cv2
import numpy as np

# ─────────────────────────────────────────
# FEATURE EXTRACTION MODULE
# Used by: train_model.py, evaluate_model.py, app.py
# ─────────────────────────────────────────

PIXELS_PER_CM = 37.8  # Change based on your camera calibration

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

# Minimum contour area in pixels to count as a fruit (filters out noise)
MIN_FRUIT_AREA_PX = 800


def preprocess_image(image_path):
    """Read and preprocess image from file path."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    img_rgb     = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (512, 512))
    img_blur    = cv2.GaussianBlur(img_resized, (5, 5), 0)
    return img_blur


def preprocess_array(image_array):
    """Preprocess image from numpy array (used in Streamlit)."""
    img_resized = cv2.resize(image_array, (512, 512))
    img_blur    = cv2.GaussianBlur(img_resized, (5, 5), 0)
    return img_blur


def segment_fruit(img_blur):
    """Segment calamansi fruits using HSV color masking."""
    hsv   = cv2.cvtColor(img_blur, cv2.COLOR_RGB2HSV)
    lower = np.array([15, 40, 40])
    upper = np.array([85, 255, 255])
    mask  = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((7, 7), np.uint8)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    return mask, hsv


def compute_features_single(cnt, mask, hsv, pixels_per_cm=PIXELS_PER_CM):
    """Extract features from a single fruit contour."""
    features = {}

    area_px      = cv2.contourArea(cnt)
    perimeter_px = cv2.arcLength(cnt, True)
    (cx, cy), radius_px = cv2.minEnclosingCircle(cnt)

    # Build mask for this single fruit only
    single_mask = np.zeros_like(mask)
    cv2.drawContours(single_mask, [cnt], -1, 255, -1)

    area_cm2    = area_px / (pixels_per_cm ** 2)
    diameter_cm = (2 * radius_px) / pixels_per_cm

    features['area_cm2']             = area_cm2
    features['diameter_cm']          = diameter_cm
    features['perimeter_cm']         = perimeter_px / pixels_per_cm
    features['circularity']          = (4 * np.pi * area_px) / (perimeter_px ** 2 + 1e-5)
    features['estimated_volume_cm3'] = (4/3) * np.pi * (diameter_cm / 2) ** 3

    # Color from this fruit's region only
    fruit_pixels = hsv[single_mask > 0]
    if len(fruit_pixels) == 0:
        return None

    features['mean_hue']        = np.mean(fruit_pixels[:, 0])
    features['mean_saturation'] = np.mean(fruit_pixels[:, 1])
    features['mean_value']      = np.mean(fruit_pixels[:, 2])

    return features


def extract_all_fruits(img_blur, mask, hsv, pixels_per_cm=PIXELS_PER_CM):
    """
    Detect ALL fruits in the image.
    Returns a list of features (one dict per fruit) and their contours.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out noise — only keep contours large enough to be a real fruit
    fruit_contours = [c for c in contours if cv2.contourArea(c) > MIN_FRUIT_AREA_PX]

    all_features = []
    valid_contours = []

    for cnt in fruit_contours:
        features = compute_features_single(cnt, mask, hsv, pixels_per_cm)
        if features is not None:
            all_features.append(features)
            valid_contours.append(cnt)

    return all_features, valid_contours


# ── For training (single fruit per image) ──
def extract_features_from_path(image_path, pixels_per_cm=PIXELS_PER_CM):
    """Used by train_model.py — single fruit per training image."""
    img_blur = preprocess_image(image_path)
    if img_blur is None:
        return None
    mask, hsv = segment_fruit(img_blur)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt      = max(contours, key=cv2.contourArea)
    features = compute_features_single(cnt, mask, hsv, pixels_per_cm)
    return features


# ── For the app (multiple fruits per image) ──
def extract_features_from_array(image_array, pixels_per_cm=PIXELS_PER_CM):
    """Used by app.py — detects ALL fruits in the uploaded image."""
    img_blur  = preprocess_array(image_array)
    mask, hsv = segment_fruit(img_blur)
    all_features, contours = extract_all_fruits(img_blur, mask, hsv, pixels_per_cm)
    return all_features, contours, mask, img_blur