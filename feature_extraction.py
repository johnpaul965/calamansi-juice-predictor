import cv2
import numpy as np

# ─────────────────────────────────────────
# FEATURE EXTRACTION MODULE
# Used by: train_model.py, evaluate_model.py, app.py
# ─────────────────────────────────────────

PIXELS_PER_CM = 37.8  # Change this based on your camera calibration

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
    """Segment calamansi fruits using dual HSV color masking.
    Mask 1 covers dark green (unripe) calamansi.
    Mask 2 covers yellow-green to orange (maturing/ripe) calamansi.
    Both masks are combined to handle all ripeness stages.
    """
    hsv = cv2.cvtColor(img_blur, cv2.COLOR_RGB2HSV)

    # Mask 1: dark green calamansi (unripe)
    mask1 = cv2.inRange(hsv, np.array([35, 30, 30]), np.array([95, 255, 200]))
    # Mask 2: yellow-green to orange calamansi (maturing / ripe)
    mask2 = cv2.inRange(hsv, np.array([10, 40, 80]), np.array([40, 255, 255]))
    # Combine both masks
    mask = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones((9, 9), np.uint8)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    return mask, hsv

def compute_features(img_blur, mask, hsv, pixels_per_cm=PIXELS_PER_CM):
    """Extract shape and color features from segmented fruit."""
    features = {}

    # SHAPE FEATURES
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    cnt          = max(contours, key=cv2.contourArea)
    area_px      = cv2.contourArea(cnt)
    perimeter_px = cv2.arcLength(cnt, True)
    (cx, cy), radius_px = cv2.minEnclosingCircle(cnt)

    area_cm2    = area_px / (pixels_per_cm ** 2)
    diameter_cm = (2 * radius_px) / pixels_per_cm

    features['area_cm2']             = area_cm2
    features['diameter_cm']          = diameter_cm
    features['perimeter_cm']         = perimeter_px / pixels_per_cm
    features['circularity']          = (4 * np.pi * area_px) / (perimeter_px ** 2 + 1e-5)
    features['estimated_volume_cm3'] = (4/3) * np.pi * (diameter_cm / 2) ** 3

    # COLOR FEATURES
    fruit_pixels = hsv[mask > 0]
    if len(fruit_pixels) == 0:
        return None

    features['mean_hue']        = np.mean(fruit_pixels[:, 0])
    features['mean_saturation'] = np.mean(fruit_pixels[:, 1])
    features['mean_value']      = np.mean(fruit_pixels[:, 2])

    return features


def extract_features_from_path(image_path, pixels_per_cm=PIXELS_PER_CM):
    """Full pipeline from image file path."""
    img_blur = preprocess_image(image_path)
    if img_blur is None:
        print(f"Could not read: {image_path}")
        return None
    mask, hsv = segment_fruit(img_blur)
    features  = compute_features(img_blur, mask, hsv, pixels_per_cm)
    return features


def extract_features_from_array(image_array, pixels_per_cm=PIXELS_PER_CM):
    """Full pipeline from numpy array (Streamlit upload)."""
    img_blur  = preprocess_array(image_array)
    mask, hsv = segment_fruit(img_blur)
    features  = compute_features(img_blur, mask, hsv, pixels_per_cm)
    return features, mask