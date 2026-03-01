import cv2
import numpy as np


# ─────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────
PIXELS_PER_CM = 37.8  # adjust based on your camera setup
IMAGE_SIZE    = (512, 512)

FEATURE_COLS = [
    'area_cm2',
    'diameter_cm',
    'perimeter_cm',
    'estimated_volume_cm3',
    'circularity',
    'mean_hue',
    'mean_saturation',
    'mean_value'
]


# ─────────────────────────────────────────
# STEP 1: PREPROCESS
# ─────────────────────────────────────────
def preprocess(image_array):
    """Resize and denoise the image."""
    resized = cv2.resize(image_array, IMAGE_SIZE)
    blurred = cv2.GaussianBlur(resized, (5, 5), 0)
    return blurred


# ─────────────────────────────────────────
# STEP 2: SEGMENT FRUIT
# ─────────────────────────────────────────
def segment(image_array):
    """
    Isolate the calamansi from the background
    using HSV color masking.
    Returns the binary mask.
    """
    hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)

    # HSV range for calamansi (green to orange-yellow)
    lower = np.array([15, 40, 40])
    upper = np.array([85, 255, 255])
    mask  = cv2.inRange(hsv, lower, upper)

    # Clean up the mask
    kernel = np.ones((7, 7), np.uint8)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

    return mask


# ─────────────────────────────────────────
# STEP 3: EXTRACT FEATURES
# ─────────────────────────────────────────
def extract_features_from_array(image_array, pixels_per_cm=PIXELS_PER_CM):
    """
    Given an RGB image array, extract shape and color features.
    Returns a feature dict or None if extraction fails.
    """
    preprocessed = preprocess(image_array)
    mask         = segment(preprocessed)
    hsv          = cv2.cvtColor(preprocessed, cv2.COLOR_RGB2HSV)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None

    cnt          = max(contours, key=cv2.contourArea)
    area_px      = cv2.contourArea(cnt)
    perimeter_px = cv2.arcLength(cnt, True)

    if perimeter_px == 0:
        return None

    (cx, cy), radius_px = cv2.minEnclosingCircle(cnt)

    # Convert to real-world units
    area_cm2    = area_px      / (pixels_per_cm ** 2)
    diameter_cm = (2 * radius_px) / pixels_per_cm
    perim_cm    = perimeter_px / pixels_per_cm

    features = {
        'area_cm2'            : area_cm2,
        'diameter_cm'         : diameter_cm,
        'perimeter_cm'        : perim_cm,
        'estimated_volume_cm3': (4/3) * np.pi * (diameter_cm / 2) ** 3,
        'circularity'         : (4 * np.pi * area_px) / (perimeter_px ** 2 + 1e-5),
    }

    # Color features
    fruit_pixels = hsv[mask > 0]
    if len(fruit_pixels) == 0:
        return None

    features['mean_hue']        = float(np.mean(fruit_pixels[:, 0]))
    features['mean_saturation'] = float(np.mean(fruit_pixels[:, 1]))
    features['mean_value']      = float(np.mean(fruit_pixels[:, 2]))

    return features


def extract_features_from_path(image_path, pixels_per_cm=PIXELS_PER_CM):
    """
    Load image from file path then extract features.
    Returns a feature dict or None if loading/extraction fails.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"  [ERROR] Cannot read image: {image_path}")
        return None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return extract_features_from_array(img_rgb, pixels_per_cm)


def get_segmented_image(image_array):
    """Return the segmented (masked) image for display."""
    preprocessed = preprocess(image_array)
    mask         = segment(preprocessed)
    segmented    = cv2.bitwise_and(preprocessed, preprocessed, mask=mask)
    return segmented, mask
