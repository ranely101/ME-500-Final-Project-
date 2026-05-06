import cv2
import numpy as np
import time

# =========================
# SETTINGS
# =========================

CAMERA_INDEX = 1
WARMUP_TIME = 1.0

# -------------------------------------------------
# CAMERA FOCUS SETTINGS
# -------------------------------------------------
USE_AUTOFOCUS = False
MANUAL_FOCUS_VALUE = 30
FOCUS_SETTLE_TIME = 0.7
FOCUS_READ_FRAMES = 8

# -------------------------------------------------
# REAL-WORLD CALIBRATION
# -------------------------------------------------
# The white body width is assumed to be 1.5 inches.
# We measure the body width each run from the captured
# image and compute px/in from that image.
# -------------------------------------------------
BODY_REAL_WIDTH_IN = 1.5

# -------------------------------------------------
# PASS/FAIL TOLERANCE
# -------------------------------------------------
# This is the actual engineering tolerance in inches.
# The code converts it to pixels using the measured
# body width from THIS image.
# -------------------------------------------------
ALIGN_TOLERANCE_IN = 0.05

# -------------------------------------------------
# CROP
# -------------------------------------------------
# Matching testt.py
# -------------------------------------------------
CROP_X = 200
CROP_Y = 236
CROP_W = 280
CROP_H = 144

# -------------------------------------------------
# BODY ROI
# -------------------------------------------------
# Matching testt.py
# -------------------------------------------------
BODY_ROI_Y1 = 55
BODY_ROI_Y2 = 100

# -------------------------------------------------
# LID ROI
# -------------------------------------------------
# Matching testt.py
# -------------------------------------------------
LID_ROI_Y1 = 100
LID_ROI_Y2 = 150

# -------------------------------------------------
# BODY DETECTION
# -------------------------------------------------
BODY_THRESHOLD = 178
MIN_BODY_ROW_PIXELS = 34

# -------------------------------------------------
# IMAGE PREPROCESSING
# -------------------------------------------------
CONTRAST_ALPHA = 1.08
BRIGHTNESS_BETA = -18

# -------------------------------------------------
# LID DETECTION
# -------------------------------------------------
LID_EDGE_THRESHOLD = 20
LID_SEARCH_MARGIN_PX = 26
MIN_LID_COLUMN_SIGNAL = 3

# -------------------------------------------------
# OPTIONAL: SAVE CAPTURED IMAGE
# -------------------------------------------------
SAVE_CAPTURED_IMAGE = True
CAPTURED_IMAGE_PATH = r"C:/Users/adml-admin/Downloads/image.jpg"

# =========================
# FUNCTIONS
# =========================


def capture_image_from_webcam(camera_index=1, warmup_time=1.0):
    """
    Captures one image from the webcam.
    Keeps the focus behavior from flipfocus.py
    """
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam at index {camera_index}")

    time.sleep(warmup_time)

    if USE_AUTOFOCUS:
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    else:
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        cap.set(cv2.CAP_PROP_FOCUS, MANUAL_FOCUS_VALUE)

    time.sleep(FOCUS_SETTLE_TIME)

    for _ in range(FOCUS_READ_FRAMES):
        ret, frame = cap.read()
        if not ret:
            cap.release()
            raise RuntimeError("Could not capture image from webcam.")

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        raise RuntimeError("Could not capture image from webcam.")

    return frame


def preprocess(image):
    """
    Light contrast / brightness adjustment.
    Matching testt.py
    """
    return cv2.convertScaleAbs(image,
                               alpha=CONTRAST_ALPHA,
                               beta=BRIGHTNESS_BETA)


def detect_body_edges(roi):
    """
    Detects left/right body edges from the white body region.

    Logic:
    - Convert ROI to grayscale
    - Threshold to isolate bright white body
    - For each row, find first and last white pixel
    - Use median left/right over valid rows for robustness
    """
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, BODY_THRESHOLD, 255, cv2.THRESH_BINARY)

    left_candidates = []
    right_candidates = []

    for row in range(binary.shape[0]):
        pixels = np.where(binary[row] > 0)[0]

        if len(pixels) < MIN_BODY_ROW_PIXELS:
            continue

        left_candidates.append(int(pixels[0]))
        right_candidates.append(int(pixels[-1]))

    if not left_candidates:
        return None, None, binary

    body_l = int(np.median(left_candidates))
    body_r = int(np.median(right_candidates))

    return body_l, body_r, binary


def detect_lid_edges(roi, body_l, body_r):
    """
    Detect lid left/right edges from vertical edge strength
    near the body edges.

    Fix:
    - left lid edge: use leftmost strong candidate
    - right lid edge: use rightmost strong candidate
    """
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Slight extra contrast only for lid detection
    gray = cv2.convertScaleAbs(gray, alpha=1.35, beta=0)

    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # Vertical edge detector
    sobel = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    sobel = np.absolute(sobel)
    sobel = np.uint8(sobel)

    _, edges = cv2.threshold(sobel, LID_EDGE_THRESHOLD, 255, cv2.THRESH_BINARY)

    h, w = edges.shape

    # Search only near the body edges
    left_search_x1 = max(0, body_l - LID_SEARCH_MARGIN_PX)
    left_search_x2 = min(w, body_l + LID_SEARCH_MARGIN_PX)

    right_search_x1 = max(0, body_r - LID_SEARCH_MARGIN_PX)
    right_search_x2 = min(w, body_r + LID_SEARCH_MARGIN_PX)

    left_band = edges[:, left_search_x1:left_search_x2]
    right_band = edges[:, right_search_x1:right_search_x2]

    if left_band.size == 0 or right_band.size == 0:
        return None, None, edges

    left_strength = np.sum(left_band > 0, axis=0)
    right_strength = np.sum(right_band > 0, axis=0)

    if np.max(left_strength) < MIN_LID_COLUMN_SIGNAL:
        return None, None, edges

    if np.max(right_strength) < MIN_LID_COLUMN_SIGNAL:
        return None, None, edges

    # Strong candidate columns only
    left_candidates = np.where(left_strength >= 0.65 *
                               np.max(left_strength))[0]
    right_candidates = np.where(right_strength >= 0.65 *
                                np.max(right_strength))[0]

    if len(left_candidates) == 0 or len(right_candidates) == 0:
        return None, None, edges

    # Left lid edge = leftmost strong candidate
    lid_l = left_search_x1 + int(left_candidates[0])

    # Right lid edge = rightmost strong candidate
    lid_r = right_search_x1 + int(right_candidates[-1])

    if lid_r <= lid_l:
        return None, None, edges

    return lid_l, lid_r, edges


def classify(body_l, body_r, lid_l, lid_r):
    """
    Convert pixel measurements to inches using the measured
    body width in THIS image.
    Matching testt.py
    """
    body_width_px = body_r - body_l + 1

    px_per_in = body_width_px / BODY_REAL_WIDTH_IN
    in_per_px = 1.0 / px_per_in

    pixel_tolerance = ALIGN_TOLERANCE_IN * px_per_in

    left_offset_px = body_l - lid_l
    right_offset_px = body_r - lid_r

    left_offset_in = left_offset_px * in_per_px
    right_offset_in = right_offset_px * in_per_px

    passed = (abs(left_offset_px) <= pixel_tolerance
              and abs(right_offset_px) <= pixel_tolerance)

    return {
        "passed": passed,
        "body_width_px": body_width_px,
        "body_width_in": BODY_REAL_WIDTH_IN,
        "px_per_in": px_per_in,
        "in_per_px": in_per_px,
        "pixel_tolerance": pixel_tolerance,
        "left_offset_px": left_offset_px,
        "right_offset_px": right_offset_px,
        "left_offset_in": left_offset_in,
        "right_offset_in": right_offset_in
    }


# =========================
# MAIN
# =========================


def run_edge_detection():
    """
    Main pipeline:
    1. Capture image
    2. Preprocess
    3. Crop
    4. Extract ROIs
    5. Detect body edges
    6. Detect lid edges
    7. Convert px to inches using measured body width
    8. Pass/fail decision
    """
    image = capture_image_from_webcam(CAMERA_INDEX, WARMUP_TIME)

    if SAVE_CAPTURED_IMAGE:
        cv2.imwrite(CAPTURED_IMAGE_PATH, image)

    image = preprocess(image)

    print("Image shape:", image.shape)

    # Crop
    crop = image[CROP_Y:CROP_Y + CROP_H, CROP_X:CROP_X + CROP_W]
    print("Crop shape:", crop.shape)

    if crop.size == 0:
        raise RuntimeError("Crop is empty. Check crop values.")

    # ROIs
    body_roi = crop[BODY_ROI_Y1:BODY_ROI_Y2, :]
    lid_roi = crop[LID_ROI_Y1:LID_ROI_Y2, :]

    print("Body ROI shape:", body_roi.shape)
    print("Lid ROI shape:", lid_roi.shape)

    if body_roi.size == 0 or lid_roi.size == 0:
        raise RuntimeError("Body ROI or lid ROI is empty. Check ROI values.")

    # Detect body
    body_l, body_r, body_bin = detect_body_edges(body_roi)

    if body_l is None or body_r is None:
        result = {
            "status": "DETECTION_ERROR",
            "passed": False,
            "body_left": None,
            "body_right": None,
            "lid_left": None,
            "lid_right": None,
            "body_width_px": None,
            "body_width_in": None,
            "px_per_in": None,
            "in_per_px": None,
            "pixel_tolerance": None,
            "left_offset_px": None,
            "right_offset_px": None,
            "left_offset_in": None,
            "right_offset_in": None,
            "align_tolerance_in": ALIGN_TOLERANCE_IN
        }
        print("Vision result:", result)
        return result

    # Detect lid
    lid_l, lid_r, lid_edges = detect_lid_edges(lid_roi, body_l, body_r)

    if lid_l is None or lid_r is None:
        body_width_px = body_r - body_l + 1
        px_per_in = body_width_px / BODY_REAL_WIDTH_IN
        in_per_px = 1.0 / px_per_in
        pixel_tolerance = ALIGN_TOLERANCE_IN * px_per_in

        result = {
            "status": "DETECTION_ERROR",
            "passed": False,
            "body_left": body_l,
            "body_right": body_r,
            "lid_left": None,
            "lid_right": None,
            "body_width_px": body_width_px,
            "body_width_in": BODY_REAL_WIDTH_IN,
            "px_per_in": px_per_in,
            "in_per_px": in_per_px,
            "pixel_tolerance": pixel_tolerance,
            "left_offset_px": None,
            "right_offset_px": None,
            "left_offset_in": None,
            "right_offset_in": None,
            "align_tolerance_in": ALIGN_TOLERANCE_IN
        }
        print("Vision result:", result)
        return result

    # Classify
    classification = classify(body_l, body_r, lid_l, lid_r)

    result = {
        "status": "PASS" if classification["passed"] else "FAIL",
        "passed": classification["passed"],
        "body_left": body_l,
        "body_right": body_r,
        "lid_left": lid_l,
        "lid_right": lid_r,
        "body_width_px": classification["body_width_px"],
        "body_width_in": classification["body_width_in"],
        "px_per_in": classification["px_per_in"],
        "in_per_px": classification["in_per_px"],
        "pixel_tolerance": classification["pixel_tolerance"],
        "left_offset_px": classification["left_offset_px"],
        "right_offset_px": classification["right_offset_px"],
        "left_offset_in": classification["left_offset_in"],
        "right_offset_in": classification["right_offset_in"],
        "align_tolerance_in": ALIGN_TOLERANCE_IN
    }

    print("\n===== VISION RESULT =====")
    print(f"Status: {result['status']}")
    print(f"Body edges: left={body_l}, right={body_r}")
    print(f"Lid edges:  left={lid_l}, right={lid_r}")
    print(f"Measured body width: {result['body_width_px']} px")
    print(f"Assumed real body width: {result['body_width_in']:.4f} in")
    print(f"Measured scale: {result['px_per_in']:.4f} px/in")
    print(f"Measured scale: {result['in_per_px']:.6f} in/px")
    print(
        f"Left offset:  {result['left_offset_px']} px ({result['left_offset_in']:.4f} in)"
    )
    print(
        f"Right offset: {result['right_offset_px']} px ({result['right_offset_in']:.4f} in)"
    )
    print(
        f"Pass/fail tolerance: ±{ALIGN_TOLERANCE_IN:.4f} in (±{result['pixel_tolerance']:.2f} px)"
    )
    print("=========================\n")

    print("Vision result:", result)
    return result


if __name__ == "__main__":
    run_edge_detection()
