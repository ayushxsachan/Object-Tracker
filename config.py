"""Configuration for the smart desk object monitoring project.

Beginners: this is the main file to tweak when you want different behavior.
For example, lower the confidence threshold if the detector misses objects, or
increase the missing/misplaced thresholds if alerts happen too quickly.
"""

from pathlib import Path


# ----------------------------- Camera settings -----------------------------

# Most laptops use camera index 0. If you have multiple webcams, try 1 or 2.
CAMERA_INDEX = 0

# Requesting a smaller frame keeps detection faster. The webcam may choose the
# closest supported resolution.
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# Run the object detector every N frames. Increasing this improves FPS, but
# boxes may update a little less often. 1 = detect every frame.
DETECTION_INTERVAL = 2


# ---------------------------- Detector settings ----------------------------

# Options:
#   "auto"      -> try YOLO first, then MediaPipe
#   "yolo"      -> force YOLOv8 via ultralytics
#   "mediapipe" -> force MediaPipe Tasks object detector
DETECTOR_BACKEND = "auto"

# YOLOv8 nano is small and fast. You can replace this with a custom model path
# such as "models/my_wallet_keys_model.pt" if you train one.
YOLO_MODEL_NAME = "yolov8n.pt"
YOLO_CONFIDENCE_THRESHOLD = 0.35
YOLO_IMAGE_SIZE = 640

# Use "cpu" for maximum compatibility. If your machine has a supported GPU,
# change this to 0 or "cuda" after installing the right PyTorch build.
YOLO_DEVICE = "cpu"

# MediaPipe requires a .tflite object detection model. Download one and place it
# here if you want to use DETECTOR_BACKEND = "mediapipe".
MEDIAPIPE_MODEL_PATH = Path("models") / "efficientdet_lite0.tflite"
MEDIAPIPE_CONFIDENCE_THRESHOLD = 0.40

# Keep this set focused for a desk/table use case. If you want to detect every
# COCO class from YOLO, set WATCHED_CLASSES = set().
WATCHED_CLASSES = {
    "bottle",
    "cup",
    "phone",
    "wallet",
    "keys",
    "laptop",
    "book",
    "keyboard",
    "mouse",
    "remote",
    "backpack",
    "handbag",
    "suitcase",
    "scissors",
    "clock",
    "vase",
}

# Detector labels can be wordy. These names are displayed on screen and used by
# the tracker. Add your custom model labels here if you want friendlier names.
DISPLAY_NAME_OVERRIDES = {
    "cell phone": "phone",
    "sports ball": "ball",
}


# ---------------------------- Tracking settings ----------------------------

# First few seconds are used to learn the original desk layout. You can also
# press S at any time to save/re-save the baseline manually.
AUTO_BASELINE_SECONDS = 5.0

# A baseline object must be absent for this long before it is called missing.
MISSING_SECONDS_THRESHOLD = 2.0

# Object movement threshold. 0.12 means 12% of the frame diagonal. Increase this
# if normal detector jitter is causing "misplaced" alerts.
MISPLACED_DISTANCE_RATIO = 0.12

# Maximum distance for matching a detection to an existing track. This is wider
# than the misplaced threshold so a picked-up object can still be recognized as
# the same object after being moved across the desk.
MATCH_DISTANCE_RATIO = 0.35

# Unseen objects that were not part of the saved baseline are removed after this
# many seconds to prevent stale boxes.
FORGET_UNBASELINED_AFTER_SECONDS = 5.0

# Number of recent positions saved per object for history and debugging.
HISTORY_LENGTH = 60


# ------------------------------ Alert settings ------------------------------

ENABLE_SOUND_ALERT = True

# Avoid repeating the same beep too often for the same object/status.
ALERT_COOLDOWN_SECONDS = 6.0

# A red alert banner remains visible for this many seconds after a new event.
ALERT_BANNER_SECONDS = 4.0


# ------------------------------- Logging ------------------------------------

ENABLE_EVENT_LOGGING = True
LOG_FILE = Path("logs") / "events.csv"


# ------------------------------- UI colors ----------------------------------

# OpenCV uses BGR colors, not RGB.
COLOR_OK = (70, 220, 70)
COLOR_MISPLACED = (0, 190, 255)
COLOR_MISSING = (40, 40, 255)
COLOR_NEW = (255, 200, 70)
COLOR_TEXT = (255, 255, 255)
COLOR_PANEL = (20, 20, 20)
