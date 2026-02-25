"""
Central configuration for the ASL Sign Language Detector project.
All constants and settings are defined here.
"""
import os

# ============================================================
# Project Paths
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_PATH = os.path.join(BASE_DIR, 'model.p')
DATASET_PATH = os.path.join(BASE_DIR, 'data.pickle')
EXPORTS_DIR = os.path.join(BASE_DIR, 'exports')

# ============================================================
# ASL Labels  (A-Z + special gestures)
# ============================================================
LABELS = {}
for _i in range(26):
    LABELS[_i] = chr(65 + _i)       # 0→A, 1→B, ... 25→Z
LABELS[26] = 'DEL'                   # Delete last character
LABELS[27] = 'NOTHING'               # Idle / no action
LABELS[28] = 'SPACE'                 # Space between words

# Reverse mapping  (character → class index)
LABELS_REVERSE = {v: k for k, v in LABELS.items()}

NUM_CLASSES = len(LABELS)            # 29

# Kaggle dataset class folder names → our indices
KAGGLE_CLASS_MAP = {}
for _i in range(26):
    KAGGLE_CLASS_MAP[chr(65 + _i)] = _i
KAGGLE_CLASS_MAP['del'] = 26
KAGGLE_CLASS_MAP['nothing'] = 27
KAGGLE_CLASS_MAP['space'] = 28

# ============================================================
# Data Collection Settings
# ============================================================
DATASET_SIZE_PER_CLASS = 200         # Images per class (webcam collection)
DOWNLOAD_IMAGES_PER_CLASS = 400      # Images per class from downloaded dataset
MIN_DETECTION_CONFIDENCE = 0.3       # MediaPipe hand detection confidence

# ============================================================
# Feature Extraction Settings
# ============================================================
NUM_LANDMARKS = 21                   # MediaPipe hand landmark count
NUM_FEATURES = NUM_LANDMARKS * 2     # x, y per landmark → 42 features

# ============================================================
# Training Settings
# ============================================================
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ============================================================
# Inference / Real-time Detection Settings
# ============================================================
STABILITY_THRESHOLD = 10             # Consecutive frames to confirm a letter
CONFIDENCE_THRESHOLD = 0.35          # Minimum prediction probability
COOLDOWN_FRAMES = 6                  # Pause frames after registering a letter

# ============================================================
# ASL Number ↔ Letter Mapping
# ============================================================
# ASL numbers look like certain letters.  When Number Mode is ON
# the detected letter is remapped to the corresponding digit.
LETTER_TO_NUMBER = {
    'D': '1',    # index finger up
    'V': '2',    # peace / two fingers
    'W': '3',    # three fingers
    'B': '4',    # four fingers extended
    'A': '5',    # (open palm variant – A with thumb out ≈ 5)
    'Y': '6',    # thumb + pinky = "six" / "hang loose"
    'S': '7',    # fist-variant with index/middle tucked
    'L': '8',    # thumb-index L-shape variant
    'F': '9',    # thumb-index circle, others up
    'O': '0',    # fingers form a circle
}

# ============================================================
# UI Colors  (BGR for OpenCV)
# ============================================================
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_BLUE = (255, 180, 0)
COLOR_DARK_BG = (45, 30, 30)
COLOR_PANEL_BG = (50, 40, 40)
COLOR_ACCENT = (0, 220, 255)
