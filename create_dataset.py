"""
Extract hand-landmark features from images and save as a pickle dataset.

Improvements over the original:
  • Fixed feature vector size (42 = 21 landmarks × 2 coords) regardless of
    how many hands are visible — only the first / right hand is used.
  • Progress bar via tqdm.
  • Robust error handling and skip logging.
"""

import os
import pickle
import cv2
import mediapipe as mp
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None          # graceful fallback

from config import (
    DATA_DIR, DATASET_PATH, NUM_FEATURES,
    MIN_DETECTION_CONFIDENCE,
)


def extract_features(data_dir: str = DATA_DIR,
                     output_path: str = DATASET_PATH):
    """
    Walk through data_dir/<class>/<image> and extract MediaPipe hand
    landmarks as normalised (x, y) pairs.  Returns and saves the dataset.
    """
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,                       # one hand → consistent features
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
    )

    data = []
    labels = []
    skipped = 0

    # Gather all (class_dir, image_file) pairs for progress tracking
    tasks = []
    for dir_name in sorted(os.listdir(data_dir)):
        dir_path = os.path.join(data_dir, dir_name)
        if not os.path.isdir(dir_path):
            continue
        for img_name in os.listdir(dir_path):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                tasks.append((dir_name, os.path.join(dir_path, img_name)))

    if not tasks:
        print("[✗] No images found in", data_dir)
        return

    print(f"\n▶  Processing {len(tasks)} images from {data_dir} …\n")

    iterator = tqdm(tasks, desc="Extracting", unit="img") if tqdm else tasks

    for class_label, file_path in iterator:
        img = cv2.imread(file_path)
        if img is None:
            skipped += 1
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if not results.multi_hand_landmarks:
            skipped += 1
            continue

        # Use the first detected hand
        hand = results.multi_hand_landmarks[0]
        x_coords = [lm.x for lm in hand.landmark]
        y_coords = [lm.y for lm in hand.landmark]
        min_x, min_y = min(x_coords), min(y_coords)

        features = []
        for lm in hand.landmark:
            features.append(lm.x - min_x)
            features.append(lm.y - min_y)

        # Safety check — should always be 42
        if len(features) != NUM_FEATURES:
            skipped += 1
            continue

        data.append(features)
        labels.append(class_label)

    hands.close()

    print(f"\n[✓] Extracted {len(data)} samples  ({skipped} images skipped)")

    # Save dataset
    with open(output_path, 'wb') as f:
        pickle.dump({'data': data, 'labels': labels}, f)

    print(f"[✓] Dataset saved to {output_path}\n")
    return data, labels


# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    extract_features()
