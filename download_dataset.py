"""
Download and organise the ASL Alphabet dataset from Kaggle.
Uses the `kagglehub` library for a hassle-free download experience.

Dataset: "grassknoted/asl-alphabet"
  – 87 000 images, 29 classes (A-Z + del + nothing + space)
  – 200 × 200 px colour images
"""

import os
import sys
import shutil
import random

from config import (
    DATA_DIR, KAGGLE_CLASS_MAP, DOWNLOAD_IMAGES_PER_CLASS
)


def download_and_prepare(images_per_class: int = DOWNLOAD_IMAGES_PER_CLASS):
    """Download the Kaggle ASL Alphabet dataset and copy a subset to ./data/."""

    # ── 1. Download via kagglehub ────────────────────────────
    try:
        import kagglehub
    except ImportError:
        print("[!] 'kagglehub' is not installed. Installing now …")
        os.system(f'"{sys.executable}" -m pip install kagglehub')
        import kagglehub

    print("\n▶  Downloading 'grassknoted/asl-alphabet' from Kaggle …")
    print("   (first run may ask you to log in at kaggle.com)\n")

    try:
        dataset_path = kagglehub.dataset_download("grassknoted/asl-alphabet")
    except Exception as e:
        print(f"\n[✗] Download failed: {e}\n")
        print("To fix this, make sure you have a Kaggle account:")
        print("  1. Go to  https://www.kaggle.com  and sign up / log in.")
        print("  2. Click your profile → Settings → API → 'Create New Token'.")
        print("  3. Place the downloaded  kaggle.json  in:")
        print("       Windows : C:\\Users\\<YOU>\\.kaggle\\kaggle.json")
        print("       Linux   : ~/.kaggle/kaggle.json")
        print("  4. Run this script again.\n")
        return False

    print(f"[✓] Dataset downloaded to: {dataset_path}\n")

    # ── 2. Locate the training folder ────────────────────────
    #    The Kaggle dataset nests folders as:
    #      .../asl_alphabet_train/asl_alphabet_train/A/
    #    We need the innermost folder that contains class sub-dirs.
    train_dir = None
    for root, dirs, _files in os.walk(dataset_path):
        # Check if this directory contains the expected class folders
        if 'A' in dirs and 'B' in dirs and 'Z' in dirs:
            train_dir = root
            break

    if train_dir is None:
        print("[✗] Could not locate the training images inside the download.")
        print(f"    Download root: {dataset_path}")
        print("    Please check the folder structure and adjust the script.")
        return False

    print(f"[✓] Training images found at: {train_dir}")

    # ── 3. Copy images into ./data/<class_index>/ ────────────
    os.makedirs(DATA_DIR, exist_ok=True)

    total_copied = 0
    for folder_name, class_idx in KAGGLE_CLASS_MAP.items():
        src_folder = os.path.join(train_dir, folder_name)
        if not os.path.isdir(src_folder):
            print(f"  [!] Source folder not found: {src_folder}  – skipping")
            continue

        dst_folder = os.path.join(DATA_DIR, str(class_idx))
        os.makedirs(dst_folder, exist_ok=True)

        # Gather valid image files
        all_images = [
            f for f in os.listdir(src_folder)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        random.shuffle(all_images)
        selected = all_images[:images_per_class]

        copied = 0
        for img_name in selected:
            src = os.path.join(src_folder, img_name)
            dst = os.path.join(dst_folder, img_name)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
            copied += 1

        total_copied += copied
        label = folder_name.upper() if len(folder_name) == 1 else folder_name
        print(f"  Class {class_idx:>2d} ({label:>7s}): {copied} images → {dst_folder}")

    print(f"\n[✓] Done!  {total_copied} images copied to {DATA_DIR}")
    print("    You can now run  create_dataset.py  to extract features.\n")
    return True


# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    download_and_prepare()
