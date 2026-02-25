"""
Collect sign-language training images from your webcam.

Features:
  • Supports all 29 ASL classes (A-Z + DEL, NOTHING, SPACE)
  • Choose which classes to collect
  • Mirror mode for natural experience
  • Resume-friendly: skips classes that already have enough images
  • Live progress overlay
"""

import os
import cv2

from config import (
    DATA_DIR, LABELS, NUM_CLASSES,
    DATASET_SIZE_PER_CLASS,
    COLOR_GREEN, COLOR_WHITE, COLOR_BLACK,
)


def collect_images(
    classes: list | None = None,
    dataset_size: int = DATASET_SIZE_PER_CLASS,
    camera_index: int = 0,
    mirror: bool = True,
):
    """
    Collect webcam images for the specified ASL classes.

    Parameters
    ----------
    classes : list[int] | None
        Class indices to collect (e.g. [0,1,2] for A,B,C).
        None → collect ALL classes.
    dataset_size : int
        Number of images to capture per class.
    camera_index : int
        OpenCV camera device index.
    mirror : bool
        Flip frame horizontally for a natural "mirror" feel.
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    if classes is None:
        classes = list(range(NUM_CLASSES))

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("[✗] Cannot open camera. Check your webcam connection.")
        return

    for class_idx in classes:
        label = LABELS.get(class_idx, str(class_idx))
        class_dir = os.path.join(DATA_DIR, str(class_idx))
        os.makedirs(class_dir, exist_ok=True)

        # Skip classes that already have enough images
        existing = len([
            f for f in os.listdir(class_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        if existing >= dataset_size:
            print(f"  Class {class_idx} ({label}) already has {existing} images – skipping.")
            continue

        print(f"\n▶  Class {class_idx} – '{label}'  ({existing}/{dataset_size} images exist)")

        # ── Wait for user to be ready ──
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            if mirror:
                frame = cv2.flip(frame, 1)

            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], 80), COLOR_BLACK, -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

            msg = f"Class {class_idx}: '{label}'  –  Press Q to start"
            cv2.putText(frame, msg, (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_GREEN, 2, cv2.LINE_AA)
            cv2.imshow('Collect Images', frame)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # ── Capture images ──
        counter = existing
        while counter < dataset_size:
            ret, frame = cap.read()
            if not ret:
                continue
            if mirror:
                frame = cv2.flip(frame, 1)

            # Progress overlay
            progress = f"{counter + 1}/{dataset_size}"
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], 50), COLOR_BLACK, -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            cv2.putText(frame, f"'{label}'  {progress}", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_WHITE, 2, cv2.LINE_AA)

            # Progress bar
            bar_w = int((counter + 1) / dataset_size * (frame.shape[1] - 40))
            cv2.rectangle(frame, (20, 45), (20 + bar_w, 48), COLOR_GREEN, -1)

            cv2.imshow('Collect Images', frame)

            img_path = os.path.join(class_dir, f'{counter}.jpg')
            cv2.imwrite(img_path, frame)
            counter += 1
            cv2.waitKey(25)

        print(f"  [✓] Collected {counter} images for '{label}'")

    cap.release()
    cv2.destroyAllWindows()
    print("\n[✓] Image collection complete!\n")


# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 50)
    print("  ASL Image Collector")
    print("=" * 50)
    print(f"\nAvailable classes ({NUM_CLASSES}):")
    for idx, lbl in LABELS.items():
        print(f"  {idx:>2d} → {lbl}")

    choice = input(
        "\nEnter class numbers to collect (comma-separated),\n"
        "or press ENTER to collect ALL: "
    ).strip()

    if choice:
        selected = [int(c.strip()) for c in choice.split(',') if c.strip().isdigit()]
    else:
        selected = None

    size = input(f"Images per class [{DATASET_SIZE_PER_CLASS}]: ").strip()
    size = int(size) if size.isdigit() else DATASET_SIZE_PER_CLASS

    collect_images(classes=selected, dataset_size=size)
