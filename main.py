"""
Main launcher – menu-driven entry point for the ASL Sign Language Detector.

Run:  python main.py
"""

import os
import sys

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


BANNER = r"""
╔══════════════════════════════════════════════════════╗
║       ASL  Sign  Language  Detector  ★ 2026 ★       ║
║       ─────────────────────────────────────          ║
║       Hand signs  →  Text  →  Speech                 ║
║                                                       ║
║       Neon HUD • Particles • Autocomplete             ║
║       Recording • Screenshots • TTS                   ║
╚══════════════════════════════════════════════════════╝
"""

MENU = """
  1 │ Download ASL Dataset (Kaggle)
  2 │ Collect Custom Dataset (Webcam)
  3 │ Create Feature Dataset  (images → landmarks)
  4 │ Train Classifier
  5 │ Run Sign-to-Text Detector  ★
  ──┼──────────────────────────────
  0 │ Exit
"""


def main():
    while True:
        print(BANNER)
        print(MENU)
        choice = input("  Select an option [0-5]: ").strip()

        if choice == '1':
            from download_dataset import download_and_prepare
            download_and_prepare()

        elif choice == '2':
            from collect_imgs import collect_images
            from config import LABELS, DATASET_SIZE_PER_CLASS, NUM_CLASSES

            print(f"\nAvailable classes ({NUM_CLASSES}):")
            for idx, lbl in LABELS.items():
                print(f"  {idx:>2d} → {lbl}")

            sel = input(
                "\nEnter class numbers (comma-separated) or ENTER for all: "
            ).strip()
            if sel:
                classes = [int(c.strip()) for c in sel.split(',')
                           if c.strip().isdigit()]
            else:
                classes = None

            sz = input(f"Images per class [{DATASET_SIZE_PER_CLASS}]: ").strip()
            sz = int(sz) if sz.isdigit() else DATASET_SIZE_PER_CLASS

            collect_images(classes=classes, dataset_size=sz)

        elif choice == '3':
            from create_dataset import extract_features
            extract_features()

        elif choice == '4':
            from train_classifier import train
            train()

        elif choice == '5':
            from inference_classifier import run
            run()

        elif choice == '0':
            print("\nGoodbye!\n")
            break
        else:
            print("\n  [!] Invalid option. Try again.\n")

        input("\n  Press ENTER to return to the menu …")


if __name__ == '__main__':
    main()
