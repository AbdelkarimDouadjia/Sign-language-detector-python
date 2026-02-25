"""
Generate a beautiful ASL reference chart image.
Run once:  python generate_reference.py
This creates  asl_reference.png  in the project folder.

If you already have your own ASL chart image, simply save it as
'asl_reference.png' in the project root and the detector will use it.
"""

import cv2
import numpy as np
import os

# ── ASL hand‐shape descriptions for each letter/number ──────
LETTERS = {
    'A': 'Fist, thumb beside',
    'B': '4 fingers up, thumb in',
    'C': 'Curved hand, C shape',
    'D': 'Index up, others circle',
    'E': 'Fingers curled, thumb in',
    'F': 'OK sign, 3 fingers up',
    'G': 'Point sideways',
    'H': 'Point 2 sideways',
    'I': 'Pinky up, fist',
    'J': 'Pinky up, draw J',
    'K': 'Index+mid up, thumb mid',
    'L': 'L shape, thumb+index',
    'M': '3 fingers over thumb',
    'N': '2 fingers over thumb',
    'O': 'Finger circle, O shape',
    'P': 'K hand, point down',
    'Q': 'G hand, point down',
    'R': 'Cross index+middle',
    'S': 'Fist, thumb over',
    'T': 'Thumb between idx+mid',
    'U': '2 fingers up together',
    'V': 'Peace sign / 2 apart',
    'W': '3 fingers up spread',
    'X': 'Index hooked',
    'Y': 'Thumb + pinky out',
    'Z': 'Index draws Z in air',
}

NUMBERS = {
    '1': 'Index finger up',
    '2': 'Index + middle up',
    '3': 'Index + mid + ring',
    '4': '4 fingers up, no thumb',
    '5': 'Open hand, all spread',
    '6': 'Thumb + pinky touch',
    '7': 'Thumb + ring touch',
    '8': 'Thumb + middle touch',
    '9': 'Thumb + index touch',
    '0': 'Fingers form circle',
}

# ── Colours (BGR) ────────────────────────────────────────────
BG            = (20, 18, 18)
PANEL         = (35, 30, 30)
CYAN          = (255, 255, 0)
PINK          = (203, 0, 255)
YELLOW        = (0, 255, 255)
GREEN         = (57, 255, 20)
WHITE         = (230, 230, 230)
DIM           = (130, 130, 130)
BORDER        = (70, 65, 65)

def generate(save_path: str = None):
    if save_path is None:
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'asl_reference.png')

    W, H = 720, 980
    img = np.full((H, W, 3), BG, dtype=np.uint8)

    # Title
    cv2.putText(img, "ASL ALPHABET & NUMBERS", (100, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, CYAN, 2, cv2.LINE_AA)
    cv2.putText(img, "Reference Guide", (245, 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, DIM, 1, cv2.LINE_AA)
    cv2.line(img, (20, 80), (W - 20, 80), BORDER, 1, cv2.LINE_AA)

    # Letters section
    cv2.putText(img, "LETTERS", (20, 108),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, PINK, 1, cv2.LINE_AA)

    y = 130
    col_w = W // 2
    keys = list(LETTERS.keys())
    for i, letter in enumerate(keys):
        col = i % 2
        row = i // 2
        x = 20 + col * col_w
        cy = y + row * 28

        # Letter box
        cv2.rectangle(img, (x, cy - 16), (x + 24, cy + 4), PANEL, -1)
        cv2.rectangle(img, (x, cy - 16), (x + 24, cy + 4), BORDER, 1)
        cv2.putText(img, letter, (x + 4, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, YELLOW, 2, cv2.LINE_AA)

        # Description
        cv2.putText(img, LETTERS[letter], (x + 32, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, WHITE, 1, cv2.LINE_AA)

    # Numbers section
    num_y = y + (len(keys) // 2 + 1) * 28 + 20
    cv2.line(img, (20, num_y - 15), (W - 20, num_y - 15), BORDER, 1, cv2.LINE_AA)
    cv2.putText(img, "NUMBERS (press # to switch)", (20, num_y + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, GREEN, 1, cv2.LINE_AA)

    num_y += 30
    nkeys = list(NUMBERS.keys())
    for i, num in enumerate(nkeys):
        col = i % 2
        row = i // 2
        x = 20 + col * col_w
        cy = num_y + row * 28

        cv2.rectangle(img, (x, cy - 16), (x + 24, cy + 4), PANEL, -1)
        cv2.rectangle(img, (x, cy - 16), (x + 24, cy + 4), BORDER, 1)
        cv2.putText(img, num, (x + 6, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, YELLOW, 2, cv2.LINE_AA)

        cv2.putText(img, NUMBERS[num], (x + 32, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, WHITE, 1, cv2.LINE_AA)

    # Special gestures
    spec_y = num_y + 5 * 28 + 25
    cv2.line(img, (20, spec_y - 10), (W - 20, spec_y - 10), BORDER, 1, cv2.LINE_AA)
    cv2.putText(img, "SPECIAL GESTURES", (20, spec_y + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, CYAN, 1, cv2.LINE_AA)

    specials = [
        ("SPACE", "Open hand, palm down"),
        ("DEL",   "Pinch fingers together"),
        ("NOTHING", "No hand / idle"),
    ]
    for i, (name, desc) in enumerate(specials):
        cy = spec_y + 35 + i * 28
        cv2.putText(img, name, (30, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, YELLOW, 1, cv2.LINE_AA)
        cv2.putText(img, desc, (120, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.38, WHITE, 1, cv2.LINE_AA)

    # Keyboard shortcuts
    kb_y = spec_y + 35 + 3 * 28 + 20
    cv2.line(img, (20, kb_y - 10), (W - 20, kb_y - 10), BORDER, 1, cv2.LINE_AA)
    cv2.putText(img, "KEYBOARD SHORTCUTS", (20, kb_y + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, CYAN, 1, cv2.LINE_AA)

    shortcuts = [
        ("Q/ESC", "Quit"),        ("S", "Speak (TTS)"),
        ("C", "Clear all"),       ("X", "Delete letter"),
        ("E", "Export text"),     ("F", "Screenshot"),
        ("R", "Record video"),    ("#", "Numbers ON/OFF"),
        ("G", "This guide"),      ("M", "Quiz mode"),
        ("Tab", "Autocomplete"),  ("1-3", "Pick suggestion"),
    ]
    for i, (k, d) in enumerate(shortcuts):
        col = i % 2
        row = i // 2
        x = 30 + col * col_w
        cy = kb_y + 35 + row * 22
        cv2.putText(img, f"[{k}]", (x, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, GREEN, 1, cv2.LINE_AA)
        cv2.putText(img, d, (x + 65, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, DIM, 1, cv2.LINE_AA)

    # Footer
    cv2.putText(img, "ASL Sign Language Detector - 2026 Edition", (155, H - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, DIM, 1, cv2.LINE_AA)

    cv2.imwrite(save_path, img)
    print(f"[✓] Reference chart saved → {save_path}")
    return img


if __name__ == '__main__':
    generate()
