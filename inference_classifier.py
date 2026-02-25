"""
ASL Sign Language → Text → Speech  ·  2026 Edition
═══════════════════════════════════════════════════

A premium real-time sign language detector with a futuristic HUD interface.

Features
────────
  ★  Neon glow hand skeleton with pulsing effects
  ★  Particle burst animation when a letter is confirmed
  ★  Top-3 prediction bar with live confidence meters
  ★  Smart word autocomplete suggestions
  ★  Session statistics dashboard (letters/min, accuracy)
  ★  Video recording mode (press R)
  ★  Screenshot capture (press F)
  ★  Frosted glass UI panels
  ★  Text-to-Speech (press S)
  ★  Export text (press E)
  ★  Animated stability ring

Keyboard shortcuts
──────────────────
  Q / ESC      Quit
  S            Speak sentence (TTS – modern AI voice)
  C            Clear sentence
  X            Delete last letter
  E            Export to file
  F            Screenshot
  R            Toggle recording
  G            Toggle ASL reference guide
  M            Toggle Quiz Mode
  N            Toggle Number mode / Skip quiz letter
  1 / 2 / 3   Accept autocomplete suggestion
  Backspace    Delete last character
  Tab          Accept top suggestion
"""

import os
import pickle
import time
import math
import random
import threading
import datetime
import asyncio
import tempfile

import cv2
import numpy as np
import mediapipe as mp

from config import (
    BASE_DIR, MODEL_PATH, LABELS, NUM_FEATURES, EXPORTS_DIR,
    STABILITY_THRESHOLD, CONFIDENCE_THRESHOLD, COOLDOWN_FRAMES,
    MIN_DETECTION_CONFIDENCE, LETTER_TO_NUMBER,
)

from word_completer import WordCompleter

# ── Modern Neural TTS (edge-tts) with fallback ──────────────
_TTS_ENGINE = None   # 'edge' or 'pyttsx3' or None
_VOICE = "en-US-AriaNeural"  # modern expressive AI voice

try:
    import edge_tts
    import pygame
    pygame.mixer.init(frequency=24000)
    _TTS_ENGINE = 'edge'
except Exception:
    try:
        import pyttsx3
        _TTS_ENGINE = 'pyttsx3'
    except Exception:
        pass

TTS_AVAILABLE = _TTS_ENGINE is not None


def _speak(text: str):
    """Speak text using modern neural AI voice (edge-tts) with pyttsx3 fallback."""
    if not TTS_AVAILABLE or not text.strip():
        return
    def _worker():
        try:
            if _TTS_ENGINE == 'edge':
                loop = asyncio.new_event_loop()
                tmp = os.path.join(tempfile.gettempdir(), 'asl_tts_output.mp3')
                async def _gen():
                    communicate = edge_tts.Communicate(text, _VOICE)
                    await communicate.save(tmp)
                loop.run_until_complete(_gen())
                loop.close()
                pygame.mixer.music.load(tmp)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
                try:
                    pygame.mixer.music.unload()
                    os.unlink(tmp)
                except Exception:
                    pass
            else:
                engine = pyttsx3.init()
                engine.setProperty('rate', 150)
                engine.say(text)
                engine.runAndWait()
                engine.stop()
        except Exception:
            pass
    threading.Thread(target=_worker, daemon=True).start()


def _export_text(text: str) -> str:
    os.makedirs(EXPORTS_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(EXPORTS_DIR, f'sign_text_{ts}.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"  [✓] Exported → {path}")
    return path


# ═════════════════════════════════════════════════════════════
#  COLOUR PALETTE  (BGR)
# ═════════════════════════════════════════════════════════════
NEON_CYAN     = (255, 255, 0)
NEON_GREEN    = (57, 255, 20)
NEON_PINK     = (203, 0, 255)
NEON_ORANGE   = (0, 165, 255)
NEON_YELLOW   = (0, 255, 255)
NEON_BLUE     = (255, 100, 0)
GLASS_BG      = (30, 25, 25)
GLASS_BORDER  = (80, 70, 70)
TEXT_WHITE     = (240, 240, 240)
TEXT_DIM       = (140, 140, 140)
TEXT_MUTED     = (90, 90, 90)
SUCCESS_GREEN  = (80, 220, 80)
DANGER_RED     = (60, 60, 220)


# ═════════════════════════════════════════════════════════════
#  CONFIDENCE SMOOTHER  –  multi-frame prediction averaging
# ═════════════════════════════════════════════════════════════
class ConfidenceSmoother:
    """Averages prediction probabilities over recent frames.

    This dramatically improves detection of hard / similar letters
    (Z, J, G vs. Q, etc.) by stabilising noisy per-frame predictions.
    """

    def __init__(self, window: int = 6, decay: float = 0.65):
        self.window = window
        self.decay = decay
        self.history: list[np.ndarray] = []

    def update(self, proba: np.ndarray) -> np.ndarray:
        self.history.append(proba.copy())
        if len(self.history) > self.window:
            self.history.pop(0)
        result = np.zeros_like(proba)
        total = 0.0
        for i, p in enumerate(self.history):
            w = self.decay ** (len(self.history) - 1 - i)
            result += w * p
            total += w
        return result / total

    def clear(self):
        self.history.clear()


# ═════════════════════════════════════════════════════════════
#  PARTICLE SYSTEM  –  burst effect when a letter is confirmed
# ═════════════════════════════════════════════════════════════
class Particle:
    __slots__ = ('x', 'y', 'vx', 'vy', 'life', 'max_life', 'color', 'size')

    def __init__(self, x, y, color):
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(2, 8)
        self.x = float(x)
        self.y = float(y)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.life = random.uniform(0.4, 1.0)
        self.max_life = self.life
        self.color = color
        self.size = random.randint(2, 5)

    def update(self, dt):
        self.x += self.vx
        self.y += self.vy
        self.vy += 12 * dt          # gravity
        self.vx *= 0.97             # drag
        self.life -= dt
        return self.life > 0

    def draw(self, frame):
        alpha = max(self.life / self.max_life, 0)
        r = max(int(self.size * alpha), 1)
        c = tuple(int(v * alpha) for v in self.color)
        cv2.circle(frame, (int(self.x), int(self.y)), r, c, -1, cv2.LINE_AA)


class ParticleSystem:
    def __init__(self):
        self.particles: list[Particle] = []

    def burst(self, x, y, color=NEON_CYAN, count=30):
        for _ in range(count):
            self.particles.append(Particle(x, y, color))

    def update_and_draw(self, frame, dt):
        self.particles = [p for p in self.particles if p.update(dt)]
        for p in self.particles:
            p.draw(frame)


# ═════════════════════════════════════════════════════════════
#  NOTIFICATION TOAST  –  pop-up messages that fade away
# ═════════════════════════════════════════════════════════════
class Toast:
    def __init__(self, text, color=NEON_CYAN, duration=2.0):
        self.text = text
        self.color = color
        self.birth = time.time()
        self.duration = duration

    @property
    def alive(self):
        return (time.time() - self.birth) < self.duration

    @property
    def alpha(self):
        age = time.time() - self.birth
        if age < 0.3:
            return age / 0.3
        remaining = self.duration - age
        if remaining < 0.5:
            return max(remaining / 0.5, 0)
        return 1.0


class ToastManager:
    def __init__(self):
        self.toasts: list[Toast] = []

    def show(self, text, color=NEON_CYAN, duration=2.0):
        self.toasts.append(Toast(text, color, duration))

    def draw(self, frame):
        self.toasts = [t for t in self.toasts if t.alive]
        H, W = frame.shape[:2]
        y_offset = H // 2 - 20 * len(self.toasts)
        for t in self.toasts:
            a = t.alpha
            c = tuple(int(v * a) for v in t.color)
            size = cv2.getTextSize(t.text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            tx = (W - size[0]) // 2
            cv2.putText(frame, t.text, (tx, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, c, 2, cv2.LINE_AA)
            y_offset += 35


# ═════════════════════════════════════════════════════════════
#  LETTER HISTORY  –  animated trail of recently typed letters
# ═════════════════════════════════════════════════════════════
class LetterHistory:
    def __init__(self, max_len=12):
        self.letters: list[tuple[str, float]] = []  # (letter, timestamp)
        self.max_len = max_len

    def add(self, letter):
        self.letters.append((letter, time.time()))
        if len(self.letters) > self.max_len:
            self.letters.pop(0)

    def draw(self, frame, x, y):
        now = time.time()
        for i, (letter, ts) in enumerate(self.letters):
            age = now - ts
            alpha = max(1.0 - age / 8.0, 0.15)
            # Recent letters are brighter and bigger
            scale = 0.5 + 0.3 * alpha
            c = tuple(int(v * alpha) for v in NEON_CYAN)
            px = x + i * 30
            cv2.putText(frame, letter, (px, y),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, c, 1, cv2.LINE_AA)


# ═════════════════════════════════════════════════════════════
#  SESSION STATISTICS
# ═════════════════════════════════════════════════════════════
class SessionStats:
    def __init__(self):
        self.start_time = time.time()
        self.letters_count = 0
        self.words_count = 0
        self.detections = 0
        self.frames = 0

    def add_letter(self):
        self.letters_count += 1

    def add_word(self):
        self.words_count += 1

    @property
    def elapsed(self):
        return time.time() - self.start_time

    @property
    def lpm(self):
        mins = self.elapsed / 60
        return self.letters_count / max(mins, 0.01)

    @property
    def wpm(self):
        mins = self.elapsed / 60
        return self.words_count / max(mins, 0.01)


# ═════════════════════════════════════════════════════════════
#  QUIZ MODE  –  interactive ASL learning game
# ═════════════════════════════════════════════════════════════
class QuizMode:
    """Interactive ASL quiz — the system shows a letter, you sign it!

    Tracks score, streaks, accuracy, and has a countdown timer.
    Phenomenal for learning & practicing sign language.
    """

    def __init__(self):
        self.active = False
        self.target_letter = None
        self.score = 0
        self.streak = 0
        self.best_streak = 0
        self.total_attempts = 0
        self.correct = 0
        self.start_time = None
        self.time_limit = 10.0
        self.last_result = None      # 'correct' or 'timeout'
        self.result_time = 0.0
        self.pool = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        # Start with easier letters, unlock harder ones
        self.easy = list("ABCDEFHIKLMNORSTUVWXY")
        self.hard = list("GJPQZ")
        self.level_pool = list(self.easy)

    def toggle(self):
        if self.active:
            self.active = False
        else:
            self.active = True
            self.score = 0
            self.streak = 0
            self.best_streak = 0
            self.total_attempts = 0
            self.correct = 0
            self.level_pool = list(self.easy)
            self.next_letter()

    def next_letter(self):
        # After 10 correct answers, add hard letters
        if self.correct >= 10 and self.level_pool == self.easy:
            self.level_pool = list(self.pool)
        self.target_letter = random.choice(self.level_pool)
        self.start_time = time.time()
        self.last_result = None

    def check(self, detected_letter, confidence):
        """Returns True if detected letter matches target."""
        if not self.active or not detected_letter or not self.target_letter:
            return False
        if detected_letter == self.target_letter and confidence >= 0.30:
            self.score += 10 + self.streak * 2
            self.streak += 1
            self.best_streak = max(self.best_streak, self.streak)
            self.correct += 1
            self.total_attempts += 1
            self.last_result = 'correct'
            self.result_time = time.time()
            return True
        return False

    def check_timeout(self):
        if self.start_time and (time.time() - self.start_time) > self.time_limit:
            self.streak = 0
            self.total_attempts += 1
            self.last_result = 'timeout'
            self.result_time = time.time()
            self.next_letter()
            return True
        return False

    def skip(self):
        self.streak = 0
        self.total_attempts += 1
        self.next_letter()

    @property
    def remaining(self):
        if self.start_time:
            return max(self.time_limit - (time.time() - self.start_time), 0)
        return self.time_limit

    @property
    def accuracy(self):
        return (self.correct / self.total_attempts * 100) if self.total_attempts else 0


# ═════════════════════════════════════════════════════════════
#  DRAWING  HELPERS
# ═════════════════════════════════════════════════════════════

def draw_glass_panel(frame, x1, y1, x2, y2, opacity=0.65, border=True):
    """Draw a frosted-glass style panel."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), GLASS_BG, -1)
    cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0, frame)
    if border:
        cv2.rectangle(frame, (x1, y1), (x2, y2), GLASS_BORDER, 1, cv2.LINE_AA)


def draw_neon_text(frame, text, pos, scale=1.0, color=NEON_CYAN, thickness=2):
    """Draw text with a subtle glow effect."""
    x, y = pos
    # Glow layers (wider, dimmer)
    glow_color = tuple(max(int(v * 0.3), 0) for v in color)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                glow_color, thickness + 4, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, thickness, cv2.LINE_AA)


def draw_neon_hand(frame, landmarks, connections, mirror, W, H):
    """Draw hand skeleton with neon glow effect."""
    pts = []
    for lm in landmarks:
        px = int((1.0 - lm.x) * W) if mirror else int(lm.x * W)
        py = int(lm.y * H)
        pts.append((px, py))

    # Draw connections with glow
    for conn in connections:
        p1, p2 = pts[conn[0]], pts[conn[1]]
        # Outer glow
        cv2.line(frame, p1, p2, (80, 40, 0), 6, cv2.LINE_AA)
        # Inner line
        cv2.line(frame, p1, p2, NEON_CYAN, 2, cv2.LINE_AA)

    # Draw joints
    for i, pt in enumerate(pts):
        # Fingertips get bigger dots
        is_tip = i in (4, 8, 12, 16, 20)
        r = 7 if is_tip else 4
        # Outer glow
        cv2.circle(frame, pt, r + 3, (80, 40, 0), -1, cv2.LINE_AA)
        # Inner dot
        color = NEON_PINK if is_tip else NEON_CYAN
        cv2.circle(frame, pt, r, color, -1, cv2.LINE_AA)
        # Bright center
        cv2.circle(frame, pt, max(r - 2, 1), TEXT_WHITE, -1, cv2.LINE_AA)

    return pts


def draw_confidence_meter(frame, x, y, w, confidence, label, rank_color):
    """Draw a single confidence bar with label."""
    bar_h = 16
    # Background
    cv2.rectangle(frame, (x, y), (x + w, y + bar_h), (50, 45, 45), -1)
    # Fill
    fill = int(w * confidence)
    cv2.rectangle(frame, (x, y), (x + fill, y + bar_h), rank_color, -1)
    # Border
    cv2.rectangle(frame, (x, y), (x + w, y + bar_h), GLASS_BORDER, 1)
    # Label
    cv2.putText(frame, f"{label} {confidence*100:.0f}%",
                (x + 4, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, TEXT_WHITE, 1, cv2.LINE_AA)


def draw_stability_ring(frame, cx, cy, radius, pct, color=NEON_GREEN):
    """Animated circular progress ring."""
    # Background ring
    cv2.ellipse(frame, (cx, cy), (radius, radius), 0, 0, 360, (50, 50, 50), 2, cv2.LINE_AA)
    # Progress arc
    if pct > 0:
        angle = int(360 * pct)
        # Glow
        cv2.ellipse(frame, (cx, cy), (radius + 2, radius + 2), -90, 0, angle,
                    tuple(int(v * 0.3) for v in color), 5, cv2.LINE_AA)
        cv2.ellipse(frame, (cx, cy), (radius, radius), -90, 0, angle,
                    color, 3, cv2.LINE_AA)
    # Percentage text
    txt = f"{int(pct * 100)}%"
    sz = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
    cv2.putText(frame, txt, (cx - sz[0] // 2, cy + sz[1] // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_WHITE, 1, cv2.LINE_AA)


def draw_suggestions(frame, suggestions, x, y):
    """Draw autocomplete suggestions with number keys."""
    if not suggestions:
        return
    draw_glass_panel(frame, x - 5, y - 18, x + 200, y + len(suggestions) * 24 + 5, 0.7)
    cv2.putText(frame, "Suggestions:", (x, y - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, TEXT_DIM, 1, cv2.LINE_AA)
    for i, word in enumerate(suggestions):
        key_color = NEON_YELLOW
        cv2.putText(frame, f"[{i+1}]", (x, y + 18 + i * 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, key_color, 1, cv2.LINE_AA)
        cv2.putText(frame, word, (x + 30, y + 18 + i * 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_WHITE, 1, cv2.LINE_AA)


def draw_recording_indicator(frame, W):
    """Pulsing red recording dot."""
    pulse = (math.sin(time.time() * 4) + 1) / 2
    r = int(8 + 3 * pulse)
    color = (0, 0, int(180 + 75 * pulse))
    cv2.circle(frame, (W - 30, 25), r, color, -1, cv2.LINE_AA)
    cv2.putText(frame, "REC", (W - 65, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)


def format_time(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


# ═════════════════════════════════════════════════════════════
#  QUIZ MODE  OVERLAY
# ═════════════════════════════════════════════════════════════

def draw_quiz_overlay(frame, quiz, current_letter, confidence):
    """Draw the interactive quiz mode overlay on top of the camera feed."""
    H, W = frame.shape[:2]

    # Subtle darkening overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 55), (W, H - 100), (10, 8, 8), -1)
    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

    # Quiz panel (centered)
    pw, ph = 400, 300
    px = (W - pw) // 2
    py = (H - ph) // 2 - 15
    draw_glass_panel(frame, px, py, px + pw, py + ph, 0.82)

    # Title
    draw_neon_text(frame, "QUIZ MODE", (px + 120, py + 32), 0.8, NEON_PINK, 2)

    # Target letter (BIG with pulse)
    letter = quiz.target_letter or "?"
    sz = cv2.getTextSize(letter, cv2.FONT_HERSHEY_SIMPLEX, 3.5, 5)[0]
    lx = px + (pw - sz[0]) // 2
    pulse = (math.sin(time.time() * 3) + 1) / 2
    lc = tuple(int(v * (0.7 + 0.3 * pulse)) for v in NEON_YELLOW)
    draw_neon_text(frame, letter, (lx, py + 125), 3.5, lc, 5)

    # Prompt
    cv2.putText(frame, "Sign this letter!", (px + 120, py + 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_DIM, 1, cv2.LINE_AA)

    # Timer bar
    pct = quiz.remaining / quiz.time_limit
    bar_w = pw - 40
    tc = NEON_GREEN if pct > 0.4 else NEON_ORANGE if pct > 0.15 else DANGER_RED
    cv2.rectangle(frame, (px + 20, py + 165), (px + 20 + bar_w, py + 178), (40, 40, 40), -1)
    cv2.rectangle(frame, (px + 20, py + 165), (px + 20 + int(bar_w * pct), py + 178), tc, -1)
    cv2.putText(frame, f"{quiz.remaining:.1f}s", (px + 20 + bar_w + 5, py + 177),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, TEXT_DIM, 1, cv2.LINE_AA)

    # Score
    cv2.putText(frame, f"Score: {quiz.score}", (px + 20, py + 210),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, NEON_CYAN, 1, cv2.LINE_AA)

    # Streak
    streak_text = f"Streak: {quiz.streak}"
    sc = NEON_ORANGE
    if quiz.streak >= 10:
        streak_text += " LEGENDARY!"
        sc = NEON_PINK
    elif quiz.streak >= 5:
        streak_text += " ON FIRE!"
        sc = NEON_YELLOW
    cv2.putText(frame, streak_text, (px + 210, py + 210),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, sc, 1, cv2.LINE_AA)

    # Accuracy + best streak
    cv2.putText(frame, f"Accuracy: {quiz.accuracy:.0f}%  |  Best: {quiz.best_streak}",
                (px + 20, py + 245),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_DIM, 1, cv2.LINE_AA)

    # Current detection feedback
    if current_letter and confidence > 0.2:
        det_color = NEON_GREEN if current_letter == quiz.target_letter else DANGER_RED
        cv2.putText(frame, f"Detecting: {current_letter} ({confidence*100:.0f}%)",
                    (px + 20, py + 270),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, det_color, 1, cv2.LINE_AA)

    # Result flash
    if quiz.last_result and (time.time() - quiz.result_time) < 1.2:
        a = max(1.0 - (time.time() - quiz.result_time) / 1.2, 0)
        if quiz.last_result == 'correct':
            txt, fc = "CORRECT!", tuple(int(v * a) for v in NEON_GREEN)
        else:
            txt, fc = "TIME'S UP!", tuple(int(v * a) for v in DANGER_RED)
        tsz = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
        draw_neon_text(frame, txt, ((W - tsz[0]) // 2, py - 10), 1.2, fc, 3)

    # Controls
    cv2.putText(frame, "M:Exit Quiz   N:Skip Letter", (px + 85, py + ph - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, TEXT_MUTED, 1, cv2.LINE_AA)


# ═════════════════════════════════════════════════════════════
#  MAIN  HUD  RENDERER
# ═════════════════════════════════════════════════════════════

def draw_hud(frame, *, sentence, current_letter, confidence, top3,
             fps, stability_pct, is_cooldown, suggestions, stats,
             letter_history, is_recording, number_mode=False):
    """Draw the complete futuristic HUD overlay."""
    H, W = frame.shape[:2]

    # ── TOP BAR ──────────────────────────────────────────────
    draw_glass_panel(frame, 0, 0, W, 55, 0.6, border=False)

    # FPS (top-left)
    fps_color = NEON_GREEN if fps > 20 else NEON_ORANGE if fps > 10 else DANGER_RED
    cv2.putText(frame, f"FPS {fps:.0f}", (12, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, fps_color, 1, cv2.LINE_AA)

    # Session time
    elapsed = format_time(stats.elapsed)
    cv2.putText(frame, elapsed, (12, 43),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_DIM, 1, cv2.LINE_AA)

    # Title (centered)
    title = "ASL SIGN LANGUAGE DETECTOR"
    title_sz = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0]
    draw_neon_text(frame, title, ((W - title_sz[0]) // 2, 22), 0.55, NEON_CYAN, 1)

    # Stats (top-right area)
    cv2.putText(frame, f"LPM: {stats.lpm:.0f}", (W - 130, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_DIM, 1, cv2.LINE_AA)
    cv2.putText(frame, f"Letters: {stats.letters_count}", (W - 130, 43),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_DIM, 1, cv2.LINE_AA)

    # Recording indicator
    if is_recording:
        draw_recording_indicator(frame, W)

    # ── CURRENT LETTER + CONFIDENCE (right side panel) ───────
    if current_letter and confidence > 0:
        panel_w = 190
        panel_x = W - panel_w - 10
        panel_y = 65
        draw_glass_panel(frame, panel_x, panel_y, W - 10, panel_y + 160, 0.65)

        # Big letter
        lbl = current_letter
        if current_letter in ('DEL', 'SPACE', 'NOTHING'):
            lbl = current_letter[:3]
        lbl_sz = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 1.8, 3)[0]
        lbl_x = panel_x + (panel_w - lbl_sz[0]) // 2
        letter_color = NEON_GREEN if confidence >= CONFIDENCE_THRESHOLD else DANGER_RED
        draw_neon_text(frame, lbl, (lbl_x, panel_y + 50), 1.8, letter_color, 3)

        # Top-3 predictions
        rank_colors = [NEON_GREEN, NEON_YELLOW, NEON_ORANGE]
        for i, (label, conf) in enumerate(top3[:3]):
            draw_confidence_meter(frame, panel_x + 8, panel_y + 68 + i * 24,
                                  panel_w - 16, conf, label, rank_colors[i])

        # Stability ring
        if stability_pct > 0 and not is_cooldown:
            ring_cx = panel_x + panel_w // 2
            ring_cy = panel_y + 148
            draw_stability_ring(frame, ring_cx, ring_cy, 14, stability_pct)

    # ── LETTER TRAIL (left side) ─────────────────────────────
    letter_history.draw(frame, 12, 80)

    # ── BOTTOM PANEL ─────────────────────────────────────────
    panel_h = 100
    draw_glass_panel(frame, 0, H - panel_h, W, H, 0.7, border=False)

    # Sentence with blinking cursor
    display_text = sentence if sentence else "Start signing to build your sentence ..."
    cursor = "|" if int(time.time() * 2) % 2 == 0 else " "
    text_color = TEXT_WHITE if sentence else TEXT_MUTED

    # Truncate if too long for display
    max_chars = W // 12
    if len(display_text) > max_chars:
        display_text = "..." + display_text[-(max_chars - 3):]

    cv2.putText(frame, display_text + cursor, (15, H - panel_h + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, text_color, 1, cv2.LINE_AA)

    # Word count & character count
    words = len(sentence.split()) if sentence.strip() else 0
    chars = len(sentence)
    info_text = f"Words: {words}  |  Chars: {chars}"
    cv2.putText(frame, info_text, (15, H - panel_h + 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, TEXT_DIM, 1, cv2.LINE_AA)

    # Autocomplete suggestions
    if suggestions:
        draw_suggestions(frame, suggestions, W - 220, H - panel_h + 25)

    # Controls bar
    tts = "S:Speak " if TTS_AVAILABLE else ""
    rec = "R:Rec " if not is_recording else "R:Stop "
    num = "N:Numbers " if not number_mode else "N:Letters "
    controls = f"Q:Quit  {tts}C:Clear  X:Del  E:Export  F:Photo  {rec}{num}G:Guide  M:Quiz  Tab/1-3:Complete"
    cv2.putText(frame, controls, (15, H - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.33, TEXT_MUTED, 1, cv2.LINE_AA)

    # Number mode badge
    if number_mode:
        badge_text = "NUMBER MODE: D=1  V=2  W=3  B=4  A=5  Y=6  S=7  L=8  F=9  O=0"
        badge_w = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)[0][0]
        bx = (W - badge_w) // 2
        draw_glass_panel(frame, bx - 8, 44, bx + badge_w + 8, 58, 0.75)
        cv2.putText(frame, badge_text, (bx, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, NEON_YELLOW, 1, cv2.LINE_AA)

    # Decorative line separator
    cv2.line(frame, (10, H - panel_h + 2), (W - 10, H - panel_h + 2),
             GLASS_BORDER, 1, cv2.LINE_AA)

    # Subtle corner accents
    corner_len = 20
    # Top-left
    cv2.line(frame, (3, 58), (3 + corner_len, 58), NEON_CYAN, 2, cv2.LINE_AA)
    cv2.line(frame, (3, 58), (3, 58 + corner_len), NEON_CYAN, 2, cv2.LINE_AA)
    # Bottom-right
    cv2.line(frame, (W - 3, H - panel_h - 3), (W - 3 - corner_len, H - panel_h - 3),
             NEON_CYAN, 2, cv2.LINE_AA)
    cv2.line(frame, (W - 3, H - panel_h - 3), (W - 3, H - panel_h - 3 - corner_len),
             NEON_CYAN, 2, cv2.LINE_AA)


# ═════════════════════════════════════════════════════════════
#  MAIN  INFERENCE  LOOP
# ═════════════════════════════════════════════════════════════

def run(camera_index: int = 0, mirror: bool = True):
    """Launch the futuristic ASL Sign Language Detector."""

    # ── Load model ───────────────────────────────────────────
    if not os.path.exists(MODEL_PATH):
        print(f"[✗] Model not found at {MODEL_PATH}")
        print("    Run train_classifier.py first.")
        return

    with open(MODEL_PATH, 'rb') as f:
        model_data = pickle.load(f)
    model = model_data['model']
    model_name = model_data.get('model_name', 'Unknown')
    print(f"[✓] Loaded model: {model_name}")

    # ── MediaPipe ────────────────────────────────────────────
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=0.4,
    )

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("[✗] Cannot open camera.")
        return

    # Get camera resolution
    cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    WIN_NAME = 'ASL Sign Language Detector - 2026 Edition'

    # ── Load reference image (if available) ──────────────────
    # Try the actual ASL chart image first, then fallback
    _ref_image = None
    for ref_name in [
        'The-26-letters-and-10-digits-of-American-Sign-Language-ASL.png',
        'asl_reference.png',
    ]:
        ref_path = os.path.join(BASE_DIR, ref_name)
        if os.path.exists(ref_path):
            _ref_image = cv2.imread(ref_path)
            if _ref_image is not None:
                break
    # Pre-compute resized reference for side-by-side display
    _ref_cached = None
    _ref_cached_h = -1
    show_guide = False

    # ── State ────────────────────────────────────────────────
    sentence = ""
    prev_letter = None
    stable_count = 0
    cooldown = 0
    fps = 0.0
    prev_time = time.time()

    # Systems
    particles = ParticleSystem()
    toasts = ToastManager()
    completer = WordCompleter()
    stats = SessionStats()
    letter_history = LetterHistory()
    smoother = ConfidenceSmoother(window=6, decay=0.65)
    quiz = QuizMode()
    suggestions = []

    # Recording
    is_recording = False
    video_writer = None
    number_mode = False

    print()
    print("  ╔════════════════════════════════════════════════╗")
    print("  ║   ASL Sign Language Detector · 2026 Edition   ║")
    print("  ╚════════════════════════════════════════════════╝")
    print()
    print("  ▶  Q:Quit  S:Speak  X:Delete  G:Guide  N:Numbers")
    print()

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        H, W, _ = frame.shape
        dt = time.time() - prev_time

        # ── Detection on ORIGINAL frame ──────────────────────
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        # ── Display frame (mirrored) ─────────────────────────
        if mirror:
            display = cv2.flip(frame, 1)
        else:
            display = frame.copy()

        current_letter = None
        confidence = 0.0
        top3 = []

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]

            # Extract features
            x_coords = [lm.x for lm in hand.landmark]
            y_coords = [lm.y for lm in hand.landmark]
            min_x, min_y = min(x_coords), min(y_coords)

            # ── Filter out face / nose false detections ──────
            # Check 1: handedness confidence from MediaPipe
            hand_ok = True
            if results.multi_handedness:
                hand_score = results.multi_handedness[0].classification[0].score
                if hand_score < 0.55:
                    hand_ok = False

            # Check 2: hand bounding box must be reasonable size
            #   A real hand spans roughly 5-40% of the frame.
            #   Face detections often produce very large or tiny boxes.
            bbox_w = max(x_coords) - min_x
            bbox_h = max(y_coords) - min_y
            bbox_area = bbox_w * bbox_h
            if bbox_area < 0.003 or bbox_area > 0.25:
                hand_ok = False

            # Check 3: aspect ratio – a real hand is roughly square-ish
            #   Face detections often look very wide or very tall
            if bbox_w > 0 and bbox_h > 0:
                aspect = max(bbox_w, bbox_h) / min(bbox_w, bbox_h)
                if aspect > 4.0:
                    hand_ok = False

            # Check 4: the wrist (landmark 0) should generally be
            #   in the lower part of the detected region, not above
            #   the middle finger tip (landmark 12). This fails for
            #   face/nose false positives where landmarks collapse.
            wrist_y = hand.landmark[0].y
            mid_tip_y = hand.landmark[12].y
            finger_spread = abs(hand.landmark[8].x - hand.landmark[20].x)
            if finger_spread < 0.01:
                hand_ok = False   # all landmarks collapsed to a point

            if not hand_ok:
                # Skip this frame – it's probably the face
                results.multi_hand_landmarks = None

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            x_coords = [lm.x for lm in hand.landmark]
            y_coords = [lm.y for lm in hand.landmark]
            min_x, min_y = min(x_coords), min(y_coords)

            features = []
            for lm in hand.landmark:
                features.append(lm.x - min_x)
                features.append(lm.y - min_y)

            if len(features) < NUM_FEATURES:
                features += [0.0] * (NUM_FEATURES - len(features))
            features = features[:NUM_FEATURES]

            # Predict with probabilities
            try:
                proba = model.predict_proba([np.array(features)])[0]
                # ── Smooth predictions over multiple frames ──
                proba = smoother.update(proba)
                # Top-3 predictions
                top_indices = np.argsort(proba)[::-1][:3]
                for idx in top_indices:
                    lbl = LABELS.get(int(model.classes_[idx]), str(model.classes_[idx]))
                    top3.append((lbl, proba[idx]))

                pred_idx = top_indices[0]
                confidence = proba[pred_idx]
                pred_label = model.classes_[pred_idx]
                current_letter = LABELS.get(int(pred_label), str(pred_label))
            except Exception:
                pred = model.predict([np.array(features)])[0]
                current_letter = LABELS.get(int(pred), str(pred))
                confidence = 1.0
                top3 = [(current_letter, 1.0)]

            # ── Number mode: remap letter → digit ────────────
            if number_mode and current_letter in LETTER_TO_NUMBER:
                digit = LETTER_TO_NUMBER[current_letter]
                # Update top3 display to show digit
                top3 = [(LETTER_TO_NUMBER.get(l, l), c) for l, c in top3]
                current_letter = digit

            # ── Draw neon hand skeleton ──────────────────────
            connections = mp_hands.HAND_CONNECTIONS
            pts = draw_neon_hand(display, hand.landmark, connections, mirror, W, H)

            # ── Bounding box with glow ───────────────────────
            if mirror:
                disp_x = [1.0 - x for x in x_coords]
            else:
                disp_x = list(x_coords)

            x1 = int(min(disp_x) * W) - 20
            y1 = int(min(y_coords) * H) - 20
            x2 = int(max(disp_x) * W) + 20
            y2 = int(max(y_coords) * H) + 20
            box_color = NEON_GREEN if confidence >= CONFIDENCE_THRESHOLD else DANGER_RED
            # Glow box
            cv2.rectangle(display, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2),
                          tuple(int(v * 0.3) for v in box_color), 3, cv2.LINE_AA)
            cv2.rectangle(display, (x1, y1), (x2, y2), box_color, 2, cv2.LINE_AA)

            stats.detections += 1

        # ── Stability logic ──────────────────────────────────
        if cooldown > 0:
            cooldown -= 1

        if (current_letter and
                current_letter != 'NOTHING' and
                confidence >= CONFIDENCE_THRESHOLD):
            if current_letter == prev_letter:
                stable_count += 1
            else:
                stable_count = 1
                prev_letter = current_letter

            if stable_count >= STABILITY_THRESHOLD and cooldown == 0:
                # CONFIRMED LETTER
                if current_letter == 'SPACE':
                    sentence += ' '
                    stats.add_word()
                    letter_history.add('_')
                    toasts.show("SPACE", NEON_BLUE, 1.0)
                elif current_letter == 'DEL':
                    if sentence:
                        sentence = sentence[:-1]
                    letter_history.add('<')
                    toasts.show("DELETE", NEON_ORANGE, 1.0)
                else:
                    sentence += current_letter
                    stats.add_letter()
                    letter_history.add(current_letter)
                    toasts.show(f"+ {current_letter}", NEON_GREEN, 0.8)

                # Particle burst at bounding box center
                if results.multi_hand_landmarks:
                    if mirror:
                        bx = int((1.0 - np.mean(x_coords)) * W)
                    else:
                        bx = int(np.mean(x_coords) * W)
                    by = int(np.mean(y_coords) * H)
                    burst_color = NEON_GREEN if current_letter not in ('DEL', 'SPACE') else NEON_ORANGE
                    particles.burst(bx, by, burst_color, count=25)

                stable_count = 0
                cooldown = COOLDOWN_FRAMES
                print(f"  ➜  {current_letter:>5s}   │  \"{sentence}\"")

                # Update suggestions
                suggestions = completer.suggest(sentence)

                # Quiz mode check
                if quiz.active and quiz.check(current_letter, confidence):
                    toasts.show(f"CORRECT! +{10 + (quiz.streak-1)*2}", NEON_GREEN, 1.2)
                    if results.multi_hand_landmarks:
                        if mirror:
                            qx = int((1.0 - np.mean(x_coords)) * W)
                        else:
                            qx = int(np.mean(x_coords) * W)
                        qy = int(np.mean(y_coords) * H)
                        particles.burst(qx, qy, NEON_PINK, count=40)
                    quiz.next_letter()
        else:
            stable_count = 0
            prev_letter = None

        # ── FPS ──────────────────────────────────────────────
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
        prev_time = now
        stats.frames += 1

        stability_pct = min(stable_count / STABILITY_THRESHOLD, 1.0)

        # ── Quiz timeout check ───────────────────────────────
        if quiz.active and quiz.check_timeout():
            toasts.show("TIME'S UP!", DANGER_RED, 1.0)

        # ── Update particles ─────────────────────────────────
        particles.update_and_draw(display, dt)

        # ── Draw HUD ─────────────────────────────────────────
        draw_hud(
            display,
            sentence=sentence,
            current_letter=current_letter,
            confidence=confidence,
            top3=top3,
            fps=fps,
            stability_pct=stability_pct,
            is_cooldown=cooldown > 0,
            suggestions=suggestions,
            stats=stats,
            letter_history=letter_history,
            is_recording=is_recording,
            number_mode=number_mode,
        )

        # ── Toasts ───────────────────────────────────────────
        toasts.draw(display)

        # ── Quiz overlay (drawn on top of everything) ────────
        if quiz.active:
            draw_quiz_overlay(display, quiz, current_letter, confidence)

        # ── Recording ────────────────────────────────────────
        if is_recording and video_writer is not None:
            video_writer.write(display)

        # ── Show frame (with optional side-by-side guide) ────
        if show_guide and _ref_image is not None:
            # Cache the resized reference (only recompute if height changed)
            if _ref_cached is None or _ref_cached_h != H:
                rh, rw = _ref_image.shape[:2]
                scale = H / rh
                new_w = int(rw * scale)
                _ref_cached = cv2.resize(_ref_image, (new_w, H), interpolation=cv2.INTER_AREA)
                _ref_cached_h = H
            combined = np.hstack([display, _ref_cached])
            cv2.imshow(WIN_NAME, combined)
        else:
            cv2.imshow(WIN_NAME, display)

        # ── Keyboard ─────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:           # Q or ESC
            break

        elif key == ord('s'):                       # Speak
            _speak(sentence)
            toasts.show("Speaking ...", NEON_CYAN)

        elif key == ord('c'):                       # Clear
            sentence = ""
            suggestions = []
            toasts.show("Cleared", NEON_ORANGE)
            print("  [cleared]")

        elif key == ord('e'):                       # Export
            path = _export_text(sentence)
            toasts.show(f"Exported!", NEON_GREEN)

        elif key == ord('f'):                       # Screenshot
            os.makedirs(EXPORTS_DIR, exist_ok=True)
            ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            ss_path = os.path.join(EXPORTS_DIR, f'screenshot_{ts}.png')
            cv2.imwrite(ss_path, display)
            toasts.show("Screenshot saved!", NEON_GREEN)
            print(f"  [✓] Screenshot → {ss_path}")

        elif key == ord('r'):                       # Record toggle
            if not is_recording:
                os.makedirs(EXPORTS_DIR, exist_ok=True)
                ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                vid_path = os.path.join(EXPORTS_DIR, f'recording_{ts}.avi')
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                video_writer = cv2.VideoWriter(vid_path, fourcc, 20.0, (W, H))
                is_recording = True
                toasts.show("Recording started", (0, 0, 255))
                print(f"  [✓] Recording → {vid_path}")
            else:
                is_recording = False
                if video_writer:
                    video_writer.release()
                    video_writer = None
                toasts.show("Recording saved", NEON_GREEN)
                print("  [✓] Recording saved")

        elif key == 8 or key == ord('x'):           # Backspace / X → delete last letter
            if sentence:
                sentence = sentence[:-1]
                suggestions = completer.suggest(sentence)
                toasts.show("Deleted letter", NEON_ORANGE, 0.6)

        elif key == 9:                              # Tab → accept top suggestion
            if suggestions:
                sentence = completer.complete(sentence, suggestions[0])
                toasts.show(f"-> {suggestions[0]}", NEON_YELLOW, 1.0)
                suggestions = completer.suggest(sentence)

        elif key in (ord('1'), ord('2'), ord('3')):  # Number for suggestion
            idx = key - ord('1')
            if idx < len(suggestions):
                sentence = completer.complete(sentence, suggestions[idx])
                toasts.show(f"-> {suggestions[idx]}", NEON_YELLOW, 1.0)
                suggestions = completer.suggest(sentence)
            else:
                toasts.show("No suggestion", TEXT_DIM, 0.7)

        elif key == ord('m'):                       # Toggle Quiz Mode
            quiz.toggle()
            if quiz.active:
                toasts.show("QUIZ MODE ON!", NEON_PINK, 1.5)
                print("  [Quiz Mode started]")
            else:
                toasts.show("Quiz ended", NEON_ORANGE, 1.0)
                if quiz.total_attempts > 0:
                    print(f"  [Quiz] Score: {quiz.score}  Accuracy: {quiz.accuracy:.0f}%  Best streak: {quiz.best_streak}")

        elif key == ord('n'):                       # N key
            if quiz.active:
                quiz.skip()
                toasts.show("Skipped", NEON_ORANGE, 0.7)
            else:
                number_mode = not number_mode
                if number_mode:
                    toasts.show("NUMBERS ON: D=1 V=2 W=3 B=4 ...", NEON_YELLOW, 2.5)
                    print("  [Number Mode ON] Signs for D,V,W,B,A,Y,S,L,F,O now show as 1-9,0")
                else:
                    toasts.show("LETTERS MODE (A-Z)", NEON_CYAN, 1.5)
                    print("  [Letter Mode ON]")

        elif key == ord('g'):                       # Toggle Reference Guide
            show_guide = not show_guide
            if show_guide:
                if _ref_image is not None:
                    toasts.show("Guide ON  (G to hide)", NEON_GREEN, 1.0)
                else:
                    toasts.show("No asl_reference.png found!", NEON_ORANGE, 1.5)
                    show_guide = False
            else:
                toasts.show("Guide OFF", NEON_ORANGE, 0.7)

    # ── Cleanup ──────────────────────────────────────────────
    if is_recording and video_writer:
        video_writer.release()
    cap.release()
    cv2.destroyAllWindows()
    hands.close()

    # Final summary
    print()
    print("  ┌─────────────────────────────────────┐")
    print(f"  │  Session Summary                     │")
    print("  ├─────────────────────────────────────┤")
    print(f"  │  Duration  : {format_time(stats.elapsed):>10s}             │")
    print(f"  │  Letters   : {stats.letters_count:>10d}             │")
    print(f"  │  Words     : {stats.words_count:>10d}             │")
    print(f"  │  Speed     : {stats.lpm:>7.1f} LPM           │")
    print("  └─────────────────────────────────────┘")
    if sentence.strip():
        print(f"\n  Final text: \"{sentence}\"\n")


# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    run()
