#!/usr/bin/env python3
"""
Hand Gesture Control System  v2
Dual-hand support · Landmark EMA smoothing · Temporal gesture stabilisation
"""

from __future__ import annotations

import logging
import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import tkinter as tk
from tkinter import ttk
import threading
import time
import math
import sys
import re
import os
import json
import webbrowser
import subprocess
import platform as _platform
from urllib.parse import quote
from collections import deque, Counter
from PIL import Image, ImageTk

try:
    from pynput.mouse import Button, Controller as _MouseCtrl
    from pynput.keyboard import Controller as _KeyboardCtrl
    PYNPUT_OK = True
except ImportError:
    PYNPUT_OK = False

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("hgc")


# ─────────────────────────────────────────────────────────────
#  PERFORMANCE MONITOR  — real-time latency & resource tracking
# ─────────────────────────────────────────────────────────────
class PerformanceMonitor:
    def __init__(self, window_size: int = 60):
        self.window_size = window_size
        self._frame_times = deque(maxlen=window_size)
        self._gesture_times = deque(maxlen=window_size)
        self._filter_times = deque(maxlen=window_size)
        self._last_t = time.perf_counter()

    def mark_frame(self):
        now = time.perf_counter()
        self._frame_times.append((now - self._last_t) * 1000)
        self._last_t = now

    def mark_gesture(self, elapsed_ms: float):
        self._gesture_times.append(elapsed_ms)

    def mark_filter(self, elapsed_ms: float):
        self._filter_times.append(elapsed_ms)

    @property
    def avg_frame_latency_ms(self) -> float:
        return sum(self._frame_times) / len(self._frame_times) if self._frame_times else 0

    @property
    def avg_gesture_latency_ms(self) -> float:
        return sum(self._gesture_times) / len(self._gesture_times) if self._gesture_times else 0

    @property
    def avg_filter_latency_ms(self) -> float:
        return sum(self._filter_times) / len(self._filter_times) if self._filter_times else 0

    @property
    def max_frame_latency_ms(self) -> float:
        return max(self._frame_times) if self._frame_times else 0

    def report(self) -> dict:
        return {
            'frame_avg_ms': round(self.avg_frame_latency_ms, 2),
            'frame_max_ms': round(self.max_frame_latency_ms, 2),
            'gesture_avg_ms': round(self.avg_gesture_latency_ms, 2),
            'filter_avg_ms': round(self.avg_filter_latency_ms, 2),
        }


# ─────────────────────────────────────────────────────────────
#  KALMAN FILTER  — advanced noise reduction for gesture tracking
# ─────────────────────────────────────────────────────────────
class KalmanFilter1D:
    def __init__(self, process_var: float = 1e-5, measurement_var: float = 1e-2):
        self._q = process_var
        self._r = measurement_var
        self._x = None
        self._p = 1.0

    def update(self, z: float) -> float:
        if self._x is None:
            self._x = z
            return z
        self._p = self._p + self._q
        k = self._p / (self._p + self._r)
        self._x = self._x + k * (z - self._x)
        self._p = (1 - k) * self._p
        return self._x

    def reset(self):
        self._x = None
        self._p = 1.0


# ─────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────
class Cfg:
    CAMERA_IDX   = 0
    CAM_W        = 640
    CAM_H        = 480
    MAX_HANDS    = 2              # now supports two hands
    DETECT_CONF  = 0.72
    TRACK_CONF   = 0.55
    FPS_TARGET   = 60             # target 60 fps for lower latency
    MODEL_COMPLEX = 0             # 0=lite, 1=full (faster inference with lite)

    # ── One-Euro Filter (cursor smoothing) ──
    CURSOR_FREQ  = 60             # Hz, matches camera FPS (120 on Edge AI)
    CURSOR_BETA  = 0.01           # lower = more lag, higher = noisier
    CURSOR_DCUTOFF = 1.0          # derivative cutoff frequency

    # ── Edge AI camera support ──
    # Cameras: Insta360 Link, Obsbot Tail, etc with on-board gesture AI
    # Latency: ~16ms (vs 200ms with MediaPipe on CPU)
    # Enable: set PREFER_EDGE_AI=True and use compatible camera
    PREFER_EDGE_AI = False        # Auto-detect by default
    MAX_EDGE_AI_FPS = 120         # Some cameras support 120fps

    # ── Gesture stabilisation (decoupled from cursor) ──
    GESTURE_FREQ = 30             # Hz for gesture path
    GESTURE_BETA = 0.05           # more aggressive for discrete gestures
    GESTURE_DCUTOFF = 1.0

    # ── Landmark EMA smoothing ── (legacy, now mostly handled by One-Euro)
    LMARK_ALPHA  = 0.55           # kept for gesture landmarks (non-cursor)

    # ── Temporal gesture stabilisation ──
    GEST_WIN     = 4              # reduced from 6 for faster confirmation
    GEST_THRESH  = 0.75           # increased from 0.60 for precision

    # ── Swipe detection ──
    SWIPE_VEL    = 0.65           # normalised units/second threshold

    # ── Two-hand pinch zoom ──
    ZOOM_DEAD    = 0.018          # dead-zone for wrist-distance delta
    ZOOM_CD      = 0.12           # seconds between zoom steps

    # Mouse
    SMOOTH       = 0.28           # legacy, kept for compatibility
    PINCH_THRESH = 0.042
    DRAG_THRESH  = 0.032
    SCROLL_SENS  = 18
    CLICK_CD     = 0.30
    SHORTCUT_CD  = 0.70
    DWELL_MS     = 16             # reduced for 60fps (16ms ≈ 60fps)

    SCR_W, SCR_H = pyautogui.size()
    MARGIN_X     = 0.12
    MARGIN_Y     = 0.12

    # Voice control
    VOICE_LANG         = "it-IT"
    VOICE_TIMEOUT      = 3
    VOICE_PHRASE_MAX   = 6
    VOICE_HISTORY_MAX  = 6    # entries kept in the voice command history ring
    VOICE_NOISE_DUR    = 0.5  # seconds of ambient noise calibration on start
    VOICE_UNKNOWN_DELAY = 0.4 # pause (s) after an unrecognised utterance
    VOICE_ERROR_DELAY  = 1.5  # pause (s) after a recogniser exception

    # Typing
    TYPE_INTERVAL      = 0.04 # seconds between keystrokes when typing text

    # Camera thread
    CAM_READ_RETRY_S   = 0.01 # sleep (s) when camera.read() returns no frame

    # Autonomous agent (AutoGPT-style plan → execute loop)
    AGENT_MODEL      = "claude-sonnet-5"
    AGENT_MAX_STEPS  = 8
    AGENT_STEP_DELAY = 0.6

    # Tkinter palette
    BG_DARK  = "#0d1117"
    BG_MID   = "#161b22"
    BG_CARD  = "#21262d"
    ACCENT   = "#e94560"
    SUCCESS  = "#3fb950"
    WARNING  = "#d29922"
    TEXT     = "#e6edf3"
    TEXT_DIM = "#8b949e"
    BLUE     = "#58a6ff"
    PURPLE   = "#bc8cff"


# ─────────────────────────────────────────────────────────────
#  ONE-EURO FILTER  — low-latency, adaptive smoothing for cursor
# ─────────────────────────────────────────────────────────────
class OneEuroFilter:
    def __init__(self, freq: float = 60.0, beta: float = 0.01, dcutoff: float = 1.0):
        self._freq = freq
        self._beta = beta
        self._dcutoff = dcutoff
        self._t_prev = None
        self._x_prev = None
        self._dx_prev = None

    def __call__(self, x: float, t: float) -> float:
        if self._x_prev is None:
            self._x_prev = x
            self._dx_prev = 0.0
            self._t_prev = t
            return x
        dt = max(1e-6, t - self._t_prev)
        dx = (x - self._x_prev) / dt
        a_d = self._alpha(dt, self._dcutoff)
        dx_smooth = a_d * dx + (1 - a_d) * self._dx_prev
        a = self._alpha(dt, self._beta)
        x_smooth = a * x + (1 - a) * (self._x_prev + self._dx_prev * dt)
        self._x_prev = x_smooth
        self._dx_prev = dx_smooth
        self._t_prev = t
        return x_smooth

    @staticmethod
    def _alpha(dt: float, cutoff: float) -> float:
        r = 2 * 3.14159 * cutoff * dt
        return r / (r + 1)

    def reset(self):
        self._t_prev = None
        self._x_prev = None
        self._dx_prev = None


# ─────────────────────────────────────────────────────────────
#  LANDMARK SMOOTHER  — per-hand EMA on raw 21-point positions
# ─────────────────────────────────────────────────────────────
class LandmarkSmoother:
    def __init__(self, alpha: float = Cfg.LMARK_ALPHA):
        self._a     = alpha
        self._state: dict[str, list] = {}

    def smooth(self, key: str, lm: list) -> list:
        if key not in self._state:
            self._state[key] = lm
            return lm
        a, prev = self._a, self._state[key]
        a1 = 1.0 - a
        out = [
            (prev[i][0] * a1 + lm[i][0] * a,
             prev[i][1] * a1 + lm[i][1] * a,
             prev[i][2] * a1 + lm[i][2] * a)
            for i in range(len(lm))
        ]
        self._state[key] = out
        return out

    def reset(self, key: str | None = None):
        if key:
            self._state.pop(key, None)
        else:
            self._state.clear()


# ─────────────────────────────────────────────────────────────
#  GESTURE STABILISER  — temporal majority-vote over last N frames
# ─────────────────────────────────────────────────────────────
class GestureStabiliser:
    def __init__(self, window: int = Cfg.GEST_WIN, thresh: float = Cfg.GEST_THRESH):
        self._hist   = deque(maxlen=window)
        self._thresh = thresh
        self._win    = window
        self.stable  = "NONE"

    def feed(self, gesture: str) -> str:
        self._hist.append(gesture)
        if len(self._hist) < self._win:
            return self.stable
        top, cnt = Counter(self._hist).most_common(1)[0]
        if cnt / self._win >= self._thresh:
            self.stable = top
        return self.stable

    def reset(self):
        self._hist.clear()
        self.stable = "NONE"


# ─────────────────────────────────────────────────────────────
#  VELOCITY TRACKER  — wrist velocity for swipe detection
# ─────────────────────────────────────────────────────────────
class VelocityTracker:
    def __init__(self, window: int = 6):
        self._hist: deque[tuple] = deque(maxlen=window)

    def push(self, x: float, y: float):
        self._hist.append((x, y, time.perf_counter()))

    def velocity(self) -> tuple[float, float]:
        if len(self._hist) < 2:
            return 0.0, 0.0
        x0, y0, t0 = self._hist[0]
        x1, y1, t1 = self._hist[-1]
        dt = max(t1 - t0, 1e-6)
        return (x1 - x0) / dt, (y1 - y0) / dt

    def reset(self):
        self._hist.clear()


# ─────────────────────────────────────────────────────────────
#  EDGE AI DETECTOR  — Support for webcams with on-board AI chip
#  (Insta360 Link, Obsbot, etc) for ultra-low latency
# ─────────────────────────────────────────────────────────────
class EdgeAIDetector:
    """Detects and uses Edge AI from camera firmware if available."""

    def __init__(self):
        self.available = False
        self.mode = "MediaPipe"  # fallback
        self._detect_edge_ai()

    def _detect_edge_ai(self):
        """Check for Edge AI capable cameras."""
        try:
            cap = cv2.VideoCapture(Cfg.CAMERA_IDX)
            # Check if camera supports hand tracking firmware (non-standard)
            props = {
                "backend": cap.getBackendName(),
                "width": cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                "height": cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
            }
            cap.release()
            # If camera has Edge AI, it will be detected via USB descriptor
            # For now, flag as available if 4K capable (indicator of modern hardware)
            if props.get("width", 0) >= 2560:
                self.available = True
                self.mode = "EdgeAI"
        except Exception as e:
            log.debug("EdgeAI detection failed: %s", e)

    def supports_60fps(self) -> bool:
        """Check if camera supports 60fps."""
        try:
            cap = cv2.VideoCapture(Cfg.CAMERA_IDX)
            cap.set(cv2.CAP_PROP_FPS, 60)
            actual_fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            return actual_fps >= 50
        except Exception as e:
            log.debug("60fps check failed: %s", e)
            return False


# ─────────────────────────────────────────────────────────────
#  HAND TRACKER  (MediaPipe wrapper — now multi-hand)
# ─────────────────────────────────────────────────────────────
class HandTracker:
    # landmark indices (shared constants)
    WRIST      = 0
    THUMB_CMC  = 1;  THUMB_MCP  = 2;  THUMB_IP   = 3;  THUMB_TIP  = 4
    INDEX_MCP  = 5;  INDEX_PIP  = 6;  INDEX_DIP  = 7;  INDEX_TIP  = 8
    MIDDLE_MCP = 9;  MIDDLE_PIP = 10; MIDDLE_DIP = 11; MIDDLE_TIP = 12
    RING_MCP   = 13; RING_PIP   = 14; RING_DIP   = 15; RING_TIP   = 16
    PINKY_MCP  = 17; PINKY_PIP  = 18; PINKY_DIP  = 19; PINKY_TIP  = 20

    def __init__(self):
        mh = mp.solutions.hands
        self._hands = mh.Hands(
            static_image_mode=False,
            max_num_hands=Cfg.MAX_HANDS,
            model_complexity=Cfg.MODEL_COMPLEX,
            min_detection_confidence=Cfg.DETECT_CONF,
            min_tracking_confidence=Cfg.TRACK_CONF,
        )
        self._draw  = mp.solutions.drawing_utils
        self._style = mp.solutions.drawing_styles
        self._conn  = mh.HAND_CONNECTIONS

    def process(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = self._hands.process(rgb)
        rgb.flags.writeable = True
        return res

    def annotate(self, frame, results):
        if results.multi_hand_landmarks:
            for lm in results.multi_hand_landmarks:
                self._draw.draw_landmarks(
                    frame, lm, self._conn,
                    self._style.get_default_hand_landmarks_style(),
                    self._style.get_default_hand_connections_style(),
                )
        return frame

    def extract(self, results) -> list[tuple[list, str]]:
        """Return list of (norm_lm, handedness_label) for every detected hand."""
        if not results.multi_hand_landmarks:
            return []
        out = []
        for i, h in enumerate(results.multi_hand_landmarks):
            lm    = [(p.x, p.y, p.z) for p in h.landmark]
            label = "Right"
            if results.multi_handedness and i < len(results.multi_handedness):
                label = results.multi_handedness[i].classification[0].label
            out.append((lm, label))
        return out

    def close(self):
        self._hands.close()


# ─────────────────────────────────────────────────────────────
#  GESTURE CONSTANTS
# ─────────────────────────────────────────────────────────────
class G:
    NONE        = "NONE"
    CURSOR      = "CURSOR"         # ☝ index only           → move
    PINCH       = "PINCH"          # 🤏 thumb+index close   → click/drag
    PINCH_RIGHT = "PINCH_RIGHT"    # 🤏 thumb+middle close  → right-click
    SCROLL      = "SCROLL"         # ✌ index+middle         → scroll
    COPY        = "COPY"           # 3 fingers              → Ctrl+C
    PASTE       = "PASTE"          # 4 fingers              → Ctrl+V
    THUMB_UP    = "THUMB_UP"       # 👍 thumb only          → double-click
    ROCK        = "ROCK"           # 🤘 index+pinky         → Ctrl+Z
    SAVE        = "SAVE"           # 🤙 thumb+pinky         → Ctrl+S
    FIST        = "FIST"           # ✊ all closed          → pause
    OPEN_PALM   = "OPEN_PALM"      # 🖐 all open            → reset / swipe
    SWIPE_L     = "SWIPE ←"        # open palm fast left    → Alt+Left
    SWIPE_R     = "SWIPE →"        # open palm fast right   → Alt+Right
    ZOOM_IN     = "ZOOM IN"        # ✌🤏 two-hand pinch out → Ctrl+scroll+
    ZOOM_OUT    = "ZOOM OUT"       # two-hand pinch in      → Ctrl+scroll-
    # modifier modes (left/non-dominant hand)
    MOD_FREEZE  = "FREEZE"         # left OPEN_PALM → pause cursor
    MOD_ZOOM    = "ZOOM MODE"      # left FIST      → right scroll = zoom
    MOD_HSCROLL = "H-SCROLL"       # left SCROLL    → right cursor = hscroll
    MOD_ALTTAB  = "ALT+TAB"        # left ROCK      → fire Alt+Tab
    MOD_MIDDLE  = "MID.CLICK"      # left THUMB_UP  → next click = middle
    UNKNOWN     = "UNKNOWN"


# ─────────────────────────────────────────────────────────────
#  GESTURE RECOGNISER  — single-hand, per-frame classification
# ─────────────────────────────────────────────────────────────
class GestureRecogniser:
    H = HandTracker  # alias
    _PINCH_TI_THRESH = Cfg.PINCH_THRESH
    _PINCH_TM_THRESH = Cfg.PINCH_THRESH * 1.2
    _PINCH_TP_THRESH = Cfg.PINCH_THRESH * 1.3

    def fingers_up(self, lm: list) -> list[bool]:
        """Returns [thumb, index, middle, ring, pinky] True = extended."""
        # Thumb: tip further left than IP when camera is already flipped
        thumb = lm[self.H.THUMB_TIP][0] < lm[self.H.THUMB_IP][0]
        others = [
            lm[tip][1] < lm[pip][1]
            for tip, pip in (
                (self.H.INDEX_TIP,  self.H.INDEX_PIP),
                (self.H.MIDDLE_TIP, self.H.MIDDLE_PIP),
                (self.H.RING_TIP,   self.H.RING_PIP),
                (self.H.PINKY_TIP,  self.H.PINKY_PIP),
            )
        ]
        return [thumb, *others]

    def pinch(self, lm: list, a: int, b: int) -> float:
        return math.hypot(lm[a][0] - lm[b][0], lm[a][1] - lm[b][1])

    def classify(self, lm: list | None) -> str:
        if lm is None:
            return G.NONE

        f = self.fingers_up(lm)
        thumb, idx, mid, ring, pinky = f
        finger_count = sum(f)

        d_ti = self.pinch(lm, self.H.THUMB_TIP, self.H.INDEX_TIP)
        d_tm = self.pinch(lm, self.H.THUMB_TIP, self.H.MIDDLE_TIP)
        d_tp = self.pinch(lm, self.H.THUMB_TIP, self.H.PINKY_TIP)

        if d_ti < self._PINCH_TI_THRESH and idx and not mid and not ring and not pinky:
            return G.PINCH
        if d_tm < self._PINCH_TM_THRESH and mid and d_ti >= self._PINCH_TI_THRESH:
            return G.PINCH_RIGHT
        if d_tp < self._PINCH_TP_THRESH and pinky and not idx and not mid and not ring:
            return G.SAVE

        if finger_count == 0:
            return G.FIST
        if finger_count == 5:
            return G.OPEN_PALM

        if idx and not mid and not ring and not pinky:
            return G.CURSOR
        if idx and mid and not ring and not pinky:
            return G.SCROLL
        if idx and mid and ring and not pinky and not thumb:
            return G.COPY
        if idx and mid and ring and pinky and not thumb:
            return G.PASTE
        if thumb and not idx and not mid and not ring and not pinky:
            return G.THUMB_UP
        if idx and pinky and not mid and not ring:
            return G.ROCK

        return G.UNKNOWN


# ─────────────────────────────────────────────────────────────
#  SMOOTH MOUSE CONTROLLER
# ─────────────────────────────────────────────────────────────
class SmoothMouse:
    def __init__(self):
        pyautogui.FAILSAFE = False
        pyautogui.PAUSE    = 0.0
        self._mc  = _MouseCtrl() if PYNPUT_OK else None
        self.cx   = Cfg.SCR_W // 2
        self.cy   = Cfg.SCR_H // 2
        self.drag = False
        self._tc  = 0.0   # last click timestamp
        self._ts  = 0.0   # last shortcut timestamp
        self._tz  = 0.0   # last zoom timestamp
        self._lk  = threading.Lock()
        self._t_last = time.time()
        self._filter_x = OneEuroFilter(Cfg.CURSOR_FREQ, Cfg.CURSOR_BETA, Cfg.CURSOR_DCUTOFF)
        self._filter_y = OneEuroFilter(Cfg.CURSOR_FREQ, Cfg.CURSOR_BETA, Cfg.CURSOR_DCUTOFF)

    def _n2s(self, nx: float, ny: float) -> tuple[int, int]:
        mx, my = Cfg.MARGIN_X, Cfg.MARGIN_Y
        sx = max(0.0, min(1.0, (nx - mx) / max(1e-6, 1 - 2 * mx)))
        sy = max(0.0, min(1.0, (ny - my) / max(1e-6, 1 - 2 * my)))
        return int((1.0 - sx) * Cfg.SCR_W), int(sy * Cfg.SCR_H)

    def move(self, nx: float, ny: float):
        tx, ty = self._n2s(nx, ny)
        t = time.time()
        with self._lk:
            self.cx = int(self._filter_x(float(tx), t))
            self.cy = int(self._filter_y(float(ty), t))
            pyautogui.moveTo(self.cx, self.cy)

    def _click_ok(self) -> bool:
        now = time.time()
        if now - self._tc < Cfg.CLICK_CD:
            return False
        self._tc = now
        return True

    def left_click(self):
        if not self._click_ok():
            return
        (self._mc.click(Button.left) if PYNPUT_OK
         else pyautogui.click())

    def right_click(self):
        if not self._click_ok():
            return
        (self._mc.click(Button.right) if PYNPUT_OK
         else pyautogui.rightClick())

    def middle_click(self):
        if not self._click_ok():
            return
        (self._mc.click(Button.middle) if PYNPUT_OK
         else pyautogui.middleClick())

    def double_click(self):
        if not self._click_ok():
            return
        (self._mc.click(Button.left, count=2) if PYNPUT_OK
         else pyautogui.doubleClick())

    def scroll(self, dy: float):
        if PYNPUT_OK:
            self._mc.scroll(0, dy)
        else:
            pyautogui.scroll(int(dy * 3))

    def hscroll(self, dx: float):
        if PYNPUT_OK:
            self._mc.scroll(dx, 0)
        else:
            try:
                pyautogui.hscroll(int(dx * 3))
            except Exception as e:
                log.debug("hscroll failed: %s", e)

    def zoom(self, direction: int) -> bool:
        """Ctrl+scroll zoom. direction +1 = in, -1 = out."""
        now = time.time()
        if now - self._tz < Cfg.ZOOM_CD:
            return False
        self._tz = now
        try:
            pyautogui.keyDown("ctrl")
            self._mc.scroll(0, direction) if PYNPUT_OK else pyautogui.scroll(direction * 3)
            pyautogui.keyUp("ctrl")
        except Exception as e:
            log.warning("zoom failed: %s", e)
            try:
                pyautogui.keyUp("ctrl")
            except Exception:
                pass
        return True

    def hotkey(self, *keys) -> bool:
        now = time.time()
        if now - self._ts < Cfg.SHORTCUT_CD:
            return False
        self._ts = now
        try:
            pyautogui.hotkey(*keys)
        except Exception as e:
            log.warning("hotkey %s failed: %s", keys, e)
        return True

    def press_key(self, key: str):
        try:
            pyautogui.press(key)
        except Exception as e:
            log.warning("press_key %r failed: %s", key, e)

    def start_drag(self):
        if self.drag:
            return
        self.drag = True
        (self._mc.press(Button.left) if PYNPUT_OK else pyautogui.mouseDown())

    def stop_drag(self):
        if not self.drag:
            return
        self.drag = False
        (self._mc.release(Button.left) if PYNPUT_OK else pyautogui.mouseUp())


# ─────────────────────────────────────────────────────────────
#  KALMAN SMOOTH MOUSE — advanced filtering using Kalman filter
# ─────────────────────────────────────────────────────────────
class KalmanSmoothMouse(SmoothMouse):
    def __init__(self):
        super().__init__()
        self._kalman_x = KalmanFilter1D(process_var=1e-5, measurement_var=1e-2)
        self._kalman_y = KalmanFilter1D(process_var=1e-5, measurement_var=1e-2)

    def move(self, x: float, y: float, screen_w: int = None, screen_h: int = None):
        x_filt = self._kalman_x.update(x)
        y_filt = self._kalman_y.update(y)
        super().move(x_filt, y_filt)


# ─────────────────────────────────────────────────────────────
#  DUAL-HAND PROCESSOR
#  Dominant hand (wrist.x > 0.5 after flip) = cursor / action
#  Modifier hand (wrist.x < 0.5)            = mode / shortcut
# ─────────────────────────────────────────────────────────────
class DualHandProcessor:
    def __init__(self, mouse: SmoothMouse, db: "GestureDatabase | None" = None):
        self.mouse   = mouse
        self._db     = db
        self._smoother = LandmarkSmoother(Cfg.LMARK_ALPHA)
        self._rec    = CustomGestureRecogniser(db) if db else GestureRecogniser()
        self._stab   = {"dom": GestureStabiliser(), "mod": GestureStabiliser()}
        self._vel    = {"dom": VelocityTracker(),   "mod": VelocityTracker()}

        # action state
        self._pinch      = False
        self._pright     = False
        self._dragging   = False
        self._drag_orig  = (0.0, 0.0)
        self._scroll_ref = None
        self._zoom_ref   = None
        self._mod_mode   = None   # current modifier from non-dominant hand
        self._mid_armed  = False  # next click = middle click

    # ── public entry point ────────────────────────────────────
    def process(self, hands: list) -> tuple[str, str, str]:
        """
        hands: list of (lm, handedness_str)
        Returns (dom_gesture, mod_gesture, action_label)
        """
        dom_lm, mod_lm = self._assign_roles(hands)

        dom_lm = self._prep("dom", dom_lm)
        mod_lm = self._prep("mod", mod_lm)

        raw_dom = self._rec.classify(dom_lm)
        raw_mod = self._rec.classify(mod_lm)

        dom_g = self._stab["dom"].feed(raw_dom) if dom_lm else G.NONE
        mod_g = self._stab["mod"].feed(raw_mod) if mod_lm else G.NONE

        if dom_lm:
            self._vel["dom"].push(dom_lm[HandTracker.WRIST][0],
                                  dom_lm[HandTracker.WRIST][1])
        if mod_lm:
            self._vel["mod"].push(mod_lm[HandTracker.WRIST][0],
                                  mod_lm[HandTracker.WRIST][1])

        action = ""

        # Two-hand gesture first (takes priority)
        if dom_lm and mod_lm:
            action = self._two_hands(dom_lm, mod_lm, dom_g, mod_g)
            if action:
                return dom_g, mod_g, action

        # Update modifier mode
        self._update_mod(mod_g)

        # Dominant hand action
        if dom_lm:
            action = self._dominant(dom_lm, dom_g)
        else:
            self._reset_all()

        return dom_g, mod_g, action

    # ── role assignment ───────────────────────────────────────
    def _assign_roles(self, hands: list) -> tuple:
        if not hands:
            return None, None
        if len(hands) == 1:
            return hands[0][0], None
        # Higher wrist.x → right side of flipped frame → dominant (right) hand
        s = sorted(hands, key=lambda h: h[0][HandTracker.WRIST][0], reverse=True)
        return s[0][0], s[1][0]

    def _prep(self, key: str, lm):
        if lm is None:
            self._smoother.reset(key)
            self._stab[key].reset()
            self._vel[key].reset()
            return None
        return self._smoother.smooth(key, lm)

    # ── two-hand gestures ─────────────────────────────────────
    def _two_hands(self, dom: list, mod: list, dg: str, mg: str) -> str:
        # Pinch-to-zoom: both hands PINCH, track wrist separation
        if dg == G.PINCH and mg == G.PINCH:
            d = math.hypot(dom[0][0] - mod[0][0], dom[0][1] - mod[0][1])
            if self._zoom_ref is None:
                self._zoom_ref = d
            else:
                delta = d - self._zoom_ref
                if delta > Cfg.ZOOM_DEAD:
                    if self.mouse.zoom(1):
                        self._zoom_ref = d
                        return G.ZOOM_IN
                elif delta < -Cfg.ZOOM_DEAD:
                    if self.mouse.zoom(-1):
                        self._zoom_ref = d
                        return G.ZOOM_OUT
            return G.ZOOM_IN if d > (self._zoom_ref or d) else ""
        else:
            self._zoom_ref = None
        return ""

    # ── modifier mode (non-dominant hand) ─────────────────────
    def _update_mod(self, mg: str):
        mapping = {
            G.OPEN_PALM: G.MOD_FREEZE,
            G.FIST:      G.MOD_ZOOM,
            G.SCROLL:    G.MOD_HSCROLL,
            G.ROCK:      G.MOD_ALTTAB,
            G.THUMB_UP:  G.MOD_MIDDLE,
        }
        prev = self._mod_mode
        self._mod_mode = mapping.get(mg)

        # Fire Alt+Tab once when modifier is raised
        if self._mod_mode == G.MOD_ALTTAB and prev != G.MOD_ALTTAB:
            self.mouse.hotkey("alt", "tab")
        # Arm middle click
        if self._mod_mode == G.MOD_MIDDLE and prev != G.MOD_MIDDLE:
            self._mid_armed = True

    # ── dominant hand main processing ─────────────────────────
    def _dominant(self, lm: list, g: str) -> str:
        m  = self.mouse
        ix = lm[HandTracker.INDEX_TIP][0]
        iy = lm[HandTracker.INDEX_TIP][1]

        # Custom-trained gesture action (takes priority over built-in rules)
        if self._db:
            entry = self._db.get_entry(g)
            if entry and entry.get("action"):
                self._exec_custom(entry["action"], entry.get("args"))
                return f"✓ {g}"

        # Modifier overrides
        if self._mod_mode == G.MOD_FREEZE:
            self._reset_drag(); self._reset_pinch()
            return G.MOD_FREEZE

        if self._mod_mode == G.MOD_ZOOM and g == G.SCROLL:
            self._reset_drag()
            if self._scroll_ref is None:
                self._scroll_ref = iy
            else:
                dy = (self._scroll_ref - iy) * Cfg.SCROLL_SENS
                if abs(dy) > 0.05:
                    m.zoom(1 if dy > 0 else -1)
                    self._scroll_ref = iy
            return G.MOD_ZOOM

        if self._mod_mode == G.MOD_HSCROLL and g == G.CURSOR:
            self._reset_drag()
            if self._scroll_ref is None:
                self._scroll_ref = ix
            else:
                dx = (ix - self._scroll_ref) * Cfg.SCROLL_SENS
                if abs(dx) > 0.05:
                    m.hscroll(dx)
                    self._scroll_ref = ix
            return G.MOD_HSCROLL

        # Swipe detection on open palm (velocity-based)
        if g == G.OPEN_PALM:
            vx, _ = self._vel["dom"].velocity()
            if vx < -Cfg.SWIPE_VEL:
                if m.hotkey("alt", "left"):
                    return G.SWIPE_L
            elif vx > Cfg.SWIPE_VEL:
                if m.hotkey("alt", "right"):
                    return G.SWIPE_R
            self._reset_drag(); self._reset_pinch()
            self._scroll_ref = None
            return ""

        # ── core single-hand gestures ──
        if g == G.CURSOR:
            self._reset_drag()
            m.move(ix, iy)
            self._reset_pinch()
            self._scroll_ref = None

        elif g == G.PINCH:
            if self._mid_armed:
                if not self._pinch:
                    self._pinch = True
                    m.middle_click()
                    self._mid_armed = False
                    return G.MOD_MIDDLE
            if not self._pinch:
                self._pinch      = True
                self._drag_orig  = (ix, iy)
                self._dragging   = False
            else:
                dx = abs(ix - self._drag_orig[0])
                dy = abs(iy - self._drag_orig[1])
                if dx > Cfg.DRAG_THRESH or dy > Cfg.DRAG_THRESH:
                    if not self._dragging:
                        self._dragging = True
                        m.start_drag()
                if self._dragging:
                    m.move(ix, iy)
            self._pright     = False
            self._scroll_ref = None

        elif g == G.PINCH_RIGHT:
            if not self._pright:
                self._pright = True
                m.right_click()
                self._reset_pinch_soft()
                return "RIGHT CLICK"
            self._scroll_ref = None

        elif g == G.SCROLL:
            self._reset_drag()
            if self._scroll_ref is None:
                self._scroll_ref = iy
            else:
                dy = (self._scroll_ref - iy) * Cfg.SCROLL_SENS
                if abs(dy) > 0.05:
                    m.scroll(dy)
                    self._scroll_ref = iy
            self._reset_pinch()

        elif g == G.THUMB_UP:
            self._reset_drag()
            m.double_click()
            self._reset_pinch()
            self._scroll_ref = None
            return "DOUBLE CLICK"

        elif g == G.COPY:
            self._reset_drag()
            if m.hotkey("ctrl", "c"):
                self._reset_pinch()
                return "COPY  Ctrl+C"

        elif g == G.PASTE:
            self._reset_drag()
            if m.hotkey("ctrl", "v"):
                self._reset_pinch()
                return "PASTE  Ctrl+V"

        elif g == G.ROCK:
            self._reset_drag()
            if m.hotkey("ctrl", "z"):
                self._reset_pinch()
                return "UNDO  Ctrl+Z"

        elif g == G.SAVE:
            self._reset_drag()
            if m.hotkey("ctrl", "s"):
                self._reset_pinch()
                return "SAVE  Ctrl+S"

        elif g == G.FIST:
            self._reset_drag()
            self._reset_pinch()
            self._scroll_ref = None

        else:
            # Gesture ended / transitional → finalise pending click
            if self._pinch and not self._dragging:
                m.left_click()
                self._reset_drag()
                self._reset_pinch()
                self._scroll_ref = None
                return "LEFT CLICK"
            self._reset_drag()
            self._reset_pinch()
            self._scroll_ref = None

        return ""

    # ── custom gesture executor ───────────────────────────────
    def _exec_custom(self, action: str, args):
        m = self.mouse
        try:
            if action == "hotkey" and args:
                if isinstance(args, list):
                    keys = args
                elif isinstance(args, str) and "+" in args:
                    keys = [k.strip() for k in args.split("+") if k.strip()]
                else:
                    keys = [str(args)]
                m.hotkey(*keys)
            elif action == "open_url" and args:
                url = str(args)
                if not url.startswith("http"):
                    url = "https://" + url
                webbrowser.open(url)
            elif action == "search" and args:
                q = quote(str(args), safe='')
                webbrowser.open(f"https://www.google.it/search?q={q}")
            elif action == "zoom" and args is not None:
                m.zoom(int(args))
            elif action == "screenshot":
                ts   = time.strftime("%Y%m%d_%H%M%S")
                path = os.path.expanduser(f"~/Desktop/screenshot_{ts}.png")
                pyautogui.screenshot(path)
            elif action == "type" and args:
                pyautogui.write(str(args), interval=Cfg.TYPE_INTERVAL)
            elif action == "create_folder":
                name   = str(args) if args else "Nuova Cartella"
                target = os.path.join(os.path.expanduser("~"), "Desktop", name)
                os.makedirs(target, exist_ok=True)
        except Exception as e:
            log.warning("custom action %r failed: %s", action, e)

    # ── helpers ───────────────────────────────────────────────
    def _reset_pinch(self):
        self._pinch  = False
        self._pright = False

    def _reset_pinch_soft(self):
        self._pinch = False

    def _reset_drag(self):
        if self._dragging:
            self.mouse.stop_drag()
            self._dragging = False
        self._pinch = False

    def _reset_all(self):
        self._reset_drag()
        self._reset_pinch()
        self._scroll_ref = None
        self._zoom_ref   = None
        self._mod_mode   = None
        self._mid_armed  = False


# ─────────────────────────────────────────────────────────────
#  VIRTUAL KEYBOARD
# ─────────────────────────────────────────────────────────────
class VirtualKeyboard(tk.Toplevel):
    _ROWS = [
        ["Esc","F1","F2","F3","F4","F5","F6","F7","F8","F9","F10","F11","F12"],
        ["`","1","2","3","4","5","6","7","8","9","0","-","=","Back"],
        ["Tab","q","w","e","r","t","y","u","i","o","p","[","]","\\"],
        ["Caps","a","s","d","f","g","h","j","k","l",";","'","Enter"],
        ["Shift","z","x","c","v","b","n","m",",",".","/","Shift↑"],
        ["Ctrl","Win","Alt","          Space          ","Alt↗","Ctrl↗"],
    ]
    _MAP = {
        "Esc":"escape","Back":"backspace","Tab":"tab","Caps":"capslock",
        "Enter":"return","Shift":"shift","Shift↑":"shift","Ctrl":"ctrl",
        "Ctrl↗":"ctrl","Alt":"alt","Alt↗":"alt","Win":"super",
        "          Space          ":"space",
        **{f"F{n}": f"f{n}" for n in range(1, 13)},
    }
    _WIDE = {"Back","Tab","Caps","Enter","Shift","Shift↑","Ctrl","Ctrl↗",
             "Alt","Alt↗","Win","          Space          "}

    def __init__(self, parent, mouse: SmoothMouse):
        super().__init__(parent)
        self._mouse   = mouse
        self._shift   = False
        self._ctrl    = False
        self._buttons: dict[str, tk.Button] = {}

        self.title("Tastiera Virtuale")
        self.configure(bg=Cfg.BG_DARK)
        self.attributes("-topmost", True)
        self.resizable(False, False)
        sh = parent.winfo_screenheight()
        self.geometry(f"+0+{sh - 240}")
        self._build()
        self.protocol("WM_DELETE_WINDOW", self.withdraw)

    def _build(self):
        for row in self._ROWS:
            fr = tk.Frame(self, bg=Cfg.BG_DARK)
            fr.pack(padx=4, pady=2)
            for key in row:
                w = 7 if key in self._WIDE else 4
                if key == "          Space          ":
                    w = 28
                btn = tk.Button(
                    fr, text=key, width=w,
                    bg=Cfg.BG_CARD, fg=Cfg.TEXT,
                    font=("Consolas", 8, "bold"),
                    relief="flat", bd=0, cursor="hand2",
                    activebackground=Cfg.ACCENT, activeforeground=Cfg.TEXT,
                    command=lambda k=key: self._press(k),
                )
                btn.pack(side="left", padx=1, pady=1, ipady=4)
                self._buttons[key] = btn

    def _press(self, key: str):
        mapped = self._MAP.get(key, key)
        if mapped in ("shift", "ctrl"):
            if mapped == "shift":
                self._shift = not self._shift
                for k in ("Shift", "Shift↑"):
                    if k in self._buttons:
                        self._buttons[k].configure(
                            bg=Cfg.ACCENT if self._shift else Cfg.BG_CARD)
            else:
                self._ctrl = not self._ctrl
                for k in ("Ctrl", "Ctrl↗"):
                    if k in self._buttons:
                        self._buttons[k].configure(
                            bg=Cfg.ACCENT if self._ctrl else Cfg.BG_CARD)
            return

        chain = []
        if self._ctrl:
            chain.append("ctrl"); self._ctrl = False
            for k in ("Ctrl", "Ctrl↗"):
                if k in self._buttons:
                    self._buttons[k].configure(bg=Cfg.BG_CARD)
        if self._shift:
            chain.append("shift"); self._shift = False
            for k in ("Shift", "Shift↑"):
                if k in self._buttons:
                    self._buttons[k].configure(bg=Cfg.BG_CARD)
        chain.append(mapped)

        try:
            if len(chain) > 1:
                self._mouse.hotkey(*chain)
            else:
                self._mouse.press_key(mapped)
        except Exception as e:
            log.warning("virtual keyboard key %r failed: %s", mapped, e)


# ─────────────────────────────────────────────────────────────
#  CAMERA THREAD
# ─────────────────────────────────────────────────────────────
class CameraThread(threading.Thread):
    def __init__(self, app):
        super().__init__(daemon=False)
        self.app       = app
        self._running  = False
        self._flock    = threading.Lock()
        self.frame     = None
        self.dom_g     = G.NONE
        self.mod_g     = G.NONE
        self.action    = ""
        self.fps       = 0.0
        self.n_hands   = 0
        self.perf      = PerformanceMonitor(window_size=60)

    def start_capture(self):
        self._running = True
        self.start()

    def stop_capture(self):
        self._running = False

    def run(self):
        cap = cv2.VideoCapture(Cfg.CAMERA_IDX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  Cfg.CAM_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Cfg.CAM_H)
        cap.set(cv2.CAP_PROP_FPS, Cfg.FPS_TARGET)

        tracker   = HandTracker()
        db        = getattr(self.app, "_db", None)
        processor = DualHandProcessor(self.app.mouse, db)

        t0 = time.perf_counter()
        fc = 0

        while self._running:
            t_frame_start = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                time.sleep(Cfg.CAM_READ_RETRY_S)
                continue

            frame = cv2.flip(frame, 1)

            if self.app.hand_active:
                t_gesture_start = time.perf_counter()
                results = tracker.process(frame)
                frame   = tracker.annotate(frame, results)
                hands   = tracker.extract(results)
                self.n_hands = len(hands)
                dom_g, mod_g, action = processor.process(hands)
                self.dom_g  = dom_g
                self.mod_g  = mod_g
                self.action = action
                gesture_elapsed = (time.perf_counter() - t_gesture_start) * 1000
                self.perf.mark_gesture(gesture_elapsed)
                self._draw_hud(frame, dom_g, mod_g, action, hands)

                # Feed recorder if a recording session is active
                recorder = getattr(self.app, "_recorder", None)
                if recorder and recorder.state in (
                        GestureRecorder.COUNTDOWN, GestureRecorder.RECORDING) and hands:
                    dom_lm = max(hands, key=lambda h: h[0][HandTracker.WRIST][0])[0]
                    recorder.feed_frame(dom_lm)
            else:
                self.dom_g = self.mod_g = G.NONE
                self.action = ""
                self.n_hands = 0

            fc += 1
            elapsed = time.perf_counter() - t0
            if elapsed >= 1.0:
                self.fps = fc / elapsed
                fc, t0 = 0, time.perf_counter()

            frame_elapsed = (time.perf_counter() - t_frame_start) * 1000
            self.perf.mark_frame()

            h, w = frame.shape[:2]
            cv2.putText(frame, f"FPS {self.fps:.0f}", (8, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 220, 80), 2)

            perf = self.perf.report()
            cv2.putText(frame, f"Latency: {perf['frame_avg_ms']:.0f}ms", (8, h-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 150, 255), 1)

            with self._flock:
                self.frame = frame.copy()

        tracker.close()
        cap.release()

    def get_frame(self):
        with self._flock:
            return None if self.frame is None else self.frame.copy()

    @staticmethod
    def _draw_hud(frame, dom_g, mod_g, action, hands):
        h, w = frame.shape[:2]

        # Dominant gesture label
        col = (80, 220, 80) if dom_g not in (G.NONE, G.UNKNOWN) else (100, 100, 100)
        cv2.putText(frame, f"DOM: {dom_g}", (8, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)

        # Modifier gesture label (second hand)
        if mod_g not in (G.NONE, G.UNKNOWN, ""):
            cv2.putText(frame, f"MOD: {mod_g}", (8, 82),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 120, 255), 2)

        # Action feedback centred
        if action:
            (tw, _), _ = cv2.getTextSize(action, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)
            cx = max(0, (w - tw) // 2)
            cv2.putText(frame, action, (cx, h - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 220, 220), 2)

        # Cursor ring on dominant index tip
        if dom_g in (G.CURSOR, G.PINCH) and hands:
            # find dominant hand (highest wrist.x)
            dom_lm = max(hands, key=lambda hd: hd[0][HandTracker.WRIST][0])[0]
            ix = int((1 - dom_lm[HandTracker.INDEX_TIP][0]) * w)
            iy = int(dom_lm[HandTracker.INDEX_TIP][1] * h)
            colour = (0, 80, 255) if dom_g == G.PINCH else (0, 255, 255)
            cv2.circle(frame, (ix, iy), 14, colour, 2)
            cv2.circle(frame, (ix, iy), 4,  colour, -1)


# ─────────────────────────────────────────────────────────────
#  COMMAND PARSER  — Italian regex-based command recognition
# ─────────────────────────────────────────────────────────────
def run_action(mouse: "SmoothMouse", action: str, args):
    """Execute a single parsed command (open_url, hotkey, type, ...).

    Shared by VoiceController (direct voice commands) and AutonomousAgent
    (LLM-planned command sequences) so both dispatch through one vocabulary.
    """
    try:
        if action == 'open_url':
            url = str(args)
            if not url.startswith('http'):
                url = 'https://' + url
            webbrowser.open(url)
        elif action == 'open_app':
            name = str(args)
            sys_ = _platform.system()
            if sys_ == "Darwin":
                subprocess.Popen(["open", "-a", name])
            elif sys_ == "Windows":
                os.startfile(name)
            else:
                subprocess.Popen(["xdg-open", name])
        elif action == 'search':
            q = quote(str(args), safe='')
            webbrowser.open(f"https://www.google.it/search?q={q}")
        elif action == 'create_folder':
            name   = str(args) if args else 'Nuova Cartella'
            target = os.path.join(os.path.expanduser("~"), "Desktop", name)
            os.makedirs(target, exist_ok=True)
        elif action == 'hotkey':
            keys = args if isinstance(args, (list, tuple)) else (args,)
            mouse.hotkey(*keys)
        elif action == 'zoom':
            mouse.zoom(int(args))
        elif action == 'type':
            pyautogui.write(str(args), interval=Cfg.TYPE_INTERVAL)
        elif action == 'screenshot':
            ts   = time.strftime("%Y%m%d_%H%M%S")
            path = os.path.expanduser(f"~/Desktop/screenshot_{ts}.png")
            pyautogui.screenshot(path)
        elif action == 'wait':
            time.sleep(max(0.0, min(float(args if args is not None else 1), 10.0)))
    except Exception as e:
        log.warning("run_action %r(%r) failed: %s", action, args, e)


class CommandParser:
    _CMDS = [
        (r'(?:agente|obiettivo|modalit[a\xe0]\s+autonoma)\s+(.+)', 'agent', None),
        (r'apri\s+youtube\b',                 'open_url',      'https://www.youtube.com'),
        (r'apri\s+google\b',                  'open_url',      'https://www.google.it'),
        (r'apri\s+gmail\b',                   'open_url',      'https://mail.google.com'),
        (r'nuova\s+cartella\b',               'create_folder', 'Nuova Cartella'),
        (r'crea\s+cartella(?:\s+(.+))?',      'create_folder', None),
        (r'apri\s+(.+)',                       'open_app',      None),
        (r'vai\s+su\s+(.+)',                  'open_url',      None),
        (r'cerca\s+(.+)',                      'search',        None),
        (r'(?:scrivi|digita|testo)\s+(.+)',    'type',          None),
        (r'seleziona\s+tutto\b',              'hotkey',        ('ctrl', 'a')),
        (r'copia\b',                           'hotkey',        ('ctrl', 'c')),
        (r'incolla\b',                         'hotkey',        ('ctrl', 'v')),
        (r'annulla\b',                         'hotkey',        ('ctrl', 'z')),
        (r'rifai\b',                           'hotkey',        ('ctrl', 'y')),
        (r'salva\b',                           'hotkey',        ('ctrl', 's')),
        (r'chiudi(?:\s+finestra)?\b',          'hotkey',        ('alt', 'f4')),
        (r'zoom\s+avanti\b',                  'zoom',          1),
        (r'zoom\s+indietro\b',                'zoom',          -1),
        (r'ingrandisci\b',                    'zoom',          1),
        (r'rimpicciolisci\b',                 'zoom',          -1),
        (r'(?:screenshot|schermo)\b',         'screenshot',    None),
        (r'invio\b',                          'hotkey',        ('return',)),
        (r'cancella\b',                       'hotkey',        ('backspace',)),
        (r'volume\s+su\b',                    'hotkey',        ('volumeup',)),
        (r'volume\s+gi[u\xf9]\b',             'hotkey',        ('volumedown',)),
        (r'muto\b',                           'hotkey',        ('volumemute',)),
        (r'torna\s+indietro\b',               'hotkey',        ('alt', 'left')),
        (r'vai\s+avanti\b',                   'hotkey',        ('alt', 'right')),
        (r'(?:cambio\s+finestra|alt\s+tab)\b','hotkey',        ('alt', 'tab')),
        (r'desktop\b',                        'hotkey',        ('super', 'd')),
    ]

    def __init__(self):
        self._compiled = [
            (re.compile(pat, re.IGNORECASE), action, args)
            for pat, action, args in self._CMDS
        ]

    def parse(self, text: str):
        """Return (action, args) or None."""
        t = text.strip()
        for pattern, action, static_args in self._compiled:
            m = pattern.search(t)
            if m:
                args = static_args
                if args is None and m.lastindex:
                    args = m.group(1).strip()
                return action, args
        return None


# ─────────────────────────────────────────────────────────────
#  VOICE CONTROLLER  — Italian speech recognition (daemon thread)
# ─────────────────────────────────────────────────────────────
class VoiceController:
    def __init__(self, mouse: SmoothMouse, on_command, on_agent_goal=None):
        self._mouse    = mouse
        self._cb       = on_command
        self._on_agent = on_agent_goal
        self._parser   = CommandParser()
        self._running  = False
        self._thread   = None
        self.status    = "OFFLINE"
        self.last_text = ""
        self.history   = deque(maxlen=Cfg.VOICE_HISTORY_MAX)

        try:
            import speech_recognition  # noqa: F401
            self.available = True
        except ImportError:
            self.available = False

    def start(self):
        if not self.available or self._running:
            return
        self._running = True
        self.status   = "AVVIO..."
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        self.status   = "OFFLINE"

    def _loop(self):
        import speech_recognition as sr
        rec = sr.Recognizer()
        try:
            with sr.Microphone() as mic:
                rec.adjust_for_ambient_noise(mic, duration=Cfg.VOICE_NOISE_DUR)
        except Exception as e:
            log.error("Microphone init failed: %s", e)
            self.status   = "MIC ERR"
            self._running = False
            return

        self.status = "ASCOLTO"
        while self._running:
            try:
                with sr.Microphone() as mic:
                    audio = rec.listen(
                        mic,
                        timeout=Cfg.VOICE_TIMEOUT,
                        phrase_time_limit=Cfg.VOICE_PHRASE_MAX,
                    )
                self.status = "RICONOSCIMENTO..."
                text = rec.recognize_google(audio, language=Cfg.VOICE_LANG)
                self.last_text = text
                self.history.appendleft(text)
                result = self._parser.parse(text)
                if result:
                    action, args = result
                    if action == 'agent' and self._on_agent:
                        self._on_agent(str(args))
                    else:
                        self._execute(action, args)
                    self._cb(text, action)
                else:
                    self._cb(text, None)
                self.status = "ASCOLTO"
            except sr.WaitTimeoutError:
                self.status = "ASCOLTO"
            except sr.UnknownValueError:
                self.status = "NON CAPITO"
                time.sleep(Cfg.VOICE_UNKNOWN_DELAY)
                self.status = "ASCOLTO"
            except Exception as e:
                self.status = f"ERR: {str(e)[:18]}"
                time.sleep(Cfg.VOICE_ERROR_DELAY)
                self.status = "ASCOLTO"

    def _execute(self, action: str, args):
        run_action(self._mouse, action, args)


# ─────────────────────────────────────────────────────────────
#  AUTONOMOUS AGENT  — AutoGPT-style plan → execute loop
#
#  Given a natural-language goal, an LLM plans a bounded sequence of
#  steps drawn from the same action vocabulary as CommandParser/
#  VoiceController, then the steps run one at a time. Triggered via
#  voice ("agente ...", "obiettivo ...") or the Agente tab.
# ─────────────────────────────────────────────────────────────
class AutonomousAgent:
    _ACTIONS = (
        "open_url(url), open_app(name), search(query), "
        "create_folder(name), hotkey(keys: list of key names), "
        "zoom(direction: 1 or -1), type(text), screenshot(), "
        "wait(seconds)"
    )

    def __init__(self, mouse: SmoothMouse, on_update=None):
        self._mouse   = mouse
        self._cb      = on_update or (lambda: None)
        self._running = False
        self._thread  = None
        self.status   = "IDLE"
        self.goal     = ""
        self.log: list[dict] = []
        self.last_error = ""

        self.api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        try:
            import anthropic  # noqa: F401
            self._has_sdk = True
        except ImportError:
            self._has_sdk = False

    @property
    def available(self) -> bool:
        # re-read key at call time so the user can set it after startup
        self.api_key = os.environ.get("ANTHROPIC_API_KEY", self.api_key)
        return self._has_sdk and bool(self.api_key)

    def run(self, goal: str):
        goal = goal.strip()
        if not goal:
            return
        if self._running:
            self.status = "OCCUPATO"
            self._cb()
            return
        if not self.available:
            self.status = "NON DISPONIBILE"
            self._cb()
            return
        self.goal        = goal
        self.log         = []
        self.last_error  = ""
        self._running    = True
        self._thread     = threading.Thread(target=self._loop, args=(goal,), daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        self.status = "IDLE"

    def _loop(self, goal: str):
        self.status = "PIANIFICO..."
        self._cb()
        plan = self._plan(goal)
        if not plan:
            self.status = "ERRORE PIANO"
            self._running = False
            self._cb()
            return

        total = len(plan)
        for i, step in enumerate(plan):
            if not self._running:
                break
            action, args = step.get("action"), step.get("args")
            self.status = f"STEP {i + 1}/{total}: {action}"
            entry = {"action": action, "args": args, "done": False}
            self.log.append(entry)
            self._cb()
            run_action(self._mouse, action, args)
            entry["done"] = True
            self._cb()
            time.sleep(Cfg.AGENT_STEP_DELAY)

        self.status   = "COMPLETATO" if self._running else "INTERROTTO"
        self._running = False
        self._cb()

    def _plan(self, goal: str) -> list[dict] | None:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key)
            prompt = (
                "Sei un agente che controlla un PC eseguendo una sequenza fissa "
                "di azioni predefinite.\n"
                f"Azioni disponibili: {self._ACTIONS}.\n"
                f"Obiettivo dell'utente: {goal}\n\n"
                f"Rispondi SOLO con un array JSON (max {Cfg.AGENT_MAX_STEPS} elementi) "
                'di step nella forma {"action": "<nome>", "args": <valore o null>}. '
                "Nessun testo prima o dopo il JSON."
            )
            msg = client.messages.create(
                model=Cfg.AGENT_MODEL,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(
                block.text for block in msg.content if getattr(block, "type", "") == "text"
            ).strip()
            text = re.sub(r'^```(?:json)?|```$', '', text, flags=re.MULTILINE).strip()
            plan = json.loads(text)
            if not isinstance(plan, list):
                return None
            return plan[:Cfg.AGENT_MAX_STEPS]
        except Exception as e:
            self.last_error = str(e)[:80]
            return None


# ─────────────────────────────────────────────────────────────
#  GESTURE DATABASE  — persistent storage of user-trained gestures
# ─────────────────────────────────────────────────────────────
class GestureDatabase:
    DB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "user_gestures.json")

    def __init__(self):
        self._d: dict = {}
        self._load()

    # ── public API ────────────────────────────────────────────
    def add_sample(self, name: str, fv: list):
        if name not in self._d:
            self._d[name] = {"samples": [], "action": None, "args": None}
        self._d[name]["samples"].append(fv)

    def set_action(self, name: str, action: str, args=None):
        if name not in self._d:
            self._d[name] = {"samples": [], "action": None, "args": None}
        self._d[name]["action"] = action
        self._d[name]["args"]   = args

    def get_entry(self, name: str) -> dict | None:
        return self._d.get(name)

    def remove(self, name: str):
        self._d.pop(name, None)

    def list_names(self) -> list:
        return sorted(self._d.keys())

    def sample_count(self, name: str) -> int:
        return len(self._d.get(name, {}).get("samples", []))

    def save(self):
        try:
            with open(self.DB_FILE, "w", encoding="utf-8") as f:
                json.dump(self._d, f, indent=2, ensure_ascii=False)
        except Exception as e:
            log.error("GestureDatabase save failed: %s", e)

    def _load(self):
        try:
            with open(self.DB_FILE, encoding="utf-8") as f:
                self._d = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self._d = {}


# ─────────────────────────────────────────────────────────────
#  GESTURE RECORDER  — collects landmark samples for learning
# ─────────────────────────────────────────────────────────────
class GestureRecorder:
    RECORD_FRAMES  = 30
    COUNTDOWN_SECS = 3

    IDLE        = "IDLE"
    COUNTDOWN   = "COUNTDOWN"
    RECORDING   = "RECORDING"
    DONE        = "DONE"

    def __init__(self, db: GestureDatabase, recogniser):
        self._db         = db
        self._rec        = recogniser
        self._state      = self.IDLE
        self._name       = ""
        self._samples    = []
        self._start_time = 0.0
        self._on_done    = None

    def start(self, name: str, on_done=None):
        self._name       = name.strip()
        self._samples    = []
        self._state      = self.COUNTDOWN
        self._start_time = time.time()
        self._on_done    = on_done

    def feed_frame(self, lm: list):
        if self._state == self.COUNTDOWN:
            if time.time() - self._start_time >= self.COUNTDOWN_SECS:
                self._state      = self.RECORDING
                self._start_time = time.time()
        elif self._state == self.RECORDING:
            fv = self._rec.feature_vector(lm)
            self._samples.append(fv)
            if len(self._samples) >= self.RECORD_FRAMES:
                self._finish()

    def _finish(self):
        for fv in self._samples:
            self._db.add_sample(self._name, fv)
        self._db.save()
        self._state = self.DONE
        if self._on_done:
            self._on_done(self._name)

    def cancel(self):
        self._state = self.IDLE

    @property
    def state(self) -> str:
        return self._state

    @property
    def progress(self) -> float:
        if self._state == self.RECORDING:
            return min(1.0, len(self._samples) / self.RECORD_FRAMES)
        if self._state == self.COUNTDOWN:
            return 0.0
        if self._state == self.DONE:
            return 1.0
        return 0.0

    @property
    def countdown_remaining(self) -> int:
        if self._state != self.COUNTDOWN:
            return 0
        return max(0, self.COUNTDOWN_SECS - int(time.time() - self._start_time))


# ─────────────────────────────────────────────────────────────
#  CUSTOM GESTURE RECOGNISER  — k-NN over user-trained samples
#  Falls back to hardcoded rules when no custom match found
# ─────────────────────────────────────────────────────────────
class CustomGestureRecogniser(GestureRecogniser):
    K           = 3
    CONF_THRESH = 0.12   # max L2 distance to accept a custom match

    def __init__(self, db: GestureDatabase):
        super().__init__()
        self._db = db

    def feature_vector(self, lm: list) -> list:
        H    = HandTracker
        f    = [float(b) for b in self.fingers_up(lm)]
        tips = [H.THUMB_TIP, H.INDEX_TIP, H.MIDDLE_TIP, H.RING_TIP, H.PINKY_TIP]
        inter = [self.pinch(lm, tips[i], tips[j])
                 for i in range(5) for j in range(i + 1, 5)]
        wrist = [self.pinch(lm, H.WRIST, t) for t in tips]
        return [*f, *inter, *wrist]   # 20 dimensions

    def classify(self, lm) -> str:
        if lm is None:
            return G.NONE
        fv   = self.feature_vector(lm)
        name = self._knn(fv)
        return name if name else super().classify(lm)

    def _knn(self, fv: list) -> str | None:
        candidates: list[tuple[float, str]] = []
        for name, entry in self._db._d.items():
            for sample in entry.get("samples", []):
                d_sq = sum((a - b) ** 2 for a, b in zip(fv, sample))
                if d_sq < self.CONF_THRESH ** 2:
                    candidates.append((d_sq, name))
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0])
        k_nearest = candidates[: self.K]
        best_name, _ = Counter(name for _, name in k_nearest).most_common(1)[0]
        return best_name


# ─────────────────────────────────────────────────────────────
#  DASHBOARD
# ─────────────────────────────────────────────────────────────
class Dashboard(tk.Tk):
    def __init__(self):
        super().__init__()
        self.mouse       = SmoothMouse()
        self.hand_active = True           # attivo di default all'avvio
        self._vk         = None
        self._cam        = None
        self._db         = GestureDatabase()
        self._recogniser = CustomGestureRecogniser(self._db)
        self._recorder   = GestureRecorder(self._db, self._recogniser)
        self._agent      = AutonomousAgent(self.mouse, self._on_agent_update)
        self._voice      = VoiceController(self.mouse, self._on_voice_cmd,
                                            on_agent_goal=self._agent.run)

        self._setup_window()
        self._build_ui()
        self._start_camera()
        self._loop()

    def _setup_window(self):
        self.title("Hand Gesture Control  ✋  v2")
        self.configure(bg=Cfg.BG_DARK)
        self.geometry("1160x800")
        self.minsize(860, 620)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):
        # Header
        hdr = tk.Frame(self, bg=Cfg.BG_DARK)
        hdr.pack(fill="x", padx=12, pady=(10, 4))
        tk.Label(hdr, text="✋  Hand Gesture Control  v2",
                 font=("Segoe UI", 19, "bold"),
                 bg=Cfg.BG_DARK, fg=Cfg.TEXT).pack(side="left")
        self._status_lbl = tk.Label(hdr, text="●  ATTIVO",
                 font=("Segoe UI", 11), bg=Cfg.BG_DARK, fg=Cfg.SUCCESS)
        self._status_lbl.pack(side="right", padx=16)
        self._hands_lbl = tk.Label(hdr, text="",
                 font=("Segoe UI", 10), bg=Cfg.BG_DARK, fg=Cfg.TEXT_DIM)
        self._hands_lbl.pack(side="right", padx=4)
        self._voice_hdr_lbl = tk.Label(hdr, text="🎤 OFFLINE",
                 font=("Segoe UI", 11), bg=Cfg.BG_DARK, fg="#ff4444")
        self._voice_hdr_lbl.pack(side="right", padx=16)

        tk.Frame(self, bg=Cfg.BG_CARD, height=1).pack(fill="x", padx=12)

        body = tk.Frame(self, bg=Cfg.BG_DARK)
        body.pack(fill="both", expand=True, padx=12, pady=8)

        # Camera panel (expands with window)
        self._left_panel = tk.Frame(body, bg=Cfg.BG_MID)
        self._left_panel.pack(side="left", fill="both", expand=True, padx=(0, 6))
        tk.Label(self._left_panel, text="CAMERA FEED",
                 font=("Segoe UI", 8, "bold"),
                 bg=Cfg.BG_MID, fg=Cfg.TEXT_DIM).pack(pady=(8, 2))
        self._cam_lbl = tk.Label(self._left_panel, bg="#000000")
        self._cam_lbl.pack(padx=8, pady=(0, 8), fill="both", expand=True)

        # Control panel (right, fixed 340px, tabbed)
        right = tk.Frame(body, bg=Cfg.BG_DARK, width=340)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        style = ttk.Style(self)
        style.theme_use("default")
        style.configure("Dark.TNotebook",
                        background=Cfg.BG_DARK, borderwidth=0, tabmargins=[2, 2, 2, 0])
        style.configure("Dark.TNotebook.Tab",
                        background=Cfg.BG_CARD, foreground=Cfg.TEXT_DIM,
                        font=("Segoe UI", 8, "bold"), padding=[10, 5])
        style.map("Dark.TNotebook.Tab",
                  background=[("selected", Cfg.BG_MID)],
                  foreground=[("selected", Cfg.TEXT)])

        nb = ttk.Notebook(right, style="Dark.TNotebook")
        nb.pack(fill="both", expand=True, padx=4, pady=4)

        tab_hands = tk.Frame(nb, bg=Cfg.BG_DARK)
        nb.add(tab_hands, text=" Mani ")

        tab_voice = tk.Frame(nb, bg=Cfg.BG_DARK)
        nb.add(tab_voice, text=" Voce ")

        tab_agent = tk.Frame(nb, bg=Cfg.BG_DARK)
        nb.add(tab_agent, text=" Agente ")

        tab_guide = tk.Frame(nb, bg=Cfg.BG_MID)
        nb.add(tab_guide, text=" Guida ")

        tab_learn = tk.Frame(nb, bg=Cfg.BG_DARK)
        nb.add(tab_learn, text=" Apprendi ")

        self._build_hands_tab(tab_hands)
        self._build_voice_tab(tab_voice)
        self._build_agent_tab(tab_agent)
        self._build_guide_tab(tab_guide)
        self._build_learn_tab(tab_learn)

    def _card(self, parent, title):
        f = tk.Frame(parent, bg=Cfg.BG_CARD)
        f.pack(fill="x", padx=4, pady=4)
        tk.Label(f, text=title, font=("Segoe UI", 9, "bold"),
                 bg=Cfg.BG_CARD, fg=Cfg.TEXT_DIM).pack(pady=(10, 4))
        return f

    def _build_hands_tab(self, parent):
        c1 = self._card(parent, "CONTROLLO MANI")
        tk.Label(c1, text="Riconosce entrambe le mani:\ndestra = cursore  |  sinistra = modificatore",
                 font=("Segoe UI", 8), bg=Cfg.BG_CARD,
                 fg=Cfg.TEXT_DIM, justify="center").pack()
        self._hand_btn = tk.Button(
            c1, text="⏹  DISATTIVA",   # attivo di default
            font=("Segoe UI", 11, "bold"), bg=Cfg.ACCENT, fg=Cfg.TEXT,
            relief="flat", bd=0, cursor="hand2", padx=16, pady=8,
            command=self._toggle_hand)
        self._hand_btn.pack(pady=10, ipadx=8)

        c2 = self._card(parent, "TASTIERA VIRTUALE")
        tk.Label(c2, text="Tastiera on-screen per scrivere\ncon i gesti",
                 font=("Segoe UI", 8), bg=Cfg.BG_CARD,
                 fg=Cfg.TEXT_DIM, justify="center").pack()
        self._vk_btn = tk.Button(
            c2, text="⌨  MOSTRA TASTIERA",
            font=("Segoe UI", 10, "bold"), bg=Cfg.BLUE, fg="#000000",
            relief="flat", bd=0, cursor="hand2", padx=16, pady=8,
            command=self._toggle_vk)
        self._vk_btn.pack(pady=10, ipadx=8)

        c3 = self._card(parent, "GESTI RILEVATI")
        row = tk.Frame(c3, bg=Cfg.BG_CARD)
        row.pack(fill="x", padx=10, pady=(2, 0))
        tk.Label(row, text="DOM:", font=("Segoe UI", 9),
                 bg=Cfg.BG_CARD, fg=Cfg.TEXT_DIM, width=5, anchor="w").pack(side="left")
        self._dom_lbl = tk.Label(row, text="—",
                 font=("Segoe UI", 13, "bold"), bg=Cfg.BG_CARD, fg=Cfg.ACCENT)
        self._dom_lbl.pack(side="left")
        row2 = tk.Frame(c3, bg=Cfg.BG_CARD)
        row2.pack(fill="x", padx=10, pady=(2, 4))
        tk.Label(row2, text="MOD:", font=("Segoe UI", 9),
                 bg=Cfg.BG_CARD, fg=Cfg.TEXT_DIM, width=5, anchor="w").pack(side="left")
        self._mod_lbl = tk.Label(row2, text="—",
                 font=("Segoe UI", 13, "bold"), bg=Cfg.BG_CARD, fg=Cfg.PURPLE)
        self._mod_lbl.pack(side="left")
        self._act_lbl = tk.Label(c3, text="",
                 font=("Segoe UI", 9), bg=Cfg.BG_CARD, fg=Cfg.WARNING)
        self._act_lbl.pack(pady=(0, 8))

        c4 = self._card(parent, "SENSIBILITÀ CURSORE")
        self._smooth_var = tk.DoubleVar(value=Cfg.SMOOTH)
        ttk.Scale(c4, from_=0.05, to=1.0, orient="horizontal",
                  variable=self._smooth_var,
                  command=lambda v: setattr(Cfg, "SMOOTH", float(v))
                  ).pack(fill="x", padx=16, pady=(4, 10))

    def _build_voice_tab(self, parent):
        c1 = self._card(parent, "CONTROLLO VOCALE")

        srow = tk.Frame(c1, bg=Cfg.BG_CARD)
        srow.pack(fill="x", padx=10, pady=(0, 4))
        tk.Label(srow, text="Status:", font=("Segoe UI", 9),
                 bg=Cfg.BG_CARD, fg=Cfg.TEXT_DIM).pack(side="left")
        self._voice_dot = tk.Label(srow, text="●", font=("Segoe UI", 9),
                 bg=Cfg.BG_CARD, fg="#ff4444")
        self._voice_dot.pack(side="left", padx=4)
        self._voice_status_lbl = tk.Label(srow, text="OFFLINE",
                 font=("Segoe UI", 9, "bold"), bg=Cfg.BG_CARD, fg=Cfg.TEXT_DIM)
        self._voice_status_lbl.pack(side="left")

        self._voice_btn = tk.Button(
            c1, text="▶  ATTIVA VOCE",
            font=("Segoe UI", 10, "bold"),
            bg=Cfg.SUCCESS if self._voice.available else Cfg.TEXT_DIM,
            fg="#000000",
            relief="flat", bd=0,
            cursor="hand2" if self._voice.available else "arrow",
            padx=16, pady=8,
            command=self._toggle_voice,
            state="normal" if self._voice.available else "disabled")
        self._voice_btn.pack(pady=8, ipadx=8)

        if not self._voice.available:
            tk.Label(c1,
                     text="SpeechRecognition non trovato.\nEsegui: python install.py",
                     font=("Segoe UI", 8), bg=Cfg.BG_CARD, fg=Cfg.WARNING,
                     justify="center").pack(pady=(0, 8))
        else:
            tk.Label(c1,
                     text="Lingua: Italiano (it-IT)\nParla chiaramente al microfono",
                     font=("Segoe UI", 8), bg=Cfg.BG_CARD, fg=Cfg.TEXT_DIM,
                     justify="center").pack(pady=(0, 8))

        c2 = self._card(parent, "ULTIMO RICONOSCIUTO")
        self._voice_last_lbl = tk.Label(
            c2, text="—",
            font=("Segoe UI", 9, "italic"), bg=Cfg.BG_CARD, fg=Cfg.TEXT,
            wraplength=300, justify="center")
        self._voice_last_lbl.pack(padx=8, pady=(0, 10))

        c3 = self._card(parent, "CRONOLOGIA (ultimi 6)")
        self._voice_log_labels = []
        for _ in range(6):
            lbl = tk.Label(c3, text="", font=("Consolas", 8),
                           bg=Cfg.BG_CARD, fg=Cfg.TEXT_DIM, anchor="w")
            lbl.pack(fill="x", padx=10, pady=1)
            self._voice_log_labels.append(lbl)
        tk.Label(c3, text="", bg=Cfg.BG_CARD).pack(pady=2)

        c4 = self._card(parent, "ESEMPI COMANDI VOCALI")
        for ex in ("apri youtube", "cerca meteo Milano",
                   "crea cartella lavoro", "copia / incolla / salva",
                   "screenshot", "zoom avanti / zoom indietro",
                   "scrivi ciao mondo", "volume su / muto"):
            tk.Label(c4, text=f"  • {ex}", font=("Segoe UI", 8),
                     bg=Cfg.BG_CARD, fg=Cfg.TEXT_DIM, anchor="w"
                     ).pack(fill="x", padx=4)
        tk.Label(c4, text="", bg=Cfg.BG_CARD).pack(pady=3)

    def _build_agent_tab(self, parent):
        c1 = self._card(parent, "AGENTE AUTONOMO")

        srow = tk.Frame(c1, bg=Cfg.BG_CARD)
        srow.pack(fill="x", padx=10, pady=(0, 4))
        tk.Label(srow, text="Status:", font=("Segoe UI", 9),
                 bg=Cfg.BG_CARD, fg=Cfg.TEXT_DIM).pack(side="left")
        self._agent_dot = tk.Label(srow, text="●", font=("Segoe UI", 9),
                 bg=Cfg.BG_CARD, fg="#ff4444")
        self._agent_dot.pack(side="left", padx=4)
        self._agent_status_lbl = tk.Label(srow, text="IDLE",
                 font=("Segoe UI", 9, "bold"), bg=Cfg.BG_CARD, fg=Cfg.TEXT_DIM)
        self._agent_status_lbl.pack(side="left")

        if not self._agent.available:
            hint = ("Pacchetto 'anthropic' non trovato.\nEsegui: pip install anthropic"
                     if not self._agent._has_sdk else
                     "Imposta la variabile ANTHROPIC_API_KEY\nper abilitare l'agente.")
            tk.Label(c1, text=hint, font=("Segoe UI", 8), bg=Cfg.BG_CARD,
                     fg=Cfg.WARNING, justify="center").pack(pady=(0, 8))
        else:
            tk.Label(c1,
                     text="Pianifica ed esegue obiettivi multi-step\ncon un set di azioni predefinite",
                     font=("Segoe UI", 8), bg=Cfg.BG_CARD, fg=Cfg.TEXT_DIM,
                     justify="center").pack(pady=(0, 8))

        c2 = self._card(parent, "OBIETTIVO")
        self._agent_goal_var = tk.StringVar()
        entry = tk.Entry(c2, textvariable=self._agent_goal_var,
                          font=("Segoe UI", 9), bg=Cfg.BG_MID, fg=Cfg.TEXT,
                          insertbackground=Cfg.TEXT, relief="flat")
        entry.pack(fill="x", padx=10, pady=(0, 6), ipady=4)
        entry.bind("<Return>", lambda e: self._start_agent())

        self._agent_run_btn = tk.Button(
            c2, text="▶  ESEGUI",
            font=("Segoe UI", 10, "bold"),
            bg=Cfg.SUCCESS if self._agent.available else Cfg.TEXT_DIM,
            fg="#000000",
            relief="flat", bd=0,
            cursor="hand2" if self._agent.available else "arrow",
            padx=16, pady=6,
            command=self._start_agent,
            state="normal" if self._agent.available else "disabled")
        self._agent_run_btn.pack(pady=(0, 8), ipadx=8)

        c3 = self._card(parent, "STEP PIANIFICATI")
        self._agent_log_lbl = tk.Label(
            c3, text="—", font=("Consolas", 8), bg=Cfg.BG_CARD, fg=Cfg.TEXT_DIM,
            justify="left", anchor="w", wraplength=300)
        self._agent_log_lbl.pack(fill="x", padx=10, pady=(0, 10))

        c4 = self._card(parent, "ESEMPI OBIETTIVI VOCALI")
        for ex in ("agente apri youtube e cerca meteo Milano",
                   "obiettivo crea una cartella progetti e apri il browser"):
            tk.Label(c4, text=f"  • {ex}", font=("Segoe UI", 8),
                     bg=Cfg.BG_CARD, fg=Cfg.TEXT_DIM, anchor="w",
                     wraplength=300, justify="left").pack(fill="x", padx=4, pady=2)
        tk.Label(c4, text="", bg=Cfg.BG_CARD).pack(pady=3)

    def _build_guide_tab(self, parent):
        tk.Label(parent, text="GUIDA GESTI",
                 font=("Segoe UI", 9, "bold"),
                 bg=Cfg.BG_MID, fg=Cfg.TEXT_DIM).pack(pady=(8, 4))

        DOM_COL, MOD_COL = Cfg.ACCENT, Cfg.PURPLE
        GUIDE = [
            ("☝ Indice solo",         "Cursore mouse",       DOM_COL),
            ("🤏 Pinch i+p",          "Click sx / drag",     DOM_COL),
            ("🤏 Pinch m+p",          "Click dx",            DOM_COL),
            ("✌ Due dita",            "Scroll verticale",    DOM_COL),
            ("🤌 3 dita su",          "Copia  Ctrl+C",       DOM_COL),
            ("✋ 4 dita su",          "Incolla  Ctrl+V",     DOM_COL),
            ("👍 Solo pollice",       "Doppio click",        DOM_COL),
            ("🤘 Rock",               "Annulla  Ctrl+Z",     DOM_COL),
            ("🤙 Pollice+mignolo",    "Salva  Ctrl+S",       DOM_COL),
            ("🖐 Palmo+velocità",     "Swipe ←→  Alt+←/→",  DOM_COL),
            ("──── Mano sinistra ────","(modificatore)",      MOD_COL),
            ("🖐 L: palmo",           "Congela cursore",     MOD_COL),
            ("✊ L: pugno",           "Scroll = zoom",       MOD_COL),
            ("✌ L: due dita",        "Cursore = h-scroll",  MOD_COL),
            ("🤘 L: rock",            "Alt+Tab",             MOD_COL),
            ("👍 L: pollice",         "Arma click centrale", MOD_COL),
            ("🤏+🤏 entrambi pinch", "Zoom in/out",         Cfg.SUCCESS),
        ]
        for g, a, col in GUIDE:
            r = tk.Frame(parent, bg=Cfg.BG_MID)
            r.pack(fill="x", padx=8, pady=1)
            tk.Label(r, text=g, font=("Segoe UI", 8),
                     bg=Cfg.BG_MID, fg=col, width=22, anchor="w").pack(side="left")
            tk.Label(r, text=a, font=("Segoe UI", 8),
                     bg=Cfg.BG_MID, fg=Cfg.TEXT_DIM, anchor="w").pack(side="left")

    def _build_learn_tab(self, parent):
        _ACTIONS = [
            ("hotkey",        "Tasto rapido (es. ctrl+c)"),
            ("open_url",      "Apri URL"),
            ("search",        "Cerca su Google"),
            ("type",          "Digita testo"),
            ("screenshot",    "Screenshot"),
            ("zoom",          "Zoom (+1 o -1)"),
            ("create_folder", "Crea cartella"),
        ]

        # ── registra gesto ────────────────────────────────────
        c1 = self._card(parent, "REGISTRA NUOVO GESTO")
        row_n = tk.Frame(c1, bg=Cfg.BG_CARD)
        row_n.pack(fill="x", padx=10, pady=2)
        tk.Label(row_n, text="Nome:", font=("Segoe UI", 9),
                 bg=Cfg.BG_CARD, fg=Cfg.TEXT_DIM, width=7, anchor="w").pack(side="left")
        self._learn_name_var = tk.StringVar()
        tk.Entry(row_n, textvariable=self._learn_name_var,
                 font=("Segoe UI", 9), bg=Cfg.BG_MID, fg=Cfg.TEXT,
                 insertbackground=Cfg.TEXT, relief="flat", width=14
                 ).pack(side="left", padx=4)

        row_a = tk.Frame(c1, bg=Cfg.BG_CARD)
        row_a.pack(fill="x", padx=10, pady=2)
        tk.Label(row_a, text="Azione:", font=("Segoe UI", 9),
                 bg=Cfg.BG_CARD, fg=Cfg.TEXT_DIM, width=7, anchor="w").pack(side="left")
        self._learn_action_var = tk.StringVar(value=_ACTIONS[0][0])
        opt = tk.OptionMenu(row_a, self._learn_action_var,
                            *[a[0] for a in _ACTIONS])
        opt.configure(bg=Cfg.BG_MID, fg=Cfg.TEXT, font=("Segoe UI", 8),
                      activebackground=Cfg.BG_CARD, relief="flat", bd=0)
        opt["menu"].configure(bg=Cfg.BG_MID, fg=Cfg.TEXT)
        opt.pack(side="left", padx=4)

        row_v = tk.Frame(c1, bg=Cfg.BG_CARD)
        row_v.pack(fill="x", padx=10, pady=2)
        tk.Label(row_v, text="Arg:", font=("Segoe UI", 9),
                 bg=Cfg.BG_CARD, fg=Cfg.TEXT_DIM, width=7, anchor="w").pack(side="left")
        self._learn_arg_var = tk.StringVar()
        tk.Entry(row_v, textvariable=self._learn_arg_var,
                 font=("Segoe UI", 9), bg=Cfg.BG_MID, fg=Cfg.TEXT,
                 insertbackground=Cfg.TEXT, relief="flat", width=14
                 ).pack(side="left", padx=4)

        self._learn_rec_btn = tk.Button(
            c1, text="● REGISTRA  (3 sec countdown)",
            font=("Segoe UI", 9, "bold"), bg=Cfg.ACCENT, fg=Cfg.TEXT,
            relief="flat", bd=0, cursor="hand2", pady=6,
            command=self._start_recording)
        self._learn_rec_btn.pack(pady=8, fill="x", padx=12)

        # progress bar (canvas-based)
        self._learn_prog_frame = tk.Frame(c1, bg=Cfg.BG_CARD)
        self._learn_prog_frame.pack(fill="x", padx=12, pady=(0, 6))
        self._learn_prog_lbl = tk.Label(
            self._learn_prog_frame, text="",
            font=("Segoe UI", 8), bg=Cfg.BG_CARD, fg=Cfg.TEXT_DIM)
        self._learn_prog_lbl.pack(side="right")
        self._learn_prog_bar = tk.Canvas(
            self._learn_prog_frame, height=6, bg=Cfg.BG_MID,
            highlightthickness=0)
        self._learn_prog_bar.pack(fill="x", side="left", expand=True, padx=(0, 6))

        # ── lista gesti appresi ───────────────────────────────
        c2 = self._card(parent, "GESTI APPRESI")
        self._learn_list_frame = tk.Frame(c2, bg=Cfg.BG_CARD)
        self._learn_list_frame.pack(fill="x", padx=8, pady=4)
        self._learn_listbox = tk.Listbox(
            self._learn_list_frame, bg=Cfg.BG_MID, fg=Cfg.TEXT,
            font=("Consolas", 8), relief="flat", selectbackground=Cfg.ACCENT,
            selectforeground=Cfg.TEXT, height=6, bd=0)
        self._learn_listbox.pack(fill="x")
        tk.Button(
            c2, text="🗑  Elimina selezionato",
            font=("Segoe UI", 8), bg=Cfg.BG_MID, fg=Cfg.ACCENT,
            relief="flat", bd=0, cursor="hand2",
            command=self._delete_gesture
        ).pack(pady=(2, 8))
        self._refresh_gesture_list()

    def _start_recording(self):
        name = self._learn_name_var.get().strip()
        if not name:
            self._learn_prog_lbl.configure(text="⚠ inserisci un nome", fg=Cfg.WARNING)
            return
        action = self._learn_action_var.get()
        arg    = self._learn_arg_var.get().strip() or None
        self._db.set_action(name, action, arg)
        self._learn_rec_btn.configure(state="disabled", bg=Cfg.TEXT_DIM)
        self._recorder.start(name, on_done=self._on_recording_done)
        self._poll_recording()

    def _poll_recording(self):
        state = self._recorder.state
        if state == GestureRecorder.COUNTDOWN:
            sec = self._recorder.countdown_remaining
            self._learn_prog_lbl.configure(
                text=f"⏱ {sec}s", fg=Cfg.WARNING)
            self._draw_progress_bar(0.0)
            self.after(100, self._poll_recording)
        elif state == GestureRecorder.RECORDING:
            p = self._recorder.progress
            n = int(p * GestureRecorder.RECORD_FRAMES)
            self._learn_prog_lbl.configure(
                text=f"{n}/{GestureRecorder.RECORD_FRAMES}", fg=Cfg.SUCCESS)
            self._draw_progress_bar(p)
            self.after(100, self._poll_recording)
        # DONE or IDLE → handled by on_done callback

    def _draw_progress_bar(self, frac: float):
        bar = self._learn_prog_bar
        bar.update_idletasks()
        w = bar.winfo_width()
        bar.delete("all")
        bar.create_rectangle(0, 0, int(w * frac), 6,
                             fill=Cfg.SUCCESS, outline="")

    def _on_recording_done(self, name: str):
        self.after(0, self._recording_done_ui, name)

    def _recording_done_ui(self, name: str):
        n = self._db.sample_count(name)
        self._learn_prog_lbl.configure(
            text=f"✓ {name}: {n} campioni", fg=Cfg.SUCCESS)
        self._draw_progress_bar(1.0)
        self._learn_rec_btn.configure(state="normal", bg=Cfg.ACCENT)
        self._recorder.cancel()
        self._refresh_gesture_list()

    def _delete_gesture(self):
        sel = self._learn_listbox.curselection()
        if not sel:
            return
        name = self._learn_listbox.get(sel[0]).split("  →")[0].strip()
        self._db.remove(name)
        self._db.save()
        self._refresh_gesture_list()

    def _refresh_gesture_list(self):
        self._learn_listbox.delete(0, "end")
        for name in self._db.list_names():
            entry  = self._db.get_entry(name)
            action = entry.get("action") or "—"
            arg    = entry.get("args") or ""
            n      = self._db.sample_count(name)
            label  = f"{name}  →  {action} {arg}  [{n}]"
            self._learn_listbox.insert("end", label)

    # ── toggles ───────────────────────────────────────────────
    def _toggle_hand(self):
        self.hand_active = not self.hand_active
        if self.hand_active:
            self._hand_btn.configure(text="⏹  DISATTIVA", bg=Cfg.ACCENT, fg=Cfg.TEXT)
            self._status_lbl.configure(text="●  ATTIVO", fg=Cfg.SUCCESS)
        else:
            self._hand_btn.configure(text="▶  ATTIVA", bg=Cfg.SUCCESS, fg="#000000")
            self._status_lbl.configure(text="●  OFFLINE", fg="#ff4444")
            self._dom_lbl.configure(text="—")
            self._mod_lbl.configure(text="—")
            self._act_lbl.configure(text="")
            self._hands_lbl.configure(text="")
            self.mouse.stop_drag()

    def _toggle_vk(self):
        if self._vk is None or not self._vk.winfo_exists():
            self._vk = VirtualKeyboard(self, self.mouse)
            self._vk_btn.configure(text="⌨  NASCONDI", bg=Cfg.ACCENT, fg=Cfg.TEXT)
        else:
            if self._vk.state() == "normal":
                self._vk.withdraw()
                self._vk_btn.configure(text="⌨  MOSTRA TASTIERA", bg=Cfg.BLUE, fg="#000000")
            else:
                self._vk.deiconify()
                self._vk_btn.configure(text="⌨  NASCONDI", bg=Cfg.ACCENT, fg=Cfg.TEXT)

    def _toggle_voice(self):
        if not self._voice.available:
            return
        if self._voice._running:
            self._voice.stop()
            self._voice_btn.configure(text="▶  ATTIVA VOCE", bg=Cfg.SUCCESS, fg="#000000")
        else:
            self._voice.start()
            self._voice_btn.configure(text="⏹  DISATTIVA VOCE", bg=Cfg.ACCENT, fg=Cfg.TEXT)

    def _on_voice_cmd(self, text: str, action):
        self.after(0, self._update_voice_ui, text, action)

    def _update_voice_ui(self, text: str, action):
        icon = "✓" if action else "✗"
        self._voice_last_lbl.configure(text=f'{icon} "{text}"')
        texts = list(self._voice.history)
        for i, lbl in enumerate(self._voice_log_labels):
            lbl.configure(text=f"  {texts[i]}" if i < len(texts) else "")

    def _start_agent(self):
        goal = self._agent_goal_var.get().strip()
        if goal:
            self._agent.run(goal)

    def _on_agent_update(self):
        self.after(0, self._update_agent_ui)

    def _update_agent_ui(self):
        lines = []
        for step in self._agent.log[-8:]:
            mark = "✓" if step["done"] else "…"
            lines.append(f"{mark} {step['action']}({step['args']})")
        if not lines and self._agent.last_error:
            lines = [f"✗ {self._agent.last_error}"]
        self._agent_log_lbl.configure(text="\n".join(lines) if lines else "—")

    # ── main loop ─────────────────────────────────────────────
    def _start_camera(self):
        self._cam = CameraThread(self)
        self._cam.start_capture()

    def _loop(self):
        try:
            if self._cam:
                frame = self._cam.get_frame()
                if frame is not None:
                    h, w  = frame.shape[:2]
                    avail = self._left_panel.winfo_width()
                    dw    = max(300, avail - 16) if avail > 10 else 630
                    dh    = int(h * dw / w)
                    small = cv2.resize(frame, (dw, dh))
                    photo = ImageTk.PhotoImage(
                        Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))
                    self._cam_lbl.configure(image=photo)
                    self._cam_lbl.image = photo

                if self.hand_active:
                    self._dom_lbl.configure(text=self._cam.dom_g or "—")
                    self._mod_lbl.configure(text=self._cam.mod_g or "—")
                    self._act_lbl.configure(text=self._cam.action)
                    n = self._cam.n_hands
                    self._hands_lbl.configure(
                        text=(f"{'✋' * n}  {n} man{'i' if n != 1 else 'o'}"
                              if n else ""))
        except Exception as e:
            log.debug("UI camera update error: %s", e)

        try:
            # Voice status polling
            status = self._voice.status
            _COL = {
                "ASCOLTO":           Cfg.SUCCESS,
                "RICONOSCIMENTO...": Cfg.WARNING,
                "AVVIO...":          Cfg.BLUE,
                "NON CAPITO":        Cfg.WARNING,
                "OFFLINE":           "#ff4444",
                "MIC ERR":           Cfg.ACCENT,
            }
            col = _COL.get(status, Cfg.TEXT_DIM)
            self._voice_dot.configure(fg=col)
            self._voice_status_lbl.configure(text=status, fg=col)
            self._voice_hdr_lbl.configure(text=f"🎤 {status}", fg=col)
        except Exception as e:
            log.debug("UI voice update error: %s", e)

        try:
            # Agent status polling
            status = self._agent.status
            if status.startswith("STEP"):
                col = Cfg.BLUE
            elif status == "COMPLETATO":
                col = Cfg.SUCCESS
            elif status in ("ERRORE PIANO", "INTERROTTO"):
                col = Cfg.ACCENT
            elif status == "PIANIFICO...":
                col = Cfg.WARNING
            else:
                col = Cfg.TEXT_DIM
            self._agent_dot.configure(fg=col)
            self._agent_status_lbl.configure(text=status, fg=col)
        except Exception as e:
            log.debug("UI agent update error: %s", e)

        self.after(Cfg.DWELL_MS, self._loop)

    def _on_close(self):
        if self._cam:
            self._cam.stop_capture()
            self._cam.join(timeout=2.0)
        self._voice.stop()
        self._agent.stop()
        self._db.save()
        self.quit()
        self.destroy()


# ─────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = Dashboard()
    app.mainloop()
