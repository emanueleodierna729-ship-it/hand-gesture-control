#!/usr/bin/env python3
"""
Hand Gesture Control System  v2
Dual-hand support · Landmark EMA smoothing · Temporal gesture stabilisation
"""

from __future__ import annotations

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
from collections import deque, Counter
from PIL import Image, ImageTk

try:
    from pynput.mouse import Button, Controller as _MouseCtrl
    from pynput.keyboard import Controller as _KeyboardCtrl
    PYNPUT_OK = True
except ImportError:
    PYNPUT_OK = False


# ─────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────
class Cfg:
    CAMERA_IDX   = 0
    CAM_W        = 640
    CAM_H        = 480
    MAX_HANDS    = 2              # now supports two hands
    DETECT_CONF  = 0.65
    TRACK_CONF   = 0.60

    # Mouse
    SMOOTH       = 0.35           # EMA alpha for cursor position
    PINCH_THRESH = 0.042
    DRAG_THRESH  = 0.032
    SCROLL_SENS  = 18
    CLICK_CD     = 0.30
    SHORTCUT_CD  = 0.70
    DWELL_MS     = 16

    # One Euro Filter (state-of-the-art adaptive smoothing)
    OEF_MIN_CUTOFF = 1.5          # min cutoff freq — lower = smoother
    OEF_BETA       = 0.007        # speed coefficient — higher = more responsive
    OEF_D_CUTOFF   = 1.0          # derivative cutoff

    # Landmark smoothing
    LMARK_ALPHA  = 0.40           # fallback EMA alpha

    # Gesture stabilisation with hysteresis
    GEST_WIN     = 5              # frames in voting window
    GEST_THRESH  = 0.60           # fraction needed to ENTER a new gesture
    GEST_EXIT    = 0.40           # fraction needed to LEAVE current gesture

    # Swipe detection
    SWIPE_VEL    = 0.65

    # Zoom
    ZOOM_DEAD    = 0.018
    ZOOM_CD      = 0.12

    # Dead zone — cursor jitter elimination
    DEAD_ZONE    = 0.004          # normalised units — movements below this are ignored

    SCR_W, SCR_H = pyautogui.size()
    MARGIN_X     = 0.12
    MARGIN_Y     = 0.12

    # Voice control
    VOICE_LANG       = "it-IT"
    VOICE_TIMEOUT    = 3
    VOICE_PHRASE_MAX = 6

    # Tkinter palette — identità propria
    BG_DARK  = "#0a0e1a"
    BG_MID   = "#121828"
    BG_CARD  = "#1a2036"
    ACCENT   = "#7c5cfc"
    SUCCESS  = "#00d4aa"
    WARNING  = "#ffb347"
    TEXT     = "#e8ecf4"
    TEXT_DIM = "#5c6488"
    BLUE     = "#4da6ff"
    PURPLE   = "#a78bfa"
    BORDER   = "#252d48"


# ─────────────────────────────────────────────────────────────
#  ONE EURO FILTER — state-of-the-art adaptive smoothing
#  Casiez, Roussel, Vogel — CHI 2012
#  Used by Meta Quest, Apple Vision Pro, SteamVR
#  Adapts: slow = heavy smooth (no jitter), fast = light (no lag)
# ─────────────────────────────────────────────────────────────
class OneEuroFilter:
    def __init__(self, min_cutoff: float = Cfg.OEF_MIN_CUTOFF,
                 beta: float = Cfg.OEF_BETA,
                 d_cutoff: float = Cfg.OEF_D_CUTOFF):
        self._min_cutoff = min_cutoff
        self._beta       = beta
        self._d_cutoff   = d_cutoff
        self._x_prev     = None
        self._dx_prev    = 0.0
        self._t_prev     = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / max(dt, 1e-6))

    def __call__(self, x: float, t: float | None = None) -> float:
        if t is None:
            t = time.perf_counter()
        if self._t_prev is None:
            self._x_prev = x
            self._dx_prev = 0.0
            self._t_prev = t
            return x

        dt = max(t - self._t_prev, 1e-6)
        self._t_prev = t

        a_d = self._alpha(self._d_cutoff, dt)
        dx = (x - self._x_prev) / dt
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev
        self._dx_prev = dx_hat

        cutoff = self._min_cutoff + self._beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self._x_prev
        self._x_prev = x_hat
        return x_hat

    def reset(self):
        self._x_prev = None
        self._dx_prev = 0.0
        self._t_prev = None


class LandmarkSmoother:
    """Per-hand One Euro Filter on all 21 landmark points (x, y, z)."""
    def __init__(self):
        self._filters: dict[str, list[list[OneEuroFilter]]] = {}

    def smooth(self, key: str, lm: list) -> list:
        t = time.perf_counter()
        if key not in self._filters:
            self._filters[key] = [
                [OneEuroFilter(), OneEuroFilter(), OneEuroFilter()]
                for _ in range(21)
            ]
        filters = self._filters[key]
        return [
            (filters[i][0](lm[i][0], t),
             filters[i][1](lm[i][1], t),
             filters[i][2](lm[i][2], t))
            for i in range(min(len(lm), 21))
        ]

    def reset(self, key: str | None = None):
        if key:
            self._filters.pop(key, None)
        else:
            self._filters.clear()


# ─────────────────────────────────────────────────────────────
#  ADAPTIVE HAND CALIBRATOR — auto-scales thresholds to hand size
# ─────────────────────────────────────────────────────────────
class HandCalibrator:
    """Measures palm width and normalises gesture thresholds."""
    def __init__(self):
        self._ref_palm = 0.12
        self._ema_palm = None
        self._alpha    = 0.15

    def update(self, lm: list) -> float:
        palm_w = math.hypot(
            lm[HandTracker.WRIST][0]      - lm[HandTracker.MIDDLE_MCP][0],
            lm[HandTracker.WRIST][1]      - lm[HandTracker.MIDDLE_MCP][1],
        )
        if self._ema_palm is None:
            self._ema_palm = palm_w
        else:
            self._ema_palm += self._alpha * (palm_w - self._ema_palm)
        return self.scale()

    def scale(self) -> float:
        if self._ema_palm is None:
            return 1.0
        return max(0.5, min(2.0, self._ema_palm / self._ref_palm))

    def pinch_thresh(self) -> float:
        return Cfg.PINCH_THRESH * self.scale()

    def drag_thresh(self) -> float:
        return Cfg.DRAG_THRESH * self.scale()

    def reset(self):
        self._ema_palm = None


# ─────────────────────────────────────────────────────────────
#  GESTURE STABILISER — hysteresis-based (different enter/exit)
#  Prevents flickering: harder to ENTER a new gesture, easier to STAY
# ─────────────────────────────────────────────────────────────
class GestureStabiliser:
    def __init__(self, window: int = Cfg.GEST_WIN,
                 enter_thresh: float = Cfg.GEST_THRESH,
                 exit_thresh: float = Cfg.GEST_EXIT):
        self._hist         = deque(maxlen=window)
        self._enter_thresh = enter_thresh
        self._exit_thresh  = exit_thresh
        self._win          = window
        self.stable        = "NONE"

    def feed(self, gesture: str) -> str:
        self._hist.append(gesture)
        if len(self._hist) < self._win:
            return self.stable

        counts = Counter(self._hist)
        top, top_cnt = counts.most_common(1)[0]
        top_ratio = top_cnt / self._win

        if top == self.stable:
            cur_ratio = counts.get(self.stable, 0) / self._win
            if cur_ratio < self._exit_thresh:
                self.stable = top
        elif top_ratio >= self._enter_thresh:
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
    H = HandTracker

    def __init__(self):
        self._calib = HandCalibrator()

    def fingers_up(self, lm: list) -> list[bool]:
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

    def _dist(self, lm: list, a: int, b: int) -> float:
        return math.hypot(lm[a][0] - lm[b][0], lm[a][1] - lm[b][1])

    def classify(self, lm: list | None) -> str:
        if lm is None:
            return G.NONE

        self._calib.update(lm)
        pt = self._calib.pinch_thresh()

        f = self.fingers_up(lm)
        thumb, idx, mid, ring, pinky = f

        d_ti = self._dist(lm, self.H.THUMB_TIP, self.H.INDEX_TIP)
        d_tm = self._dist(lm, self.H.THUMB_TIP, self.H.MIDDLE_TIP)
        d_tp = self._dist(lm, self.H.THUMB_TIP, self.H.PINKY_TIP)

        if d_ti < pt:
            return G.PINCH
        if d_tm < pt * 1.2:
            return G.PINCH_RIGHT
        if d_tp < pt * 1.3 and not idx and not mid and not ring:
            return G.SAVE

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
        if sum(f) == 0:
            return G.FIST
        if sum(f) == 5:
            return G.OPEN_PALM

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
        self._tc  = 0.0
        self._ts  = 0.0
        self._tz  = 0.0
        self._lk  = threading.Lock()
        self._oef_x = OneEuroFilter(min_cutoff=2.5, beta=0.01)
        self._oef_y = OneEuroFilter(min_cutoff=2.5, beta=0.01)
        self._raw_prev = (0.5, 0.5)

    def _n2s(self, nx: float, ny: float) -> tuple[int, int]:
        mx, my = Cfg.MARGIN_X, Cfg.MARGIN_Y
        sx = max(0.0, min(1.0, (nx - mx) / max(1e-6, 1 - 2 * mx)))
        sy = max(0.0, min(1.0, (ny - my) / max(1e-6, 1 - 2 * my)))
        return int((1.0 - sx) * Cfg.SCR_W), int(sy * Cfg.SCR_H)

    def move(self, nx: float, ny: float):
        dx = abs(nx - self._raw_prev[0])
        dy = abs(ny - self._raw_prev[1])
        if dx < Cfg.DEAD_ZONE and dy < Cfg.DEAD_ZONE:
            return
        self._raw_prev = (nx, ny)

        t = time.perf_counter()
        fx = self._oef_x(nx, t)
        fy = self._oef_y(ny, t)

        tx, ty = self._n2s(fx, fy)
        with self._lk:
            if abs(tx - self.cx) < 1 and abs(ty - self.cy) < 1:
                return
            self.cx, self.cy = tx, ty
            if self._mc:
                self._mc.position = (self.cx, self.cy)
            else:
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
            except Exception:
                pass

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
        except Exception:
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
        except Exception:
            pass
        return True

    def press_key(self, key: str):
        try:
            pyautogui.press(key)
        except Exception:
            pass

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
#  DUAL-HAND PROCESSOR
#  Dominant hand (wrist.x > 0.5 after flip) = cursor / action
#  Modifier hand (wrist.x < 0.5)            = mode / shortcut
# ─────────────────────────────────────────────────────────────
class DualHandProcessor:
    def __init__(self, mouse: SmoothMouse, db: "GestureDatabase | None" = None):
        self.mouse   = mouse
        self._db     = db
        self._smoother = LandmarkSmoother()
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
                webbrowser.open(f"https://www.google.it/search?q={str(args).replace(' ', '+')}")
            elif action == "zoom" and args is not None:
                m.zoom(int(args))
            elif action == "screenshot":
                ts   = time.strftime("%Y%m%d_%H%M%S")
                path = os.path.expanduser(f"~/Desktop/screenshot_{ts}.png")
                pyautogui.screenshot(path)
            elif action == "type" and args:
                pyautogui.write(str(args), interval=0.04)
            elif action == "create_folder":
                name   = str(args) if args else "Nuova Cartella"
                target = os.path.join(os.path.expanduser("~"), "Desktop", name)
                os.makedirs(target, exist_ok=True)
        except Exception:
            pass

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
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────
#  CAMERA THREAD
# ─────────────────────────────────────────────────────────────
class CameraThread(threading.Thread):
    def __init__(self, app):
        super().__init__(daemon=True)
        self.app       = app
        self._running  = False
        self._flock    = threading.Lock()
        self.frame     = None
        self._frame_id = 0
        self.dom_g     = G.NONE
        self.mod_g     = G.NONE
        self.action    = ""
        self.fps       = 0.0
        self.n_hands   = 0

    def start_capture(self):
        self._running = True
        self.start()

    def stop_capture(self):
        self._running = False

    def _open_camera(self):
        cap = cv2.VideoCapture(Cfg.CAMERA_IDX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  Cfg.CAM_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Cfg.CAM_H)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def run(self):
        cap = self._open_camera()
        self._fail_count = 0

        tracker   = HandTracker()
        db        = getattr(self.app, "_db", None)
        processor = DualHandProcessor(self.app.mouse, db)

        t0 = time.perf_counter()
        fc = 0

        while self._running:
            try:
                ok, frame = cap.read()
                if not ok:
                    self._fail_count += 1
                    if self._fail_count > 30:
                        cap.release()
                        self.action = "⚠ RICONNESSIONE CAMERA..."
                        time.sleep(1.0)
                        cap = self._open_camera()
                        self._fail_count = 0
                    else:
                        time.sleep(0.01)
                    continue
                self._fail_count = 0

                frame = cv2.flip(frame, 1)

                if self.app.hand_active:
                    results = tracker.process(frame)
                    frame   = tracker.annotate(frame, results)
                    hands   = tracker.extract(results)
                    self.n_hands = len(hands)
                    dom_g, mod_g, action = processor.process(hands)
                    self.dom_g  = dom_g
                    self.mod_g  = mod_g
                    self.action = action
                    self._draw_hud(frame, dom_g, mod_g, action, hands)

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

                cv2.putText(frame, f"FPS {self.fps:.0f}", (8, 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 220, 80), 2)

                with self._flock:
                    self.frame = frame
                    self._frame_id += 1

            except Exception:
                time.sleep(0.02)

        try:
            tracker.close()
            cap.release()
        except Exception:
            pass

    def get_frame(self):
        with self._flock:
            return self._frame_id, self.frame

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
class CommandParser:
    _CMDS = [
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
    def __init__(self, mouse: SmoothMouse, on_command):
        self._mouse    = mouse
        self._cb       = on_command
        self._parser   = CommandParser()
        self._running  = False
        self._thread   = None
        self.status    = "OFFLINE"
        self.last_text = ""
        self.history   = deque(maxlen=6)

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
                rec.adjust_for_ambient_noise(mic, duration=0.5)
        except Exception:
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
                    self._execute(action, args)
                    self._cb(text, action)
                else:
                    self._cb(text, None)
                self.status = "ASCOLTO"
            except sr.WaitTimeoutError:
                self.status = "ASCOLTO"
            except sr.UnknownValueError:
                self.status = "NON CAPITO"
                time.sleep(0.4)
                self.status = "ASCOLTO"
            except Exception as e:
                self.status = f"ERR: {str(e)[:18]}"
                time.sleep(1.5)
                self.status = "ASCOLTO"

    def _execute(self, action: str, args):
        m = self._mouse
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
                q = str(args).replace(' ', '+')
                webbrowser.open(f"https://www.google.it/search?q={q}")
            elif action == 'create_folder':
                name   = str(args) if args else 'Nuova Cartella'
                target = os.path.join(os.path.expanduser("~"), "Desktop", name)
                os.makedirs(target, exist_ok=True)
            elif action == 'hotkey':
                m.hotkey(*args)
            elif action == 'zoom':
                m.zoom(int(args))
            elif action == 'type':
                pyautogui.write(str(args), interval=0.04)
            elif action == 'screenshot':
                ts   = time.strftime("%Y%m%d_%H%M%S")
                path = os.path.expanduser(f"~/Desktop/screenshot_{ts}.png")
                pyautogui.screenshot(path)
        except Exception:
            pass


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
        except Exception:
            pass

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
        inter = [self._dist(lm, tips[i], tips[j])
                 for i in range(5) for j in range(i + 1, 5)]
        wrist = [self._dist(lm, H.WRIST, t) for t in tips]
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
                d = math.sqrt(sum((a - b) ** 2 for a, b in zip(fv, sample)))
                candidates.append((d, name))
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0])
        k_nearest = candidates[: self.K]
        if k_nearest[0][0] >= self.CONF_THRESH:
            return None
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
        self._last_fid   = -1
        self._cached_dw  = 0
        self._db         = GestureDatabase()
        self._recogniser = CustomGestureRecogniser(self._db)
        self._recorder   = GestureRecorder(self._db, self._recogniser)
        self._voice      = VoiceController(self.mouse, self._on_voice_cmd)

        self._setup_window()
        self.withdraw()
        self._splash = SplashScreen(self)
        self.update()
        self._build_ui()
        self._start_camera()
        self.after(600, self._finish_splash)
        self._loop()

    def _setup_window(self):
        self.title("HGC — Hand Gesture Control")
        self.configure(bg=Cfg.BG_DARK)
        self.geometry("1200x820")
        self.minsize(900, 640)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):
        _FONT = "Segoe UI"

        # ── HEADER — minimal, pulito ──────────────────────────
        hdr = tk.Frame(self, bg=Cfg.BG_DARK)
        hdr.pack(fill="x", padx=20, pady=(14, 0))

        title_row = tk.Frame(hdr, bg=Cfg.BG_DARK)
        title_row.pack(side="left")
        tk.Label(title_row, text="HGC",
                 font=(_FONT, 22, "bold"),
                 bg=Cfg.BG_DARK, fg=Cfg.ACCENT).pack(side="left")
        tk.Label(title_row, text="  Hand Gesture Control",
                 font=(_FONT, 14),
                 bg=Cfg.BG_DARK, fg=Cfg.TEXT).pack(side="left", pady=(4, 0))

        indicators = tk.Frame(hdr, bg=Cfg.BG_DARK)
        indicators.pack(side="right")

        self._status_lbl = tk.Label(indicators, text="● ATTIVO",
                 font=(_FONT, 10, "bold"), bg=Cfg.BG_DARK, fg=Cfg.SUCCESS)
        self._status_lbl.pack(side="right", padx=(16, 0))
        self._hands_lbl = tk.Label(indicators, text="",
                 font=(_FONT, 10), bg=Cfg.BG_DARK, fg=Cfg.TEXT_DIM)
        self._hands_lbl.pack(side="right", padx=8)
        self._voice_hdr_lbl = tk.Label(indicators, text="MIC OFF",
                 font=(_FONT, 9), bg=Cfg.BG_DARK, fg=Cfg.TEXT_DIM)
        self._voice_hdr_lbl.pack(side="right", padx=8)

        # ── Separator line ────────────────────────────────────
        tk.Frame(self, bg=Cfg.BORDER, height=1).pack(fill="x", padx=20, pady=(10, 0))

        # ── BODY — camera + pannello ──────────────────────────
        body = tk.Frame(self, bg=Cfg.BG_DARK)
        body.pack(fill="both", expand=True, padx=20, pady=12)

        # Camera panel
        self._left_panel = tk.Frame(body, bg=Cfg.BG_MID)
        self._left_panel.pack(side="left", fill="both", expand=True, padx=(0, 10))
        self._cam_lbl = tk.Label(self._left_panel, bg=Cfg.BG_MID)
        self._cam_lbl.pack(padx=6, pady=6, fill="both", expand=True)

        # Control panel (right, tabbed)
        right = tk.Frame(body, bg=Cfg.BG_DARK, width=360)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        style = ttk.Style(self)
        style.theme_use("default")
        style.configure("App.TNotebook",
                        background=Cfg.BG_DARK, borderwidth=0,
                        tabmargins=[0, 0, 0, 0])
        style.configure("App.TNotebook.Tab",
                        background=Cfg.BG_MID, foreground=Cfg.TEXT_DIM,
                        font=(_FONT, 9, "bold"), padding=[14, 7])
        style.map("App.TNotebook.Tab",
                  background=[("selected", Cfg.ACCENT)],
                  foreground=[("selected", "#ffffff")])

        nb = ttk.Notebook(right, style="App.TNotebook")
        nb.pack(fill="both", expand=True)

        tab_hands = tk.Frame(nb, bg=Cfg.BG_DARK)
        nb.add(tab_hands, text="  Controllo  ")

        tab_voice = tk.Frame(nb, bg=Cfg.BG_DARK)
        nb.add(tab_voice, text="  Voce  ")

        tab_guide = tk.Frame(nb, bg=Cfg.BG_DARK)
        nb.add(tab_guide, text="  Guida  ")

        tab_learn = tk.Frame(nb, bg=Cfg.BG_DARK)
        nb.add(tab_learn, text="  Apprendi  ")

        self._build_hands_tab(tab_hands)
        self._build_voice_tab(tab_voice)
        self._build_guide_tab(tab_guide)
        self._build_learn_tab(tab_learn)

        # ── STATUS BAR — in basso ─────────────────────────────
        sbar = tk.Frame(self, bg=Cfg.BG_MID, height=28)
        sbar.pack(fill="x", side="bottom")
        sbar.pack_propagate(False)
        self._fps_lbl = tk.Label(sbar, text="FPS: —",
                 font=(_FONT, 8), bg=Cfg.BG_MID, fg=Cfg.TEXT_DIM)
        self._fps_lbl.pack(side="left", padx=12)
        tk.Label(sbar, text="HGC v2",
                 font=(_FONT, 8), bg=Cfg.BG_MID, fg=Cfg.TEXT_DIM
                 ).pack(side="right", padx=12)

    def _card(self, parent, title):
        outer = tk.Frame(parent, bg=Cfg.BORDER)
        outer.pack(fill="x", padx=8, pady=5)
        f = tk.Frame(outer, bg=Cfg.BG_CARD)
        f.pack(fill="x", padx=1, pady=1)
        tk.Label(f, text=title.upper(), font=("Segoe UI", 8, "bold"),
                 bg=Cfg.BG_CARD, fg=Cfg.TEXT_DIM, anchor="w"
                 ).pack(fill="x", padx=14, pady=(10, 4))
        return f

    def _btn(self, parent, text, color, command, fg="#ffffff"):
        return tk.Button(parent, text=text,
            font=("Segoe UI", 10, "bold"), bg=color, fg=fg,
            activebackground=color, activeforeground=fg,
            relief="flat", bd=0, cursor="hand2", padx=20, pady=7,
            command=command)

    def _build_hands_tab(self, parent):
        c1 = self._card(parent, "Riconoscimento")
        tk.Label(c1, text="Destra = cursore    Sinistra = modificatore",
                 font=("Segoe UI", 9), bg=Cfg.BG_CARD,
                 fg=Cfg.TEXT_DIM).pack(padx=14, anchor="w")
        self._hand_btn = self._btn(c1, "DISATTIVA", Cfg.ACCENT, self._toggle_hand)
        self._hand_btn.pack(pady=10, padx=14, fill="x")

        c2 = self._card(parent, "Tastiera Virtuale")
        self._vk_btn = self._btn(c2, "MOSTRA TASTIERA", Cfg.BG_MID, self._toggle_vk, fg=Cfg.TEXT)
        self._vk_btn.pack(pady=10, padx=14, fill="x")

        c3 = self._card(parent, "Gesti Rilevati")
        grid = tk.Frame(c3, bg=Cfg.BG_CARD)
        grid.pack(fill="x", padx=14, pady=(4, 0))
        tk.Label(grid, text="MANO DX", font=("Segoe UI", 8),
                 bg=Cfg.BG_CARD, fg=Cfg.TEXT_DIM, anchor="w").grid(row=0, column=0, sticky="w")
        self._dom_lbl = tk.Label(grid, text="—",
                 font=("Segoe UI", 16, "bold"), bg=Cfg.BG_CARD, fg=Cfg.ACCENT)
        self._dom_lbl.grid(row=1, column=0, sticky="w", pady=(0, 4))
        tk.Label(grid, text="MANO SX", font=("Segoe UI", 8),
                 bg=Cfg.BG_CARD, fg=Cfg.TEXT_DIM, anchor="w").grid(row=0, column=1, sticky="w", padx=(24, 0))
        self._mod_lbl = tk.Label(grid, text="—",
                 font=("Segoe UI", 16, "bold"), bg=Cfg.BG_CARD, fg=Cfg.PURPLE)
        self._mod_lbl.grid(row=1, column=1, sticky="w", padx=(24, 0), pady=(0, 4))
        self._act_lbl = tk.Label(c3, text="",
                 font=("Segoe UI", 9), bg=Cfg.BG_CARD, fg=Cfg.SUCCESS)
        self._act_lbl.pack(padx=14, anchor="w", pady=(0, 10))

        c4 = self._card(parent, "Sensibilità")
        sf = tk.Frame(c4, bg=Cfg.BG_CARD)
        sf.pack(fill="x", padx=14, pady=(4, 10))
        tk.Label(sf, text="Lento", font=("Segoe UI", 8),
                 bg=Cfg.BG_CARD, fg=Cfg.TEXT_DIM).pack(side="left")
        self._smooth_var = tk.DoubleVar(value=Cfg.SMOOTH)
        ttk.Scale(sf, from_=0.05, to=1.0, orient="horizontal",
                  variable=self._smooth_var,
                  command=lambda v: setattr(Cfg, "SMOOTH", float(v))
                  ).pack(fill="x", side="left", expand=True, padx=6)
        tk.Label(sf, text="Veloce", font=("Segoe UI", 8),
                 bg=Cfg.BG_CARD, fg=Cfg.TEXT_DIM).pack(side="left")

    def _build_voice_tab(self, parent):
        c1 = self._card(parent, "Controllo Vocale")

        srow = tk.Frame(c1, bg=Cfg.BG_CARD)
        srow.pack(fill="x", padx=14, pady=(0, 4))
        self._voice_dot = tk.Label(srow, text="●", font=("Segoe UI", 12),
                 bg=Cfg.BG_CARD, fg=Cfg.TEXT_DIM)
        self._voice_dot.pack(side="left")
        self._voice_status_lbl = tk.Label(srow, text="OFFLINE",
                 font=("Segoe UI", 10, "bold"), bg=Cfg.BG_CARD, fg=Cfg.TEXT_DIM)
        self._voice_status_lbl.pack(side="left", padx=6)

        vcolor = Cfg.SUCCESS if self._voice.available else Cfg.BG_MID
        vstate = "normal" if self._voice.available else "disabled"
        self._voice_btn = self._btn(c1, "ATTIVA VOCE", vcolor, self._toggle_voice)
        self._voice_btn.configure(state=vstate)
        self._voice_btn.pack(pady=8, padx=14, fill="x")

        if not self._voice.available:
            tk.Label(c1, text="SpeechRecognition non trovato. Esegui: python install.py",
                     font=("Segoe UI", 8), bg=Cfg.BG_CARD, fg=Cfg.WARNING,
                     wraplength=300).pack(padx=14, pady=(0, 8))

        c2 = self._card(parent, "Ultimo Riconosciuto")
        self._voice_last_lbl = tk.Label(
            c2, text="—",
            font=("Segoe UI", 10), bg=Cfg.BG_CARD, fg=Cfg.TEXT,
            wraplength=300, anchor="w")
        self._voice_last_lbl.pack(padx=14, pady=(0, 10), anchor="w")

        c3 = self._card(parent, "Cronologia")
        self._voice_log_labels = []
        for _ in range(6):
            lbl = tk.Label(c3, text="", font=("Consolas", 8),
                           bg=Cfg.BG_CARD, fg=Cfg.TEXT_DIM, anchor="w")
            lbl.pack(fill="x", padx=14, pady=1)
            self._voice_log_labels.append(lbl)
        tk.Frame(c3, bg=Cfg.BG_CARD, height=6).pack()

        c4 = self._card(parent, "Comandi Esempio")
        cmds = ["apri youtube", "cerca meteo Milano", "crea cartella lavoro",
                "copia / incolla / salva", "screenshot", "zoom avanti / indietro",
                "scrivi ciao mondo", "volume su / muto"]
        for ex in cmds:
            tk.Label(c4, text=f"  {ex}", font=("Segoe UI", 8),
                     bg=Cfg.BG_CARD, fg=Cfg.TEXT_DIM, anchor="w"
                     ).pack(fill="x", padx=14)
        tk.Frame(c4, bg=Cfg.BG_CARD, height=8).pack()

    def _build_guide_tab(self, parent):
        c = self._card(parent, "Mano Destra — Azioni")
        DOM = [
            ("Indice solo",       "Cursore"),
            ("Pinch indice",      "Click / Drag"),
            ("Pinch medio",       "Click destro"),
            ("Due dita",          "Scroll"),
            ("Tre dita",          "Copia"),
            ("Quattro dita",      "Incolla"),
            ("Pollice solo",      "Doppio click"),
            ("Rock",              "Annulla"),
            ("Pollice+mignolo",   "Salva"),
            ("Palmo+swipe",       "Avanti / Indietro"),
        ]
        for g, a in DOM:
            r = tk.Frame(c, bg=Cfg.BG_CARD)
            r.pack(fill="x", padx=14, pady=1)
            tk.Label(r, text=g, font=("Segoe UI", 9),
                     bg=Cfg.BG_CARD, fg=Cfg.TEXT, width=18, anchor="w").pack(side="left")
            tk.Label(r, text=a, font=("Segoe UI", 9),
                     bg=Cfg.BG_CARD, fg=Cfg.ACCENT, anchor="w").pack(side="left")
        tk.Frame(c, bg=Cfg.BG_CARD, height=6).pack()

        c2 = self._card(parent, "Mano Sinistra — Modalità")
        MOD = [
            ("Palmo aperto",   "Congela cursore"),
            ("Pugno",          "Zoom mode"),
            ("Due dita",       "Scroll orizzontale"),
            ("Rock",           "Alt+Tab"),
            ("Pollice",        "Click centrale"),
            ("Entrambi pinch", "Zoom in/out"),
        ]
        for g, a in MOD:
            r = tk.Frame(c2, bg=Cfg.BG_CARD)
            r.pack(fill="x", padx=14, pady=1)
            tk.Label(r, text=g, font=("Segoe UI", 9),
                     bg=Cfg.BG_CARD, fg=Cfg.TEXT, width=18, anchor="w").pack(side="left")
            tk.Label(r, text=a, font=("Segoe UI", 9),
                     bg=Cfg.BG_CARD, fg=Cfg.PURPLE, anchor="w").pack(side="left")
        tk.Frame(c2, bg=Cfg.BG_CARD, height=6).pack()

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

        c1 = self._card(parent, "Nuovo Gesto Personalizzato")

        def _field(parent_f, label, var):
            r = tk.Frame(parent_f, bg=Cfg.BG_CARD)
            r.pack(fill="x", padx=14, pady=3)
            tk.Label(r, text=label, font=("Segoe UI", 9),
                     bg=Cfg.BG_CARD, fg=Cfg.TEXT_DIM, width=8, anchor="w").pack(side="left")
            e = tk.Entry(r, textvariable=var,
                     font=("Segoe UI", 9), bg=Cfg.BG_MID, fg=Cfg.TEXT,
                     insertbackground=Cfg.TEXT, relief="flat", bd=2)
            e.pack(side="left", fill="x", expand=True, padx=(4, 0))
            return e

        self._learn_name_var = tk.StringVar()
        _field(c1, "Nome", self._learn_name_var)

        row_a = tk.Frame(c1, bg=Cfg.BG_CARD)
        row_a.pack(fill="x", padx=14, pady=3)
        tk.Label(row_a, text="Azione", font=("Segoe UI", 9),
                 bg=Cfg.BG_CARD, fg=Cfg.TEXT_DIM, width=8, anchor="w").pack(side="left")
        self._learn_action_var = tk.StringVar(value=_ACTIONS[0][0])
        opt = tk.OptionMenu(row_a, self._learn_action_var,
                            *[a[0] for a in _ACTIONS])
        opt.configure(bg=Cfg.BG_MID, fg=Cfg.TEXT, font=("Segoe UI", 9),
                      activebackground=Cfg.BG_CARD, relief="flat", bd=0,
                      highlightthickness=0)
        opt["menu"].configure(bg=Cfg.BG_MID, fg=Cfg.TEXT, font=("Segoe UI", 9))
        opt.pack(side="left", fill="x", expand=True, padx=(4, 0))

        self._learn_arg_var = tk.StringVar()
        _field(c1, "Argomento", self._learn_arg_var)

        self._learn_rec_btn = self._btn(c1, "REGISTRA  (3s)", Cfg.ACCENT, self._start_recording)
        self._learn_rec_btn.pack(pady=10, padx=14, fill="x")

        self._learn_prog_frame = tk.Frame(c1, bg=Cfg.BG_CARD)
        self._learn_prog_frame.pack(fill="x", padx=14, pady=(0, 6))
        self._learn_prog_lbl = tk.Label(
            self._learn_prog_frame, text="",
            font=("Segoe UI", 8), bg=Cfg.BG_CARD, fg=Cfg.TEXT_DIM)
        self._learn_prog_lbl.pack(side="right")
        self._learn_prog_bar = tk.Canvas(
            self._learn_prog_frame, height=4, bg=Cfg.BG_MID,
            highlightthickness=0)
        self._learn_prog_bar.pack(fill="x", side="left", expand=True, padx=(0, 6))

        c2 = self._card(parent, "Gesti Appresi")
        self._learn_list_frame = tk.Frame(c2, bg=Cfg.BG_CARD)
        self._learn_list_frame.pack(fill="x", padx=14, pady=4)
        self._learn_listbox = tk.Listbox(
            self._learn_list_frame, bg=Cfg.BG_MID, fg=Cfg.TEXT,
            font=("Segoe UI", 9), relief="flat", selectbackground=Cfg.ACCENT,
            selectforeground="#ffffff", height=5, bd=0)
        self._learn_listbox.pack(fill="x")
        self._btn(c2, "ELIMINA SELEZIONATO", Cfg.BG_MID, self._delete_gesture, fg=Cfg.ACCENT
                  ).pack(pady=(4, 10), padx=14, fill="x")
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

    # ── main loop ─────────────────────────────────────────────
    def _finish_splash(self):
        if hasattr(self, "_splash") and self._splash.winfo_exists():
            self._splash.done()
        self.deiconify()

    def _start_camera(self):
        self._cam = CameraThread(self)
        self._cam.start_capture()

    def _loop(self):
        try:
            if self._cam:
                fid, frame = self._cam.get_frame()
                if frame is not None and fid != self._last_fid:
                    self._last_fid = fid
                    h, w  = frame.shape[:2]
                    avail = self._left_panel.winfo_width()
                    dw    = max(300, avail - 16) if avail > 10 else 630
                    if dw != self._cached_dw:
                        self._cached_dw = dw
                    dh = int(h * dw / w)
                    small = cv2.resize(frame, (dw, dh), interpolation=cv2.INTER_LINEAR)
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
                        text=(f"{n} man{'i' if n != 1 else 'o'}"
                              if n else ""))
                    self._fps_lbl.configure(
                        text=f"FPS: {self._cam.fps:.0f}")
        except Exception:
            pass

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
        except Exception:
            pass

        self.after(Cfg.DWELL_MS, self._loop)

    def _on_close(self):
        try:
            if self._cam:
                self._cam.stop_capture()
            self._voice.stop()
            self._db.save()
        except Exception:
            pass
        self.destroy()
        sys.exit(0)

    def _show_error(self, title: str, msg: str):
        from tkinter import messagebox
        messagebox.showerror(title, msg)


# ─────────────────────────────────────────────────────────────
#  SPLASH SCREEN
# ─────────────────────────────────────────────────────────────
class SplashScreen(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.overrideredirect(True)

        w, h = 440, 200
        sx = (parent.winfo_screenwidth()  - w) // 2
        sy = (parent.winfo_screenheight() - h) // 2
        self.geometry(f"{w}x{h}+{sx}+{sy}")

        self.configure(bg=Cfg.BG_DARK)
        tk.Label(self, text="HGC",
                 font=("Segoe UI", 36, "bold"),
                 bg=Cfg.BG_DARK, fg=Cfg.ACCENT).pack(pady=(30, 0))
        tk.Label(self, text="Hand Gesture Control",
                 font=("Segoe UI", 12),
                 bg=Cfg.BG_DARK, fg=Cfg.TEXT_DIM).pack(pady=(0, 20))
        self._bar = tk.Canvas(self, width=w - 100, height=3,
                              bg=Cfg.BG_MID, highlightthickness=0)
        self._bar.pack()
        self._progress = 0
        self._animate()

    def _animate(self):
        self._progress = min(self._progress + 10, 340)
        self._bar.delete("all")
        self._bar.create_rectangle(0, 0, self._progress, 3, fill=Cfg.ACCENT, outline="")
        if self._progress < 340:
            self.after(40, self._animate)

    def done(self):
        self.destroy()


# ─────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────
def main():
    app = Dashboard()
    app.mainloop()


if __name__ == "__main__":
    main()
