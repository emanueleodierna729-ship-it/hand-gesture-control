# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec per Hand Gesture Control
Uso: pyinstaller hand_gesture_control.spec
"""

import sys
import os
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

# ── Raccolta dati mediapipe ──────────────────────────────────────────────────
mediapipe_datas = collect_data_files("mediapipe")
cv2_datas       = collect_data_files("cv2")

all_datas = mediapipe_datas + cv2_datas + [
    ("user_gestures.json", "."),   # database gesti custom (se presente)
]

# ── Binaries dinamici ────────────────────────────────────────────────────────
mediapipe_bins = collect_dynamic_libs("mediapipe")

# ── Analisi sorgente ─────────────────────────────────────────────────────────
a = Analysis(
    ["hand_gesture_control.py"],
    pathex=["."],
    binaries=mediapipe_bins,
    datas=all_datas,
    hiddenimports=[
        # mediapipe
        "mediapipe",
        "mediapipe.python",
        "mediapipe.python.solutions",
        "mediapipe.python.solutions.hands",
        "mediapipe.python.solutions.drawing_utils",
        # numpy/cv2
        "numpy",
        "numpy.core",
        "cv2",
        # GUI
        "tkinter",
        "tkinter.ttk",
        "PIL",
        "PIL.Image",
        "PIL.ImageTk",
        # input
        "pynput",
        "pynput.mouse",
        "pynput.keyboard",
        # voce (opzionale, non blocca se assente)
        "speech_recognition",
        "pyaudio",
        # altri
        "pyautogui",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "matplotlib",
        "scipy",
        "pandas",
        "IPython",
        "jupyter",
        "notebook",
    ],
    noarchive=False,
    optimize=1,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="hand-gesture-control",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,          # nessuna finestra terminale
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="hand-gesture-control",
)
