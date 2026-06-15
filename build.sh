#!/usr/bin/env bash
# Build script per Hand Gesture Control — crea eseguibile con PyInstaller
# Uso: bash build.sh
# Output: dist/hand-gesture-control/hand-gesture-control  (Linux/macOS)
#         dist\hand-gesture-control\hand-gesture-control.exe  (Windows)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "═══════════════════════════════════════════════"
echo "  Hand Gesture Control — Build Eseguibile"
echo "═══════════════════════════════════════════════"

# ── 1. Python ────────────────────────────────────────────────────────────────
if command -v python3 &>/dev/null; then
    PYTHON=python3
elif command -v python &>/dev/null; then
    PYTHON=python
else
    echo "ERRORE: Python non trovato. Installa Python 3.8+."
    exit 1
fi

PY_VER=$($PYTHON -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Python: $PY_VER"

# ── 2. Dipendenze ────────────────────────────────────────────────────────────
echo ""
echo "▸ Installazione dipendenze..."

pip install --quiet \
    pyinstaller \
    "mediapipe>=0.10.0,<0.10.14" \
    "opencv-python>=4.8.0" \
    "pyautogui>=0.9.54" \
    "pynput>=1.7.6" \
    "numpy>=1.24.0" \
    "Pillow>=10.0.0" \
    "SpeechRecognition>=3.10.0"

# pyaudio è opzionale (voce) — non bloccare se fallisce
pip install --quiet pyaudio 2>/dev/null || \
    echo "  ⚠  pyaudio non installato — controllo voce disabilitato"

echo "  ✓ dipendenze ok"

# ── 3. Verifica sorgenti ─────────────────────────────────────────────────────
if [[ ! -f "hand_gesture_control.py" ]]; then
    echo "ERRORE: hand_gesture_control.py non trovato."
    exit 1
fi

# ── 4. Pulizia precedente build ──────────────────────────────────────────────
echo ""
echo "▸ Pulizia build precedente..."
rm -rf build/ dist/

# ── 5. Build con PyInstaller ─────────────────────────────────────────────────
echo ""
echo "▸ Build in corso... (può richiedere 1-3 minuti)"

$PYTHON -m PyInstaller hand_gesture_control.spec \
    --noconfirm \
    --clean \
    2>&1

# ── 6. Verifica output ───────────────────────────────────────────────────────
echo ""
if [[ -d "dist/hand-gesture-control" ]]; then
    SIZE=$(du -sh dist/hand-gesture-control 2>/dev/null | cut -f1)
    echo "═══════════════════════════════════════════════"
    echo "  ✓  Build completata con successo!"
    echo ""
    echo "  Output:  dist/hand-gesture-control/"
    echo "  Peso:    $SIZE"
    echo ""
    echo "  Per avviare:"
    if [[ "$OSTYPE" == "msys"* || "$OSTYPE" == "win"* ]]; then
        echo "  dist\\hand-gesture-control\\hand-gesture-control.exe"
    else
        echo "  ./dist/hand-gesture-control/hand-gesture-control"
    fi
    echo "═══════════════════════════════════════════════"
else
    echo "ERRORE: build fallita. Controlla l'output sopra."
    exit 1
fi
