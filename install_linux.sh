#!/usr/bin/env bash
# Hand Gesture Control — Linux Installer
# Compatibile con: Ubuntu/Debian, Fedora/RHEL, Arch Linux, openSUSE
# Uso: bash install_linux.sh

set -euo pipefail

R="\033[91m"; G="\033[92m"; Y="\033[93m"; C="\033[96m"
B="\033[94m"; W="\033[97m"; DIM="\033[2m"; BOLD="\033[1m"; RST="\033[0m"

step() { echo -e "  ${B}→${RST}  $*"; }
ok()   { echo -e "  ${G}✓${RST}  $*"; }
warn() { echo -e "  ${Y}⚠${RST}  $*"; }
fail() { echo -e "  ${R}✗${RST}  $*"; exit 1; }

echo -e "
${C}${BOLD}╔══════════════════════════════════════════════════════════╗
║       Hand Gesture Control  v2  —  Linux Installer        ║
║       Creato da Emanuele Odierna insieme a Claude          ║
╚══════════════════════════════════════════════════════════╝${RST}
"

# Rileva distro
DISTRO="unknown"
if   command -v apt-get &>/dev/null; then DISTRO="debian"
elif command -v dnf     &>/dev/null; then DISTRO="fedora"
elif command -v pacman  &>/dev/null; then DISTRO="arch"
elif command -v zypper  &>/dev/null; then DISTRO="opensuse"
fi
echo -e "  Distribuzione rilevata: ${BOLD}$DISTRO${RST}"

# Verifica Python 3.8+
step "Verifica Python 3.8+..."
PYTHON=""
for cmd in python3.12 python3.11 python3.10 python3.9 python3.8 python3; do
    if command -v "$cmd" &>/dev/null; then
        ver=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        maj=${ver%%.*}; min=${ver##*.}
        if [ "$maj" -ge 3 ] && [ "$min" -ge 8 ]; then
            PYTHON="$cmd"; break
        fi
    fi
done
[ -z "$PYTHON" ] && fail "Python 3.8+ non trovato. Installa Python e riprova."
ok "Python $ver trovato ($PYTHON)"

# Dipendenze di sistema
step "Installazione dipendenze di sistema..."
case "$DISTRO" in
    debian)
        sudo apt-get update -qq
        sudo apt-get install -y -qq python3-tk python3-dev python3-pip \
            portaudio19-dev libx11-dev python3-xlib 2>/dev/null || true
        ok "Dipendenze Debian/Ubuntu installate"
        ;;
    fedora)
        sudo dnf install -y -q python3-tkinter python3-devel python3-pip \
            portaudio-devel libX11-devel 2>/dev/null || true
        ok "Dipendenze Fedora/RHEL installate"
        ;;
    arch)
        sudo pacman -Sy --noconfirm --quiet python-tkinter python-pip \
            portaudio libx11 python-xlib 2>/dev/null || true
        ok "Dipendenze Arch Linux installate"
        ;;
    opensuse)
        sudo zypper install -y -q python3-tk python3-pip \
            portaudio-devel libX11-devel 2>/dev/null || true
        ok "Dipendenze openSUSE installate"
        ;;
    *)
        warn "Distro non riconosciuta. Installa manualmente: python3-tk, portaudio-dev"
        ;;
esac

# Aggiorna pip
step "Aggiornamento pip..."
"$PYTHON" -m pip install --upgrade pip --quiet
ok "pip aggiornato"

# Pacchetti Python
PACKAGES=(
    "opencv-python>=4.8.0"
    "mediapipe>=0.10.0,<0.10.14"
    "pyautogui>=0.9.54"
    "numpy>=1.24.0"
    "Pillow>=10.0.0"
    "pynput>=1.7.6"
    "SpeechRecognition>=3.10.0"
    "anthropic>=0.40.0"
)
for pkg in "${PACKAGES[@]}"; do
    name="${pkg%%>*}"
    step "Installazione $name..."
    "$PYTHON" -m pip install "$pkg" --quiet
    ok "$name"
done

# PyAudio
step "Installazione PyAudio (controllo vocale)..."
if "$PYTHON" -m pip install pyaudio --quiet 2>/dev/null; then
    ok "PyAudio"
else
    warn "PyAudio non installabile. Per il controllo vocale:"
    warn "  sudo apt install portaudio19-dev && pip install pyaudio"
fi

# python-xlib (fallback)
step "Verifica python-xlib..."
if ! "$PYTHON" -c "import Xlib" 2>/dev/null; then
    "$PYTHON" -m pip install python-xlib --quiet 2>/dev/null || \
        warn "python-xlib non installabile via pip — prova: sudo apt install python3-xlib"
else
    ok "python-xlib già disponibile"
fi

# Collegamento Desktop
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP="$SCRIPT_DIR/hand_gesture_control.py"
DESKTOP="$HOME/Desktop"

if [ -d "$DESKTOP" ]; then
    step "Creazione collegamento Desktop..."
    cat > "$DESKTOP/HandGestureControl.desktop" <<EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=Hand Gesture Control
Comment=Controllo PC con gesti della mano — v2
Exec=$PYTHON $APP
Icon=input-gaming
Terminal=false
Categories=Utility;Accessibility;
StartupNotify=true
EOF
    chmod +x "$DESKTOP/HandGestureControl.desktop"
    ok "Collegamento creato: ~/Desktop/HandGestureControl.desktop"
fi

echo -e "
${G}${BOLD}╔══════════════════════════════════════════════════════════╗
║         Installazione completata con successo!            ║
╚══════════════════════════════════════════════════════════╝${RST}

  Avvio rapido:
    ${W}$PYTHON hand_gesture_control.py${RST}

  ${DIM}Note Linux:
    - Richiede sessione X11 (non Wayland puro) per il controllo mouse
    - Se Wayland: avvia con XWayland o usa una sessione GNOME/X11
    - Webcam: verifica che l'utente sia nel gruppo 'video'
      sudo usermod -aG video \$USER${RST}
"

read -rp "  Avviare adesso l'applicazione? [S/n] " ans
if [[ ! "$ans" =~ ^[Nn]$ ]]; then
    echo -e "\n  ${C}Avvio Hand Gesture Control...${RST}"
    "$PYTHON" "$APP" &
fi
