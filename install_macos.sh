#!/usr/bin/env bash
# Hand Gesture Control — macOS Installer
# Compatibile con: macOS 12 Monterey, 13 Ventura, 14 Sonoma, 15 Sequoia
# Uso: bash install_macos.sh

set -euo pipefail

R="\033[91m"; G="\033[92m"; Y="\033[93m"; C="\033[96m"
B="\033[94m"; W="\033[97m"; DIM="\033[2m"; BOLD="\033[1m"; RST="\033[0m"

step() { echo -e "  ${B}→${RST}  $*"; }
ok()   { echo -e "  ${G}✓${RST}  $*"; }
warn() { echo -e "  ${Y}⚠${RST}  $*"; }
fail() { echo -e "  ${R}✗${RST}  $*"; exit 1; }

echo -e "
${C}${BOLD}╔══════════════════════════════════════════════════════════╗
║       Hand Gesture Control  v2  —  macOS Installer        ║
║       Creato da Emanuele Odierna insieme a Claude          ║
╚══════════════════════════════════════════════════════════╝${RST}
"

# macOS version
MACOS_VER=$(sw_vers -productVersion)
echo -e "  macOS: ${BOLD}$MACOS_VER${RST}"

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

if [ -z "$PYTHON" ]; then
    warn "Python 3.8+ non trovato."
    echo -e "  ${Y}Scarica Python da: https://www.python.org/downloads/macos/${RST}"
    echo -e "  ${Y}Oppure installa via Homebrew: brew install python${RST}"
    exit 1
fi
ok "Python $ver trovato ($PYTHON)"

# Homebrew
step "Verifica Homebrew..."
if command -v brew &>/dev/null; then
    ok "Homebrew trovato"
    HAS_BREW=true
else
    warn "Homebrew non trovato."
    warn "Installa da: https://brew.sh"
    HAS_BREW=false
fi

# PortAudio (per PyAudio / controllo vocale)
step "Installazione PortAudio (richiesto per controllo vocale)..."
if $HAS_BREW; then
    if brew list portaudio &>/dev/null 2>&1; then
        ok "PortAudio già installato"
    else
        brew install portaudio --quiet
        ok "PortAudio installato"
    fi
else
    warn "Installa PortAudio manualmente: brew install portaudio"
fi

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
    warn "PyAudio non installabile. Assicurati che PortAudio sia installato:"
    warn "  brew install portaudio && pip install pyaudio"
fi

# Crea launcher .command (doppio clic dal Finder)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP="$SCRIPT_DIR/hand_gesture_control.py"
LAUNCHER="$SCRIPT_DIR/Avvia HandGestureControl.command"

step "Creazione launcher per Finder..."
cat > "$LAUNCHER" <<EOF
#!/bin/bash
cd "$(dirname "\$0")"
$PYTHON hand_gesture_control.py
EOF
chmod +x "$LAUNCHER"
ok "Launcher creato: 'Avvia HandGestureControl.command'"

# Nota accessibilità
echo -e "
${Y}${BOLD}  IMPORTANTE — Permessi macOS:${RST}
  ${DIM}Per controllare mouse e tastiera, macOS richiede permessi di Accessibilità.

  Alla prima esecuzione:
  1. Vai in: Impostazioni di Sistema → Privacy e Sicurezza → Accessibilità
  2. Clicca '+' e aggiungi Terminale (o la tua app Terminal)
  3. Riavvia l'applicazione

  Se usi una versione recente di macOS (14+):
  Impostazioni di Sistema → Privacy e Sicurezza → Automazione
  e aggiungi anche i permessi per la webcam.${RST}
"

echo -e "
${G}${BOLD}╔══════════════════════════════════════════════════════════╗
║         Installazione completata con successo!            ║
╚══════════════════════════════════════════════════════════╝${RST}

  Avvio rapido:
    ${W}$PYTHON hand_gesture_control.py${RST}
    oppure: doppio clic su '${W}Avvia HandGestureControl.command${RST}'
"

read -rp "  Avviare adesso l'applicazione? [S/n] " ans
if [[ ! "$ans" =~ ^[Nn]$ ]]; then
    echo -e "\n  ${C}Avvio Hand Gesture Control...${RST}"
    "$PYTHON" "$APP" &
fi
