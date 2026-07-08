#!/usr/bin/env python3
"""
Hand Gesture Control System - Auto Installer
============================================
Esegui questo script una sola volta per installare tutte le dipendenze
e avviare automaticamente l'applicazione.

  python install.py          # installa + lancia
  python install.py --only   # solo installazione, non avvia

Requisiti minimi: Python 3.8+
"""

import subprocess
import sys
import os
import platform
import shutil
import time
import argparse

# ──────────────────────────────────────────────────────────────
#  ANSI colors (disabled automatically on Windows senza supporto)
# ──────────────────────────────────────────────────────────────
if platform.system() == "Windows":
    os.system("color")   # abilita VT100 su Windows 10+

R  = "\033[91m"
G  = "\033[92m"
Y  = "\033[93m"
B  = "\033[94m"
C  = "\033[96m"
W  = "\033[97m"
DIM = "\033[2m"
BOLD= "\033[1m"
RST = "\033[0m"


def banner():
    print(f"""
{C}{BOLD}╔══════════════════════════════════════════════════════════╗
║         Hand Gesture Control System  —  Installer        ║
║         Controllo schermo con mani e dita  (webcam)       ║
╚══════════════════════════════════════════════════════════╝{RST}
""")


def step(msg):
    print(f"  {B}→{RST}  {msg}")


def ok(msg=""):
    print(f"  {G}✓{RST}  {msg}" if msg else f"  {G}✓{RST}")


def warn(msg):
    print(f"  {Y}⚠{RST}  {msg}")


def err(msg):
    print(f"  {R}✗{RST}  {msg}")


# ──────────────────────────────────────────────────────────────
#  PYTHON VERSION CHECK
# ──────────────────────────────────────────────────────────────
def check_python():
    step("Verifica versione Python …")
    major, minor = sys.version_info[:2]
    if major < 3 or (major == 3 and minor < 8):
        err(f"Python {major}.{minor} non supportato. Richiede Python 3.8+")
        sys.exit(1)
    ok(f"Python {major}.{minor} — OK")


# ──────────────────────────────────────────────────────────────
#  PIP UPGRADE
# ──────────────────────────────────────────────────────────────
def upgrade_pip():
    step("Aggiornamento pip …")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
            check=True, capture_output=True,
        )
        ok("pip aggiornato")
    except subprocess.CalledProcessError:
        warn("Impossibile aggiornare pip — continuo con la versione corrente")


# ──────────────────────────────────────────────────────────────
#  PACKAGE LIST
# ──────────────────────────────────────────────────────────────
CORE_PACKAGES = [
    ("opencv-python",      "4.8.0"),
    ("mediapipe",          "0.10.0,<0.10.14"),  # 0.10.14+ rimuove mp.solutions
    ("pyautogui",          "0.9.54"),
    ("numpy",              "1.24.0"),
    ("Pillow",             "10.0.0"),
    ("pynput",             "1.7.6"),
    ("SpeechRecognition",  "3.10.0"),
    ("anthropic",          "0.40.0"),
]

# Extra per Linux: la libreria X11 Python (pynput richiede)
LINUX_EXTRAS = [
    ("python3-xlib",    None),   # None = nessun minimo, prende latest
]


def _install(pkg_name: str, min_ver: str | None):
    spec = f"{pkg_name}>={min_ver}" if min_ver else pkg_name
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--upgrade", spec],
        check=True, capture_output=True,
    )


def _install_pyaudio():
    """PyAudio richiede portaudio di sistema; gestione per piattaforma."""
    system = platform.system()
    step(f"Installazione  {W}PyAudio{RST} (microfono per controllo vocale) …")
    if system == "Linux":
        # Prova prima a installare portaudio via apt, poi pip
        has_apt = shutil.which("apt-get") is not None
        if has_apt:
            try:
                subprocess.run(
                    ["sudo", "apt-get", "install", "-y", "portaudio19-dev"],
                    check=True, capture_output=True,
                )
            except subprocess.CalledProcessError:
                warn("portaudio19-dev non installabile automaticamente.\n"
                     "     Esegui manualmente:  sudo apt install portaudio19-dev")
    elif system == "Darwin":
        has_brew = shutil.which("brew") is not None
        if has_brew:
            try:
                subprocess.run(["brew", "install", "portaudio"],
                               check=True, capture_output=True)
            except subprocess.CalledProcessError:
                warn("portaudio non installabile via brew.\n"
                     "     Installa manualmente: brew install portaudio")
        else:
            warn("Homebrew non trovato. Installa portaudio:\n"
                 "     brew install portaudio")
    try:
        _install("pyaudio", None)
        ok("PyAudio")
    except subprocess.CalledProcessError:
        warn("PyAudio non installato — il controllo vocale non sarà disponibile.\n"
             "     Linux:   sudo apt install portaudio19-dev && pip install pyaudio\n"
             "     macOS:   brew install portaudio && pip install pyaudio\n"
             "     Windows: pip install pyaudio")


def install_packages():
    system = platform.system()
    packages = list(CORE_PACKAGES)
    if system == "Linux":
        packages += LINUX_EXTRAS

    print(f"\n  Piattaforma rilevata: {BOLD}{system}{RST}\n")

    failed = []
    for pkg, ver in packages:
        label = f"{pkg}>={ver}" if ver else pkg
        step(f"Installazione  {W}{label}{RST} …")
        try:
            _install(pkg, ver)
            ok(f"{pkg}")
        except subprocess.CalledProcessError as exc:
            # python3-xlib su Debian/Ubuntu può non essere disponibile via pip
            if pkg == "python3-xlib":
                warn(f"{pkg} non trovato via pip — provo python-xlib …")
                try:
                    _install("python-xlib", None)
                    ok("python-xlib (alternativa)")
                except subprocess.CalledProcessError:
                    warn("python-xlib non installabile via pip.\n"
                         "     Su Debian/Ubuntu esegui:\n"
                         "       sudo apt install python3-xlib")
            else:
                err(f"Impossibile installare {pkg}: {exc.stderr.decode()[:200]}")
                failed.append(pkg)

    # PyAudio ha dipendenze di sistema — gestione separata
    _install_pyaudio()

    if failed:
        err(f"\nPacchetti non installati: {', '.join(failed)}")
        warn("L'applicazione potrebbe non funzionare correttamente.")
    else:
        print(f"\n  {G}{BOLD}Tutti i pacchetti installati con successo!{RST}\n")


# ──────────────────────────────────────────────────────────────
#  SYSTEM DEPENDENCY HINTS
# ──────────────────────────────────────────────────────────────
def system_hints():
    system = platform.system()
    if system == "Linux":
        print(f"""  {DIM}Nota Linux: se l'applicazione mostra errori di display assicurati che
  tkinter sia installato:
    sudo apt install python3-tk   (Debian/Ubuntu)
    sudo dnf install python3-tkinter  (Fedora)

  Per il controllo del mouse senza permessi speciali su Wayland
  potrebbe essere necessario usare una sessione X11.

  Per il controllo vocale (microfono):
    sudo apt install portaudio19-dev python3-pyaudio{RST}
""")
    elif system == "Darwin":
        print(f"""  {DIM}Nota macOS: se Accessibility è bloccato vai in:
  Preferenze di Sistema → Privacy e Sicurezza → Accessibilità
  e aggiungi il tuo Terminale / Python.{RST}
""")
    elif system == "Windows":
        print(f"""  {DIM}Nota Windows: se il riconoscimento non funziona prova ad avviare
  il terminale come Amministratore.{RST}
""")


# ──────────────────────────────────────────────────────────────
#  SHORTCUT (opzionale)
# ──────────────────────────────────────────────────────────────
def create_shortcut():
    system = platform.system()
    app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "hand_gesture_control.py")
    if system == "Linux":
        desktop_dir = os.path.expanduser("~/Desktop")
        if os.path.isdir(desktop_dir):
            shortcut = os.path.join(desktop_dir, "HandGestureControl.desktop")
            content = f"""[Desktop Entry]
Type=Application
Name=Hand Gesture Control
Exec={sys.executable} {app_path}
Icon=input-gaming
Terminal=false
Categories=Utility;
"""
            try:
                with open(shortcut, "w", encoding="utf-8") as f:
                    f.write(content)
                os.chmod(shortcut, 0o755)
                ok("Collegamento sul Desktop creato (Linux)")
            except Exception:
                pass

    elif system == "Windows":
        try:
            # Sonda: winreg esiste solo su Windows
            import winreg  # noqa: F401  # pylint: disable=unused-import
            bat = os.path.join(os.path.dirname(app_path), "avvia.bat")
            with open(bat, "w", encoding="utf-8") as f:
                f.write(f'@echo off\n"{sys.executable}" "{app_path}"\n')
            desktop = os.path.join(os.path.expanduser("~"), "Desktop")
            dst = os.path.join(desktop, "HandGestureControl.bat")
            shutil.copy(bat, dst)
            ok("Collegamento sul Desktop creato (Windows)")
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────
#  LAUNCH
# ──────────────────────────────────────────────────────────────
def launch_app():
    app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "hand_gesture_control.py")
    if not os.path.isfile(app_path):
        err(f"File non trovato: {app_path}")
        sys.exit(1)

    print(f"\n  {G}{BOLD}Avvio Hand Gesture Control…{RST}\n")
    time.sleep(0.6)
    subprocess.Popen([sys.executable, app_path])  # pylint: disable=consider-using-with


# ──────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Auto-installer per Hand Gesture Control System"
    )
    parser.add_argument("--only", action="store_true",
                        help="Solo installazione, non avviare l'app")
    parser.add_argument("--no-shortcut", action="store_true",
                        help="Non creare collegamento sul Desktop")
    args = parser.parse_args()

    banner()
    check_python()
    upgrade_pip()
    install_packages()
    system_hints()

    if not args.no_shortcut:
        create_shortcut()

    if not args.only:
        launch_app()
    else:
        print(f"  {C}Installazione completata.{RST}")
        print(f"  Avvia manualmente con:\n    {W}python hand_gesture_control.py{RST}\n")


if __name__ == "__main__":
    main()
