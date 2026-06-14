# ✋ Hand Gesture Control v2

> Controlla il tuo PC con la **webcam** e i **gesti delle mani** — nessun hardware aggiuntivo.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://python.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-green?logo=google)](https://mediapipe.dev)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey)]()
[![License](https://img.shields.io/badge/License-MIT-orange)]()

---

## Installazione rapida

### Windows 11 Pro
```powershell
PowerShell -ExecutionPolicy Bypass -File install_windows.ps1
```

### macOS (12 Monterey – 15 Sequoia)
```bash
bash install_macos.sh
```

### Linux (Ubuntu / Debian / Fedora / Arch / openSUSE)
```bash
bash install_linux.sh
```

### Multipiattaforma (Python)
```bash
python install.py          # installa tutto + avvia
python install.py --only   # solo installazione
```

---

## Avvio diretto

```bash
python hand_gesture_control.py
```

---

## Requisiti

| Componente | Versione |
|---|---|
| Python | 3.8 o superiore |
| Webcam | USB o integrata |
| OS | Windows 10/11 · macOS 12+ · Linux (X11) |

---

## Gesti — Mano Dominante (destra)

| Gesto | Azione |
|---|---|
| ☝ Solo indice | Muovi cursore |
| 🤏 Pinch pollice+indice | Click sinistro |
| 🤏 Pinch + movimento | Drag |
| 🤏 Pinch pollice+medio | Click destro |
| ✌ Due dita | Scroll verticale |
| 🤌 3 dita su | Copia `Ctrl+C` |
| ✋ 4 dita su | Incolla `Ctrl+V` |
| 👍 Solo pollice | Doppio click |
| 🤘 Rock | Annulla `Ctrl+Z` |
| 🤙 Pollice+mignolo | Salva `Ctrl+S` |
| 🖐 Palmo + velocità | Swipe ← → (`Alt+←/→`) |

---

## Gesti — Mano Modificatore (sinistra)

| Gesto sinistra | Modalità | Effetto |
|---|---|---|
| 🖐 Palmo aperto | FREEZE | Cursore congelato |
| ✊ Pugno | ZOOM | Scroll → zoom |
| ✌ Due dita | H-SCROLL | Scroll orizzontale |
| 🤘 Rock | ALT+TAB | Cambia finestra |
| 👍 Pollice | MIDDLE | Click centrale |

---

## Gesti a Due Mani

| Gesto | Azione |
|---|---|
| 🤏+🤏 Allontanare | Zoom In `Ctrl+scroll+` |
| 🤏+🤏 Avvicinare | Zoom Out `Ctrl+scroll-` |

---

## Controllo Vocale

Parla in italiano per comandare il PC:

| Comando | Azione |
|---|---|
| "clicca" / "click" | Click sinistro |
| "apri" | Doppio click |
| "copia" / "incolla" | Ctrl+C / Ctrl+V |
| "annulla" | Ctrl+Z |
| "salva" | Ctrl+S |
| "scherma" | Screenshot |
| "pausa" / "riprendi" | Congela/sblocca cursore |

---

## Architettura

```
Webcam frame (640×480)
   ↓ cv2.flip
HandTracker (MediaPipe, max 2 mani)
   ↓ 21 landmark × 2 mani
LandmarkSmoother (EMA α=0.40 per mano)
   ↓ landmark levigati
GestureRecogniser (classificazione per-frame)
   ↓ gesto raw
GestureStabiliser (vote window=6, thresh=0.60)
   ↓ gesto stabile
DualHandProcessor
   ├── _assign_roles()    DOM = wrist.x > 0.5
   ├── _two_hands()       zoom pinch
   ├── _update_mod()      modifier mode
   └── _dominant()        azioni mouse/tastiera
   ↓
SmoothMouse (EMA cursore α=0.28)
```

---

## Dipendenze Python

```
opencv-python >= 4.8.0
mediapipe     >= 0.10.0, < 0.10.14
pyautogui     >= 0.9.54
numpy         >= 1.24.0
Pillow        >= 10.0.0
pynput        >= 1.7.6
SpeechRecognition >= 3.10.0
pyaudio       (opzionale — controllo vocale)
```

> **Nota:** mediapipe 0.10.14+ rimuove l'API `mp.solutions` — il cap `<0.10.14` è necessario.

---

## Note per piattaforma

**macOS** — Alla prima esecuzione, vai in:  
`Impostazioni di Sistema → Privacy e Sicurezza → Accessibilità` e aggiungi il tuo Terminale.

**Linux** — Richiede sessione X11 (non Wayland puro). Per il controllo mouse assicurati che l'utente sia nel gruppo `video`:
```bash
sudo usermod -aG video $USER
```

**Windows** — Se il mouse non risponde, avvia come Amministratore.

---

## Gesti Personalizzati

L'app include un **Recorder** integrato (tab "Apprendi"):
1. Scegli un nome per il gesto
2. Fai clic su "Registra" e tieni la mano ferma per 3 secondi
3. Il gesto viene salvato e riconosciuto in tempo reale

---

Creato da **Emanuele Odierna** insieme a **Claude** · Python + MediaPipe + Tkinter
