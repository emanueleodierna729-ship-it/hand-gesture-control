# Hand Gesture Control System ✋  v2

Applicazione Python che trasforma la webcam in un controller gestuale completo.
Supporta **due mani** con riconoscimento fluido grazie a:
- **LandmarkSmoother** — EMA sui 21 punti landmark grezzi per ogni mano
- **GestureStabiliser** — votazione temporale su una finestra di N frame per eliminare falsi positivi in transizione
- **VelocityTracker** — rilevamento swipe basato su velocità del polso

---

## Avvio rapido

```bash
python install.py          # installa dipendenze + avvia
python install.py --only   # solo installazione
python hand_gesture_control.py  # avvio diretto
```

---

## Requisiti

| Componente | Versione minima |
|---|---|
| Python | 3.8+ |
| Webcam | qualsiasi USB / integrata |
| OS | Windows 10+, macOS 12+, Linux (X11) |

**Linux:** `sudo apt install python3-tk python3-xlib`  
**macOS:** Preferenze → Privacy → Accessibilità → aggiungi Terminale  
**Windows:** avviare come Amministratore se il mouse non risponde

---

## Ruolo delle due mani

| Mano | Ruolo |
|---|---|
| **Destra** (wrist.x > 0.5) | Mano dominante — cursore e azioni |
| **Sinistra** (wrist.x < 0.5) | Mano modificatore — cambia modo operativo |

---

## Gesti — Mano Dominante (destra)

| Gesto | Azione |
|---|---|
| ☝ Solo indice | Muovi cursore |
| 🤏 Pinch pollice+indice | Click sinistro |
| 🤏 Pinch + movimento | Drag (trascina) |
| 🤏 Pinch pollice+medio | Click destro |
| ✌ Due dita (indice+medio) | Scroll verticale |
| 🤌 3 dita su | Copia `Ctrl+C` |
| ✋ 4 dita su | Incolla `Ctrl+V` |
| 👍 Solo pollice | Doppio click |
| 🤘 Rock (indice+mignolo) | Annulla `Ctrl+Z` |
| 🤙 Pollice+mignolo | Salva `Ctrl+S` |
| 🖐 Palmo aperto + velocità | Swipe ← → (`Alt+←/→`) |

---

## Gesti — Mano Modificatore (sinistra)

Tenere il gesto con la mano sinistra attiva una **modalità** che cambia il comportamento della mano destra:

| Gesto sinistra | Modalità | Effetto sulla mano destra |
|---|---|---|
| 🖐 Palmo aperto | **FREEZE** | Cursore congelato (precisione) |
| ✊ Pugno | **ZOOM MODE** | Scroll verticale destra → zoom |
| ✌ Due dita | **H-SCROLL** | Cursore destra → scroll orizzontale |
| 🤘 Rock | **ALT+TAB** | Scatta `Alt+Tab` immediatamente |
| 👍 Pollice su | **MIDDLE** | Prossimo click = click centrale |

---

## Gesti a Due Mani

| Gesto | Azione |
|---|---|
| 🤏+🤏 Entrambe le mani in pinch, allontanare | Zoom In `Ctrl+scroll+` |
| 🤏+🤏 Entrambe le mani in pinch, avvicinare | Zoom Out `Ctrl+scroll-` |

---

## Pipeline tecnica

```
Webcam frame (640×480)
   │ cv2.flip
   ↓
HandTracker (MediaPipe, max 2 mani)
   │ 21 landmark × 2 mani
   ↓
LandmarkSmoother (EMA α=0.40 per mano)
   │ landmark levigati
   ↓
GestureRecogniser (classificazione per-frame, per mano)
   │ gesto raw
   ↓
GestureStabiliser (vote window=6, thresh=0.60 per mano)
   │ gesto stabile
   ↓
VelocityTracker (velocità polso per swipe)
   ↓
DualHandProcessor
   ├── _assign_roles()    — DOM = wrist.x > 0.5
   ├── _two_hands()       — zoom pinch
   ├── _update_mod()      — modifier mode da mano sinistra
   └── _dominant()        — azioni mouse/tastiera
   ↓
SmoothMouse (EMA cursore α=0.28)
```

---

## Dipendenze

```
opencv-python >= 4.8
mediapipe     >= 0.10
pyautogui     >= 0.9.54
numpy         >= 1.24
Pillow        >= 10.0
pynput        >= 1.7.6
```
