# Hand Gesture Control v2 — Certificazione Completa

**Data**: 2026-05-30  
**Branch**: `claude/hand-gesture-control-DkZ5i`  
**Commit HEAD**: bcd7a66  
**Eseguito su**: Python 3.12 / Xvfb :99 (1920×1080) / Mediapipe 0.10.13

---

## Verdetto: PASS ✅

---

## Struttura del Branch

| Commit | Descrizione |
|--------|-------------|
| `bcd7a66` | chore: report test 100× (7200/7200 passed) |
| `58e41d4` | fix: 4 bug critici (maniacal review) |
| `c42fda9` | chore: report test precedente |
| `8b47871` | feat: AI gesti autonoma, apprendimento utente, mouse sempre attivo |
| `365d9f9` | feat: UI ridimensionabile, controllo vocale italiano |
| `3a9224b` | feat: dual-hand support, gesture stabilization |

---

## Moduli Certificati

| Modulo | Classe | Stato | Note |
|--------|--------|-------|------|
| Configurazione | `Cfg` | ✅ PASS | `from __future__ import annotations` aggiunto (Py 3.8/3.9 compat) |
| Smoothing landmark | `LandmarkSmoother` | ✅ PASS | EMA α=0.40, test convergenza+reset |
| Stabilizzazione gesti | `GestureStabiliser` | ✅ PASS | Majority-vote sliding window 6 frame |
| Velocità swipe | `VelocityTracker` | ✅ PASS | dx/dt con deque fisso |
| Tracking mani | `HandTracker` | ✅ PASS (runtime) | Webcam assente → CameraThread cicla senza crash |
| Riconoscimento gesti | `GestureRecogniser` | ✅ PASS | Regole fisse 12 gesti |
| AI gesti custom | `CustomGestureRecogniser` | ✅ PASS | k-NN K=3 majority-vote, fallback a regole |
| Database gesti | `GestureDatabase` | ✅ PASS | JSON persistence, load/save/remove |
| Registratore gesti | `GestureRecorder` | ✅ PASS | State machine IDLE→COUNTDOWN→RECORDING→DONE |
| Mouse fluido | `SmoothMouse` | ✅ PASS | EMA + pynput + pyautogui fallback |
| Processore dual-hand | `DualHandProcessor` | ✅ PASS | Dominant/modifier, zoom bimanuale |
| Parser comandi | `CommandParser` | ✅ PASS | 30 pattern regex italiani |
| Controllo vocale | `VoiceController` | ✅ PASS (struttura) | Thread daemon, SpeechRecognition opzionale |
| Tastiera virtuale | `VirtualKeyboard` | ✅ PASS | Toplevel Tkinter |
| Thread camera | `CameraThread` | ✅ PASS | Graceful failure senza webcam |
| Dashboard UI | `Dashboard` | ✅ PASS | 4 tab, resizable, attivo di default |

---

## Test Suite Automatizzata

```
python3 test_gesture_control.py --x100

✓ PASS  7200/7200 passed  (100.0%)  in 1.34s
```

**72 test × 100 run** — zero fallimenti su tutte le esecuzioni.

### Classi di test

| Classe | Test | Copre |
|--------|------|-------|
| `TestCommandParser` | 30 | Tutti i pattern regex italiani |
| `TestLandmarkSmoother` | 4 | Convergenza EMA, reset, multi-key |
| `TestGestureStabiliser` | 5 | Majority-vote, transizioni |
| `TestVelocityTracker` | 4 | Velocità, direzione, reset |
| `TestGestureRecogniser` | 8 | OPEN_PALM, FIST, CURSOR, SCROLL, PINCH, fingers_up |
| `TestSmoothMouseCoords` | 6 | _n2s mirror X, clamping, margini |
| `TestGestureDatabase` | 8 | CRUD, persistence JSON, errori load |
| `TestCustomGestureRecogniser` | 6 | k-NN match/miss, fallback regole, feature vector 20D |

---

## Verifica Runtime (Xvfb)

App lanciata con `python3.12 hand_gesture_control.py` sotto `DISPLAY=:99`.

### Screenshot acquisiti

| Screenshot | Evidenza |
|-----------|---------|
| `hgc_startup.png` | **● ATTIVO** + **⏹ DISATTIVA** visibili all'avvio — hand tracking attivo di default ✅ |
| `hgc_v3_voce.png` | Tab Voce: CONTROLLO VOCALE, Status ● OFFLINE, ATTIVA VOCE, esempi comandi ✅ |
| `hgc_v3_apprendi.png` | Tab Guida: tutti i 17 gesti listati con descrizione ✅ |
| `hgc_apprendi_final.png` | Tab Apprendi: form Nome/Azione/Arg, REGISTRA button, GESTI APPRESI listbox ✅ |

### Comportamenti osservati

- ✅ **Avvio**: header mostra `● ATTIVO` (verde), bottone `⏹ DISATTIVA` (rosso) — fix startup confermato visivamente
- ✅ **4 tab presenti**: Mani / Voce / Guida / Apprendi — tutte navigabili
- ✅ **Tab Voce**: riconosce assenza di SpeechRecognition e mostra messaggio `python install.py` (graceful degradation)
- ✅ **Tab Guida**: guida gesti completa con 11 gesti dominante + 5 modificatori + 1 bimanuale
- ✅ **Tab Apprendi**: form registrazione funzionante, listbox gesti appresi vuota (nessun gesto pre-caricato), bottone Elimina
- ✅ **Camera assente**: CameraThread gestisce `cap.read() = False` con `time.sleep(0.01)` senza crash — UI rimane funzionale
- ✅ **Resizable**: pannello camera si espande con la finestra (left_panel fill="both", expand=True)
- ⚠️ **Voce OFFLINE**: SpeechRecognition non installato nell'ambiente di test — comportamento atteso e gestito

---

## Bug Corretti (Revisione Maniacale)

| # | Posizione | Bug | Fix |
|---|-----------|-----|-----|
| 1 | `import` section | `str\|None`, `list[T]` → TypeError su Python 3.8/3.9 | `from __future__ import annotations` |
| 2 | `_exec_custom()` L.722 | `"ctrl+c"` passato come stringa unica a `hotkey()` | Split su `"+"` → `["ctrl","c"]` |
| 3 | `_knn()` L.1298 | 1-NN: variabile `K=3` definita ma mai usata | k-NN reale: sort candidati, majority-vote K vicini |
| 4 | `_loop()` L.1785 | Plurale IT: `mano{'i'...}` → `"manoi"` per n≠1 | `man{'i' if n!=1 else 'o'}`; n=0 → `""` |

---

## Compatibilità Dipendenze

| Libreria | Versione testata | Note |
|----------|-----------------|------|
| Python | 3.12 (app) / 3.11 (stub tests) | Tkinter disponibile su 3.12 |
| OpenCV | 4.13.0 | Headless (no webcam) |
| MediaPipe | 0.10.13 | `mp.solutions` disponibile (rimosso in 0.10.35) |
| PyAutoGUI | 0.9.54 | DISPLAY richiesto |
| Pillow | 12.2.0 | Image/ImageTk ✓ |
| pynput | 1.8.2 | X11 backend |
| NumPy | 2.4.6 | ✓ |

> **Nota installazione**: `mediapipe>=0.10.0` in `install.py` può risolvere a 0.10.35 che ha rimosso `mp.solutions`. Fissare a `mediapipe>=0.10.0,<0.10.14` o usare le nuove Tasks API.

---

*Certificazione generata automaticamente da Claude Code — sessione https://claude.ai/code/session_01BUNwKydE8YL9DeRXcWwTMJ*

---

## Aggiornamento — 2026-09-09

**Branch**: `claude/hand-control-improvement-muu918` (PR #9)
**Sessione**: https://claude.ai/code/session_01LQXf4xaVNi3YUQpLrs89HL

### Modifiche

| Area | Cosa è cambiato |
|------|------------------|
| `GestureRecogniser` | Isteresi (Schmitt trigger) sui pinch indice/medio/mignolo — soglia di rilascio 25% più ampia di quella di innesco (`Cfg.PINCH_RELEASE_RATIO`), stato tracciato per mano (`hand_key`), per eliminare lo sfarfallio del gesto vicino al bordo di soglia |
| Logo | `assets/logo.svg`, `logo_192.png`, `logo_512.png` rinnovati — palette duotone viola→ciano, sfondo con gradiente più profondo, proporzioni rifinite per leggibilità a 192px |
| Installer | `install_windows.ps1`: corretto un bug per cui `python -m pip install` falliva silenziosamente e lo script riportava comunque "OK" (mancava il controllo di `$LASTEXITCODE`) — riguardava l'aggiornamento di pip, ogni pacchetto core e PyAudio. `install_linux.sh` / `install_macos.sh`: `apt-get update` e `brew install portaudio` potevano interrompere l'intero installer (`set -e`) per un singolo comando di sistema fallito; ora sono tollerati con avviso, e il ciclo di installazione pacchetti pip riporta i fallimenti invece di uscire di colpo — comportamento allineato a quello già presente in `install.py` |

### Test

```
python3 -m unittest test_gesture_control      → 109/109 PASS  (+6 su TestPinchHysteresis)
python3 test_gesture_control.py --x100        → 10900/10900 PASS
bash -n install_linux.sh / install_macos.sh   → sintassi OK
```

Nessun test manuale con webcam reale in questa sessione (ambiente headless, senza hardware). `install_windows.ps1` non verificabile in questo ambiente (nessun runtime PowerShell disponibile) — la correzione è stata fatta per lettura del codice, non per esecuzione.

---

## Aggiornamento — Revisione maniacale, 2026-09-10

Seconda passata di revisione mirata a trovare bug reali (non solo stile), sullo stesso branch/PR.

### Bug trovati e corretti

| Area | Bug | Fix |
|------|-----|-----|
| `DualHandProcessor._two_hands` (zoom a due mani) | Una riga di fallback residua poteva restituire l'etichetta `ZOOM_IN` anche quando `mouse.zoom()` era bloccato dal cooldown (`Cfg.ZOOM_CD`) o la distanza era ancora nella dead-zone (`Cfg.ZOOM_DEAD`) — l'azione mostrata non corrispondeva a quella realmente eseguita, con possibile sfarfallio dell'etichetta per rumore | Rimossa la riga; l'azione riportata ora coincide sempre con l'azione davvero eseguita |
| Dashboard, tab "Mani" | Lo slider "SENSIBILITÀ CURSORE" scriveva su `Cfg.SMOOTH`, parametro legacy non più letto da nessuna parte da quando il cursore usa il One-Euro Filter — muoverlo non aveva alcun effetto reale | Aggiunto `OneEuroFilter.set_beta()`; lo slider ora modifica live `Cfg.CURSOR_BETA` sui filtri del mouse |
| `GestureDatabase._load()` | Caricava qualunque JSON valido senza controllare che fosse un oggetto — un file `user_gestures.json` con JSON valido ma non a forma di dict (es. `[]`, scrittura parziale/corrotta) avrebbe fatto crashare l'app ad ogni frame (`CustomGestureRecogniser._knn()` chiama `self._db._d.items()` continuamente durante il tracking) | Validato il tipo dopo il parsing: se non è un dict, fallback a `{}` |
| `CameraThread.run()` | Variabile morta `frame_elapsed`/`t_frame_start` (calcolata e mai usata) | Rimossa |
| `test_gesture_control.py` | Due metodi di test (`test_fingers_up_all/none`) erano finiti nella classe sbagliata durante un refactor precedente in questa stessa sessione | Rispostati in `TestGestureRecogniser` |

### Test

```
python3 -m unittest test_gesture_control      → 113/113 PASS  (+3 TestTwoHandZoom, +1 TestGestureDatabase)
```

Ogni fix di correttezza è stato verificato riproducendo il bug sul codice precedente (il test dedicato fallisce lì, passa col fix) prima di essere considerato chiuso.
