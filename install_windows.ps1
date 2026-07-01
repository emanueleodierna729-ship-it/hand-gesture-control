# Hand Gesture Control — Windows 11 Installer
# Esegui con: PowerShell -ExecutionPolicy Bypass -File install_windows.ps1

$Host.UI.RawUI.WindowTitle = "Hand Gesture Control — Installer"
$ErrorActionPreference = "Stop"

function Write-Step  { param($msg) Write-Host "  -> $msg" -ForegroundColor Cyan }
function Write-OK    { param($msg) Write-Host "  OK  $msg" -ForegroundColor Green }
function Write-Warn  { param($msg) Write-Host "  !!  $msg" -ForegroundColor Yellow }
function Write-Fail  { param($msg) Write-Host "  XX  $msg" -ForegroundColor Red }

Clear-Host
Write-Host @"

  ╔══════════════════════════════════════════════════════════╗
  ║      Hand Gesture Control  v2  —  Windows Installer      ║
  ║      Creato da Emanuele Odierna insieme a Claude         ║
  ╚══════════════════════════════════════════════════════════╝

"@ -ForegroundColor Cyan

# 1. Verifica Python
Write-Step "Verifica Python 3.8+..."
try {
    $pyver = python --version 2>&1
    if ($pyver -match "Python (\d+)\.(\d+)") {
        $maj = [int]$Matches[1]; $min = [int]$Matches[2]
        if ($maj -lt 3 -or ($maj -eq 3 -and $min -lt 8)) {
            Write-Fail "Python $maj.$min non supportato. Richiede Python 3.8+"
            Write-Host "  Scarica da: https://www.python.org/downloads/" -ForegroundColor Yellow
            Read-Host "Premi INVIO per uscire"; exit 1
        }
        Write-OK "Python $maj.$min trovato"
    }
} catch {
    Write-Fail "Python non trovato nel PATH."
    Write-Host "  Scarica da: https://www.python.org/downloads/" -ForegroundColor Yellow
    Write-Warn "Assicurati di spuntare 'Add Python to PATH' durante l'installazione."
    Read-Host "Premi INVIO per uscire"; exit 1
}

# 2. Aggiorna pip
Write-Step "Aggiornamento pip..."
python -m pip install --upgrade pip --quiet
Write-OK "pip aggiornato"

# 3. Installa dipendenze
$packages = @(
    "opencv-python>=4.8.0",
    "mediapipe>=0.10.0,<0.10.14",
    "pyautogui>=0.9.54",
    "numpy>=1.24.0",
    "Pillow>=10.0.0",
    "pynput>=1.7.6",
    "SpeechRecognition>=3.10.0",
    "anthropic>=0.40.0"
)

foreach ($pkg in $packages) {
    $name = ($pkg -split ">=|,")[0]
    Write-Step "Installazione $name..."
    python -m pip install "$pkg" --quiet
    Write-OK $name
}

# 4. PyAudio (opzionale — per il controllo vocale)
Write-Step "Installazione PyAudio (controllo vocale)..."
try {
    python -m pip install pyaudio --quiet
    Write-OK "PyAudio"
} catch {
    Write-Warn "PyAudio non installabile automaticamente."
    Write-Warn "Per il controllo vocale: pip install pyaudio"
}

# 5. Crea collegamento sul Desktop
Write-Step "Creazione collegamento sul Desktop..."
try {
    $AppPath  = Join-Path $PSScriptRoot "hand_gesture_control.py"
    $Desktop  = [Environment]::GetFolderPath("Desktop")
    $ShortLnk = Join-Path $Desktop "Hand Gesture Control.lnk"
    $WScript  = New-Object -ComObject WScript.Shell
    $sc = $WScript.CreateShortcut($ShortLnk)
    $sc.TargetPath       = "python"
    $sc.Arguments        = "`"$AppPath`""
    $sc.WorkingDirectory = $PSScriptRoot
    $sc.IconLocation     = "shell32.dll,45"
    $sc.Description      = "Hand Gesture Control v2 — Controllo PC con gesti"
    $sc.Save()
    Write-OK "Collegamento creato: $ShortLnk"
} catch {
    Write-Warn "Impossibile creare collegamento: $_"
}

# 6. Crea avvia.bat
$BatPath = Join-Path $PSScriptRoot "avvia.bat"
@"
@echo off
title Hand Gesture Control v2
python "%~dp0hand_gesture_control.py"
if errorlevel 1 pause
"@ | Out-File -FilePath $BatPath -Encoding ASCII
Write-OK "File avvia.bat creato"

Write-Host @"

  ╔══════════════════════════════════════════════════════════╗
  ║         Installazione completata con successo!           ║
  ╚══════════════════════════════════════════════════════════╝

  Avvio rapido:
    - Doppio clic su 'Hand Gesture Control' sul Desktop
    - Oppure: python hand_gesture_control.py

  Note Windows 11:
    - Se il mouse non risponde, avvia come Amministratore
    - Webcam: assicurati che non sia in uso da altre app

"@ -ForegroundColor Green

$launch = Read-Host "Avviare adesso l'applicazione? [S/n]"
if ($launch -ne "n" -and $launch -ne "N") {
    Write-Host "`n  Avvio Hand Gesture Control..." -ForegroundColor Cyan
    Start-Process python -ArgumentList "`"$(Join-Path $PSScriptRoot 'hand_gesture_control.py')`""
}
