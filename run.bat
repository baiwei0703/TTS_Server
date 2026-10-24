@echo off
cd /d "%~dp0"
setlocal enabledelayedexpansion

set "VENV=.venv-cosyvoice"
set "ACTIVATE=%VENV%\Scripts\activate.bat"

if not exist "%ACTIVATE%" (
    echo [ERROR] Virtual environment "%VENV%" not found. Please run install_cosyvoice.bat first.
    pause
    exit /b 1
)

echo [1/2] Activating CosyVoice virtual environment...
call "%ACTIVATE%"
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment.
    pause
    exit /b 1
)

echo [2/2] Starting CosyVoice TTS server (press Ctrl+C to stop)...
echo.

set "PYTHONPATH=%CD%\third_party\CosyVoice;%CD%\third_party\CosyVoice\third_party\Matcha-TTS;%PYTHONPATH%"

python app.py --engine cosyvoice --device auto --host 0.0.0.0 --port 8000 %*

echo.
echo [INFO] Server exited with code: %errorlevel%
pause
