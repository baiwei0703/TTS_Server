@echo off
cd /d "%~dp0"
setlocal enabledelayedexpansion

set "VENV=.venv"
set "ACTIVATE=%VENV%\Scripts\activate.bat"

if not exist "%ACTIVATE%" (
    echo [ERROR] Virtual environment "%VENV%" not found. Please run install.bat or install_gpu.bat first.
    pause
    exit /b 1
)

call "%ACTIVATE%"
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment.
    pause
    exit /b 1
)

python app.py --engine fishspeech --device auto --host 0.0.0.0 --port 8000 %*

echo.
echo [INFO] Server exited with code: %errorlevel%
pause
