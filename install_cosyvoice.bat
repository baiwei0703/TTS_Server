@echo off
cd /d "%~dp0"
setlocal enabledelayedexpansion

where uv >nul 2>nul
if errorlevel 1 (
    echo [ERROR] uv not found. Please install uv first:
    echo    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    pause
    exit /b 1
)

set "VENV=.venv-cosyvoice"
set "ACTIVATE=%VENV%\Scripts\activate.bat"

echo [1/6] Creating Python 3.10.12 virtual environment %VENV%...
uv venv --clear --python 3.10.12 %VENV%
if errorlevel 1 (
    echo Failed to create virtual environment
    pause
    exit /b 1
)

call "%ACTIVATE%"

echo [2/6] Installing PyTorch 2.3.1 + CUDA 12.1 (Aliyun mirror)...
uv pip install --index-url https://mirrors.aliyun.com/pypi/simple/ ^
  --extra-index-url https://download.pytorch.org/whl/cu121 ^
  torch==2.3.1+cu121 torchvision==0.18.1+cu121 torchaudio==2.3.1+cu121
if errorlevel 1 (
    echo PyTorch installation failed
    pause
    exit /b 1
)

echo [3/6] Cloning CosyVoice source code (using gitclone.com mirror)...
if not exist "third_party\CosyVoice" (
    git clone --recursive https://gitclone.com/github.com/FunAudioLLM/CosyVoice.git third_party\CosyVoice
    if errorlevel 1 (
        echo Failed to clone CosyVoice
        pause
        exit /b 1
    )
    if not exist "third_party\CosyVoice\third_party\Matcha-TTS\.git" (
        rmdir /s /q third_party\CosyVoice\third_party\Matcha-TTS 2>nul
        git clone https://gitclone.com/github.com/shivammehta25/Matcha-TTS.git third_party\CosyVoice\third_party\Matcha-TTS
        if errorlevel 1 (
            echo Failed to clone Matcha-TTS submodule
            pause
            exit /b 1
        )
    )
) else (
    echo third_party\CosyVoice already exists, skipping clone
)

echo [4/6] Installing CosyVoice dependencies (Windows compatible, no wetext/pynini)...
echo        Installing setuptools first to fix openai-whisper build...
uv pip install "setuptools<70"
uv pip install --no-build-isolation --index-strategy unsafe-best-match -r requirements-cosyvoice.txt
if errorlevel 1 (
    echo Failed to install CosyVoice dependencies
    pause
    exit /b 1
)

echo [5/6] Installing project dependencies...
uv pip install -e .
if errorlevel 1 (
    echo Failed to install project dependencies
    pause
    exit /b 1
)

echo [6/6] Downloading CosyVoice2-0.5B model...
%VENV%\Scripts\python.exe download_models.py
if errorlevel 1 (
    echo Model download failed
    pause
    exit /b 1
)

echo.
echo Installation complete! Run run.bat to start the CosyVoice server.
pause
