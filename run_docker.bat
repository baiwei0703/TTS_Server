@echo off
cd /d "%~dp0"
echo Building and starting CosyVoice TTS server in Docker...
echo Build log will be saved to docker-build.log
echo.

docker compose down >nul 2>&1

docker compose build --no-cache > docker-build.log 2>&1
if errorlevel 1 (
    echo.
    echo [ERROR] Docker build failed. See docker-build.log for details.
    echo.
    pause
    exit /b 1
)

docker compose up -d
echo.
echo Service started. Open http://localhost:8000
echo To view logs: docker compose logs -f
pause
