@echo off
chcp 65001 >nul
cd /d "%~dp0"

where uv >nul 2>nul
if errorlevel 1 (
    echo 未找到 uv，请先安装 uv：
    echo    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    pause
    exit /b 1
)

echo [1/4] 创建 Python 3.12 虚拟环境...
uv venv --python 3.12.12
if errorlevel 1 (
    echo 创建虚拟环境失败
    pause
    exit /b 1
)

echo [2/4] 安装 PyTorch CPU 版（使用阿里云镜像）...
uv pip install ^
  "https://mirrors.aliyun.com/pytorch-wheels/cpu/torch-2.5.1%%2Bcpu-cp312-cp312-win_amd64.whl" ^
  "https://mirrors.aliyun.com/pytorch-wheels/cpu/torchvision-0.20.1%%2Bcpu-cp312-cp312-win_amd64.whl" ^
  "https://mirrors.aliyun.com/pytorch-wheels/cpu/torchaudio-2.5.1%%2Bcpu-cp312-cp312-win_amd64.whl"
if errorlevel 1 (
    echo PyTorch 安装失败，请检查网络
    pause
    exit /b 1
)

echo [3/4] 安装 Fish Speech 与 Web 服务依赖...
uv pip install -e .
if errorlevel 1 (
    echo 依赖安装失败，请检查网络
    pause
    exit /b 1
)

echo [4/4] 下载模型权重...
.venv\Scripts\python.exe download_models.py
if errorlevel 1 (
    echo 模型下载失败
    pause
    exit /b 1
)

echo.
echo 安装完成！运行 run.bat 启动服务。
pause
