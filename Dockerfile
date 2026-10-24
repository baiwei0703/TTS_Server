# CosyVoice TTS Server - Docker (GPU)
# Requires Docker Desktop with WSL2 backend and NVIDIA Container Toolkit.

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Use Tsinghua PyPI mirror + PyTorch CUDA 12.1 index
ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ENV PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu121

WORKDIR /app

# Use Aliyun APT mirror (replace default Ubuntu sources) for faster/reliable builds in China
RUN sed -i \
    -e 's|http://archive.ubuntu.com/ubuntu/|https://mirrors.aliyun.com/ubuntu/|g' \
    -e 's|http://security.ubuntu.com/ubuntu/|https://mirrors.aliyun.com/ubuntu/|g' \
    /etc/apt/sources.list

# Install system dependencies
RUN apt-get update -o Acquire::Retries=5 -o Acquire::http::Timeout=120 && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Upgrade pip and install setuptools that still includes pkg_resources
RUN python -m pip install --no-cache-dir --upgrade pip "setuptools<70"

# Install PyTorch CUDA 12.1
# Install torch first using the Tsinghua mirror (it usually caches the big torch wheel)
RUN python -m pip install --no-cache-dir --default-timeout=1000 torch==2.3.1+cu121
# Install torchvision/torchaudio from the PyTorch CUDA 12.1 index
RUN python -m pip install --no-cache-dir --default-timeout=1000 \
    --index-url https://download.pytorch.org/whl/cu121 \
    --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    torchvision==0.18.1+cu121 torchaudio==2.3.1+cu121

# Install pynini (Linux wheel available) so wetext can be installed
RUN python -m pip install --no-cache-dir pynini==2.1.6

# Note: CosyVoice source code is mounted from host at runtime:
# ./third_party/CosyVoice -> /app/third_party/CosyVoice

# Install CosyVoice runtime dependencies (Docker-optimized: no deepspeed, no CUDA toolkit required)
COPY requirements-cosyvoice-docker.txt /app/requirements-cosyvoice-docker.txt
RUN python -m pip install --no-cache-dir --default-timeout=1000 --no-build-isolation -r /app/requirements-cosyvoice-docker.txt

# Install text frontend for Linux (wetext needs pynini)
RUN python -m pip install --no-cache-dir wetext==0.0.4

# Copy project files
COPY . /app/

# Install the project itself (web server entrypoints)
RUN python -m pip install --no-cache-dir -e .

EXPOSE 8000

CMD ["python", "app.py", "--engine", "cosyvoice", "--model-dir", "/app/pretrained_models/CosyVoice2-0.5B", "--device", "auto", "--host", "0.0.0.0", "--port", "8000"]
