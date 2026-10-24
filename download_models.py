"""
下载 CosyVoice 2.0 推理所需的模型权重。

默认使用 ModelScope 镜像（国内访问更稳定），下载到：
    pretrained_models/CosyVoice2-0.5B

用法：
    python download_models.py

环境变量：
    MODEL_ID   模型 ID，默认 iic/CosyVoice2-0.5B
    LOCAL_DIR  本地保存目录，默认 pretrained_models/CosyVoice2-0.5B
"""

from __future__ import annotations

import os
import sys

MODEL_ID = os.getenv("MODEL_ID", "iic/CosyVoice2-0.5B")
LOCAL_DIR = os.getenv("LOCAL_DIR", "pretrained_models/CosyVoice2-0.5B")


def download() -> None:
    try:
        from modelscope import snapshot_download
    except ImportError:
        print("错误：未安装 modelscope。请先执行：uv pip install modelscope 或 pip install modelscope")
        sys.exit(1)

    print(f"正在从 ModelScope 下载 {MODEL_ID} ...")
    print(f"保存位置：{LOCAL_DIR}")

    snapshot_download(
        MODEL_ID,
        local_dir=LOCAL_DIR,
    )
    print("模型下载完成。")
    print(f"启动服务：python app.py --engine cosyvoice --model-dir {LOCAL_DIR}")


if __name__ == "__main__":
    download()
