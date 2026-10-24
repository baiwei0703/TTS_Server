"""TTS 引擎抽象接口。"""

from __future__ import annotations

import argparse
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class TTSJobRequest:
    """统一的 TTS 请求对象，引擎可自行选择使用哪些字段。"""

    text: str
    reference_audio_bytes: bytes = b""
    reference_text: str = ""
    voice_id: str | None = None
    format: str = "wav"
    seed: int | None = None
    speed: float = 1.0
    text_frontend: bool = False

    # 以下字段仅 Fish Speech 使用，CosyVoice 会忽略但保留以兼容 API
    max_new_tokens: int = 1024
    chunk_length: int = 200
    top_p: float = 0.7
    repetition_penalty: float = 1.2
    temperature: float = 0.7
    normalize: bool = True
    use_memory_cache: str = "off"


class TTSEngine(ABC):
    """TTS 引擎抽象基类。"""

    @abstractmethod
    def load(self, args: argparse.Namespace) -> "TTSEngine":
        """根据命令行参数加载模型，返回 self。"""
        raise NotImplementedError

    @abstractmethod
    def tts(self, req: TTSJobRequest) -> tuple[np.ndarray, int]:
        """执行推理，返回 (audio_numpy, sample_rate)。"""
        raise NotImplementedError

    @abstractmethod
    def warmup(self) -> None:
        """启动时预热一次（可选）。"""
        raise NotImplementedError

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """输出音频采样率。"""
        raise NotImplementedError

    @property
    @abstractmethod
    def engine_type(self) -> str:
        """引擎类型标识。"""
        raise NotImplementedError

    @property
    @abstractmethod
    def device(self) -> str:
        """当前运行设备。"""
        raise NotImplementedError

    @property
    @abstractmethod
    def model_name(self) -> str:
        """模型名称或路径。"""
        raise NotImplementedError

    def to_health_dict(self) -> dict[str, Any]:
        return {
            "engine_type": self.engine_type,
            "model_name": self.model_name,
            "device": self.device,
            "sample_rate": self.sample_rate,
        }
