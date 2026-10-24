"""CosyVoice 2.0 引擎实现。"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger

from engines.base import TTSEngine, TTSJobRequest

# CosyVoice 源码以子模块形式放在 third_party/CosyVoice
_COSYVOICE_ROOT = Path("third_party/CosyVoice")
_MATCHA_TTS_ROOT = _COSYVOICE_ROOT / "third_party" / "Matcha-TTS"

for _p in (_COSYVOICE_ROOT, _MATCHA_TTS_ROOT):
    _s = str(_p.resolve())
    if _s not in sys.path:
        sys.path.insert(0, _s)


try:
    from cosyvoice.cli.cosyvoice import AutoModel
    from cosyvoice.utils.file_utils import load_wav
except Exception as _exc:
    logger.warning(f"CosyVoice 尚未安装或路径不正确: {_exc}")
    AutoModel = None  # type: ignore[misc,assignment]
    load_wav = None  # type: ignore[misc,assignment]

try:
    from cosyvoice.utils.common import set_all_random_seed
except Exception:
    set_all_random_seed = None  # type: ignore[misc,assignment]


class CosyVoiceEngine(TTSEngine):
    def __init__(self) -> None:
        self.model: Any = None
        self.model_dir: Path = Path("pretrained_models/CosyVoice2-0.5B")
        self._fp16: bool = False
        self._device: str = "cpu"
        self._sample_rate: int = 22050

    def load(self, args: argparse.Namespace) -> "CosyVoiceEngine":
        if AutoModel is None:
            raise RuntimeError(
                "CosyVoice 未安装。请先运行 install_cosyvoice.bat，"
                "或确认 third_party/CosyVoice 已存在。"
            )

        self.model_dir = Path(getattr(args, "model_dir", self.model_dir))
        self._fp16 = getattr(args, "half", False)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(
            f"加载 CosyVoice 模型: {self.model_dir}, "
            f"device={self._device}, fp16={self._fp16}"
        )

        self.model = AutoModel(
            model_dir=str(self.model_dir),
            load_jit=False,
            load_trt=False,
            load_vllm=False,
            fp16=self._fp16,
        )
        self._sample_rate = getattr(self.model, "sample_rate", 22050)
        logger.info(f"CosyVoice 加载完成，采样率: {self._sample_rate}")
        return self

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def engine_type(self) -> str:
        return "cosyvoice"

    @property
    def device(self) -> str:
        return self._device

    @property
    def model_name(self) -> str:
        return str(self.model_dir)

    def _set_seed(self, seed: int | None) -> None:
        if seed is None:
            return
        if set_all_random_seed is not None:
            try:
                set_all_random_seed(seed)
                return
            except Exception:
                pass
        # fallback
        import random

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _save_prompt_wav(self, audio_bytes: bytes) -> str:
        """把参考音频写入临时 wav 文件并返回路径。

        注意：本仓库 third_party/CosyVoice 的 frontend_zero_shot 接收的是
        「文件路径」（内部自行 load_wav + 重采样），不是已加载的张量。
        因此这里必须返回路径；调用方负责在用完后删除临时文件。
        """
        fd, tmp_path = tempfile.mkstemp(suffix=".wav")
        with os.fdopen(fd, "wb") as f:
            f.write(audio_bytes)
        return tmp_path

    def _concat_chunks(self, generator) -> np.ndarray:
        chunks = list(generator)
        if not chunks:
            raise RuntimeError("CosyVoice 没有生成音频")
        speech = torch.cat([c["tts_speech"] for c in chunks], dim=-1)
        return speech.squeeze(0).cpu().numpy().astype(np.float32)

    def _get_prompt_from_voice(self, voice_id: str) -> tuple[str, str]:
        voice_dir = Path("references") / voice_id
        audio_path = voice_dir / "ref.wav"
        lab_path = voice_dir / "ref.lab"
        if not audio_path.exists():
            raise FileNotFoundError(f"音色 {voice_id} 的参考音频不存在")
        prompt_text = ""
        if lab_path.exists():
            prompt_text = lab_path.read_text(encoding="utf-8")
        # 传路径（本仓库 CosyVoice 的 frontend 内部自行 load_wav）
        return str(audio_path), prompt_text

    def _synthesize(self, text, prompt_wav, prompt_text, speed, text_frontend):
        """根据是否有参考文本，选择最合适的克隆模式。

        - 有参考文本 → zero_shot（音色+韵律对齐，效果最好）
        - 无参考文本 → cross_lingual（CosyVoice2 专为该场景设计；
          若仍走 zero_shot 会因缺少文本引导而跑飞，生成大段无关语音）
        """
        if prompt_text and prompt_text.strip():
            return self.model.inference_zero_shot(
                text,
                prompt_text,
                prompt_wav,
                stream=False,
                speed=speed,
                text_frontend=text_frontend,
            )
        return self.model.inference_cross_lingual(
            text,
            prompt_wav,
            stream=False,
            speed=speed,
            text_frontend=text_frontend,
        )

    def _available_spks(self) -> list[str]:
        if self.model is None:
            return []
        fn = getattr(self.model, "list_available_spks", None)
        if fn is None:
            return []
        try:
            return list(fn())
        except Exception:
            return []

    def tts(self, req: TTSJobRequest) -> tuple[np.ndarray, int]:
        if self.model is None:
            raise RuntimeError("CosyVoice 模型未加载")

        self._set_seed(req.seed)

        # 1) 使用已保存音色
        if req.voice_id:
            prompt_wav, prompt_text = self._get_prompt_from_voice(req.voice_id)
            audio = self._concat_chunks(
                self._synthesize(
                    req.text,
                    prompt_wav,
                    prompt_text,
                    req.speed,
                    req.text_frontend,
                )
            )
            return audio, self.sample_rate

        # 2) 使用上传的参考音频
        if req.reference_audio_bytes:
            prompt_wav = self._save_prompt_wav(req.reference_audio_bytes)
            try:
                audio = self._concat_chunks(
                    self._synthesize(
                        req.text,
                        prompt_wav,
                        req.reference_text or "",
                        req.speed,
                        req.text_frontend,
                    )
                )
            finally:
                try:
                    os.unlink(prompt_wav)
                except Exception:
                    pass
            return audio, self.sample_rate

        # 3) 无参考音频：使用内置音色（SFT 模式）
        spks = self._available_spks()
        if not spks:
            raise RuntimeError(
                "CosyVoice 需要参考音频或已保存音色。"
                "请先上传参考音频或调用 /api/voices 保存音色。"
            )
        audio = self._concat_chunks(
            self.model.inference_sft(
                req.text,
                spk_id=spks[0],
                stream=False,
                speed=req.speed,
                text_frontend=req.text_frontend,
            )
        )
        return audio, self.sample_rate

    def warmup(self) -> None:
        logger.info("CosyVoice warmup...")
        try:
            spks = self._available_spks()
            if spks:
                self.model.inference_sft(
                    "你好，这是一个测试。",
                    spk_id=spks[0],
                    stream=False,
                    speed=1.0,
                    text_frontend=False,
                )
            else:
                # 没有内置音色时尝试使用第一个已保存音色
                voices = [d for d in Path("references").iterdir() if d.is_dir()]
                if voices:
                    prompt_wav, prompt_text = self._get_prompt_from_voice(voices[0].name)
                    self._concat_chunks(
                        self._synthesize(
                            "你好，这是一个测试。",
                            prompt_wav,
                            prompt_text,
                            1.0,
                            False,
                        )
                    )
                else:
                    logger.warning("没有可用音色，跳过 warmup")
        except Exception as exc:
            logger.warning(f"CosyVoice warmup 失败（不影响服务启动）: {exc}\n{traceback.format_exc()}")
