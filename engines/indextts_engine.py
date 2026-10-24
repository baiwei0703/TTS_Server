"""IndexTTS-2.5 引擎实现。

特点：
- 支持发音标注语法 <文字|发音>（如 <行|XING2>、<贵|GUI1>），可实现
  「正确的词 + 错误的声调」的确定性注入，是蹩脚中文合成的核心能力。
- 音色来自 references/ 下已保存音色或上传的参考音频（与 CosyVoice 共用
  同一个 references 目录与 /api/voices 管理）。
- speed 参数映射为 IndexTTS 的 duration_factor（>1 变慢）。
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from loguru import logger

from engines.base import TTSEngine, TTSJobRequest

_INDEXTTS_ROOT = Path("third_party/index-tts")
_s = str(_INDEXTTS_ROOT.resolve())
if _s not in sys.path:
    sys.path.insert(0, _s)

try:
    from indextts.infer_v2_5 import IndexTTS2
except Exception as _exc:  # pragma: no cover
    logger.warning(f"IndexTTS 尚未安装或路径不正确: {_exc}")
    IndexTTS2 = None  # type: ignore[misc,assignment]

# 发音标注 <文字|发音>，含标注的文本必须关闭 text_normalization，
# 否则 normalize 步骤可能破坏标记（normalize 先于标注解析执行）。
_HAS_ANNOTATION = re.compile(r"<[^|>\n]+\|[^>\n]+>")


class IndexTTSEngine(TTSEngine):
    def __init__(self) -> None:
        self.model: Any = None
        self.model_dir: Path = Path("pretrained_models/IndexTTS-2.5")
        self._device: str = "cpu"
        self._sample_rate: int = 22050  # IndexTTS 输出固定 22050

    # ------------------------------------------------------------------
    # TTSEngine 接口
    # ------------------------------------------------------------------
    def load(self, args: argparse.Namespace) -> "IndexTTSEngine":
        if IndexTTS2 is None:
            raise RuntimeError(
                "IndexTTS 未安装。请确认 third_party/index-tts 存在且已在"
                "其目录内执行过 uv sync。"
            )
        self.model_dir = Path(getattr(args, "model_dir", self.model_dir))
        cfg_path = self.model_dir / "config.yaml"
        if not cfg_path.exists():
            raise RuntimeError(f"IndexTTS 模型不完整，缺少 {cfg_path}")

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        use_bf16 = self._device == "cuda"
        logger.info(
            f"加载 IndexTTS-2.5: {self.model_dir}, device={self._device}, bf16={use_bf16}"
        )
        self.model = IndexTTS2(
            cfg_path=str(cfg_path),
            model_dir=str(self.model_dir),
            use_bf16=use_bf16,
            use_cuda_kernel=False,
            use_deepspeed=False,
        )
        logger.info("IndexTTS-2.5 加载完成")
        return self

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def engine_type(self) -> str:
        return "indextts"

    @property
    def device(self) -> str:
        return self._device

    @property
    def model_name(self) -> str:
        return str(self.model_dir)

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    @staticmethod
    def _detect_lang(text: str) -> str:
        """粗略判定语言：含 CJK 即按中文处理，否则英文。"""
        if re.search(r"[\u4e00-\u9fff]", text):
            return "zh"
        return "en"

    def _resolve_speaker(self, req: TTSJobRequest) -> str:
        """把 voice_id / 上传的参考音频统一解析为 wav 文件路径。"""
        if req.voice_id:
            audio_path = Path("references") / req.voice_id / "ref.wav"
            if not audio_path.exists():
                raise FileNotFoundError(f"音色 {req.voice_id} 的参考音频不存在")
            return str(audio_path)

        if req.reference_audio_bytes:
            fd, tmp_path = tempfile.mkstemp(suffix=".wav")
            with os.fdopen(fd, "wb") as f:
                f.write(req.reference_audio_bytes)
            return tmp_path

        raise RuntimeError(
            "IndexTTS 需要参考音频。请上传参考音频或使用已保存音色"
            "（/api/voices）。"
        )

    # ------------------------------------------------------------------
    # 推理
    # ------------------------------------------------------------------
    def tts(self, req: TTSJobRequest) -> tuple[np.ndarray, int]:
        if self.model is None:
            raise RuntimeError("IndexTTS 模型未加载")

        if req.seed is not None:
            torch.manual_seed(req.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(req.seed)

        spk_path = self._resolve_speaker(req)
        tmp_out_fd, tmp_out = tempfile.mkstemp(suffix=".wav")
        os.close(tmp_out_fd)

        try:
            # speed: 1.0=原速，>1 更快（CosyVoice 语义）→
            # duration_factor: >1 更慢（IndexTTS 语义），取倒数
            duration_factor = float(np.clip(1.0 / max(req.speed, 0.25), 0.5, 2.0))
            has_annotation = bool(_HAS_ANNOTATION.search(req.text))

            self.model.infer(
                spk_audio_prompt=spk_path,
                text=req.text,
                output_path=tmp_out,
                lang=self._detect_lang(req.text),
                duration_factor=duration_factor,
                # 含发音标注时关闭规范化，避免破坏 <文字|发音> 标记
                text_normalization=not has_annotation,
                verbose=False,
            )

            audio, sr = sf.read(tmp_out, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            self._sample_rate = int(sr)
            return audio, int(sr)
        except Exception:
            logger.error(f"IndexTTS 推理失败:\n{traceback.format_exc()}")
            raise
        finally:
            for p in (tmp_out, spk_path):
                try:
                    if p.startswith(str(Path(tempfile.gettempdir()))):
                        os.unlink(p)
                except Exception:
                    pass

    def warmup(self) -> None:
        logger.info("IndexTTS warmup...")
        try:
            voices = [d for d in Path("references").iterdir() if d.is_dir()] if Path("references").exists() else []
            if not voices:
                logger.warning("没有可用音色，跳过 IndexTTS warmup")
                return
            req = TTSJobRequest(
                text="你好，这是一个测试。",
                voice_id=voices[0].name,
                speed=1.0,
            )
            self.tts(req)
        except Exception as exc:
            logger.warning(
                f"IndexTTS warmup 失败（不影响服务启动）: {exc}\n{traceback.format_exc()}"
            )
