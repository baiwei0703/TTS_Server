"""Fish Speech 引擎实现（保留作为可选项）。"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger

from engines.base import TTSEngine, TTSJobRequest

try:
    from fish_speech.inference_engine import TTSInferenceEngine
    from fish_speech.models.dac.inference import load_model as load_decoder_model
    from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
    from fish_speech.utils.schema import ServeReferenceAudio, ServeTTSRequest
except Exception as _exc:
    logger.warning(f"Fish Speech 尚未安装: {_exc}")
    TTSInferenceEngine = None  # type: ignore[misc,assignment]
    load_decoder_model = None  # type: ignore[misc,assignment]
    launch_thread_safe_queue = None  # type: ignore[misc,assignment]
    ServeReferenceAudio = None  # type: ignore[misc,assignment]
    ServeTTSRequest = None  # type: ignore[misc,assignment]


class FishSpeechEngine(TTSEngine):
    def __init__(self) -> None:
        self.engine: Any = None
        self._device: str = "cpu"
        self._precision: torch.dtype = torch.float32
        self._sample_rate: int = 44100
        self.model_dir: Path = Path("checkpoints/openaudio-s1-mini")

    @staticmethod
    def resolve_device(preference: str) -> str:
        if preference != "auto":
            return preference
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return "xpu"
        return "cpu"

    @staticmethod
    def resolve_precision(device: str, half: bool) -> torch.dtype:
        if half:
            return torch.half
        if device == "cpu":
            return torch.float32
        return torch.bfloat16

    def load(self, args: argparse.Namespace) -> "FishSpeechEngine":
        if TTSInferenceEngine is None:
            raise RuntimeError(
                "Fish Speech 未安装。请在原 .venv 中运行，"
                "或使用 --engine cosyvoice。"
            )

        self.model_dir = Path(
            getattr(args, "llama_checkpoint_path", os.getenv("LLAMA_CHECKPOINT_PATH", "checkpoints/openaudio-s1-mini"))
        )
        decoder_path = Path(
            getattr(args, "decoder_checkpoint_path", os.getenv("DECODER_CHECKPOINT_PATH", "checkpoints/openaudio-s1-mini/codec.pth"))
        )
        decoder_config = getattr(args, "decoder_config_name", os.getenv("DECODER_CONFIG_NAME", "modded_dac_vq"))

        self._device = self.resolve_device(args.device)
        self._precision = self.resolve_precision(self._device, getattr(args, "half", False))
        compile_flag = getattr(args, "compile", False)

        logger.info(
            f"加载 Fish Speech: device={self._device}, "
            f"precision={self._precision}, compile={compile_flag}"
        )

        llama_queue = launch_thread_safe_queue(
            checkpoint_path=str(self.model_dir),
            device=self._device,
            precision=self._precision,
            compile=compile_flag,
        )
        decoder_model = load_decoder_model(
            config_name=decoder_config,
            checkpoint_path=str(decoder_path),
            device=self._device,
        )
        self.engine = TTSInferenceEngine(
            llama_queue=llama_queue,
            decoder_model=decoder_model,
            precision=self._precision,
            compile=compile_flag,
        )
        self._sample_rate = (
            decoder_model.spec_transform.sample_rate
            if hasattr(decoder_model, "spec_transform")
            else decoder_model.sample_rate
        )
        logger.info(f"Fish Speech 加载完成，采样率: {self._sample_rate}")
        return self

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def engine_type(self) -> str:
        return "fishspeech"

    @property
    def device(self) -> str:
        return self._device

    @property
    def model_name(self) -> str:
        return str(self.model_dir)

    def _build_serve_request(self, req: TTSJobRequest) -> Any:
        references: list[Any] = []
        ref_id: str | None = None

        if req.voice_id:
            ref_id = req.voice_id
        elif req.reference_audio_bytes:
            references.append(
                ServeReferenceAudio(audio=req.reference_audio_bytes, text=req.reference_text or "")
            )

        return ServeTTSRequest(
            text=req.text,
            references=references,
            reference_id=ref_id,
            max_new_tokens=req.max_new_tokens,
            chunk_length=req.chunk_length,
            top_p=req.top_p,
            repetition_penalty=req.repetition_penalty,
            temperature=req.temperature,
            seed=req.seed,
            format=req.format,
            normalize=req.normalize,
            streaming=False,
            use_memory_cache=req.use_memory_cache,
        )

    def tts(self, req: TTSJobRequest) -> tuple[np.ndarray, int]:
        if self.engine is None:
            raise RuntimeError("Fish Speech 模型未加载")

        serve_req = self._build_serve_request(req)
        for result in self.engine.inference(serve_req):
            if result.code == "final":
                sr, audio = result.audio
                return audio, int(sr)
            if result.code == "error":
                raise RuntimeError(f"推理失败: {result.error}")
        raise RuntimeError("没有生成音频")

    def warmup(self) -> None:
        logger.info("Fish Speech warmup...")
        req = self._build_serve_request(
            TTSJobRequest(
                text="你好，这是一个测试。",
                use_memory_cache="off",
            )
        )
        for result in self.engine.inference(req):
            if result.code == "final":
                logger.info("Fish Speech warmup 完成")
                return
            if result.code == "error":
                raise RuntimeError(f"Warmup 失败: {result.error}")
        logger.warning("Fish Speech warmup 没有返回音频")
