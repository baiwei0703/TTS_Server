"""
基于 CosyVoice 2.0 的 Web TTS 语音克隆服务。

提供：
- 浏览器可直接访问的 HTML 调试页面（GET /）
- HTTP API：POST /api/tts（multipart/form-data，支持上传参考音频）
- HTTP API：POST /api/tts/base64（JSON，支持 base64 参考音频）
- 健康检查：GET /api/health

启动示例：
    python app.py --engine cosyvoice --model-dir pretrained_models/CosyVoice2-0.5B --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import argparse
import base64
import io
import os
import re
import shutil
import sys
import traceback
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from loguru import logger

from engines.base import TTSJobRequest
from engines.cosyvoice_engine import CosyVoiceEngine
from engines.fishspeech_engine import FishSpeechEngine
from engines.indextts_engine import IndexTTSEngine

# 可选：用于 mp3 输出
try:
    from pydub import AudioSegment
except Exception:  # pragma: no cover
    AudioSegment = None

ENGINE_REGISTRY = {
    "cosyvoice": CosyVoiceEngine,
    "fishspeech": FishSpeechEngine,
    "indextts": IndexTTSEngine,
}

# ---------------------------------------------------------------------------
# 音色（参考音频）持久化
# ---------------------------------------------------------------------------
VOICES_DIR = Path("references")
VOICES_DIR.mkdir(parents=True, exist_ok=True)

AUDIO_EXTENSIONS_FOR_VOICE = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac"}


def sanitize_voice_id(name: str) -> str:
    """把用户输入的音色 ID 规范化，只允许安全字符。"""
    name = name.strip()
    if not name:
        return uuid.uuid4().hex[:8]
    name = re.sub(r"[^a-zA-Z0-9_\-\u4e00-\u9fa5]", "_", name)
    name = name[:64]
    if name in (".", ".."):
        name = "voice_" + name
    return name


def save_voice(voice_id: str, audio_bytes: bytes, text: str) -> Path:
    voice_dir = VOICES_DIR / voice_id
    if voice_dir.exists():
        shutil.rmtree(voice_dir)
    voice_dir.mkdir(parents=True, exist_ok=True)

    audio_path = voice_dir / "ref.wav"
    lab_path = voice_dir / "ref.lab"

    with open(audio_path, "wb") as f:
        f.write(audio_bytes)
    with open(lab_path, "w", encoding="utf-8") as f:
        f.write(text)

    return voice_dir


def list_voice_ids() -> list[dict]:
    voices = []
    if not VOICES_DIR.exists():
        return voices
    for voice_dir in sorted(VOICES_DIR.iterdir()):
        if not voice_dir.is_dir():
            continue
        audio_files = [
            f
            for f in voice_dir.iterdir()
            if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS_FOR_VOICE
        ]
        if not audio_files:
            continue
        lab_file = voice_dir / (audio_files[0].stem + ".lab")
        text = ""
        if lab_file.exists():
            try:
                text = lab_file.read_text(encoding="utf-8")
            except Exception:
                pass
        voices.append(
            {
                "voice_id": voice_dir.name,
                "text": text,
                "audio_name": audio_files[0].name,
            }
        )
    return voices


def delete_voice_dir(voice_id: str) -> bool:
    voice_dir = VOICES_DIR / voice_id
    if not voice_dir.exists() or not voice_dir.is_dir():
        return False
    shutil.rmtree(voice_dir)
    return True


# ---------------------------------------------------------------------------
# 命令行参数
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CosyVoice / Fish Speech Web TTS Server")
    parser.add_argument(
        "--engine",
        type=str,
        choices=list(ENGINE_REGISTRY.keys()),
        default=os.getenv("ENGINE", "cosyvoice"),
        help="TTS 后端引擎：cosyvoice（默认）/ fishspeech / indextts",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(os.getenv("MODEL_DIR", "pretrained_models/CosyVoice2-0.5B")),
        help="CosyVoice 模型目录（默认 pretrained_models/CosyVoice2-0.5B）",
    )
    # 保留 Fish Speech 专用参数，供 --engine fishspeech 时使用
    parser.add_argument(
        "--llama-checkpoint-path",
        type=Path,
        default=Path(os.getenv("LLAMA_CHECKPOINT_PATH", "checkpoints/openaudio-s1-mini")),
        help="Fish Speech：LLaMA 模型目录（仅 fishspeech 有效）",
    )
    parser.add_argument(
        "--decoder-checkpoint-path",
        type=Path,
        default=Path(
            os.getenv("DECODER_CHECKPOINT_PATH", "checkpoints/openaudio-s1-mini/codec.pth")
        ),
        help="Fish Speech：Codec 权重路径（仅 fishspeech 有效）",
    )
    parser.add_argument(
        "--decoder-config-name",
        type=str,
        default=os.getenv("DECODER_CONFIG_NAME", "modded_dac_vq"),
        help="Fish Speech：解码器配置名（仅 fishspeech 有效）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=os.getenv("DEVICE", "auto"),
        help='运行设备：auto / cuda / cpu / mps / xpu（默认 auto，目前主要影响日志）',
    )
    parser.add_argument(
        "--half",
        action="store_true",
        default=os.getenv("HALF", "0").lower() in ("1", "true", "yes"),
        help="使用 FP16（GPU 推荐，可减少显存占用）",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        default=os.getenv("COMPILE", "0").lower() in ("1", "true", "yes"),
        help="启用 torch.compile（仅 Fish Speech，Windows 上通常不支持）",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=os.getenv("HOST", "0.0.0.0"),
        help="监听地址",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("PORT", "8000")),
        help="监听端口",
    )
    parser.add_argument(
        "--max-text-length",
        type=int,
        default=int(os.getenv("MAX_TEXT_LENGTH", "0")),
        help="允许的最大文本长度，0 表示不限制",
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
        default=os.getenv("WARMUP", "1").lower() in ("1", "true", "yes"),
        help="启动时执行一次 warmup 推理",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=float(os.getenv("SPEED", "1.0")),
        help="语速倍率（仅 CosyVoice 有效，默认 1.0）",
    )
    parser.add_argument(
        "--text-frontend",
        action="store_true",
        default=os.getenv("TEXT_FRONTEND", "0").lower() in ("1", "true", "yes"),
        help="开启 CosyVoice 文本前端（Windows 默认关闭，避免 pynini 依赖）",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# FastAPI 生命周期
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    args = getattr(app.state, "args", None)
    if args is None:
        args = parse_args()
        app.state.args = args

    engine_cls = ENGINE_REGISTRY.get(args.engine, CosyVoiceEngine)
    try:
        engine = engine_cls().load(args)
        app.state.engine = engine
        app.state.max_text_length = args.max_text_length
        if args.warmup:
            engine.warmup()
        logger.info(f"服务已就绪：http://{args.host}:{args.port}")
    except Exception as exc:
        logger.error(f"模型加载失败: {exc}\n{traceback.format_exc()}")
        app.state.engine = None
    yield
    logger.info("服务关闭中...")


app = FastAPI(
    title="CosyVoice TTS Server",
    description="基于 CosyVoice 2.0 的 Web 语音克隆 TTS 服务",
    version="0.2.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# 音频编码辅助函数
# ---------------------------------------------------------------------------
def encode_audio_to_bytes(
    audio: np.ndarray, sample_rate: int, fmt: str
) -> tuple[bytes, str]:
    """将 numpy 音频编码为指定格式，返回 (bytes, mime_type)。"""

    fmt = fmt.lower()
    if fmt == "wav":
        buffer = io.BytesIO()
        sf.write(buffer, audio, sample_rate, format="WAV")
        return buffer.getvalue(), "audio/wav"

    if fmt == "pcm":
        return audio.astype(np.float32).tobytes(), "audio/pcm"

    if fmt == "mp3":
        if AudioSegment is None:
            logger.warning("pydub 未安装，mp3 输出回退为 wav")
            return encode_audio_to_bytes(audio, sample_rate, "wav")
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, audio, sample_rate, format="WAV")
        wav_buffer.seek(0)
        segment = AudioSegment.from_wav(wav_buffer)
        mp3_buffer = io.BytesIO()
        segment.export(mp3_buffer, format="mp3", bitrate="192k")
        return mp3_buffer.getvalue(), "audio/mpeg"

    raise HTTPException(status_code=400, detail=f"不支持的音频格式: {fmt}")


async def read_uploaded_audio(upload: UploadFile) -> bytes:
    if upload is None:
        return b""
    try:
        return await upload.read()
    finally:
        await upload.close()


# ---------------------------------------------------------------------------
# 请求构造
# ---------------------------------------------------------------------------
def build_request(
    text: str,
    reference_audio_bytes: bytes,
    reference_text: str,
    voice_id: str | None,
    fmt: str,
    max_new_tokens: int,
    chunk_length: int,
    top_p: float,
    repetition_penalty: float,
    temperature: float,
    seed: int | None,
    normalize: bool,
    use_memory_cache: str,
    speed: float,
    text_frontend: bool,
) -> TTSJobRequest:
    return TTSJobRequest(
        text=text,
        reference_audio_bytes=reference_audio_bytes,
        reference_text=reference_text,
        voice_id=voice_id,
        format=fmt,
        max_new_tokens=max_new_tokens,
        chunk_length=chunk_length,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        temperature=temperature,
        seed=seed,
        normalize=normalize,
        use_memory_cache=use_memory_cache,
        speed=speed,
        text_frontend=text_frontend,
    )


def check_text_length(app: FastAPI, text: str) -> None:
    max_len = app.state.max_text_length
    if max_len > 0 and len(text) > max_len:
        raise HTTPException(
            status_code=400,
            detail=f"文本过长，最大允许 {max_len} 个字符",
        )


# ---------------------------------------------------------------------------
# 推理执行
# ---------------------------------------------------------------------------
def run_tts(app: FastAPI, req: TTSJobRequest) -> tuple[np.ndarray, int]:
    engine = app.state.engine
    if engine is None:
        raise HTTPException(status_code=503, detail="TTS 模型尚未加载完成或加载失败")
    return engine.tts(req)


# ---------------------------------------------------------------------------
# API 路由
# ---------------------------------------------------------------------------
@app.get("/api/health")
async def health(request: Request):
    engine = request.app.state.engine
    result = {
        "status": "ok",
        "model_loaded": engine is not None,
    }
    if engine is not None:
        result.update(engine.to_health_dict())
    return result


@app.post("/api/tts")
async def tts_endpoint(
    request: Request,
    text: str = Form(..., description="要合成的文本"),
    reference_audio: UploadFile | None = File(
        None, description="参考音频文件（wav / mp3 / flac）"
    ),
    reference_text: str = Form("", description="参考音频对应的文本"),
    voice_id: str = Form("", description="已保存的音色 ID（与 reference_audio 二选一）"),
    format: str = Form("wav", description="输出格式：wav / pcm / mp3"),
    max_new_tokens: int = Form(1024, ge=1, le=4096),
    chunk_length: int = Form(200, ge=100, le=300),
    top_p: float = Form(0.7, ge=0.1, le=1.0),
    repetition_penalty: float = Form(1.2, ge=0.9, le=2.0),
    temperature: float = Form(0.7, ge=0.1, le=1.0),
    seed: int | None = Form(None, ge=0),
    normalize: bool = Form(True),
    use_memory_cache: str = Form("off", description="是否缓存参考音频编码：on / off"),
    speed: float = Form(1.0, ge=0.5, le=2.0),
    text_frontend: bool = Form(False),
):
    check_text_length(request.app, text)
    ref_bytes = await read_uploaded_audio(reference_audio) if reference_audio else b""

    if voice_id and not (VOICES_DIR / voice_id).exists():
        raise HTTPException(status_code=404, detail=f"音色 {voice_id} 不存在")

    if voice_id and not use_memory_cache:
        use_memory_cache = "on"

    req = build_request(
        text=text,
        reference_audio_bytes=ref_bytes,
        reference_text=reference_text,
        voice_id=voice_id or None,
        fmt=format,
        max_new_tokens=max_new_tokens,
        chunk_length=chunk_length,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        temperature=temperature,
        seed=seed,
        normalize=normalize,
        use_memory_cache=use_memory_cache,
        speed=speed,
        text_frontend=text_frontend,
    )

    audio, sample_rate = run_tts(request.app, req)
    data, mime = encode_audio_to_bytes(audio, sample_rate, format)

    return StreamingResponse(
        iter([data]),
        media_type=mime,
        headers={"Content-Disposition": f"attachment; filename=audio.{format.lower()}"},
    )


class Base64TTSRequest:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.text: str = payload.get("text", "")
        self.reference_audio: str = payload.get("reference_audio", "")
        self.reference_text: str = payload.get("reference_text", "")
        self.voice_id: str = payload.get("voice_id", "")
        self.format: str = payload.get("format", "wav")
        self.max_new_tokens: int = int(payload.get("max_new_tokens", 1024))
        self.chunk_length: int = int(payload.get("chunk_length", 200))
        self.top_p: float = float(payload.get("top_p", 0.7))
        self.repetition_penalty: float = float(payload.get("repetition_penalty", 1.2))
        self.temperature: float = float(payload.get("temperature", 0.7))
        self.seed: int | None = payload.get("seed")
        self.normalize: bool = bool(payload.get("normalize", True))
        self.use_memory_cache: str = payload.get("use_memory_cache", "off")
        self.speed: float = float(payload.get("speed", 1.0))
        self.text_frontend: bool = bool(payload.get("text_frontend", False))


@app.post("/api/tts/base64")
async def tts_base64_endpoint(request: Request):
    payload = await request.json()
    body = Base64TTSRequest(payload)

    if not body.text:
        raise HTTPException(status_code=400, detail="text 字段不能为空")
    check_text_length(request.app, body.text)

    ref_bytes = b""
    if body.reference_audio:
        try:
            ref_bytes = base64.b64decode(body.reference_audio)
        except Exception:
            raise HTTPException(status_code=400, detail="reference_audio 不是有效的 base64")

    if body.voice_id and not (VOICES_DIR / body.voice_id).exists():
        raise HTTPException(status_code=404, detail=f"音色 {body.voice_id} 不存在")

    if body.voice_id and not body.use_memory_cache:
        body.use_memory_cache = "on"

    req = build_request(
        text=body.text,
        reference_audio_bytes=ref_bytes,
        reference_text=body.reference_text,
        voice_id=body.voice_id or None,
        fmt=body.format,
        max_new_tokens=body.max_new_tokens,
        chunk_length=body.chunk_length,
        top_p=body.top_p,
        repetition_penalty=body.repetition_penalty,
        temperature=body.temperature,
        seed=body.seed,
        normalize=body.normalize,
        use_memory_cache=body.use_memory_cache,
        speed=body.speed,
        text_frontend=body.text_frontend,
    )

    audio, sample_rate = run_tts(request.app, req)
    data, mime = encode_audio_to_bytes(audio, sample_rate, body.format)

    return StreamingResponse(
        iter([data]),
        media_type=mime,
        headers={"Content-Disposition": f"attachment; filename=audio.{body.format.lower()}"},
    )


# ---------------------------------------------------------------------------
# 音色管理 API
# ---------------------------------------------------------------------------
@app.post("/api/voices")
async def create_voice(
    name: str = Form("", description="音色名称（留空则自动生成）"),
    audio: UploadFile = File(..., description="参考音频文件"),
    text: str = Form("", description="参考音频对应的文本"),
):
    audio_bytes = await read_uploaded_audio(audio)
    if len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="音频文件不能为空")

    voice_id = sanitize_voice_id(name)
    original_voice_id = voice_id
    counter = 1
    while (VOICES_DIR / voice_id).exists():
        voice_id = f"{original_voice_id}_{counter}"
        counter += 1

    save_voice(voice_id, audio_bytes, text)
    return {
        "voice_id": voice_id,
        "text": text,
        "audio_size": len(audio_bytes),
        "message": "音色已保存",
    }


@app.get("/api/voices")
async def get_voices():
    return {"voices": list_voice_ids()}


@app.delete("/api/voices/{voice_id}")
async def remove_voice(voice_id: str):
    if not delete_voice_dir(voice_id):
        raise HTTPException(status_code=404, detail=f"音色 {voice_id} 不存在")
    return {"deleted": voice_id}


# ---------------------------------------------------------------------------
# HTML 调试页面
# ---------------------------------------------------------------------------
INDEX_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CosyVoice 语音克隆 TTS</title>
    <style>
        :root { color-scheme: light dark; }
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Microsoft YaHei", sans-serif; max-width: 720px; margin: 40px auto; padding: 0 20px; line-height: 1.6; }
        h1 { text-align: center; }
        h2 { margin-top: 32px; border-bottom: 1px solid #ccc; padding-bottom: 8px; }
        label { display: block; margin-top: 16px; font-weight: 600; }
        textarea, input[type="text"], input[type="number"], select { width: 100%; padding: 8px; box-sizing: border-box; }
        textarea { min-height: 100px; }
        .row { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }
        button { margin-top: 20px; padding: 12px 24px; font-size: 16px; cursor: pointer; }
        .inline-btn { margin-top: 8px; padding: 6px 12px; font-size: 14px; }
        #status, #voiceStatus { margin-top: 16px; color: #666; }
        audio { width: 100%; margin-top: 16px; }
        .error { color: #c00; }
        .success { color: #080; }
        #voiceList { margin-top: 12px; }
        #voiceList li { margin: 4px 0; }
        .hint { font-size: 13px; color: #888; font-weight: normal; }
    </style>
</head>
<body>
    <h1>🎙️ CosyVoice 语音克隆 TTS</h1>

    <h2>1️⃣ 保存音色</h2>
    <p>上传一段 10~30 秒的参考音频并命名，之后就可以反复用这个音色生成语音。</p>
    <form id="voiceForm" enctype="multipart/form-data">
        <label for="voiceName">音色名称</label>
        <input type="text" name="name" id="voiceName" placeholder="例如：小明" required>

        <label for="voiceAudio">参考音频</label>
        <input type="file" name="audio" id="voiceAudio" accept="audio/*" required>

        <label for="voiceText">参考音频文本</label>
        <input type="text" name="text" id="voiceText" placeholder="参考音频里说了什么">

        <button type="submit">💾 保存音色</button>
    </form>
    <div id="voiceStatus"></div>
    <div>
        <h3>已保存音色</h3>
        <button type="button" class="inline-btn" onclick="loadVoices()">🔄 刷新列表</button>
        <ul id="voiceList"></ul>
    </div>

    <h2>2️⃣ 生成语音</h2>
    <form id="ttsForm" enctype="multipart/form-data">
        <label for="text">要合成的文本</label>
        <textarea name="text" id="text" required>你好，这是 CosyVoice 的语音克隆测试。</textarea>

        <label for="voice_id">使用已保存音色（可选）</label>
        <select name="voice_id" id="voice_id">
            <option value="">不使用 / 直接上传参考音频</option>
        </select>

        <label for="reference_audio">或直接上传参考音频（可选）</label>
        <input type="file" name="reference_audio" id="reference_audio" accept="audio/*">

        <label for="reference_text">参考音频文本</label>
        <input type="text" name="reference_text" id="reference_text" placeholder="参考音频里说了什么">

        <div class="row">
            <div>
                <label for="format">输出格式</label>
                <select name="format" id="format">
                    <option value="wav" selected>WAV</option>
                    <option value="mp3">MP3</option>
                    <option value="pcm">PCM</option>
                </select>
            </div>
            <div>
                <label for="speed">语速 <span class="hint">(CosyVoice)</span></label>
                <input type="number" name="speed" id="speed" step="0.1" min="0.5" max="2.0" value="1.0">
            </div>
            <div>
                <label for="seed">随机种子 <span class="hint">(CosyVoice)</span></label>
                <input type="number" name="seed" id="seed" placeholder="留空则随机">
            </div>
        </div>

        <div class="row">
            <div>
                <label for="text_frontend">文本前端 <span class="hint">(CosyVoice，Windows 建议关闭)</span></label>
                <select name="text_frontend" id="text_frontend">
                    <option value="false" selected>关闭</option>
                    <option value="true">开启</option>
                </select>
            </div>
            <div>
                <label for="top_p">Top P <span class="hint">(FishSpeech 有效)</span></label>
                <input type="number" name="top_p" id="top_p" step="0.05" min="0.1" max="1" value="0.7">
            </div>
            <div>
                <label for="temperature">Temperature <span class="hint">(FishSpeech 有效)</span></label>
                <input type="number" name="temperature" id="temperature" step="0.05" min="0.1" max="1" value="0.7">
            </div>
        </div>

        <div class="row">
            <div>
                <label for="repetition_penalty">重复惩罚 <span class="hint">(FishSpeech 有效)</span></label>
                <input type="number" name="repetition_penalty" id="repetition_penalty" step="0.05" min="0.9" max="2" value="1.2">
            </div>
            <div>
                <label for="max_new_tokens">最大 Token <span class="hint">(FishSpeech 有效)</span></label>
                <input type="number" name="max_new_tokens" id="max_new_tokens" min="1" max="4096" value="1024">
            </div>
            <div>
                <label for="chunk_length">Chunk Length <span class="hint">(FishSpeech 有效)</span></label>
                <input type="number" name="chunk_length" id="chunk_length" min="100" max="300" value="200">
            </div>
        </div>

        <button type="submit">🎙️ 生成语音</button>
    </form>

    <div id="status"></div>
    <audio id="player" controls hidden></audio>

    <script>
        const voiceForm = document.getElementById('voiceForm');
        const voiceStatus = document.getElementById('voiceStatus');
        const voiceList = document.getElementById('voiceList');
        const voiceSelect = document.getElementById('voice_id');

        async function loadVoices() {
            try {
                const resp = await fetch('/api/voices');
                const data = await resp.json();
                voiceList.innerHTML = '';
                voiceSelect.innerHTML = '<option value="">不使用 / 直接上传参考音频</option>';
                if (data.voices.length === 0) {
                    voiceList.innerHTML = '<li>暂无已保存音色</li>';
                    return;
                }
                data.voices.forEach(v => {
                    const li = document.createElement('li');
                    li.textContent = `${v.voice_id} （${v.audio_name}${v.text ? '：' + v.text : ''}）`;
                    voiceList.appendChild(li);

                    const opt = document.createElement('option');
                    opt.value = v.voice_id;
                    opt.textContent = v.voice_id;
                    voiceSelect.appendChild(opt);
                });
            } catch (err) {
                voiceList.innerHTML = `<li class="error">加载失败：${err.message}</li>`;
            }
        }

        voiceForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            voiceStatus.textContent = '正在保存...';
            voiceStatus.className = '';
            const formData = new FormData(voiceForm);
            try {
                const resp = await fetch('/api/voices', { method: 'POST', body: formData });
                const data = await resp.json();
                if (!resp.ok) throw new Error(data.detail || `保存失败 (${resp.status})`);
                voiceStatus.textContent = `音色已保存：${data.voice_id}`;
                voiceStatus.className = 'success';
                voiceForm.reset();
                loadVoices();
            } catch (err) {
                voiceStatus.textContent = err.message;
                voiceStatus.className = 'error';
            }
        });

        const form = document.getElementById('ttsForm');
        const statusDiv = document.getElementById('status');
        const player = document.getElementById('player');

        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            statusDiv.textContent = '正在生成，请稍候...';
            statusDiv.className = '';
            player.hidden = true;

            const formData = new FormData(form);
            if (!formData.get('seed')) formData.delete('seed');
            if (!formData.get('voice_id')) formData.delete('voice_id');
            if (!formData.get('reference_audio').size) formData.delete('reference_audio');

            try {
                const resp = await fetch('/api/tts', { method: 'POST', body: formData });
                if (!resp.ok) {
                    const text = await resp.text();
                    throw new Error(`生成失败 (${resp.status}): ${text}`);
                }
                const blob = await resp.blob();
                const url = URL.createObjectURL(blob);
                player.src = url;
                player.hidden = false;
                statusDiv.textContent = '生成成功！';
                statusDiv.className = 'success';
            } catch (err) {
                statusDiv.textContent = err.message;
                statusDiv.className = 'error';
            }
        });

        loadVoices();
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return INDEX_HTML


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    app.state.args = args
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
