# -*- coding: utf-8 -*-
"""IndexTTS-2.5 蹩脚中文三句对照测试（直接调引擎，不起服务）。

用法：
  third_party/index-tts/.venv/Scripts/python.exe tests/test_indextts_broken_cn.py
输出到 outputs/test_broken_cn/*.wav
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import numpy as np
import soundfile as sf
import torch

from engines.base import TTSJobRequest
from engines.indextts_engine import IndexTTSEngine

OUT_DIR = Path("outputs/test_broken_cn")
VOICE = "narrator"  # 测试用中文音色

CASES = [
    # (文件名, 文本, speed)
    ("01_baseline", "这个太贵了，便宜一点吧。", 1.0),
    # 声调注入：贵 GUI4(四声) → GUI1(一声)，便 BIAN4 → BIAN2
    ("02_tone_inject", "这个太<贵|GUI1>了，<便|BIAN2>宜一点吧。", 0.85),
    # 声调注入 + 放慢（模拟外国人的犹豫节奏）
    ("03_tone_slow", "这个太<贵|GUI1>了，<便|BIAN2>宜一点吧。", 0.7),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="pretrained_models/IndexTTS-2.5")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    engine = IndexTTSEngine().load(args)

    for name, text, speed in CASES:
        t0 = time.perf_counter()
        audio, sr = engine.tts(
            TTSJobRequest(text=text, voice_id=VOICE, speed=speed, seed=42)
        )
        dt = time.perf_counter() - t0
        out_path = OUT_DIR / f"{name}.wav"
        sf.write(out_path, audio, sr)
        dur = len(audio) / sr
        print(f"[OK] {name}: {dur:.2f}s audio, {dt:.1f}s render -> {out_path}")

    # 顺手测一个词表外的声调组合，验证约束（mai 无一声）
    try:
        engine.tts(
            TTSJobRequest(text="<买|MAI1>这个。", voice_id=VOICE, speed=1.0)
        )
        print("[OK] MAI1(词表外) 未报错")
    except Exception as exc:
        print(f"[EXPECTED-FAIL] MAI1(词表外): {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
