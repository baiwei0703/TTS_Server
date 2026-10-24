# TTS Server

本地多引擎语音克隆 TTS 服务，统一 REST API，可插拔引擎：

| 引擎 | 端口 | 特点 |
|---|---|---|
| [CosyVoice 2](https://github.com/FunAudioLLM/CosyVoice)（默认） | 8000 | 零样本声音克隆、中/日/英多语言 |
| [IndexTTS 2.5](https://github.com/index-tts/index-tts) | 8001 | **拼音级发音标注**（`<文字\|PINYIN1>` 强制指定声调）、情感控制、语速控制 |
| Fish Speech（旧） | 8002 | 旧版回退方案 |

支持：
- 浏览器直接访问的 HTML 调试页面
- REST API：`/api/tts`、`/api/tts/base64`、`/api/voices`、`/api/health`
- 零样本声音克隆（上传参考音频或保存音色后复用），音色库跨引擎共享
- 多语言合成，包括中文、日语、英语等
- IndexTTS 引擎：逐字声调标注——把 `太贵了` 写成 `太<贵|GUI1>了` 即可让"贵"读成一声，用于生成"外国人蹩脚中文"等可控错误发音

## 环境要求

- Windows 10/11
- NVIDIA 显卡，显存 ≥ 6GB（RTX 3060 6GB 可运行，建议开启 `--half`）
- [uv](https://github.com/astral-sh/uv) 包管理器
- Git（用于拉取 CosyVoice 源码）

## 安装

双击运行：

```bat
install_cosyvoice.bat
```

脚本会：
1. 创建 Python 3.10 虚拟环境 `.venv-cosyvoice`
2. 安装 PyTorch 2.3.1 + CUDA 12.1
3. 克隆 CosyVoice 源码到 `third_party/CosyVoice`
4. 安装项目依赖
5. 下载 `CosyVoice2-0.5B` 模型到 `pretrained_models/CosyVoice2-0.5B`

> 安装耗时较长，请保持网络畅通。如果下载模型失败，可单独重跑 `download_models.py`。

## 启动服务

### 方式一：Windows 本地（.venv-cosyvoice）

```bat
run.bat
```

或手动（需要先在 `.venv-cosyvoice` 中）：

```bat
set PYTHONPATH=%CD%\third_party\CosyVoice;%CD%\third_party\CosyVoice\third_party\Matcha-TTS;%PYTHONPATH%
python app.py --engine cosyvoice --device auto --host 0.0.0.0 --port 8000
```

启动后打开浏览器：

```
http://localhost:8000
```

### ⚠️ 关于 `--half`（FP16）——本机请勿使用

本机（RTX 3060 Laptop 6GB）实测：**开启 `--half` 会导致输出出现长段低电平噪声尾、时长失控**
（如 4 秒的句子生成 16 秒，后段全为 −28dB 左右的噪声）。

```bat
run.bat --half   &REM ← 不要这样启动
```

FP32 下显存占用约 2.9GB，完全够用，请直接 `run.bat`（不加参数）。

### 方式二：Docker（推荐，环境更干净）

需要：
- Docker Desktop（WSL2 backend）
- NVIDIA GPU 支持（安装 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)）

模型已下载到 `pretrained_models/CosyVoice2-0.5B`，直接构建并运行：

```bat
run_docker.bat
```

或手动：

```bat
docker compose up --build
```

容器会挂载宿主机的模型、音色目录和输出目录：
- `pretrained_models/CosyVoice2-0.5B`
- `references/`
- `outputs/`

## 回退到 Fish Speech

如果已经安装过原来的 Fish Speech 环境，仍可用旧模型：

```bat
run_fishspeech.bat
```

## IndexTTS 2.5 引擎（声调标注 / 蹩脚中文）

与 CosyVoice 依赖不兼容（Python 3.11 + torch 2.8 vs Python 3.10 + torch 2.3.1），因此使用独立 venv 和端口，API 契约与 CosyVoice 完全一致。

### 安装

```bat
:: 1. 源码（国内网络可用 ghfast.top 等加速镜像）
git clone https://github.com/index-tts/index-tts.git third_party/index-tts

:: 2. 依赖（需 uv；torch 2.8.0+cu128 约几 GB，国内可加阿里云镜像：
::    uv pip install torch==2.8.0+cu128 torchaudio==2.8.0+cu128 -f https://mirrors.aliyun.com/pytorch-wheels/cu128/）
cd third_party/index-tts && uv sync

:: 3. 模型（约 5.2GB；HuggingFace: IndexTeam/IndexTTS-2.5，国内推荐 ModelScope）
uvx --from modelscope modelscope download --model IndexTeam/IndexTTS-2.5 --local_dir pretrained_models/IndexTTS-2.5

:: 4. 服务端依赖
uv pip install --python .venv/Scripts/python.exe fastapi "uvicorn[standard]" soundfile loguru python-multipart
```

### 启动

```bat
run_indextts.bat    &REM 端口 8001，CUDA bf16；首次加载约 90 秒
```

### 拼音级发音标注

文本中用 `<文字|拼音+调号>` 强制指定读音，含声调。用法：多音字消歧、生僻读音，以及**故意注入错误声调**（生成外国人学中文的"蹩脚感"）：

```bash
# "贵"(GUI4 四声) 故意读成一声、"手机"指定读音
curl -X POST http://localhost:8001/api/tts \
  -F "text=这个太<贵|GUI1>了，我不小心把<手机|SHOU3 JI1>落在出租车上了。" \
  -F "voice_id=narrator" \
  -F "speed=0.9" \
  --output broken_cn.wav
```

实现细节（`engines/indextts_engine.py`）：
- 检测到发音标注时自动关闭 text_normalization（否则标注会被规范化吃掉）
- `speed` 参数映射为 `duration_factor`（取倒数）
- 音色与 CosyVoice 共享 `references/` 目录
- 实测：6GB 显存 bf16 可运行（未 OOM）；词表外拼音组合（如 MAI1）不报错、正常出声，实际可注入范围比官方文档更宽

### 已知部署坑（Windows）

| 坑 | 解法 |
|---|---|
| jieba 构建失败（GBK 解码 setup.py 中文注释） | 设 `PYTHONUTF8=1` 后重跑 `uv sync` |
| download.pytorch.org 拉 torch cu128 极慢 | 阿里云镜像：`-f https://mirrors.aliyun.com/pytorch-wheels/cu128/` |
| hf-mirror 下载撞 Xet 401 | 改走 ModelScope |
| 首跑自动补下 w2v-bert-2.0 后 `.msc` 只读报错 | `chmod 644 .msc` 并删除 `._____temp` 目录 |

> 许可证注意：IndexTTS 采用 bilibili 专用协议（非 MIT/Apache），商业化使用前需核对条款。

## API 使用

### 1. 保存音色

```bash
curl -X POST http://localhost:8000/api/voices \
  -F "name=xiaoming" \
  -F "audio=@reference.wav" \
  -F "text=你好，这是我的声音。"
```

返回：

```json
{ "voice_id": "xiaoming", "text": "你好，这是我的声音。", "audio_size": 123456, "message": "音色已保存" }
```

音色文件保存在 `references/<voice_id>/`，服务重启后仍然可用。

### 2. 查看已保存音色

```bash
curl http://localhost:8000/api/voices
```

### 3. 使用保存的音色生成语音

```bash
curl -X POST http://localhost:8000/api/tts \
  -F "text=你好，这是用保存音色生成的语音。" \
  -F "voice_id=xiaoming" \
  -F "format=wav" \
  --output output.wav
```

### 4. 直接上传参考音频（不保存）

```bash
curl -X POST http://localhost:8000/api/tts \
  -F "text=你好，这是语音克隆测试。" \
  -F "reference_audio=@reference.wav" \
  -F "reference_text=你好，这是我的声音。" \
  -F "format=wav" \
  --output output.wav
```

### 5. JSON + Base64 方式

```bash
curl -X POST http://localhost:8000/api/tts/base64 \
  -H "Content-Type: application/json" \
  -d '{
    "text": "你好，这是语音克隆测试。",
    "voice_id": "xiaoming",
    "format": "wav"
  }' \
  --output output.wav
```

### 6. 删除音色

```bash
curl -X DELETE http://localhost:8000/api/voices/xiaoming
```

### 7. 健康检查

```bash
curl http://localhost:8000/api/health
```

## 常用参数

启动 `app.py` 时可用命令行参数：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--engine` | `cosyvoice` | 后端引擎：`cosyvoice` / `indextts` / `fishspeech` |
| `--model-dir` | `pretrained_models/CosyVoice2-0.5B` | 模型目录（IndexTTS 用 `pretrained_models/IndexTTS-2.5`） |
| `--device` | `auto` | 推理设备：auto / cuda / cpu |
| `--half` | False | 使用 FP16（GPU 推荐，减少显存） |
| `--speed` | `1.0` | 语速倍率（仅 CosyVoice 有效） |
| `--text-frontend` | False | 是否开启文本前端（Windows 默认关闭） |
| `--host` | `0.0.0.0` | 监听地址 |
| `--port` | `8000` | 监听端口 |

> `temperature`、`top_p`、`repetition_penalty`、`max_new_tokens`、`chunk_length` 是 Fish Speech 专用参数，CosyVoice 会忽略但保留在 API 中以兼容旧调用。

## 目录结构

```
.
├── app.py                      # FastAPI Web 服务主程序（单进程单引擎，--engine 选择）
├── engines/                    # TTS 引擎抽象层（实现 base.TTSEngine 即可接入）
│   ├── base.py
│   ├── cosyvoice_engine.py     # CosyVoice 2（端口 8000，默认）
│   ├── fishspeech_engine.py    # Fish Speech（旧版回退）
│   └── indextts_engine.py      # IndexTTS 2.5（端口 8001，拼音标注）
├── tests/
│   └── test_indextts_broken_cn.py  # 声调注入对照测试
├── download_models.py          # CosyVoice 模型下载脚本
├── install_cosyvoice.bat       # CosyVoice 一键安装
├── requirements-cosyvoice.txt  # CosyVoice Windows 依赖
├── run.bat                     # 启动 CosyVoice 服务（8000）
├── run_indextts.bat            # 启动 IndexTTS 服务（8001）
├── run_fishspeech.bat          # 启动旧 Fish Speech 服务
├── Dockerfile                  # Docker 镜像构建
├── docker-compose.yml          # Docker Compose 配置
├── run_docker.bat              # 一键 Docker 启动
├── pyproject.toml              # 项目依赖
├── README.md
├── third_party/                # 引擎源码（安装后生成，不入库）
│   └── index-tts/
├── pretrained_models/          # 模型权重（安装后生成，不入库，约 26GB）
│   ├── CosyVoice2-0.5B/
│   └── IndexTTS-2.5/
├── references/                 # 已保存音色（运行时生成，不入库）
│   └── <voice_id>/
└── outputs/                    # 合成音频输出（运行时生成，不入库）
```

## 常见问题

1. **安装时报 pynini 错误**：本项目的 `requirements-cosyvoice.txt` 已移除 `wetext`，默认关闭文本前端，不需要 pynini。
2. **显存不足（OOM）**：启动时加 `--half`；如果仍不足，可改用更小的 `CosyVoice-300M` 模型。
3. **Windows 下 `torch.compile` 报错**：CosyVoice 默认不使用 compile，无需处理。
4. **日语合成效果不佳**：确保参考音频是日语，且 `reference_text` 为日语音素/文本；语速和文本前端可微调。
