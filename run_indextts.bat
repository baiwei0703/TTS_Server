@echo off
REM IndexTTS-2.5 engine launcher (separate venv from cosyvoice, port 8001)
cd /d D:\Codes\TTS_Server
third_party\index-tts\.venv\Scripts\python.exe app.py --engine indextts --model-dir pretrained_models/IndexTTS-2.5 --host 0.0.0.0 --port 8001
