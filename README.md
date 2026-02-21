# openclaw-local-asr

[中文](#中文) | [English](#english)

---

## 中文

### 簡介

本地 GPU 加速的語音轉逐字稿工具包，支援兩種轉錄引擎，可作為獨立 CLI 工具或 AI Agent Skill 使用。

### 兩種 ASR 模式

| | speaches | whisperx |
|---|---|---|
| 引擎 | speaches Docker (faster-whisper) | WhisperX (本地 Python) |
| VAD | ffmpeg silencedetect 智能切割 | Silero VAD 內建 |
| 時間戳 | segment 級別 | word 級別（wav2vec2 對齊）|
| 說話者辨識 | 無 | 有（pyannote，需 HF_TOKEN）|
| 字幕切割 | 無 | 有（`--max-chars`，word-level 精確切割）|
| 繁體轉換 | 無 | 有（OpenCC s2twp）|
| 熱詞加強 | 無 | 有（faster-whisper native hotwords）|
| 字詞糾正 | 無 | 有（corrections.json）|
| 速度 | 較快（Docker API） | 稍慢（首次載入模型）|
| 需要 Docker | 是 | 否 |

### 前置需求

**通用：**
- NVIDIA GPU（支援 CUDA）
- ffmpeg v5+
- gdown（`pip install gdown`）

**speaches 模式額外需要：**
- Docker + Docker Compose
- NVIDIA Container Toolkit v1.18+

**whisperx 模式額外需要：**
- Python 3.12（不支援 3.14）
- RTX 50 系列需 PyTorch nightly + CUDA 12.8

### 安裝

#### 1. Clone

```bash
git clone https://github.com/Kinolian1107/openclaw-local-asr.git
cd openclaw-local-asr
```

#### 2. speaches 模式設定

```bash
# 安裝 NVIDIA Container Toolkit（如尚未安裝）
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update && sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# 啟動 speaches 容器（根據 speaches/docker/docker-compose.yml 調整路徑）
sudo docker compose -f speaches/docker/docker-compose.yml up -d
```

#### 3. whisperx 模式設定

```bash
# 建立虛擬環境
/usr/bin/python3.12 -m venv /path/to/asr-venv

# 安裝 WhisperX
/path/to/asr-venv/bin/pip install whisperx gdown opencc-python-reimplemented

# RTX 50 系列（Blackwell/sm_120）必要步驟
/path/to/asr-venv/bin/pip install --pre torch torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu128 \
    --force-reinstall --no-deps
/path/to/asr-venv/bin/pip install --pre nvidia-cudnn-cu12 nvidia-nccl-cu12 \
    --index-url https://download.pytorch.org/whl/nightly/cu128 \
    --force-reinstall --no-deps

# （選用）說話者辨識需 HuggingFace Token
export HF_TOKEN=hf_your_token_here
```

### 使用方式

#### CLI

```bash
# speaches 模式
python3 speaches/scripts/transcribe_smart.py /path/to/audio.mp3 --lang zh

# whisperx 模式
python3 whisperx/scripts/transcribe_whisperx.py /path/to/audio.mp3 --lang zh --format srt

# whisperx + 說話者辨識 + 字幕切割
HF_TOKEN=hf_xxx python3 whisperx/scripts/transcribe_whisperx.py \
    /path/to/audio.mp3 --lang zh --format srt --diarize --max-chars 25 \
    --topic "投資分享會"
```

#### AI Agent 整合

**OpenClaw：**
```bash
# 建立 3 個 symlink（路由器 + 兩個引擎）
ln -sf /path/to/openclaw-local-asr ~/.openclaw/skills/gfile-asr
ln -sf /path/to/openclaw-local-asr/speaches ~/.openclaw/skills/gfile-asr-speaches
ln -sf /path/to/openclaw-local-asr/whisperx ~/.openclaw/skills/gfile-asr-whisperx
```

然後在 Telegram 中傳送 Google Drive 連結並說「轉逐字稿」。

**Cursor / Claude Code / Codex / Gemini CLI：**
Clone 此 repo 到專案目錄。Agent 會自動讀取 `SKILL.md` / `CLAUDE.md` / `AGENTS.md` / `GEMINI.md`。

### 設定檔

所有設定檔在 `config/` 目錄：

| 檔案 | 用途 |
|------|------|
| `asr_config.json` | ASR 模式選擇、路徑設定 |
| `corrections.json` | 字詞糾正對照表（ASR 常見錯誤修正）|
| `hotwords.txt` | 熱詞清單（提升 WhisperX 辨識準確度）|

### 專案結構

```
openclaw-local-asr/
├── SKILL.md                  # 路由器 skill（判斷模式，導向子 skill）
├── README.md
├── AGENTS.md                 # AI agent 通用指引
├── CLAUDE.md                 # Claude Code 指引
├── GEMINI.md                 # Gemini 指引
├── .github/
│   └── copilot-instructions.md
├── config/
│   ├── asr_config.json       # 模式切換設定
│   ├── corrections.json      # 字詞糾正表
│   └── hotwords.txt          # 熱詞清單
├── speaches/
│   ├── SKILL.md              # speaches 專屬 skill 指引
│   ├── scripts/
│   │   ├── transcribe_smart.py
│   │   ├── transcribe_optimized.py
│   │   └── transcribe.sh
│   ├── docker/
│   │   └── docker-compose.yml
│   └── references/
│       └── speaches-api.md
└── whisperx/
    ├── SKILL.md              # whisperx 專屬 skill 指引
    └── scripts/
        ├── transcribe_whisperx.py
        └── speaker_embed.py
```

---

## English

### Overview

Local GPU-accelerated speech-to-text toolkit with two transcription engines. Works as a standalone CLI tool or as an AI Agent Skill.

### Two ASR Modes

| | speaches | whisperx |
|---|---|---|
| Engine | speaches Docker (faster-whisper) | WhisperX (local Python) |
| VAD | ffmpeg silencedetect | Silero VAD built-in |
| Timestamps | segment-level | word-level (wav2vec2 alignment) |
| Speaker ID | No | Yes (pyannote, needs HF_TOKEN) |
| Subtitle split | No | Yes (`--max-chars`, word-level precision) |
| Traditional Chinese | No | Yes (OpenCC s2twp) |
| Hotwords | No | Yes (faster-whisper native) |
| Corrections | No | Yes (corrections.json) |
| Speed | Faster (Docker API) | Slightly slower (model loading) |
| Requires Docker | Yes | No |

### Quick Start

```bash
git clone https://github.com/Kinolian1107/openclaw-local-asr.git
cd openclaw-local-asr

# speaches mode
python3 speaches/scripts/transcribe_smart.py /path/to/audio.mp3 --lang zh

# whisperx mode
python3 whisperx/scripts/transcribe_whisperx.py /path/to/audio.mp3 --lang zh --format srt
```

### AI Agent Installation

**OpenClaw:**
```bash
ln -sf /path/to/openclaw-local-asr ~/.openclaw/skills/gfile-asr
ln -sf /path/to/openclaw-local-asr/speaches ~/.openclaw/skills/gfile-asr-speaches
ln -sf /path/to/openclaw-local-asr/whisperx ~/.openclaw/skills/gfile-asr-whisperx
```

**Cursor / Claude Code / Codex / Gemini CLI:**
Clone this repo into your project directory. The agent will read the appropriate instruction file automatically.

## License

MIT
