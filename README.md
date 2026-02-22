# openclaw-local-asr-skill

[中文](#中文) | [English](#english)

---

## 中文

### 簡介

本地 GPU 加速的語音轉逐字稿 AI Agent Skill，支援兩種轉錄引擎。可接收 Google Drive 連結、Telegram 音訊/影片檔案、或本地檔案路徑，自動完成轉錄並回傳結果。

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

### 架構

```
輸入來源（GDrive / Telegram / 本地檔案）
    ↓
路由器 SKILL.md → 讀取 config/asr_config.json
    ↓
mode == "speaches"?
├─ YES → speaches/SKILL.md → speaches/scripts/transcribe_smart.py
└─ NO  → whisperx/SKILL.md → whisperx/scripts/transcribe_whisperx.py
    ↓
輸出 SRT/TXT → 透過 Telegram 傳送
```

### 工作目錄結構

```
/home/kino/asr/                     ← 運行時工作目錄（--output-dir）
├── downloads/                      ← 下載的來源檔（mp3, mp4 等）
├── tmp/                            ← 中間檔（WAV, chunks）— 轉寫完自動清理
├── output/                         ← 最終輸出（SRT, TXT, JSON）
├── speaker_embeddings/             ← 已註冊的說話者聲紋
├── speaker_samples/                ← 自動擷取的說話者音檔樣本
├── .venv/                          ← speaches 模式 Python venv
└── .venv-whisperx/                 ← whisperx 模式 Python venv
```

### 資料夾分工

| 位置 | 用途 |
|------|------|
| 本 repo (`config/`) | 設定檔：模式切換、熱詞、字詞糾正 |
| 本 repo (`speaches/`, `whisperx/`) | 轉錄腳本、子 skill 指引 |
| `{ASR_DIR}/downloads/` | 下載的來源檔案 |
| `{ASR_DIR}/tmp/` | 中間產物（WAV、chunks），轉寫完自動清理 |
| `{ASR_DIR}/output/` | 最終輸出（SRT, TXT, JSON）|
| `{ASR_DIR}/speaker_embeddings/` | 已註冊的說話者聲紋 |
| `{ASR_DIR}/speaker_samples/` | 自動擷取的說話者音檔樣本 |

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
git clone https://github.com/Kinolian1107/openclaw-local-asr-skill.git
cd openclaw-local-asr-skill
```

#### 2. 建立工作目錄

```bash
mkdir -p /home/kino/asr/{downloads,tmp,output,speaker_embeddings,speaker_samples}
```

#### 3. speaches 模式設定

```bash
# 安裝 NVIDIA Container Toolkit（如尚未安裝）
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update && sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# 建立 speaches venv
python3 -m venv /home/kino/asr/.venv
/home/kino/asr/.venv/bin/pip install torch silero-vad gdown

# 啟動 speaches 容器
sudo docker compose -f speaches/docker/docker-compose.yml up -d
```

#### 4. whisperx 模式設定

```bash
# 建立虛擬環境
/usr/bin/python3.12 -m venv /home/kino/asr/.venv-whisperx

# 安裝 WhisperX
/home/kino/asr/.venv-whisperx/bin/pip install whisperx gdown opencc-python-reimplemented

# RTX 50 系列（Blackwell/sm_120）必要步驟
/home/kino/asr/.venv-whisperx/bin/pip install --pre torch torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu128 \
    --force-reinstall --no-deps
/home/kino/asr/.venv-whisperx/bin/pip install --pre nvidia-cudnn-cu12 nvidia-nccl-cu12 \
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
/home/kino/asr/.venv-whisperx/bin/python3 whisperx/scripts/transcribe_whisperx.py \
    /path/to/audio.mp3 --lang zh --format srt

# whisperx + 說話者辨識 + 字幕切割
HF_TOKEN=hf_xxx /home/kino/asr/.venv-whisperx/bin/python3 whisperx/scripts/transcribe_whisperx.py \
    /path/to/audio.mp3 --lang zh --format srt --diarize --max-chars 25 \
    --topic "投資分享會"
```

#### AI Agent 整合

**OpenClaw：**
```bash
# 只需 1 個 symlink（路由器會自動呼叫子 skill）
ln -sf /path/to/openclaw-local-asr-skill ~/.openclaw/skills/gfile-asr
```

然後在 Telegram 中傳送 Google Drive 連結或音檔並說「轉逐字稿」。

**Cursor / Claude Code / Codex / Gemini CLI：**
Clone 此 repo 到專案目錄。Agent 會自動讀取 `SKILL.md` / `CLAUDE.md` / `AGENTS.md` / `GEMINI.md`。

### 設定檔

所有設定檔在 `config/` 目錄：

| 檔案 | 用途 |
|------|------|
| `asr_config.json` | ASR 模式選擇、引擎設定 |
| `corrections.json` | 字詞糾正對照表（ASR 常見錯誤修正）|
| `hotwords.txt` | 熱詞清單（提升 WhisperX 辨識準確度）|

### 專案結構

```
openclaw-local-asr-skill/
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

### 相關 Repo

| Repo | 用途 |
|------|------|
| [openclaw-asr-speaches-skill](https://github.com/Kinolian1107/openclaw-asr-speaches-skill) | speaches 模式獨立 repo（已合併至此，archived） |
| [openclaw-asr-whisperX-skill](https://github.com/Kinolian1107/openclaw-asr-whisperX-skill) | whisperX 模式獨立 repo（已合併至此，archived） |
| [googlefile-asr-faster-whisper-skill](https://github.com/Kinolian1107/googlefile-asr-faster-whisper-skill) | 最初版本（archived） |

---

## English

### Overview

Local GPU-accelerated speech-to-text AI Agent Skill with two transcription engines. Accepts Google Drive links, Telegram audio/video files, or local file paths, and automatically transcribes and delivers results.

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
git clone https://github.com/Kinolian1107/openclaw-local-asr-skill.git
cd openclaw-local-asr-skill

# speaches mode
python3 speaches/scripts/transcribe_smart.py /path/to/audio.mp3 --lang zh

# whisperx mode
/path/to/venv-whisperx/bin/python3 whisperx/scripts/transcribe_whisperx.py \
    /path/to/audio.mp3 --lang zh --format srt
```

### AI Agent Installation

**OpenClaw:**
```bash
# Only 1 symlink needed (router auto-delegates to sub-skills)
ln -sf /path/to/openclaw-local-asr-skill ~/.openclaw/skills/gfile-asr
```

**Cursor / Claude Code / Codex / Gemini CLI:**
Clone this repo into your project directory. The agent will read the appropriate instruction file automatically.

### Related Repos

| Repo | Purpose |
|------|---------|
| [openclaw-asr-speaches-skill](https://github.com/Kinolian1107/openclaw-asr-speaches-skill) | Standalone speaches repo (merged here, archived) |
| [openclaw-asr-whisperX-skill](https://github.com/Kinolian1107/openclaw-asr-whisperX-skill) | Standalone whisperX repo (merged here, archived) |
| [googlefile-asr-faster-whisper-skill](https://github.com/Kinolian1107/googlefile-asr-faster-whisper-skill) | Original version (archived) |

## License

MIT
