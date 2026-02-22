---
name: openclaw-asr-whisperx
description: >
  Transcribe audio/video using WhisperX (faster-whisper + wav2vec2 alignment +
  speaker diarization). GPU-accelerated. Accepts Google Drive links, Telegram files, or local paths.
  Features: OpenCC 繁體輸出, hotwords, corrections, speaker embedding matching.
  Sub-skill of openclaw-local-asr-skill.
  Triggers on keywords: 轉逐字稿, 轉文字, transcribe, transcript, 語音轉文字, ASR, 字幕, subtitle,
  辨識成文字, 語音辨識.
metadata:
  openclaw:
    emoji: "🎙️"
    requires:
      bins: ["ffmpeg", "gdown", "python3"]
    os: ["linux"]
---

# ASR — WhisperX Mode (v2)

Transcribe audio/video using **WhisperX** (faster-whisper + wav2vec2 alignment + speaker diarization). Accepts Google Drive links, Telegram files, or local paths.

## v2 Features

- **Topic-guided initial_prompt** — improves accuracy for domain-specific content
- **Audio denoising** — optional ffmpeg-based noise reduction
- **OpenCC s2twp** — auto-converts simplified → traditional Chinese (Taiwan usage)
- **Hotwords** — faster-whisper native hotword boosting from `config/hotwords.txt`
- **Corrections dictionary** — post-processing replacements from `config/corrections.json`
- **Speaker embedding** — auto-extract speaker samples, match against registered DB
- **Speaker diarization** — pyannote speaker-diarization-3.1
- **Subtitle splitting** — `--max-chars` for limiting characters per subtitle line (word-level timestamp precision)

## Pre-Transcription Interaction (IMPORTANT)

**Before starting transcription, check if the user has provided enough context:**

### 1. Topic / 主題
If the user did NOT mention the audio topic/subject, ask:
```
這個音檔的主題是什麼？（例如：財經討論、會議記錄、課堂講座、日常對話等）
提供主題可以提升辨識準確度 📈
如果不確定，直接說「不用」我就開始轉了。
```
Use the user's answer as `--topic` parameter.

### 2. max-chars / 字幕字數限制
Based on the topic, auto-determine `--max-chars`:

| Topic Type | Default max-chars | Notes |
|------------|------------------|-------|
| 會議記錄 | not set (0) | longer segments, no splitting |
| 演講 | 25 | suitable for video subtitles |
| 分享會 | 25 | suitable for video subtitles |
| Podcast | 25 | suitable for video subtitles |
| Unspecified | not set (0) | default no splitting |
| User-specified | user's number | highest priority |

**Priority:** user-specified number > topic default > no splitting (0)

### 3. Denoising / 降噪
If the user explicitly mentions 降噪、雜音多、音質不好、背景噪音, add `--denoise` flag.
**Do NOT proactively ask about denoising** — only apply when user mentions it.

### 4. Speaker Diarization
If user asks to identify speakers (辨識說話者, 分辨講者, diarize, 誰在說話), add `--diarize`.

## Prerequisites

Python venv at `/home/kino/asr/.venv-whisperx/` with whisperx + PyTorch nightly (CUDA 12.8 for RTX 50 series).

Required packages: `whisperx`, `gdown`, `opencc-python-reimplemented`, `soundfile`, `numpy`, `pyannote.audio`

## Workflow

**CRITICAL: Run ALL steps without stopping. Deliver results via Telegram when done.**

### Step 1: Acquire File

**If called from the asr-local router with a local file path, skip this step.**

For Google Drive links:
```bash
gdown "https://drive.google.com/uc?id={FILE_ID}" -O /home/kino/asr/downloads/{filename}
```

For Telegram files or local paths, use the file path directly.

### Step 2: Run WhisperX Transcription

Resolve config file paths relative to the **repo root** (parent of this SKILL.md's directory):

```bash
REPO_DIR="$(dirname "${SKILL_DIR}")"
```

Output goes to `/home/kino/asr/output/`. Intermediate WAV in `/home/kino/asr/tmp/` is auto-cleaned.

Basic (with topic):
```bash
/home/kino/asr/.venv-whisperx/bin/python3 "${SKILL_DIR}/scripts/transcribe_whisperx.py" \
    /home/kino/asr/downloads/{filename} --lang zh --format srt --topic "主題描述" \
    --hotwords-file "${REPO_DIR}/config/hotwords.txt" \
    --corrections-file "${REPO_DIR}/config/corrections.json"
```

With denoising:
```bash
/home/kino/asr/.venv-whisperx/bin/python3 "${SKILL_DIR}/scripts/transcribe_whisperx.py" \
    /home/kino/asr/downloads/{filename} --lang zh --format srt --topic "主題" --denoise \
    --hotwords-file "${REPO_DIR}/config/hotwords.txt" \
    --corrections-file "${REPO_DIR}/config/corrections.json"
```

With speaker diarization:
```bash
HF_TOKEN=hf_xxx /home/kino/asr/.venv-whisperx/bin/python3 "${SKILL_DIR}/scripts/transcribe_whisperx.py" \
    /home/kino/asr/downloads/{filename} --lang zh --format srt --topic "主題" --diarize \
    --hotwords-file "${REPO_DIR}/config/hotwords.txt" \
    --corrections-file "${REPO_DIR}/config/corrections.json"
```

With subtitle splitting:
```bash
/home/kino/asr/.venv-whisperx/bin/python3 "${SKILL_DIR}/scripts/transcribe_whisperx.py" \
    /home/kino/asr/downloads/{filename} --lang zh --format srt --topic "主題" --max-chars 25 \
    --hotwords-file "${REPO_DIR}/config/hotwords.txt" \
    --corrections-file "${REPO_DIR}/config/corrections.json"
```

The script automatically:
- Loads hotwords (boosts accuracy for domain terms)
- Loads corrections (fixes known ASR errors)
- Converts output to traditional Chinese via OpenCC (s2twp mode)
- When `--diarize`: extracts speaker audio samples → matches against speaker DB → saves unknown speakers for future matching

### Step 3: Report Results & Deliver via Telegram

1. Send via Telegram `message` tool:
   ```
   action: send
   message: "轉寫完成！{basename}.srt（WhisperX，{duration}s）"
   filePath: /home/kino/asr/output/{basename}.srt
   ```

2. If `--diarize` was used and there are unmatched speakers, inform the user:
   ```
   辨識出 {n} 位說話者。
   未匹配的說話者音檔已保存在：{speaker_samples_dir}
   你可以之後告訴我「把 SPEAKER_00 命名為 XXX」來註冊聲紋。
   ```

## Speaker Embedding Management

### Registering a speaker
```bash
/home/kino/asr/.venv-whisperx/bin/python3 "${SKILL_DIR}/scripts/speaker_embed.py" \
    register --name "名字" --audio /path/to/audio.wav
```

### Renaming a SPEAKER_XX from a previous session
```bash
/home/kino/asr/.venv-whisperx/bin/python3 "${SKILL_DIR}/scripts/speaker_embed.py" \
    rename --sample-dir /home/kino/asr/speaker_samples/{session_dir} \
    --speaker SPEAKER_00 --name "名字"
```

### Listing registered speakers
```bash
/home/kino/asr/.venv-whisperx/bin/python3 "${SKILL_DIR}/scripts/speaker_embed.py" list
```

### Deleting a registered speaker
```bash
/home/kino/asr/.venv-whisperx/bin/python3 "${SKILL_DIR}/scripts/speaker_embed.py" \
    delete --name "名字"
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `WHISPERX_MODEL` | `large-v3-turbo` | Model size |
| `WHISPERX_DEVICE` | `auto` | Device (cuda/cpu/auto) |
| `ASR_COMPUTE_TYPE` | `int8` | Compute type |
| `WHISPERX_BATCH_SIZE` | `16` | Batch size for inference |
| `HF_TOKEN` | (none) | HuggingFace token for diarization |
| `HF_HOME` | `/home/kino/ollama-models/huggingface-hub` | Model cache |
| `--topic` | (none) | Topic description for initial_prompt |
| `--denoise` | false | Apply audio denoising |
| `--no-opencc` | false | Disable OpenCC traditional Chinese conversion |
| `--max-chars` | 0 (disabled) | Max characters per subtitle segment (recommended: 20-25 for Chinese) |

## Config Files

| File | Location | Purpose |
|------|----------|---------|
| `corrections.json` | `config/` (repo root) | Error→correct word mappings |
| `hotwords.txt` | `config/` (repo root) | Hotword list (one per line) |
| `speakers.json` | `/home/kino/asr/speaker_embeddings/` | Registered speaker metadata |

## Supported Input

- **Audio**: MP3, WAV, M4A, FLAC, OGG, AAC, WMA
- **Video**: MP4, MKV, AVI, MOV, WebM, FLV
- **Sources**: Google Drive links, local file paths

## References

- [WhisperX](https://github.com/m-bain/whisperX) — INTERSPEECH 2023
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — hotwords support in v1.2+
- [Silero VAD](https://github.com/snakers4/silero-vad)
- [OpenCC](https://github.com/BYVoid/OpenCC) — Chinese conversion
- [pyannote-audio](https://github.com/pyannote/pyannote-audio) — speaker diarization & embedding
