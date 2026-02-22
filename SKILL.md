---
name: gfile-asr
description: >
  ASR router skill — reads asr_config.json to determine the active transcription
  mode (speaches or whisperx), then delegates to the corresponding sub-skill.
  Supports Google Drive links, Telegram audio/video files, and local file paths.
  Part of openclaw-local-asr-skill.
  Triggers on keywords: 轉逐字稿, 轉文字, transcribe, transcript, 語音轉文字, ASR, 字幕, subtitle,
  辨識成文字, 語音辨識.
metadata:
  openclaw:
    emoji: "🎙️"
    requires:
      bins: ["ffmpeg", "gdown", "python3"]
    os: ["linux"]
---

# ASR Router — Speech-to-Text

This is the **entry point** for all ASR (speech-to-text) tasks. It handles file acquisition from multiple sources, then delegates transcription to the configured engine.

## How It Works

1. **Acquire the file** (see Input Sources below)
2. Read `config/asr_config.json` (resolve relative to this SKILL.md's directory)
3. Check the `"mode"` field
4. Delegate to the corresponding sub-skill:
   - `"speaches"` → read and follow `speaches/SKILL.md` (in this same repo)
   - `"whisperx"` → read and follow `whisperx/SKILL.md` (in this same repo)
5. **Pass the local file path** to the sub-skill (skip the sub-skill's download step)

## Trigger Conditions

Activate when ANY of the following are true:

1. User sends a **Google Drive link** + mentions: 轉逐字稿, 轉文字, transcribe, transcript, 語音轉文字, ASR, 字幕, subtitle, 摘要, summary, 分析, 辨識成文字, 語音辨識
2. User sends an **audio/video file via Telegram** (appears as media attachment or `<telegram_large_file>` tag)
3. User provides a **local file path** to audio/video and asks for transcription
4. User says "transcribe" or "轉逐字稿" referencing a previously downloaded file

## Input Sources & File Acquisition

### Source 1: Google Drive Link
The script handles downloads automatically into `/home/kino/asr/downloads/`:
```bash
# Just pass the URL as the input argument — gdown is called internally
```

### Source 2: Telegram File (OpenClaw media attachment)
When a user sends an audio/video file via Telegram, OpenClaw downloads it automatically (via Local Bot API if configured). The file path appears in the conversation as a media attachment — use it directly as a local file path.

### Source 3: Telegram Large File (`<telegram_large_file>` tag)
If OpenClaw cannot download the file (e.g. >20MB without Local Bot API), a `<telegram_large_file>` tag is injected into the message text containing `file_id`, `file_size`, `file_name`, and `mime_type`.

Extract the `file_id` and download using the tg-dl-localapi skill:
```bash
FILE_PATH=$(bash ~/.openclaw/skills/tg-dl-localapi/scripts/tg-download.sh "{file_id}" -o /home/kino/asr/downloads)
```

### Source 4: Local File Path
Use the path directly — no download needed.

## Directory Structure

The working directory `/home/kino/asr/` is organized as:

```
/home/kino/asr/
├── downloads/          ← Downloaded source files (mp3, mp4, etc.)
├── tmp/                ← Intermediate files (WAV, chunks) — auto-cleaned
├── output/             ← Final output (SRT, TXT, JSON)
├── speaker_embeddings/ ← Registered speaker voice prints
├── speaker_samples/    ← Extracted speaker audio samples
├── .venv/              ← speaches Python venv
└── .venv-whisperx/     ← whisperx Python venv
```

Intermediate WAV files and chunk directories in `tmp/` are automatically cleaned up after transcription completes.

## Video → Audio Conversion

Both sub-skills automatically convert video files to WAV (16kHz mono) using ffmpeg before transcription. No manual conversion is needed. Supported video formats: MP4, MKV, AVI, MOV, WebM, FLV.

## Mode Routing

```
Input (any source) → local file path
                         ↓
              Read config/asr_config.json
                         ↓
              mode == "speaches"?
              ├─ YES → Read speaches/SKILL.md → Follow it (skip Step 1 if file is local)
              └─ NO  → Read whisperx/SKILL.md → Follow it (skip Step 1 if file is local)
```

After determining the mode, **read the corresponding sub-skill's SKILL.md and follow its instructions completely**. Do not mix instructions from different modes. If the file is already downloaded locally, skip the sub-skill's Step 1 (download) and go directly to Step 2 (transcription).

## /asrmode Command

When user types `/asrmode` (with or without argument):

**Without argument (`/asrmode`):**
1. Read `config/asr_config.json`
2. Show current mode (✅ marked) and all available modes
3. Use inline buttons to let user select
4. After selection, update `config/asr_config.json` `"mode"` field
5. Confirm the switch

**With argument (`/asrmode speaches` or `/asrmode whisperx`):**
1. Update `config/asr_config.json` `"mode"` field directly
2. Confirm the switch

## Available Modes

| Mode | Engine | Key Features |
|------|--------|-------------|
| `speaches` | speaches Docker (faster-whisper) | ffmpeg silencedetect, hallucination filtering, no speaker ID |
| `whisperx` | WhisperX (local Python) | word-level timestamps, speaker diarization, hotwords, corrections |

## Config Files

All config files are in `config/` relative to this skill's directory:

| File | Purpose |
|------|---------|
| `config/asr_config.json` | Mode selection & paths |
| `config/corrections.json` | Post-processing word replacements |
| `config/hotwords.txt` | Hotword list for WhisperX accuracy boost |

## Hotword Management

When user says "增加熱詞 XXX":
- Append to `config/hotwords.txt` (one word per line)
- Confirm: "已新增熱詞：XXX ✅"

## Correction Management

When user says "把 A 改成 B":
- Add `"A": "B"` to `config/corrections.json`
- Confirm: "已新增字詞糾正：A → B ✅"
