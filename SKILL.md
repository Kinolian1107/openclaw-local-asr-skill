---
name: gfile-asr
description: >
  ASR router skill — reads asr_config.json to determine the active transcription
  mode (speaches or whisperx), then delegates to the corresponding sub-skill.
  Triggers on keywords: 轉逐字稿, 轉文字, transcribe, transcript, 語音轉文字, ASR, 字幕, subtitle,
  辨識成文字, 語音辨識.
metadata:
  openclaw:
    emoji: "🎙️"
    requires:
      bins: ["ffmpeg", "gdown", "python3"]
    os: ["linux"]
---

# Google Drive ASR — Router

This is the **entry point** for all ASR (speech-to-text) tasks. It determines which transcription engine to use and delegates to the appropriate sub-skill.

## How It Works

1. Read `config/asr_config.json` (resolve relative to this SKILL.md's directory)
2. Check the `"mode"` field
3. Delegate to the corresponding sub-skill:
   - `"speaches"` → read and follow `speaches/SKILL.md` (in this same repo)
   - `"whisperx"` → read and follow `whisperx/SKILL.md` (in this same repo)

## Trigger Conditions

Activate when ANY of the following are true:

1. User sends a **Google Drive link** + mentions: 轉逐字稿, 轉文字, transcribe, transcript, 語音轉文字, ASR, 字幕, subtitle, 摘要, summary, 分析, 辨識成文字, 語音辨識
2. User provides a **local file path** to audio/video and asks for transcription
3. User says "transcribe" or "轉逐字稿" referencing a previously downloaded file

## Mode Routing

```
User request → Read config/asr_config.json
                    ↓
              mode == "speaches"?
              ├─ YES → Read speaches/SKILL.md → Follow it
              └─ NO  → Read whisperx/SKILL.md → Follow it
```

After determining the mode, **read the corresponding sub-skill's SKILL.md and follow its instructions completely**. Do not mix instructions from different modes.

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
