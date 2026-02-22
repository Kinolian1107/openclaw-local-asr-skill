---
name: openclaw-asr-speaches
description: >
  Transcribe audio/video files using ffmpeg silence-detection + speaches Docker
  (faster-whisper with GPU). Accepts Google Drive links, Telegram files, or local paths.
  Outputs SRT subtitles, plain text, or JSON. Sub-skill of openclaw-local-asr-skill.
  Triggers on keywords: 轉逐字稿, 轉文字, transcribe, transcript, 語音轉文字, ASR, 字幕, subtitle.
metadata:
  openclaw:
    emoji: "🎙️"
    requires:
      bins: ["ffmpeg", "curl", "gdown", "python3"]
    os: ["linux"]
---

# ASR — Speaches Mode

Transcribe audio/video using **ffmpeg silence-detection + speaches Docker API** (faster-whisper, GPU-accelerated). Accepts Google Drive links, Telegram files, or local paths.

## How It Works

1. **Silero VAD** (Voice Activity Detection) detects precise speech segments (threshold=0.3)
2. Merges nearby speech into processing chunks (max 5 min each, with 2s gap tolerance)
3. Sends only speech-containing chunks to **speaches Docker API** (faster-whisper large-v3-turbo, int8, CUDA)
4. Combines results with corrected timestamps into SRT

This eliminates the hallucination problem in silence-heavy audio and significantly reduces processing time by skipping silent regions.

## Prerequisites

Verify speaches Docker container is running:

```bash
docker ps --filter name=speaches --format '{{.Status}}'
```

If not running:

```bash
sudo docker compose -f /opt/docker/docker-compose.yml up -d speaches
```

## Workflow

**CRITICAL: Run ALL steps in sequence without stopping. Do NOT wait for user prompts between steps. The entire pipeline must complete autonomously. Proactively report results via Telegram when done.**

### Step 1: Acquire File

**If called from the asr-local router with a local file path, skip this step.**

For Google Drive links:
```bash
gdown "https://drive.google.com/uc?id={FILE_ID}" -O /home/kino/asr/downloads/{filename}
```

For Telegram files or local paths, use the file path directly.

### Step 2: Run Smart Transcription

```bash
python3 "${SKILL_DIR}/scripts/transcribe_smart.py" \
    /home/kino/asr/downloads/{filename} --lang zh --format srt
```

Output goes to `/home/kino/asr/output/`. Intermediate WAV and chunks in `/home/kino/asr/tmp/` are auto-cleaned.

The script handles everything automatically:
- Detects file type (audio/video), converts to WAV if needed (ffmpeg)
- **ffmpeg silencedetect** with adaptive thresholds detects silence boundaries
- Smart chunking: speech-segment extraction for high-silence audio, split-at-silence for continuous audio
- Sends chunks to speaches Docker API (faster-whisper large-v3-turbo, int8, CUDA)
- Hallucination filtering (regex patterns for common Whisper artifacts)
- Combines results with corrected timestamps into SRT

### Step 3: Report Results & Deliver via Telegram

**IMPORTANT: After transcription completes, you MUST proactively notify the user.**

1. Send via Telegram using the `message` tool:
   ```
   action: send
   channel: telegram
   message: "轉寫完成！{basename}.srt（{duration}s，{n_segments} 條字幕）"
   filePath: /home/kino/asr/output/{basename}.srt
   ```

2. Display summary in conversation:
   ```
   轉寫完成！
   📁 {basename}.srt
   ⏱️ {duration}s
   🔤 語言：{language}
   📨 已透過 Telegram 傳送
   ```

3. If user also asked for 摘要/分析, proceed to analyze the transcript.

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SPEACHES_URL` | `http://localhost:18996` | speaches API endpoint |
| `ASR_MODEL` | `deepdml/faster-whisper-large-v3-turbo-ct2` | Whisper model |
| `ASR_DIR` | `/home/kino/asr` | Working directory |
| `--format` | `srt` | Output format: srt, text, json, all |
| `--lang` | `zh` | Language code (zh, en, ja, auto) |

## Supported Input

- **Audio**: MP3, WAV, M4A, FLAC, OGG, AAC, WMA
- **Video**: MP4, MKV, AVI, MOV, WebM, FLV
- **Sources**: Google Drive links, local file paths

## Known Issues & Solutions

| Issue | Solution |
|-------|----------|
| VAD too aggressive on silence-heavy audio | Silero VAD (threshold=0.3) only sends speech segments |
| Whisper hallucinations | `temperature=0` + `condition_on_previous_text=false` + silence skipping |
| Language detection errors | Always specify `--lang zh` for Chinese content |
| Docker needs sudo | Some environments require `sudo docker` |
| Python venv required | Use `/home/kino/asr/.venv/bin/python3` (has torch, silero-vad, faster-whisper) |

## References

- [speaches (faster-whisper-server)](https://github.com/speaches-ai/speaches)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
