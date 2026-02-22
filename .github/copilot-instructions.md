# openclaw-local-asr-skill — Copilot Instructions

Local ASR toolkit with two modes: **speaches** (Docker-based) and **whisperx** (local Python).

## Quick Start

1. Read `config/asr_config.json` to check current mode
2. Based on `"mode"` value, follow the corresponding skill:
   - `"speaches"` → `speaches/SKILL.md`
   - `"whisperx"` → `whisperx/SKILL.md`

## Speaches Mode

```bash
python3 speaches/scripts/transcribe_smart.py <file_or_url> --lang zh --format srt
```

## WhisperX Mode

```bash
python3 whisperx/scripts/transcribe_whisperx.py <file_or_url> --lang zh --format srt \
    --hotwords-file config/hotwords.txt --corrections-file config/corrections.json
```

Optional: `--topic "主題"`, `--denoise`, `--diarize`, `--max-chars 25`

## CRITICAL

- Run the entire pipeline without stopping
- Config files are in `config/` directory
