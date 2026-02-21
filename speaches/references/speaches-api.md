# speaches API Quick Reference

speaches is an OpenAI API-compatible server for Speech-to-Text powered by faster-whisper.

## Transcription Endpoint

```
POST /v1/audio/transcriptions
```

### Form Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `file` | file | (required) | Audio file to transcribe |
| `model` | string | (required) | Model name (e.g. `deepdml/faster-whisper-large-v3-turbo-ct2`) |
| `language` | string | auto | Language code (zh, en, ja, etc.) |
| `response_format` | string | json | Output format: text, srt, vtt, json, verbose_json |
| `temperature` | float | 0.0 | Sampling temperature (0 = deterministic) |
| `condition_on_previous_text` | bool | true | Use previous text as context (set false to reduce hallucinations) |

### Response Formats

- **text**: Plain text transcript
- **srt**: SRT subtitle format with timestamps
- **vtt**: WebVTT subtitle format
- **json**: `{"text": "..."}`
- **verbose_json**: Detailed JSON with segments, timestamps, confidence

### verbose_json Response Structure

```json
{
  "text": "full transcript text",
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 5.5,
      "text": "segment text",
      "tokens": [...],
      "temperature": 0.0,
      "avg_logprob": -0.25,
      "compression_ratio": 1.2,
      "no_speech_prob": 0.01
    }
  ],
  "language": "zh",
  "duration": 64.31
}
```

## Health Check

```
GET /health
```

Returns 200 when the server is ready.

## Models Endpoint

```
GET /v1/models
```

Lists available/loaded models.

## Recommended Settings for Chinese Audio

```bash
curl -s -X POST http://localhost:18996/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "model=deepdml/faster-whisper-large-v3-turbo-ct2" \
  -F "response_format=verbose_json" \
  -F "language=zh" \
  -F "condition_on_previous_text=false" \
  -F "temperature=0"
```

## Known Issues

1. **VAD too aggressive**: Whisper's Voice Activity Detection can filter out speech with natural pauses. Solution: chunk audio into 15-second segments before transcription.
2. **Hallucinations**: When audio is mostly silent, Whisper may hallucinate text. Solution: use `temperature=0` and `condition_on_previous_text=false`.
3. **Language detection**: Auto-detection may fail for short segments. Solution: always specify `language` explicitly.

## Links

- [speaches GitHub](https://github.com/speaches-ai/speaches)
- [faster-whisper GitHub](https://github.com/SYSTRAN/faster-whisper)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference/audio/createTranscription)
