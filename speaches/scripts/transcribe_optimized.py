#!/usr/bin/env python3
"""
Optimized ASR Pipeline: Silero VAD preprocessing + speaches API.

Strategy:
1. Silero VAD detects speech segments (threshold=0.3, handles silence-heavy audio)
2. Merge nearby speech into processing chunks (max 5min each)
3. Send each chunk to speaches Docker API (faster-whisper with GPU via Docker)
4. Combine results with corrected timestamps into SRT

This avoids Whisper's aggressive built-in VAD while leveraging the existing
speaches Docker container for GPU-accelerated inference.

Usage:
    python3 transcribe_optimized.py <input_file> [--lang zh] [--format srt]
"""

import argparse
import json
import os
import struct
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

ASR_DIR = os.environ.get("ASR_DIR", "/home/kino/asr")
SPEACHES_URL = os.environ.get("SPEACHES_URL", "http://localhost:18996")
MODEL = os.environ.get("ASR_MODEL", "deepdml/faster-whisper-large-v3-turbo-ct2")


def get_duration(filepath: str) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", filepath],
        capture_output=True, text=True
    )
    return float(result.stdout.strip())


def extract_audio(input_path: str, output_path: str) -> str:
    subprocess.run(
        ["ffmpeg", "-y", "-i", input_path, "-vn", "-acodec", "pcm_s16le",
         "-ar", "16000", "-ac", "1", output_path],
        capture_output=True, check=True
    )
    return output_path


def detect_mime(filepath: str) -> str:
    result = subprocess.run(
        ["file", "--brief", "--mime-type", filepath],
        capture_output=True, text=True
    )
    return result.stdout.strip()


def format_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def load_wav_torch(filepath: str):
    """Load 16kHz mono WAV as torch tensor without sox dependency."""
    import torch

    with wave.open(filepath, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sampwidth == 2:
        samples = struct.unpack(f"<{n_frames * n_channels}h", raw)
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth}")

    tensor = torch.FloatTensor(samples) / 32768.0
    if n_channels > 1:
        tensor = tensor.view(-1, n_channels).mean(dim=1)
    return tensor


def merge_speech_segments(timestamps, max_gap=2.0, max_length=300.0):
    """Merge nearby speech segments, respecting max chunk length."""
    if not timestamps:
        return []

    merged = []
    current = {"start": timestamps[0]["start"], "end": timestamps[0]["end"]}

    for ts in timestamps[1:]:
        gap = ts["start"] - current["end"]
        new_length = ts["end"] - current["start"]

        if gap <= max_gap and new_length <= max_length:
            current["end"] = ts["end"]
        else:
            merged.append(dict(current))
            current = {"start": ts["start"], "end": ts["end"]}

    merged.append(dict(current))
    return merged


def transcribe_chunk_via_api(chunk_path: str, language: str) -> dict:
    """Send an audio chunk to speaches API and get verbose_json response."""
    cmd = [
        "curl", "-s", "-X", "POST",
        f"{SPEACHES_URL}/v1/audio/transcriptions",
        "-F", f"file=@{chunk_path}",
        "-F", f"model={MODEL}",
        "-F", "response_format=verbose_json",
        "-F", "condition_on_previous_text=false",
        "-F", "temperature=0",
    ]
    if language != "auto":
        cmd += ["-F", f"language={language}"]

    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        print(f"  WARNING: Failed to parse API response: {result.stdout[:200]}")
        return {"segments": [], "text": ""}


def extract_chunk(source: str, start: float, duration: float, output: str):
    """Extract a time range from audio file."""
    subprocess.run(
        ["ffmpeg", "-y", "-i", source,
         "-ss", str(start), "-t", str(duration),
         "-ar", "16000", "-ac", "1", "-acodec", "pcm_s16le", output],
        capture_output=True, check=True
    )


def write_srt(segments, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            f.write(f"{i}\n")
            f.write(f"{format_timestamp(seg['start'])} --> {format_timestamp(seg['end'])}\n")
            f.write(f"{seg['text']}\n\n")


def transcribe_with_vad(audio_path: str, language: str, output_format: str):
    """Main pipeline: Silero VAD detection → speaches API transcription."""
    import torch
    from silero_vad import load_silero_vad, get_speech_timestamps

    print("Loading Silero VAD model...")
    vad_model = load_silero_vad()

    print(f"Loading audio: {audio_path}")
    wav = load_wav_torch(audio_path)
    duration = len(wav) / 16000
    print(f"Duration: {duration:.1f}s")

    print("Detecting speech segments with Silero VAD (threshold=0.3)...")
    speech_timestamps = get_speech_timestamps(
        wav, vad_model,
        threshold=0.3,
        min_silence_duration_ms=300,
        speech_pad_ms=200,
        min_speech_duration_ms=250,
        return_seconds=True
    )

    if not speech_timestamps:
        print("WARNING: No speech detected. Processing full audio.")
        speech_timestamps = [{"start": 0.0, "end": duration}]

    total_speech = sum(ts["end"] - ts["start"] for ts in speech_timestamps)
    silence_ratio = 1 - (total_speech / duration) if duration > 0 else 0
    print(f"Speech: {total_speech:.1f}s / {duration:.1f}s (silence ratio: {silence_ratio:.0%})")
    print(f"Detected {len(speech_timestamps)} speech segments")

    merged = merge_speech_segments(speech_timestamps, max_gap=2.0, max_length=300.0)
    print(f"Merged into {len(merged)} processing chunks")

    all_segments = []
    tmp_dir = tempfile.mkdtemp(prefix="asr_chunks_")

    try:
        for i, seg_range in enumerate(merged):
            seg_start = seg_range["start"]
            seg_end = seg_range["end"]
            seg_duration = seg_end - seg_start

            print(f"  Chunk {i+1}/{len(merged)}: "
                  f"{format_timestamp(seg_start)} - {format_timestamp(seg_end)} "
                  f"({seg_duration:.1f}s)", end="", flush=True)

            chunk_path = os.path.join(tmp_dir, f"chunk_{i:04d}.wav")
            extract_chunk(audio_path, seg_start, seg_duration, chunk_path)

            response = transcribe_chunk_via_api(chunk_path, language)

            if "segments" in response and response["segments"]:
                for seg in response["segments"]:
                    text = seg.get("text", "").strip()
                    if not text:
                        continue
                    all_segments.append({
                        "start": seg["start"] + seg_start,
                        "end": seg["end"] + seg_start,
                        "text": text
                    })
            elif "text" in response and response["text"].strip():
                all_segments.append({
                    "start": seg_start,
                    "end": seg_end,
                    "text": response["text"].strip()
                })

            n_segs = len(response.get("segments", []))
            print(f" → {n_segs} segments")

    finally:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"\nTotal: {len(all_segments)} segments")

    basename = Path(audio_path).stem
    output_base = os.path.join(ASR_DIR, basename)

    srt_path = f"{output_base}.srt"
    write_srt(all_segments, srt_path)
    print(f"SRT: {srt_path}")

    txt_path = f"{output_base}.txt"
    full_text = " ".join(s["text"] for s in all_segments)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(full_text)
    print(f"TXT: {txt_path}")

    if output_format in ("json", "all"):
        json_path = f"{output_base}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "text": full_text,
                "segments": all_segments,
                "duration": duration,
                "language": language,
                "model": MODEL,
                "vad": "silero-vad",
                "speech_ratio": 1 - silence_ratio,
            }, f, ensure_ascii=False, indent=2)
        print(f"JSON: {json_path}")

    return all_segments


def main():
    parser = argparse.ArgumentParser(
        description="Optimized ASR: Silero VAD + speaches API (faster-whisper Docker)")
    parser.add_argument("input", help="Input audio/video file path")
    parser.add_argument("--lang", default="zh", help="Language code (zh, en, ja, auto)")
    parser.add_argument("--format", default="srt", choices=["srt", "text", "json", "all"],
                        help="Output format")
    args = parser.parse_args()

    input_path = args.input
    if not os.path.exists(input_path):
        print(f"ERROR: File not found: {input_path}")
        sys.exit(1)

    mime = detect_mime(input_path)
    print(f"Input: {input_path} ({mime})")

    basename = Path(input_path).stem
    wav_path = os.path.join(ASR_DIR, f"{basename}.wav")

    if not input_path.endswith(".wav"):
        print("Converting to WAV (16kHz mono)...")
        extract_audio(input_path, wav_path)
    else:
        wav_path = input_path

    segments = transcribe_with_vad(wav_path, args.lang, args.format)
    print(f"\nDone! {len(segments)} segments transcribed.")


if __name__ == "__main__":
    main()
