#!/usr/bin/env python3
"""
Smart ASR Pipeline: Google Drive → ffmpeg → silence-based chunking → speaches API → SRT
Uses ffmpeg silencedetect for intelligent audio splitting and hallucination filtering.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ASR_DIR = os.environ.get("ASR_DIR", "/home/kino/asr")
SPEACHES_URL = os.environ.get("SPEACHES_URL", "http://localhost:18996")
MODEL = os.environ.get("ASR_MODEL", "deepdml/faster-whisper-large-v3-turbo-ct2")
GDOWN_PATH = os.environ.get("GDOWN_PATH", "gdown")


def run(cmd, **kwargs):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, **kwargs)
    return result.stdout.strip(), result.stderr.strip(), result.returncode


def extract_gdrive_id(url: str) -> str:
    patterns = [
        r"drive\.google\.com/file/d/([a-zA-Z0-9_-]+)",
        r"drive\.google\.com/open\?id=([a-zA-Z0-9_-]+)",
        r"id=([a-zA-Z0-9_-]+)",
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    return ""


def download_gdrive(url: str, output_dir: str) -> str:
    file_id = extract_gdrive_id(url)
    if not file_id:
        raise ValueError(f"Cannot extract Google Drive file ID from: {url}")
    output = os.path.join(output_dir, f"gdrive_{file_id}")
    print(f"Downloading from Google Drive (file_id: {file_id})...")
    stdout, stderr, rc = run(
        f'{GDOWN_PATH} "https://drive.google.com/uc?id={file_id}" -O "{output}"'
    )
    if rc != 0:
        raise RuntimeError(f"gdown failed: {stderr}")
    ext = detect_extension(output)
    if ext:
        new_path = f"{output}.{ext}"
        os.rename(output, new_path)
        return new_path
    return output


def detect_extension(path: str) -> str:
    stdout, _, _ = run(f'file --brief --mime-type "{path}"')
    mime_map = {
        "audio/mpeg": "mp3",
        "audio/mp4": "m4a",
        "audio/x-wav": "wav",
        "audio/wav": "wav",
        "audio/flac": "flac",
        "audio/ogg": "ogg",
        "video/mp4": "mp4",
        "video/x-matroska": "mkv",
        "video/quicktime": "mov",
        "video/webm": "webm",
    }
    return mime_map.get(stdout.strip(), "")


def get_duration(path: str) -> float:
    stdout, _, _ = run(
        f'ffprobe -v error -show_entries format=duration -of csv=p=0 "{path}"'
    )
    return float(stdout.strip())


def extract_audio(input_path: str, output_path: str):
    print(f"Extracting audio → {output_path}")
    run(
        f'ffmpeg -i "{input_path}" -vn -acodec pcm_s16le -ar 16000 -ac 1 "{output_path}" -y',
    )


def detect_mean_volume(wav_path: str) -> float:
    """Detect mean volume of audio file."""
    stdout, stderr, _ = run(
        f'ffmpeg -i "{wav_path}" -af volumedetect -f null -'
    )
    output = stdout + "\n" + stderr
    m = re.search(r"mean_volume:\s*([-\d.]+)\s*dB", output)
    return float(m.group(1)) if m else -30.0


def detect_silence(wav_path: str, noise_db: float = None, min_silence: float = 0.3) -> list:
    """Use ffmpeg silencedetect with adaptive threshold."""
    if noise_db is None:
        mean_vol = detect_mean_volume(wav_path)
        noise_db = mean_vol + 8
        print(f"  Audio mean volume: {mean_vol:.1f}dB, silence threshold: {noise_db:.1f}dB")

    stdout, stderr, _ = run(
        f'ffmpeg -i "{wav_path}" -af silencedetect=noise={noise_db}dB:d={min_silence} -f null -'
    )
    output = stdout + "\n" + stderr
    silences = []
    starts = re.findall(r"silence_start: ([\d.]+)", output)
    ends = re.findall(r"silence_end: ([\d.]+)", output)
    for s, e in zip(starts, ends):
        silences.append((float(s), float(e)))
    return silences


def extract_speech_segments(duration: float, silences: list, min_speech: float = 0.3) -> list:
    """Extract speech segments (inverse of silence)."""
    if not silences:
        return [(0, duration)]
    
    segments = []
    prev_end = 0
    for s_start, s_end in sorted(silences):
        if s_start > prev_end + min_speech:
            segments.append((prev_end, s_start))
        prev_end = max(prev_end, s_end)
    if prev_end < duration - min_speech:
        segments.append((prev_end, duration))
    return segments


def group_speech_segments(speech_segs: list, max_group: float = 300, padding: float = 0.5) -> list:
    """Group adjacent speech segments into chunks, adding padding around each."""
    if not speech_segs:
        return []
    
    groups = []
    group_start = max(0, speech_segs[0][0] - padding)
    group_end = speech_segs[0][1] + padding
    
    for start, end in speech_segs[1:]:
        padded_start = max(0, start - padding)
        padded_end = end + padding
        
        if padded_start - group_end < 1.0 and (padded_end - group_start) <= max_group:
            group_end = padded_end
        else:
            groups.append((group_start, group_end))
            group_start = padded_start
            group_end = padded_end
    
    groups.append((group_start, group_end))
    return groups


def compute_silence_ratio(silences: list, duration: float) -> float:
    total_silence = sum(e - s for s, e in silences)
    return total_silence / duration if duration > 0 else 0


def smart_chunk_boundaries(duration: float, silences: list, max_chunk: float = 300, min_chunk: float = 10) -> list:
    """Determine chunk boundaries by splitting at silence gaps.
    For high-silence audio, extracts only speech segments to avoid hallucinations.
    Returns list of (start, end) tuples."""
    silence_ratio = compute_silence_ratio(silences, duration)

    if silence_ratio > 0.35:
        print(f"  High silence ratio ({silence_ratio:.0%}), using speech-segment extraction")
        speech_segs = extract_speech_segments(duration, silences)
        if speech_segs:
            groups = group_speech_segments(speech_segs, max_group=max_chunk)
            print(f"  Found {len(speech_segs)} speech segments → {len(groups)} groups")
            return groups

    if duration <= max_chunk:
        return [(0, duration)]

    silence_midpoints = [(s + e) / 2 for s, e in silences]
    chunks = []
    chunk_start = 0

    while chunk_start < duration - 1:
        chunk_end = min(chunk_start + max_chunk, duration)

        if chunk_end >= duration:
            chunks.append((chunk_start, duration))
            break

        best_split = None
        target = chunk_start + max_chunk * 0.8
        for mp in silence_midpoints:
            if chunk_start + min_chunk < mp < chunk_end:
                if best_split is None or abs(mp - target) < abs(best_split - target):
                    best_split = mp

        if best_split is not None:
            chunks.append((chunk_start, best_split))
            chunk_start = best_split
        else:
            chunks.append((chunk_start, chunk_end))
            chunk_start = chunk_end

    return chunks


def extract_chunk(wav_path: str, start: float, end: float, output_path: str):
    duration = end - start
    run(
        f'ffmpeg -y -ss {start} -i "{wav_path}" -t {duration} -ar 16000 -ac 1 -acodec pcm_s16le "{output_path}"'
    )


def transcribe_chunk(chunk_path: str, lang: str) -> dict:
    lang_param = f'-F "language={lang}"' if lang != "auto" else ""
    stdout, stderr, rc = run(
        f'curl -s -X POST {SPEACHES_URL}/v1/audio/transcriptions '
        f'-F "file=@{chunk_path}" '
        f'-F "model={MODEL}" '
        f'-F "response_format=verbose_json" '
        f'-F "condition_on_previous_text=false" '
        f'-F "temperature=0" '
        f"{lang_param}"
    )
    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        print(f"  WARNING: Failed to parse API response: {stdout[:200]}")
        return {}


def is_hallucination(text: str) -> bool:
    """Detect common Whisper hallucination patterns."""
    text = text.strip()
    if not text:
        return True
    if len(text) < 2:
        return True

    hallucination_patterns = [
        r"^(字幕志[愿願]者?\s*)+",
        r"^(請不吝點讚\s*)+",
        r"^(訂閱\s*)+",
        r"^(感謝觀看\s*)+",
        r"^(Thank you\.?\s*)+$",
        r"^(Thanks for watching\.?\s*)+$",
        r"^(Subtitles?\s*)+$",
        r"^(\.+\s*)+$",
        r"^(\.\.\.\s*)+$",
    ]
    for pattern in hallucination_patterns:
        if re.match(pattern, text, re.IGNORECASE):
            return True

    words = re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z]+", text)
    if len(words) >= 4:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.2:
            return True

    return False


def format_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def build_srt(all_segments: list) -> str:
    lines = []
    for i, seg in enumerate(all_segments, 1):
        lines.append(str(i))
        lines.append(f"{format_timestamp(seg['start'])} --> {format_timestamp(seg['end'])}")
        lines.append(seg["text"].strip())
        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Smart ASR Pipeline")
    parser.add_argument("input", help="File path or Google Drive URL")
    parser.add_argument("--lang", default="zh", help="Language code (default: zh)")
    parser.add_argument("--format", default="srt", choices=["srt", "text", "json"],
                        help="Output format (default: srt)")
    parser.add_argument("--max-chunk", type=float, default=300,
                        help="Max chunk size in seconds (default: 300)")
    parser.add_argument("--output-dir", default=ASR_DIR, help="Output directory")
    args = parser.parse_args()

    input_path = args.input

    if "drive.google.com" in input_path:
        input_path = download_gdrive(input_path, args.output_dir)

    if not os.path.exists(input_path):
        print(f"ERROR: File not found: {input_path}")
        sys.exit(1)

    basename = Path(input_path).stem
    mime, _, _ = run(f'file --brief --mime-type "{input_path}"')
    print(f"File: {input_path} ({mime})")

    wav_path = os.path.join(args.output_dir, f"{basename}.wav")
    if mime.startswith("video/") or (mime.startswith("audio/") and not input_path.endswith(".wav")):
        extract_audio(input_path, wav_path)
    elif input_path.endswith(".wav"):
        wav_path = input_path

    duration = get_duration(wav_path)
    print(f"Duration: {duration:.1f}s ({duration/60:.1f} min)")

    print("Detecting silence boundaries...")
    silences = detect_silence(wav_path)
    silence_ratio = compute_silence_ratio(silences, duration)
    print(f"  Found {len(silences)} silence gaps, silence ratio: {silence_ratio:.0%}")

    boundaries = smart_chunk_boundaries(duration, silences, max_chunk=args.max_chunk)
    print(f"  Splitting into {len(boundaries)} smart chunks")

    chunk_dir = os.path.join(args.output_dir, f"chunks_{basename}")
    os.makedirs(chunk_dir, exist_ok=True)

    all_segments = []
    full_text = []
    hallucination_count = 0

    for idx, (start, end) in enumerate(boundaries):
        chunk_path = os.path.join(chunk_dir, f"chunk_{idx:04d}.wav")
        extract_chunk(wav_path, start, end, chunk_path)

        chunk_dur = end - start
        print(f"  [{idx+1}/{len(boundaries)}] {format_timestamp(start)} - {format_timestamp(end)} ({chunk_dur:.0f}s)...", end="", flush=True)

        result = transcribe_chunk(chunk_path, args.lang)

        if "segments" in result and result["segments"]:
            for seg in result["segments"]:
                text = seg.get("text", "").strip()
                if not text:
                    continue
                if is_hallucination(text):
                    hallucination_count += 1
                    continue
                all_segments.append({
                    "start": seg["start"] + start,
                    "end": seg["end"] + start,
                    "text": text,
                })
                full_text.append(text)
        elif "text" in result:
            text = result["text"].strip()
            if text and not is_hallucination(text):
                all_segments.append({
                    "start": start,
                    "end": end,
                    "text": text,
                })
                full_text.append(text)
        print(" done")

    import shutil
    shutil.rmtree(chunk_dir, ignore_errors=True)

    if hallucination_count > 0:
        print(f"  Filtered {hallucination_count} hallucinated segments")

    srt_path = os.path.join(args.output_dir, f"{basename}.srt")
    txt_path = os.path.join(args.output_dir, f"{basename}.txt")
    json_path = os.path.join(args.output_dir, f"{basename}.json")

    srt_content = build_srt(all_segments)
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write(srt_content)

    text_content = " ".join(full_text)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text_content)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"text": text_content, "segments": all_segments}, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*50}")
    print(f"Transcription complete!")
    print(f"  Duration: {duration:.0f}s ({duration/60:.1f} min)")
    print(f"  Segments: {len(all_segments)}")
    print(f"  Hallucinations filtered: {hallucination_count}")
    print(f"  SRT: {srt_path}")
    print(f"  TXT: {txt_path}")
    print(f"  JSON: {json_path}")


if __name__ == "__main__":
    main()
