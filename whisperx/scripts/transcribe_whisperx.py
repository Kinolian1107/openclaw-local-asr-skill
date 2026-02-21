#!/usr/bin/env python3
"""
WhisperX ASR Pipeline v2

Features:
- Topic-based initial_prompt for better accuracy
- Audio denoising via ffmpeg
- OpenCC simplified→traditional Chinese (s2twp)
- Hotwords support (faster-whisper native)
- Custom corrections dictionary
- Speaker embedding: auto-extract unknown speakers + match against known
- Speaker diarization with pyannote

Usage:
    python3 transcribe_whisperx.py <input_file> --lang zh --format srt
    python3 transcribe_whisperx.py <input_file> --lang zh --topic "財經討論" --diarize
    python3 transcribe_whisperx.py <input_file> --lang zh --denoise --diarize
"""

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ASR_DIR = os.environ.get("ASR_DIR", "/home/kino/asr")
HF_CACHE = os.environ.get("HF_HOME", "/home/kino/ollama-models/huggingface-hub")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
COMPUTE_TYPE = os.environ.get("WHISPERX_COMPUTE_TYPE", "int8")
MODEL_SIZE = os.environ.get("WHISPERX_MODEL", "large-v3-turbo")
BATCH_SIZE = int(os.environ.get("WHISPERX_BATCH_SIZE", "16"))
GDOWN_PATH = os.environ.get("GDOWN_PATH", "gdown")

WORKSPACE = "/home/kino/.openclaw/workspace"
DEFAULT_HOTWORDS_FILE = os.path.join(WORKSPACE, "whisperx_hotwords.txt")
DEFAULT_CORRECTIONS_FILE = os.path.join(WORKSPACE, "asr_corrections.json")
SPEAKER_DB = os.environ.get("SPEAKER_DB", os.path.join(ASR_DIR, "speaker_embeddings"))
SPEAKER_SAMPLES = os.environ.get("SPEAKER_SAMPLES", os.path.join(ASR_DIR, "speaker_samples"))
EMBED_MODEL = "pyannote/wespeaker-voxceleb-resnet34-LM"
SPEAKER_MATCH_THRESHOLD = 0.65


def run_cmd(cmd, **kwargs):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, **kwargs)
    return result.stdout.strip(), result.stderr.strip(), result.returncode


def detect_best_device():
    try:
        import torch
        if not torch.cuda.is_available():
            return "cpu"
        try:
            t = torch.zeros(1, device="cuda")
            del t
            torch.cuda.empty_cache()
            return "cuda"
        except Exception:
            print("WARNING: CUDA unavailable, falling back to CPU")
            return "cpu"
    except ImportError:
        return "cpu"


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
    stdout, stderr, rc = run_cmd(
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
    stdout, _, _ = run_cmd(f'file --brief --mime-type "{path}"')
    mime_map = {
        "audio/mpeg": "mp3", "audio/mp4": "m4a", "audio/x-wav": "wav",
        "audio/wav": "wav", "audio/flac": "flac", "audio/ogg": "ogg",
        "video/mp4": "mp4", "video/x-matroska": "mkv",
        "video/quicktime": "mov", "video/webm": "webm",
    }
    return mime_map.get(stdout.strip(), "")


def detect_mime(filepath: str) -> str:
    stdout, _, _ = run_cmd(f'file --brief --mime-type "{filepath}"')
    return stdout.strip()


def extract_audio(input_path: str, output_path: str, denoise: bool = False):
    if denoise:
        print(f"Extracting audio with denoising → {output_path}")
        af = "highpass=f=200,lowpass=f=3000,afftdn=nf=-25"
        subprocess.run(
            ["ffmpeg", "-y", "-i", input_path, "-vn", "-af", af,
             "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", output_path],
            capture_output=True, check=True
        )
    else:
        print(f"Extracting audio → {output_path}")
        subprocess.run(
            ["ffmpeg", "-y", "-i", input_path, "-vn", "-acodec", "pcm_s16le",
             "-ar", "16000", "-ac", "1", output_path],
            capture_output=True, check=True
        )


def format_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def load_hotwords(hotwords_file: str) -> str:
    if not os.path.exists(hotwords_file):
        return ""
    words = []
    with open(hotwords_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                words.append(line)
    if words:
        hotwords_str = ", ".join(words)
        print(f"Loaded {len(words)} hotwords from {hotwords_file}")
        return hotwords_str
    return ""


def load_corrections(corrections_file: str) -> dict:
    if not os.path.exists(corrections_file):
        return {}
    with open(corrections_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    data.pop("_comment", None)
    if data:
        print(f"Loaded {len(data)} corrections from {corrections_file}")
    return data


def apply_corrections(text: str, corrections: dict) -> str:
    for wrong, right in corrections.items():
        text = text.replace(wrong, right)
    return text


def clean_repetitions(text: str) -> str:
    """Remove repeated character/word sequences caused by Whisper hallucinations."""
    # Detect character-level repetition: same char repeated 4+ times (e.g. 馬馬馬馬馬)
    text = re.sub(r'(.)\1{3,}', r'\1\1', text)

    # Detect short phrase repetition: 2-10 char phrases repeated 3+ times
    text = re.sub(r'(.{2,10}?)\1{2,}', r'\1', text)

    # Detect comma-separated number repetition (e.g. "000,000,000,000,...")
    text = re.sub(r'((?:\d{1,3},){3,})\d{1,3}(?:,\d{1,3})*', r'\1', text)

    return text.strip()


def is_hallucination(text: str) -> bool:
    """Detect if a segment is a hallucination (too repetitive or nonsensical)."""
    cleaned = text.strip()
    if not cleaned:
        return True

    if len(cleaned) < 2:
        return False

    # Check if >60% of text is the same character
    from collections import Counter
    char_counts = Counter(cleaned.replace(" ", ""))
    if char_counts and char_counts.most_common(1)[0][1] / len(cleaned.replace(" ", "")) > 0.6:
        return True

    # Check for very high compression ratio (repetitive content)
    unique_chars = len(set(cleaned))
    if len(cleaned) > 20 and unique_chars < len(cleaned) * 0.1:
        return True

    return False


def convert_to_traditional(text: str, converter) -> str:
    return converter.convert(text)


def split_long_segments(segments, max_chars):
    """Split segments exceeding max_chars using word-level timestamps."""
    if not max_chars or max_chars <= 0:
        return segments

    new_segments = []
    for seg in segments:
        text = seg["text"].strip()
        if len(text) <= max_chars:
            new_segments.append(seg)
            continue

        words = seg.get("words", [])
        valid_words = [w for w in words if w.get("word")]
        if not valid_words:
            new_segments.append(seg)
            continue

        current_text = ""
        current_words = []
        sub_start = None

        for w in valid_words:
            word_text = w.get("word", "")
            candidate = current_text + word_text

            if len(candidate) > max_chars and current_text.strip():
                last_end = current_words[-1].get("end") if current_words else None
                new_seg = {
                    "start": sub_start if sub_start is not None else seg["start"],
                    "end": last_end if last_end is not None else seg["end"],
                    "text": current_text.strip(),
                    "words": current_words,
                }
                if "speaker" in seg:
                    new_seg["speaker"] = seg["speaker"]
                new_segments.append(new_seg)

                current_text = word_text
                current_words = [w]
                sub_start = w.get("start")
            else:
                if sub_start is None:
                    sub_start = w.get("start", seg["start"])
                current_text = candidate
                current_words.append(w)

        if current_text.strip():
            last_end = current_words[-1].get("end") if current_words else None
            new_seg = {
                "start": sub_start if sub_start is not None else seg["start"],
                "end": last_end if last_end is not None else seg["end"],
                "text": current_text.strip(),
                "words": current_words,
            }
            if "speaker" in seg:
                new_seg["speaker"] = seg["speaker"]
            new_segments.append(new_seg)

    return new_segments


def write_srt(segments, filepath, speaker_map=None):
    with open(filepath, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            f.write(f"{i}\n")
            f.write(f"{format_timestamp(seg['start'])} --> {format_timestamp(seg['end'])}\n")
            text = seg["text"].strip()
            if speaker_map and "speaker" in seg:
                name = speaker_map.get(seg["speaker"], seg["speaker"])
                text = f"[{name}] {text}"
            f.write(f"{text}\n\n")


def extract_speaker_samples(audio, segments, basename, output_dir):
    """Extract representative audio clips for each speaker."""
    import soundfile as sf

    speakers = {}
    for seg in segments:
        spk = seg.get("speaker")
        if not spk:
            continue
        if spk not in speakers:
            speakers[spk] = []
        speakers[spk].append(seg)

    session_dir = os.path.join(output_dir, f"{datetime.now().strftime('%Y%m%d')}_{basename}")
    os.makedirs(session_dir, exist_ok=True)

    sample_paths = {}
    for spk, segs in speakers.items():
        segs_sorted = sorted(segs, key=lambda s: s["end"] - s["start"], reverse=True)
        collected = np.array([], dtype=np.float32)
        target_duration = 15.0  # seconds

        for seg in segs_sorted:
            start_sample = int(seg["start"] * 16000)
            end_sample = int(seg["end"] * 16000)
            chunk = audio[start_sample:end_sample]
            collected = np.concatenate([collected, chunk])
            if len(collected) / 16000 >= target_duration:
                break

        if len(collected) / 16000 < 3.0:
            print(f"  {spk}: too short ({len(collected)/16000:.1f}s), skipping sample")
            continue

        wav_path = os.path.join(session_dir, f"{spk}.wav")
        sf.write(wav_path, collected, 16000)
        sample_paths[spk] = wav_path
        print(f"  {spk}: saved {len(collected)/16000:.1f}s sample → {wav_path}")

    return session_dir, sample_paths


def match_speakers_against_db(sample_paths: dict, device: str) -> dict:
    """Match extracted speaker samples against registered speaker embeddings."""
    os.environ["HF_HOME"] = HF_CACHE
    os.environ.setdefault("TRANSFORMERS_CACHE", HF_CACHE)

    metadata_file = os.path.join(SPEAKER_DB, "speakers.json")
    if not os.path.exists(metadata_file):
        print("  No registered speakers in DB, skipping matching")
        return {label: label for label in sample_paths}

    with open(metadata_file, "r", encoding="utf-8") as f:
        meta = json.load(f)

    known_speakers = meta.get("speakers", {})
    if not known_speakers:
        print("  No registered speakers in DB, skipping matching")
        return {label: label for label in sample_paths}

    known_embeds = {}
    for name, info in known_speakers.items():
        embed_path = os.path.join(SPEAKER_DB, info["embedding_file"])
        if os.path.exists(embed_path):
            known_embeds[name] = np.load(embed_path)

    if not known_embeds:
        return {label: label for label in sample_paths}

    from pyannote.audio import Inference
    embed_model = Inference(EMBED_MODEL, window="whole", device=device)

    mapping = {}
    for label, wav_path in sample_paths.items():
        query_embed = np.array(embed_model(wav_path))

        best_name = label
        best_sim = 0.0
        for name, known_embed in known_embeds.items():
            sim = float(np.dot(query_embed, known_embed) /
                       (np.linalg.norm(query_embed) * np.linalg.norm(known_embed) + 1e-8))
            if sim > best_sim:
                best_sim = sim
                best_name = name

        if best_sim >= SPEAKER_MATCH_THRESHOLD:
            mapping[label] = best_name
            print(f"  {label} → {best_name} (similarity: {best_sim:.4f})")
        else:
            mapping[label] = label
            print(f"  {label} → unknown (best match: {best_sim:.4f} < {SPEAKER_MATCH_THRESHOLD})")

    del embed_model
    return mapping


def save_unknown_embeddings(sample_paths: dict, speaker_map: dict, device: str):
    """Save embeddings for unmatched speakers for future registration."""
    unmatched = {label: path for label, path in sample_paths.items()
                 if speaker_map.get(label, label) == label}
    if not unmatched:
        return

    os.environ["HF_HOME"] = HF_CACHE
    from pyannote.audio import Inference
    embed_model = Inference(EMBED_MODEL, window="whole", device=device)

    for label, wav_path in unmatched.items():
        embed = np.array(embed_model(wav_path))
        embed_path = wav_path.rsplit(".", 1)[0] + ".npy"
        np.save(embed_path, embed)
        print(f"  Saved embedding for {label} → {embed_path}")

    del embed_model


def transcribe(audio_path: str, language: str, output_format: str,
               diarize: bool = False, output_dir: str = ASR_DIR,
               device: str = None, topic: str = None,
               hotwords_file: str = None, corrections_file: str = None,
               no_opencc: bool = False, max_chars: int = 0):
    import whisperx
    import torch

    os.environ["HF_HOME"] = HF_CACHE
    os.environ.setdefault("TRANSFORMERS_CACHE", HF_CACHE)

    if device is None:
        device = detect_best_device()

    compute = COMPUTE_TYPE if device == "cuda" else "int8"

    hotwords_str = load_hotwords(hotwords_file or DEFAULT_HOTWORDS_FILE)
    corrections = load_corrections(corrections_file or DEFAULT_CORRECTIONS_FILE)

    asr_options = {
        "no_repeat_ngram_size": 3,
        "repetition_penalty": 1.2,
        "hallucination_silence_threshold": 2.0,
        "compression_ratio_threshold": 2.4,
        "condition_on_previous_text": False,
    }
    if topic:
        asr_options["initial_prompt"] = topic
        print(f"Using topic as initial_prompt: {topic}")
    if hotwords_str:
        asr_options["hotwords"] = hotwords_str

    print(f"Loading WhisperX model: {MODEL_SIZE} (device={device}, compute={compute})")

    model = whisperx.load_model(
        MODEL_SIZE,
        device=device,
        compute_type=compute,
        download_root=os.path.join(HF_CACHE, "whisperx"),
        asr_options=asr_options if asr_options else None,
    )

    print(f"Loading audio: {audio_path}")
    audio = whisperx.load_audio(audio_path)
    duration = len(audio) / 16000
    print(f"Duration: {duration:.1f}s ({duration/60:.1f} min)")

    print(f"Transcribing with batch_size={BATCH_SIZE}...")
    result = model.transcribe(
        audio,
        batch_size=BATCH_SIZE,
        language=language if language != "auto" else None,
    )

    detected_lang = result.get("language", language)
    print(f"Detected language: {detected_lang}")
    print(f"Initial segments: {len(result['segments'])}")

    print("Aligning with wav2vec2 for precise timestamps...")
    try:
        align_model, align_metadata = whisperx.load_align_model(
            language_code=detected_lang,
            device=device,
        )
        result = whisperx.align(
            result["segments"],
            align_model,
            align_metadata,
            audio,
            device,
            return_char_alignments=False,
        )
        print(f"Aligned segments: {len(result['segments'])}")
        del align_model
    except Exception as e:
        print(f"WARNING: Alignment failed ({e}), using original timestamps")

    speaker_map = None
    session_dir = None
    if diarize:
        print("Running speaker diarization...")
        try:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["HUGGINGFACE_HUB_CACHE"] = HF_CACHE
            os.environ["HF_HOME"] = HF_CACHE
            from whisperx.diarize import DiarizationPipeline
            diarize_model = DiarizationPipeline(
                model_name="pyannote/speaker-diarization-3.1",
                token=HF_TOKEN or None,
                device=device,
                cache_dir=HF_CACHE,
            )
            diarize_segments = diarize_model(audio)
            result = whisperx.assign_word_speakers(diarize_segments, result)
            speakers_found = set(s.get("speaker", "") for s in result["segments"] if s.get("speaker"))
            print(f"Identified {len(speakers_found)} speakers: {', '.join(sorted(speakers_found))}")
            del diarize_model

            basename = Path(audio_path).stem
            print("Extracting speaker samples...")
            session_dir, sample_paths = extract_speaker_samples(
                audio, result["segments"], basename, SPEAKER_SAMPLES
            )

            if sample_paths:
                print("Matching speakers against registered DB...")
                speaker_map = match_speakers_against_db(sample_paths, device)

                print("Saving embeddings for unknown speakers...")
                save_unknown_embeddings(sample_paths, speaker_map, device)

        except Exception as e:
            print(f"WARNING: Diarization failed ({e}), continuing without speaker labels")

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    segments = result["segments"]
    basename = Path(audio_path).stem

    # --- Post-processing: hallucination filter + corrections + OpenCC ---
    pre_filter_count = len(segments)
    segments = [seg for seg in segments if not is_hallucination(seg.get("text", ""))]
    hallucination_count = pre_filter_count - len(segments)
    if hallucination_count > 0:
        print(f"Filtered {hallucination_count} hallucinated segments ({pre_filter_count} → {len(segments)})")

    opencc_converter = None
    if not no_opencc and detected_lang in ("zh", "zh-cn", "chinese"):
        try:
            import opencc
            opencc_converter = opencc.OpenCC("s2twp")
            print("OpenCC: converting simplified → traditional Chinese (s2twp)")
        except ImportError:
            print("WARNING: opencc-python-reimplemented not installed, skipping conversion")

    for seg in segments:
        text = seg["text"].strip()
        text = clean_repetitions(text)
        if corrections:
            text = apply_corrections(text, corrections)
        if opencc_converter:
            text = convert_to_traditional(text, opencc_converter)
        seg["text"] = text

    if max_chars and max_chars > 0:
        pre_split = len(segments)
        segments = split_long_segments(segments, max_chars)
        print(f"Split segments by max_chars={max_chars}: {pre_split} → {len(segments)}")

    srt_path = os.path.join(output_dir, f"{basename}.srt")
    write_srt(segments, srt_path, speaker_map=speaker_map)
    print(f"SRT: {srt_path}")

    txt_path = os.path.join(output_dir, f"{basename}.txt")
    full_text_parts = []
    for seg in segments:
        text = seg["text"].strip()
        if speaker_map and "speaker" in seg:
            name = speaker_map.get(seg["speaker"], seg["speaker"])
            text = f"[{name}] {text}"
        full_text_parts.append(text)
    full_text = "\n".join(full_text_parts)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(full_text)
    print(f"TXT: {txt_path}")

    if output_format in ("json", "all"):
        json_path = os.path.join(output_dir, f"{basename}.json")
        json_segments = []
        for seg in segments:
            entry = {
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"].strip(),
            }
            if "speaker" in seg:
                raw_spk = seg["speaker"]
                entry["speaker_raw"] = raw_spk
                if speaker_map:
                    entry["speaker"] = speaker_map.get(raw_spk, raw_spk)
                else:
                    entry["speaker"] = raw_spk
            if "words" in seg:
                entry["words"] = [
                    {"word": w.get("word", ""), "start": w.get("start", 0), "end": w.get("end", 0)}
                    for w in seg["words"] if "word" in w
                ]
            json_segments.append(entry)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "text": full_text,
                "segments": json_segments,
                "duration": duration,
                "language": detected_lang,
                "model": MODEL_SIZE,
                "engine": "whisperx",
                "device": device,
                "diarization": diarize,
                "topic": topic,
                "hotwords_file": hotwords_file or DEFAULT_HOTWORDS_FILE,
                "opencc": "s2twp" if opencc_converter else None,
                "speaker_map": speaker_map,
                "speaker_samples_dir": session_dir,
            }, f, ensure_ascii=False, indent=2)
        print(f"JSON: {json_path}")

    print(f"\n{'='*50}")
    print(f"Transcription complete!")
    print(f"  Duration: {duration:.0f}s ({duration/60:.1f} min)")
    print(f"  Segments: {len(segments)}")
    print(f"  Language: {detected_lang}")
    print(f"  Engine: WhisperX ({MODEL_SIZE}) on {device}")
    if topic:
        print(f"  Topic: {topic}")
    if opencc_converter:
        print(f"  OpenCC: s2twp (繁體中文)")
    if diarize and speaker_map:
        matched = sum(1 for v in speaker_map.values() if not v.startswith("SPEAKER_"))
        print(f"  Speakers: {len(speaker_map)} ({matched} matched to known)")
        if session_dir:
            print(f"  Speaker samples: {session_dir}")
    elif diarize:
        speakers_found = set(s.get("speaker", "") for s in segments if s.get("speaker"))
        print(f"  Speakers: {len(speakers_found)}")
    print(f"  SRT: {srt_path}")

    return segments


def main():
    parser = argparse.ArgumentParser(
        description="WhisperX ASR Pipeline v2 with enhanced features")
    parser.add_argument("input", help="Input audio/video file path or Google Drive URL")
    parser.add_argument("--lang", default="zh", help="Language code (zh, en, ja, auto)")
    parser.add_argument("--format", default="srt", choices=["srt", "text", "json", "all"],
                        help="Output format (default: srt)")
    parser.add_argument("--diarize", action="store_true",
                        help="Enable speaker diarization (requires HF_TOKEN)")
    parser.add_argument("--output-dir", default=ASR_DIR, help="Output directory")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Batch size for inference")
    parser.add_argument("--device", default=None, choices=["cuda", "cpu"],
                        help="Force device (default: auto-detect)")
    parser.add_argument("--topic", default=None,
                        help="Topic description for initial_prompt (improves accuracy)")
    parser.add_argument("--denoise", action="store_true",
                        help="Apply audio denoising before transcription")
    parser.add_argument("--hotwords-file", default=DEFAULT_HOTWORDS_FILE,
                        help="Path to hotwords file (one word per line)")
    parser.add_argument("--corrections-file", default=DEFAULT_CORRECTIONS_FILE,
                        help="Path to corrections JSON file")
    parser.add_argument("--no-opencc", action="store_true",
                        help="Disable OpenCC simplified→traditional conversion")
    parser.add_argument("--max-chars", type=int, default=0,
                        help="Max characters per subtitle segment (0=disabled, recommended: 20 for Chinese)")
    args = parser.parse_args()

    input_path = args.input

    if "drive.google.com" in input_path:
        input_path = download_gdrive(input_path, args.output_dir)

    if not os.path.exists(input_path):
        print(f"ERROR: File not found: {input_path}")
        sys.exit(1)

    mime = detect_mime(input_path)
    print(f"Input: {input_path} ({mime})")

    basename = Path(input_path).stem
    wav_path = os.path.join(args.output_dir, f"{basename}.wav")

    if mime.startswith("video/") or (mime.startswith("audio/") and not input_path.endswith(".wav")):
        extract_audio(input_path, wav_path, denoise=args.denoise)
    elif input_path.endswith(".wav"):
        if args.denoise:
            denoised_path = os.path.join(args.output_dir, f"{basename}_denoised.wav")
            extract_audio(input_path, denoised_path, denoise=True)
            wav_path = denoised_path
        else:
            wav_path = input_path

    segments = transcribe(
        wav_path,
        language=args.lang,
        output_format=args.format,
        diarize=args.diarize,
        output_dir=args.output_dir,
        device=args.device,
        topic=args.topic,
        hotwords_file=args.hotwords_file,
        corrections_file=args.corrections_file,
        no_opencc=args.no_opencc,
        max_chars=args.max_chars,
    )
    print(f"\nDone! {len(segments)} segments transcribed.")


if __name__ == "__main__":
    main()
