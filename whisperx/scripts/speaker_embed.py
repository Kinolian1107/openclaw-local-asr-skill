#!/usr/bin/env python3
"""
Speaker Embedding Manager for WhisperX ASR Pipeline

Operations:
  register  — Extract embedding from audio file and save with a name
  list      — List all registered speakers
  match     — Match an audio file against known speakers
  delete    — Remove a registered speaker
  rename    — Rename SPEAKER_XX samples to a real name (post-hoc labeling)

Usage:
    python3 speaker_embed.py register --name "Alice" --audio speaker_alice.wav
    python3 speaker_embed.py list
    python3 speaker_embed.py match --audio unknown.wav [--threshold 0.65]
    python3 speaker_embed.py delete --name "Alice"
    python3 speaker_embed.py rename --sample-dir /path/to/speaker_samples/session/ --speaker SPEAKER_00 --name "Alice"
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

SPEAKER_DB = os.environ.get("SPEAKER_DB", "/home/kino/asr/speaker_embeddings")
SPEAKER_SAMPLES = os.environ.get("SPEAKER_SAMPLES", "/home/kino/asr/speaker_samples")
HF_CACHE = os.environ.get("HF_HOME", "/home/kino/ollama-models/huggingface-hub")
METADATA_FILE = os.path.join(SPEAKER_DB, "speakers.json")
EMBED_MODEL = "pyannote/wespeaker-voxceleb-resnet34-LM"
DEFAULT_THRESHOLD = 0.65


def ensure_dirs():
    os.makedirs(SPEAKER_DB, exist_ok=True)
    os.makedirs(SPEAKER_SAMPLES, exist_ok=True)


def load_metadata() -> dict:
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"speakers": {}}


def save_metadata(meta: dict):
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def get_embedding_model():
    os.environ["HF_HOME"] = HF_CACHE
    os.environ.setdefault("TRANSFORMERS_CACHE", HF_CACHE)
    from pyannote.audio import Inference
    return Inference(EMBED_MODEL, window="whole", device="cuda")


def extract_embedding(model, audio_path: str) -> np.ndarray:
    wav_path = audio_path
    if not audio_path.endswith(".wav"):
        wav_path = audio_path.rsplit(".", 1)[0] + "_tmp.wav"
        subprocess.run(
            ["ffmpeg", "-y", "-i", audio_path, "-vn", "-acodec", "pcm_s16le",
             "-ar", "16000", "-ac", "1", wav_path],
            capture_output=True, check=True
        )

    embedding = model(wav_path)

    if wav_path != audio_path and os.path.exists(wav_path):
        os.remove(wav_path)

    return np.array(embedding)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def register_speaker(name: str, audio_path: str):
    ensure_dirs()
    if not os.path.exists(audio_path):
        print(f"ERROR: Audio file not found: {audio_path}")
        sys.exit(1)

    print(f"Extracting embedding for '{name}' from {audio_path}...")
    model = get_embedding_model()
    embedding = extract_embedding(model, audio_path)

    embed_path = os.path.join(SPEAKER_DB, f"{name}.npy")
    np.save(embed_path, embedding)

    meta = load_metadata()
    meta["speakers"][name] = {
        "embedding_file": f"{name}.npy",
        "source_audio": os.path.basename(audio_path),
        "registered_at": __import__("datetime").datetime.now().isoformat(),
    }
    save_metadata(meta)

    print(f"Registered speaker '{name}' ({embedding.shape[0]}-dim embedding)")
    print(f"Embedding saved to: {embed_path}")


def list_speakers():
    meta = load_metadata()
    speakers = meta.get("speakers", {})
    if not speakers:
        print("No registered speakers.")
        return

    print(f"Registered speakers ({len(speakers)}):")
    for name, info in speakers.items():
        embed_path = os.path.join(SPEAKER_DB, info["embedding_file"])
        exists = "✓" if os.path.exists(embed_path) else "✗"
        src = info.get("source_audio", "unknown")
        reg = info.get("registered_at", "unknown")
        print(f"  {exists} {name} — source: {src}, registered: {reg}")


def match_speaker(audio_path: str, threshold: float = DEFAULT_THRESHOLD):
    if not os.path.exists(audio_path):
        print(f"ERROR: Audio file not found: {audio_path}")
        sys.exit(1)

    meta = load_metadata()
    speakers = meta.get("speakers", {})
    if not speakers:
        print("No registered speakers to match against.")
        return None

    print(f"Extracting embedding from {audio_path}...")
    model = get_embedding_model()
    query_embed = extract_embedding(model, audio_path)

    print(f"Matching against {len(speakers)} registered speakers (threshold={threshold})...")
    results = []
    for name, info in speakers.items():
        embed_path = os.path.join(SPEAKER_DB, info["embedding_file"])
        if not os.path.exists(embed_path):
            continue
        known_embed = np.load(embed_path)
        sim = cosine_similarity(query_embed, known_embed)
        results.append((name, sim))

    results.sort(key=lambda x: x[1], reverse=True)

    print("\nMatch results:")
    for name, sim in results:
        marker = "✓ MATCH" if sim >= threshold else "  "
        print(f"  {marker} {name}: {sim:.4f}")

    if results and results[0][1] >= threshold:
        best_name, best_sim = results[0]
        print(f"\nBest match: {best_name} (similarity: {best_sim:.4f})")
        return best_name
    else:
        print("\nNo match found above threshold.")
        return None


def match_all_speakers(embeddings_dict: dict, threshold: float = DEFAULT_THRESHOLD) -> dict:
    """Match a dict of {speaker_label: embedding} against known speakers.
    Returns {speaker_label: matched_name_or_original_label}"""
    meta = load_metadata()
    known_speakers = meta.get("speakers", {})
    if not known_speakers:
        return {label: label for label in embeddings_dict}

    known_embeds = {}
    for name, info in known_speakers.items():
        embed_path = os.path.join(SPEAKER_DB, info["embedding_file"])
        if os.path.exists(embed_path):
            known_embeds[name] = np.load(embed_path)

    mapping = {}
    for label, query_embed in embeddings_dict.items():
        best_name = label
        best_sim = 0.0
        for name, known_embed in known_embeds.items():
            sim = cosine_similarity(query_embed, known_embed)
            if sim > best_sim:
                best_sim = sim
                best_name = name
        if best_sim >= threshold:
            mapping[label] = best_name
            print(f"  Speaker {label} → {best_name} (similarity: {best_sim:.4f})")
        else:
            mapping[label] = label
            print(f"  Speaker {label} → unknown (best: {best_sim:.4f} < {threshold})")

    return mapping


def delete_speaker(name: str):
    meta = load_metadata()
    if name not in meta.get("speakers", {}):
        print(f"Speaker '{name}' not found.")
        return

    info = meta["speakers"].pop(name)
    embed_path = os.path.join(SPEAKER_DB, info["embedding_file"])
    if os.path.exists(embed_path):
        os.remove(embed_path)
    save_metadata(meta)
    print(f"Deleted speaker '{name}'")


def rename_sample(sample_dir: str, speaker_label: str, new_name: str):
    """Rename a SPEAKER_XX sample to a real name and register the embedding."""
    wav_path = os.path.join(sample_dir, f"{speaker_label}.wav")
    if not os.path.exists(wav_path):
        print(f"ERROR: Sample not found: {wav_path}")
        sys.exit(1)

    register_speaker(new_name, wav_path)
    print(f"Renamed {speaker_label} → {new_name} and registered embedding.")


def main():
    parser = argparse.ArgumentParser(description="Speaker Embedding Manager")
    sub = parser.add_subparsers(dest="command", required=True)

    reg = sub.add_parser("register", help="Register a speaker from audio")
    reg.add_argument("--name", required=True, help="Speaker name")
    reg.add_argument("--audio", required=True, help="Audio file path")

    sub.add_parser("list", help="List registered speakers")

    mat = sub.add_parser("match", help="Match audio against known speakers")
    mat.add_argument("--audio", required=True, help="Audio file to match")
    mat.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)

    dlt = sub.add_parser("delete", help="Delete a registered speaker")
    dlt.add_argument("--name", required=True, help="Speaker name to delete")

    ren = sub.add_parser("rename", help="Rename a SPEAKER_XX sample")
    ren.add_argument("--sample-dir", required=True, help="Directory containing speaker samples")
    ren.add_argument("--speaker", required=True, help="Speaker label (e.g. SPEAKER_00)")
    ren.add_argument("--name", required=True, help="Real name to assign")

    args = parser.parse_args()

    if args.command == "register":
        register_speaker(args.name, args.audio)
    elif args.command == "list":
        list_speakers()
    elif args.command == "match":
        match_speaker(args.audio, args.threshold)
    elif args.command == "delete":
        delete_speaker(args.name)
    elif args.command == "rename":
        rename_sample(args.sample_dir, args.speaker, args.name)


if __name__ == "__main__":
    main()
