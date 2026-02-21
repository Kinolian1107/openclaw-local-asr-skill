#!/bin/bash
# ASR Pipeline: Download → Extract Audio → Chunk → Transcribe → SRT
# Usage: ./transcribe.sh <input_file_or_url> [language] [format]
# Language: zh (default), en, ja, auto, etc.
# Format: srt (default), text, json

set -euo pipefail

ASR_DIR="${ASR_DIR:-/home/kino/asr}"
SPEACHES_URL="${SPEACHES_URL:-http://localhost:18996}"
MODEL="${ASR_MODEL:-deepdml/faster-whisper-large-v3-turbo-ct2}"
CHUNK_SECONDS="${CHUNK_SECONDS:-15}"
INPUT="${1:-}"
LANG="${2:-zh}"
FORMAT="${3:-srt}"

if [ -z "$INPUT" ]; then
    echo "Usage: $0 <file_path_or_google_drive_url> [language] [format]"
    echo "  language: zh (default), en, ja, auto, etc."
    echo "  format: srt (default), text, json"
    exit 1
fi

BASENAME=""
AUDIO_FILE=""

download_gdrive() {
    local url="$1"
    local file_id=""

    if echo "$url" | grep -q "drive.google.com/file/d/"; then
        file_id=$(echo "$url" | sed -n 's|.*drive.google.com/file/d/\([^/]*\).*|\1|p')
    elif echo "$url" | grep -q "id="; then
        file_id=$(echo "$url" | sed -n 's|.*id=\([^&]*\).*|\1|p')
    fi

    if [ -z "$file_id" ]; then
        echo "ERROR: Cannot extract Google Drive file ID from URL"
        exit 1
    fi

    echo "Downloading from Google Drive (file_id: $file_id)..."
    local output_file="${ASR_DIR}/gdrive_${file_id}"
    gdown "https://drive.google.com/uc?id=${file_id}" -O "$output_file" 2>&1
    echo "$output_file"
}

if echo "$INPUT" | grep -qi "drive.google.com"; then
    INPUT=$(download_gdrive "$INPUT")
fi

if [ ! -f "$INPUT" ]; then
    echo "ERROR: File not found: $INPUT"
    exit 1
fi

BASENAME=$(basename "$INPUT" | sed 's/\.[^.]*$//')
MIME=$(file --brief --mime-type "$INPUT")
echo "File: $INPUT ($MIME)"

# Extract audio if video
if echo "$MIME" | grep -q "video/"; then
    echo "Video detected. Extracting audio with ffmpeg..."
    AUDIO_FILE="${ASR_DIR}/${BASENAME}.wav"
    ffmpeg -i "$INPUT" -vn -acodec pcm_s16le -ar 16000 -ac 1 "$AUDIO_FILE" -y 2>/dev/null
    echo "Audio extracted: $AUDIO_FILE"
elif echo "$MIME" | grep -q "audio/"; then
    AUDIO_FILE="${ASR_DIR}/${BASENAME}.wav"
    if [ "$INPUT" != "$AUDIO_FILE" ]; then
        echo "Converting to WAV..."
        ffmpeg -i "$INPUT" -vn -acodec pcm_s16le -ar 16000 -ac 1 "$AUDIO_FILE" -y 2>/dev/null
    fi
    echo "Audio file: $AUDIO_FILE"
else
    echo "ERROR: Unsupported file type: $MIME"
    exit 1
fi

# Get duration
DURATION=$(ffprobe -i "$AUDIO_FILE" -show_entries format=duration -v quiet -of csv="p=0" | cut -d. -f1)
echo "Duration: ${DURATION}s"

# If short enough, transcribe directly without chunking
if [ "$DURATION" -le "$CHUNK_SECONDS" ]; then
    echo "Short audio, transcribing directly..."
    LANG_PARAM=""
    if [ "$LANG" != "auto" ]; then
        LANG_PARAM="-F language=$LANG"
    fi

    RESULT=$(curl -s -X POST "${SPEACHES_URL}/v1/audio/transcriptions" \
        -F "file=@${AUDIO_FILE}" \
        -F "model=${MODEL}" \
        -F "response_format=srt" \
        -F "condition_on_previous_text=false" \
        -F "temperature=0" \
        $LANG_PARAM)

    echo "$RESULT" > "${ASR_DIR}/${BASENAME}.srt"
    echo "=== Result ==="
    echo "$RESULT"
    echo "Saved: ${ASR_DIR}/${BASENAME}.srt"
    exit 0
fi

# Split into chunks for longer audio
echo "Splitting into ${CHUNK_SECONDS}s chunks..."
CHUNK_DIR="${ASR_DIR}/chunks_${BASENAME}"
mkdir -p "$CHUNK_DIR"
ffmpeg -i "$AUDIO_FILE" -f segment -segment_time "$CHUNK_SECONDS" -ar 16000 -ac 1 -acodec pcm_s16le "${CHUNK_DIR}/chunk_%04d.wav" -y 2>/dev/null

CHUNK_COUNT=$(ls "${CHUNK_DIR}"/chunk_*.wav | wc -l)
echo "Created $CHUNK_COUNT chunks"

# Transcribe each chunk and build combined SRT
FULL_TEXT=""
SRT_CONTENT=""
SEG_INDEX=1

for chunk_file in $(ls "${CHUNK_DIR}"/chunk_*.wav | sort); do
    chunk_name=$(basename "$chunk_file" .wav)
    chunk_num=$(echo "$chunk_name" | sed 's/chunk_0*//' | sed 's/^$/0/')
    chunk_offset=$((chunk_num * CHUNK_SECONDS))

    echo -n "Transcribing chunk $((chunk_num + 1))/$CHUNK_COUNT..."

    LANG_PARAM=""
    if [ "$LANG" != "auto" ]; then
        LANG_PARAM="-F language=$LANG"
    fi

    RESPONSE=$(curl -s -X POST "${SPEACHES_URL}/v1/audio/transcriptions" \
        -F "file=@${chunk_file}" \
        -F "model=${MODEL}" \
        -F "response_format=verbose_json" \
        -F "condition_on_previous_text=false" \
        -F "temperature=0" \
        $LANG_PARAM)

    CHUNK_TEXT=$(echo "$RESPONSE" | python3 -c "
import json, sys
data = json.load(sys.stdin)
offset = $chunk_offset
if 'segments' in data and data['segments']:
    for seg in data['segments']:
        start = seg['start'] + offset
        end = seg['end'] + offset
        text = seg['text'].strip()
        if text:
            sh, sm, ss = int(start//3600), int((start%3600)//60), int(start%60)
            sms = int((start%1)*1000)
            eh, em, es = int(end//3600), int((end%3600)//60), int(end%60)
            ems = int((end%1)*1000)
            print(f'SRT:{sh:02d}:{sm:02d}:{ss:02d},{sms:03d} --> {eh:02d}:{em:02d}:{es:02d},{ems:03d}|{text}')
    text = data.get('text', '').strip()
    if text:
        print(f'TXT:{text}')
elif 'text' in data:
    text = data['text'].strip()
    if text:
        start = offset
        end = offset + data.get('duration', $CHUNK_SECONDS)
        sh, sm, ss = int(start//3600), int((start%3600)//60), int(start%60)
        sms = int((start%1)*1000)
        eh, em, es = int(end//3600), int((end%3600)//60), int(end%60)
        ems = int((end%1)*1000)
        print(f'SRT:{sh:02d}:{sm:02d}:{ss:02d},{sms:03d} --> {eh:02d}:{em:02d}:{es:02d},{ems:03d}|{text}')
        print(f'TXT:{text}')
" 2>/dev/null)

    while IFS= read -r line; do
        if [[ "$line" == SRT:* ]]; then
            ts=$(echo "$line" | sed 's/^SRT:\([^|]*\)|.*/\1/')
            text=$(echo "$line" | sed 's/^SRT:[^|]*|//')
            SRT_CONTENT="${SRT_CONTENT}${SEG_INDEX}\n${ts}\n${text}\n\n"
            SEG_INDEX=$((SEG_INDEX + 1))
        elif [[ "$line" == TXT:* ]]; then
            text=$(echo "$line" | sed 's/^TXT://')
            FULL_TEXT="${FULL_TEXT}${text} "
        fi
    done <<< "$CHUNK_TEXT"

    echo " done"
done

# Save results
echo -e "$SRT_CONTENT" > "${ASR_DIR}/${BASENAME}.srt"
echo "$FULL_TEXT" > "${ASR_DIR}/${BASENAME}.txt"
echo "$FULL_TEXT" | python3 -c "import json,sys; print(json.dumps({'text': sys.stdin.read().strip()}, ensure_ascii=False, indent=2))" > "${ASR_DIR}/${BASENAME}.json"

echo ""
echo "=== Transcript ==="
echo "$FULL_TEXT"
echo ""
echo "=== Files ==="
echo "  SRT: ${ASR_DIR}/${BASENAME}.srt"
echo "  TXT: ${ASR_DIR}/${BASENAME}.txt"
echo "  JSON: ${ASR_DIR}/${BASENAME}.json"

# Cleanup chunks
rm -rf "$CHUNK_DIR"
echo "Done!"
