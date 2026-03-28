#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="${ROOT}/artifacts"
URL="${1:-https://www.youtube.com/watch?v=ibLPZsg1hu0}"
STEM="youtube_ibLPZsg1hu0"
RAW="${OUT_DIR}/${STEM}_raw.%(ext)s"
WAV="${OUT_DIR}/${STEM}_reference_16k.wav"
mkdir -p "${OUT_DIR}"
echo "Downloading: ${URL}"
yt-dlp -f "bestaudio/best" --no-playlist --cookies-from-browser chrome -o "${RAW}" "${URL}"
F="$(ls -t "${OUT_DIR}/${STEM}_raw."* 2>/dev/null | head -1)"
if [[ -z "${F}" || ! -f "${F}" ]]; then
  echo "Download failed. Update: pip install -U yt-dlp" >&2
  exit 1
fi
ffmpeg -y -i "${F}" -ar 16000 -ac 1 -c:a pcm_s16le "${WAV}"
echo "Done: ${WAV}"
ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "${WAV}"
