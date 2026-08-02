#!/usr/bin/env bash
# Set up subs2dub: virtualenv, Python deps, and the OpenVoice tone-colour
# converter used by --voice-convert.
#
# Model weights (~4 GB total) download on first run and are cached afterwards.
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3.11}"

echo "==> checking system dependencies"
missing=()
for bin in ffmpeg ffprobe rubberband espeak-ng; do
  command -v "$bin" >/dev/null 2>&1 || missing+=("$bin")
done
if [ ${#missing[@]} -gt 0 ]; then
  echo "missing: ${missing[*]}"
  echo "  macOS:  brew install ffmpeg rubberband espeak-ng"
  echo "  Debian: sudo apt install ffmpeg rubberband-cli espeak-ng"
  exit 1
fi

if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "need $PYTHON (3.10 or 3.11; most ML wheels do not ship for 3.12+ yet)"
  echo "override with: PYTHON=python3.10 $0"
  exit 1
fi

echo "==> creating virtualenv"
[ -d .venv ] || "$PYTHON" -m venv .venv
.venv/bin/pip install --quiet --upgrade pip

echo "==> installing Python dependencies (several minutes)"
.venv/bin/pip install --quiet -r requirements.txt

echo "==> vendoring the OpenVoice tone-colour converter"
# Only the converter is used. Its declared dependencies (numpy 1.22, librosa
# 0.9, faster-whisper, gradio) conflict with the rest of the pipeline, and none
# are needed: se_extractor is bypassed by supplying reference clips directly.
if [ ! -d vendor/openvoice ]; then
  rm -rf .openvoice-src
  git clone --depth 1 --quiet https://github.com/myshell-ai/OpenVoice.git .openvoice-src
  mkdir -p vendor
  cp -r .openvoice-src/openvoice vendor/
  rm -rf .openvoice-src
fi

echo "==> downloading the converter checkpoint (~125 MB)"
.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("myshell-ai/OpenVoiceV2", local_dir="checkpoints_ov",
                  allow_patterns=["converter/*"])
PY

echo
echo "done. try:"
echo "  .venv/bin/python -m subs2dub build movie.mkv -o preview.mkv --clip 60:90"
