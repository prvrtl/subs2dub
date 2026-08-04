#!/usr/bin/env bash
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
echo "  ./bin/subs2dub                       # interactive"
echo "  ./bin/subs2dub movie.mkv             # dub a file"
