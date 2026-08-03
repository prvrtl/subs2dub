"""Split the original audio into vocals and everything-else with Demucs.

One stage, two purposes:

  * The mix bed. Without separation the original dialogue is merely ducked under
    the dub and stays audible. With it, only music and effects remain.
  * Analysis. Speaker embeddings and pitch computed over music describe the
    scene rather than the voice; computed over an isolated vocal stem they
    describe the speaker.

Stems are written as FLAC - lossless and roughly half the size of WAV, which
matters for feature-length sources.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

MODEL = "htdemucs"


def _device() -> str:
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def separate(
    audio: Path,
    work: Path,
    *,
    model: str = MODEL,
    device: str | None = None,
    shifts: int = 0,
    python: str | None = None,
) -> tuple[Path, Path]:
    """Return (vocals, no_vocals). Cached - re-runs are free.

    `shifts` trades time for quality: each shift is another full pass. 0 is
    already good; 1-2 is worth it when there is compute budget to spare.
    """
    out_dir = work / "stems"
    stem_dir = out_dir / model / audio.stem
    vocals = stem_dir / "vocals.flac"
    rest = stem_dir / "no_vocals.flac"
    if vocals.exists() and rest.exists():
        return vocals, rest

    dev = device or _device()
    cmd = [
        python or sys.executable, "-m", "demucs",
        "-n", model,
        "--two-stems", "vocals",
        "--flac",
        "-d", dev,
        "-o", str(out_dir),
        str(audio),
    ]
    if shifts:
        cmd += ["--shifts", str(shifts)]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 and dev == "mps":
        shutil.rmtree(out_dir, ignore_errors=True)
        cmd[cmd.index("-d") + 1] = "cpu"
        proc = subprocess.run(cmd, capture_output=True, text=True)

    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-12:])
        raise RuntimeError(f"demucs failed:\n{tail}")

    if not (vocals.exists() and rest.exists()):
        found = list(stem_dir.glob("*")) if stem_dir.exists() else []
        raise RuntimeError(f"demucs produced no stems; found {found}")
    return vocals, rest


def extract_wav(video: Path, out: Path, stream: int = 0, sr: int | None = None) -> Path:
    """Pull an audio stream out of the container for Demucs to chew on."""
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(video),
           "-map", f"0:a:{stream}", "-c:a", "pcm_s16le"]
    if sr:
        cmd += ["-ar", str(sr)]
    cmd.append(str(out))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"audio extract failed: {proc.stderr[-300:]}")
    return out
