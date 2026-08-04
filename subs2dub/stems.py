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


def full_band_vocals(vocals: Path, source: Path, out: Path,
                     crossover: float = 190.0) -> Path:
    """Put the vocal fundamental back into the separated speech.

    Demucs is trained on music, where energy below a couple of hundred hertz is
    bass instruments, so it routes the low end of a male voice into the music
    stem: measured on one source, 60-200 Hz carries 9.9% of the original audio
    but only 1.7% of the vocal stem. Everything downstream then works from a
    voice with no foundation, and a cloning model handed such a reference
    reproduces it an octave high.

    The low band is taken from the original audio but only where the separated
    vocals say someone is speaking. Taking it everywhere pours the score's bass
    and room rumble into the voice track, which reads as a boomy voice pitched
    below the actor.
    """
    import numpy as np
    import soundfile as sf
    from scipy.signal import butter, sosfiltfilt

    if out.exists() and out.stat().st_mtime >= vocals.stat().st_mtime:
        return out

    voc, sr = sf.read(str(vocals), dtype="float32", always_2d=False)
    if voc.ndim > 1:
        voc = voc.mean(axis=1)
    raw, sr_raw = sf.read(str(source), dtype="float32", always_2d=False)
    if raw.ndim > 1:
        raw = raw.mean(axis=1)
    if sr_raw != sr:
        import librosa

        raw = librosa.resample(raw, orig_sr=sr_raw, target_sr=sr)

    n = min(len(voc), len(raw))
    voc, raw = voc[:n], raw[:n]

    low = sosfiltfilt(butter(4, crossover, "lowpass", fs=sr, output="sos"), raw)
    high = sosfiltfilt(butter(4, crossover, "highpass", fs=sr, output="sos"), voc)

    gate = _speech_gate(voc, sr)
    mixed = (low * gate + high).astype(np.float32)

    peak = float(np.max(np.abs(mixed)) or 0.0)
    if peak > 0.99:
        mixed *= 0.99 / peak

    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out, mixed, sr)
    return out


def _speech_gate(voc, sr, frame: float = 0.02, attack: float = 0.05):
    """A 0-to-1 envelope following where the separated vocals carry speech.

    Ramped rather than switched, because a hard gate on a low band clicks.
    """
    import numpy as np

    n = max(1, int(frame * sr))
    frames = np.abs(voc[: len(voc) // n * n].reshape(-1, n)).mean(axis=1)
    peak = float(frames.max()) if frames.size else 0.0
    if peak <= 0:
        return np.zeros(len(voc), dtype="float32")

    level = np.clip(frames / max(peak * 0.12, 1e-6), 0.0, 1.0)
    smooth = max(1, int(attack / frame))
    kernel = np.ones(smooth) / smooth
    level = np.convolve(level, kernel, mode="same")

    gate = np.repeat(level, n)
    if len(gate) < len(voc):
        gate = np.concatenate([gate, np.full(len(voc) - len(gate), gate[-1])])
    return gate[: len(voc)].astype("float32")
