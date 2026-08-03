"""Mix the dialogue bus over the background and mux back to video.

Uses ffmpeg rather than numpy so nothing large is held in memory, which matters
for feature-length sources.

The background is ducked beneath the dub by sidechain compression, driven by the
dialogue bus, so music and effects dip only while someone is speaking. The
original audio is kept as a second stream so the result remains watchable in the
source language.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf


def write_dialogue_wav(bus: np.ndarray, sr: int, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, bus, sr, subtype="PCM_16")
    return path


LANGS = {
    "en": ("eng", "English"), "uk": ("ukr", "Ukrainian"),
    "de": ("deu", "German"),  "fr": ("fra", "French"),
    "es": ("spa", "Spanish"), "pl": ("pol", "Polish"),
    "it": ("ita", "Italian"), "pt": ("por", "Portuguese"),
    "ja": ("jpn", "Japanese"), "ko": ("kor", "Korean"),
    "zh": ("zho", "Chinese"), "ru": ("rus", "Russian"),
}


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-15:])
        raise RuntimeError(f"ffmpeg failed:\n{tail}")


def mux(
    video: Path,
    dialogue_wav: Path,
    out: Path,
    *,
    bed: Path | None = None,
    vocals: Path | None = None,
    lang: str = "en",
    audio_stream: int = 0,
    duck_db: float = 9.0,
    dialogue_gain_db: float = 2.0,
    bg_gain_db: float = 0.0,
    voices_db: float = -16.0,
    voices_duck_db: float = 14.0,
    keep_original: bool = True,
    loudnorm: bool = True,
) -> Path:
    """Mix the dub over a background bed and mux to `out`.

    `bed` is Demucs' no_vocals stem when available. With it the original speech
    is genuinely absent, so ducking only has to make room against music and
    effects and can be much gentler than when fighting the original dialogue.

    `vocals` is the other Demucs stem, the original speech. Folding it back in
    quietly keeps the performance audible - laughter, gasps, the reactions that
    were never subtitled - which a stripped dub loses entirely. It is ducked
    harder than the bed so it sits under the dub rather than competing with it.
    """
    code, name = LANGS.get(lang, ("und", lang.upper()))

    ratio = max(2.0, min(20.0, duck_db if bed is None else duck_db * 0.55))
    bg_src = "[2:a]" if bed else f"[0:a:{audio_stream}]"
    voices_idx = 3 if bed else 2
    use_voices = vocals is not None

    chain = [
        f"{bg_src}aformat=sample_fmts=fltp:sample_rates=48000:"
        f"channel_layouts=stereo,volume={bg_gain_db}dB[bg]",
        "[1:a]aformat=sample_fmts=fltp:sample_rates=48000:"
        f"channel_layouts=stereo,volume={dialogue_gain_db}dB[dlg]",
    ]
    chain.append(
        "[dlg]asplit=3[dlgmix][key][key2]" if use_voices
        else "[dlg]asplit=2[dlgmix][key]"
    )
    chain.append(
        f"[bg][key]sidechaincompress=threshold=0.03:ratio={ratio}:"
        "attack=15:release=350:makeup=1[ducked]"
    )

    if use_voices:
        vratio = max(2.0, min(20.0, voices_duck_db))
        chain += [
            f"[{voices_idx}:a]aformat=sample_fmts=fltp:sample_rates=48000:"
            f"channel_layouts=stereo,volume={voices_db}dB[orig]",
            f"[orig][key2]sidechaincompress=threshold=0.02:ratio={vratio}:"
            "attack=5:release=500:makeup=1[origducked]",
            "[ducked][origducked]amix=inputs=2:duration=first:normalize=0[beds]",
            "[beds][dlgmix]amix=inputs=2:duration=first:normalize=0[mixed]",
        ]
    else:
        chain.append(
            "[ducked][dlgmix]amix=inputs=2:duration=first:normalize=0[mixed]"
        )
    chain.append(
        "[mixed]loudnorm=I=-16:LRA=11:TP=-1.5[aout]"
        if loudnorm
        else "[mixed]anull[aout]"
    )

    cmd = [
        "ffmpeg", "-y", "-v", "error", "-stats",
        "-i", str(video),
        "-i", str(dialogue_wav),
    ]
    if bed:
        cmd += ["-i", str(bed)]
    if use_voices:
        cmd += ["-i", str(vocals)]
    cmd += [
        "-filter_complex", ";".join(chain),
        "-map", "0:v",
        "-map", "[aout]",
    ]
    if keep_original:
        cmd += ["-map", f"0:a:{audio_stream}"]
    cmd += [
        "-c:v", "copy",
        "-c:a:0", "aac", "-b:a:0", "192k",
        "-metadata:s:a:0", f"title={name} dub",
        "-metadata:s:a:0", f"language={code}",
        "-disposition:a:0", "default",
    ]
    if keep_original:
        cmd += [
            "-c:a:1", "copy",
            "-metadata:s:a:1", "title=Original", "-disposition:a:1", "0",
        ]
    cmd += ["-map", "0:s?", "-c:s", "copy", str(out)]

    out.parent.mkdir(parents=True, exist_ok=True)
    _run(cmd)
    return out


def extract_clip(video: Path, out: Path, start: float, duration: float) -> Path:
    """Cut a short segment for fast iteration. Re-encodes video for exact cuts."""
    out.parent.mkdir(parents=True, exist_ok=True)
    _run([
        "ffmpeg", "-y", "-v", "error",
        "-ss", f"{start:.3f}", "-i", str(video), "-t", f"{duration:.3f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", "-b:a", "160k", "-c:s", "copy",
        str(out),
    ])
    return out


def probe_duration(path: Path) -> float:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True,
    )
    return float(proc.stdout.strip())
