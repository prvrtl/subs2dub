"""Cut a voice-cloning reference clip for each character.

Cloning needs ~10-15 seconds of clean speech per speaker. The vocal stem plus
diarization labels provide it: for each character, splice together their longest
and loudest lines.

Reference quality matters more than length - a short clean sample beats a longer
one containing music bleed or a second voice - so cues are ranked and marginal
ones dropped.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from .model import Cue

TARGET_SECONDS = 12.0
MIN_SECONDS = 4.0
_MIN_CUE = 1.0  # shorter cues carry too little voice to be worth splicing
_GAP = 0.25  # silence between spliced pieces


def build_references(
    cues: list[Cue],
    vocals_wav: Path,
    out_dir: Path,
    *,
    target: float = TARGET_SECONDS,
) -> dict[str, Path]:
    """Write one reference clip per speaker. Returns speaker -> wav path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    audio, sr = sf.read(str(vocals_wav), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    by_speaker: dict[str, list[Cue]] = {}
    for c in cues:
        if c.speaker and (c.end - c.start) >= _MIN_CUE:
            by_speaker.setdefault(c.speaker, []).append(c)

    refs: dict[str, Path] = {}
    silence = np.zeros(int(_GAP * sr), dtype=np.float32)

    for spk, group in by_speaker.items():
        scored = []
        for c in group:
            seg = audio[int(c.start * sr):int(c.end * sr)]
            if seg.size < _MIN_CUE * sr:
                continue
            rms = float(np.sqrt(np.mean(seg.astype(np.float64) ** 2)))
            if rms < 1e-4:
                continue
            # Favour loud, long lines - both correlate with clean solo speech.
            scored.append((rms * min(seg.size / sr, 6.0), seg))

        if not scored:
            continue
        scored.sort(key=lambda x: -x[0])

        pieces, total = [], 0.0
        for _, seg in scored:
            pieces.append(seg)
            total += seg.size / sr
            if total >= target:
                break
        if total < MIN_SECONDS:
            continue

        joined = np.concatenate(
            [p for seg in pieces for p in (seg, silence)][:-1]
        )
        peak = float(np.max(np.abs(joined)) or 0)
        if peak > 0:
            joined = joined * (0.95 / peak)

        path = out_dir / f"{spk}.wav"
        sf.write(path, joined, sr, subtype="PCM_16")
        refs[spk] = path

    return refs


def describe(refs: dict[str, Path]) -> str:
    if not refs:
        return "  no usable reference clips"
    rows = []
    for spk, p in sorted(refs.items()):
        info = sf.info(str(p))
        rows.append(f"  {spk}  {info.duration:5.1f}s  {p.name}")
    return "\n".join(rows)
