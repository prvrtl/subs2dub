"""Pitch measurement, shared and parallelised.

Measuring pitch is roughly half the wall time of a render and it ran on one
core. The work is per-cue and independent, so it spreads across the machine.

pyin rather than the far cheaper yin: yin is thirty times faster but disagrees
with pyin by about 2.6 semitones, which is larger than the register errors this
is used to detect and correct.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

FMIN = 75.0
FMAX = 400.0

SPEECH_LOW = 80.0
SPEECH_HIGH = 360.0


def plausible(hz: float) -> bool:
    """Whether a measured pitch could be someone speaking.

    Adult speech runs roughly 85-255 Hz, wider for children and for a
    performer putting on a voice. Anything outside this is the tracker
    locking onto music, a sub-harmonic or room rumble, and correcting a
    voice toward it drags it somewhere nobody talks.
    """
    return SPEECH_LOW <= hz <= SPEECH_HIGH
FRAME = 2048


def shape(audio: np.ndarray, sr: int) -> tuple[float, float]:
    """Median pitch in Hz and expressive range in semitones."""
    try:
        import librosa

        f0, _, _ = librosa.pyin(
            audio, fmin=FMIN, fmax=FMAX, sr=sr, frame_length=FRAME
        )
        voiced = f0[np.isfinite(f0)]
        if voiced.size < 8:
            return 0.0, 0.0
        median = float(np.median(voiced))
        semitones = 12.0 * np.log2(voiced / median)
        spread = float(
            np.percentile(semitones, 90) - np.percentile(semitones, 10)
        )
        return median, spread
    except Exception:
        return 0.0, 0.0


def _one(args):
    return shape(args[0], args[1])


def shapes(segments: list[np.ndarray], sr: int, workers: int = 0):
    """Measure many segments at once, across cores.

    Threads rather than processes: pyin holds the interpreter lock for part of
    its work so this gains around 1.5x rather than the core count, but spawning
    processes on macOS re-imports librosa in every worker and costs more than
    it saves.
    """
    usable = [s for s in segments if s is not None and s.size]
    if len(usable) < 4:
        return [shape(s, sr) if s is not None and s.size else (0.0, 0.0)
                for s in segments]

    workers = workers or max(1, min(8, (os.cpu_count() or 2) - 2))
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(_one, [(s, sr) for s in segments]))
    except Exception:
        return [shape(s, sr) if s is not None and s.size else (0.0, 0.0)
                for s in segments]
