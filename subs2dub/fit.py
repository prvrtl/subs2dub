"""Fit synthesized speech into fixed subtitle windows.

A subtitle cue is a hard [start, end] window; TTS output is whatever length it
happens to be. Four levers, applied in order of how much they degrade the audio:

  1. Nothing      - the line already fits. Most do.
  2. Borrow       - run past `end` into the silence before the next cue.
  3. Engine speed - re-synthesize faster. Better than post-stretching, because
                    the model re-articulates instead of resampling. Not all
                    backends support it.
  4. Time-stretch - pitch-preserving compression via rubberband. Capped, since
                    past ~15% it audibly wobbles.

Lines that still overrun are reported rather than degraded further; they need a
shorter translation, which is what the adaptation pass provides.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .model import Cue
from .tts import TTSBackend


@dataclass
class FitConfig:
    max_engine_speed: float = 1.25
    max_stretch: float = 1.15
    max_stretch_no_speed: float = 1.22  # backends without a rate parameter
    max_borrow: float = 1.50
    borrow_guard: float = 0.12  # silence always left before the next cue
    overrun_tolerance: float = 0.15  # ignore overruns smaller than this


@dataclass
class FitReport:
    total: int = 0
    clean: int = 0
    borrowed: int = 0
    resynth: int = 0
    stretched: int = 0
    overrun: list[Cue] = field(default_factory=list)

    def summary(self) -> str:
        pct = (100.0 * len(self.overrun) / self.total) if self.total else 0.0
        return (
            f"{self.total} cues: {self.clean} fit as-is, {self.borrowed} borrowed "
            f"gap time, {self.resynth} re-synthesized faster, {self.stretched} "
            f"time-stretched, {len(self.overrun)} still overrun ({pct:.1f}%)"
        )


def _time_stretch(audio: np.ndarray, sr: int, rate: float) -> np.ndarray:
    """rate > 1 makes it shorter. Falls back to the original on failure."""
    if abs(rate - 1.0) < 1e-3 or audio.size == 0:
        return audio
    try:
        import pyrubberband

        return pyrubberband.time_stretch(audio, sr, rate).astype(np.float32)
    except Exception:
        return audio


def fit_cue(
    cue: Cue,
    backend: TTSBackend,
    gap_after: float,
    cfg: FitConfig,
    report: FitReport,
) -> np.ndarray:
    """Synthesize `cue` and squeeze it toward its window. Returns the clip."""
    sr = backend.sample_rate
    voice = cue.voice or "af_heart"

    audio = backend.synth(cue.text, voice, speed=1.0, emotion=cue.emotion)
    dur = len(audio) / sr
    cue.rendered_dur = dur
    report.total += 1

    window = cue.window
    if dur <= window:
        report.clean += 1
        return audio

    # 2. Borrow from the gap before the next cue, keeping a guard of silence.
    borrow = max(0.0, min(gap_after - cfg.borrow_guard, cfg.max_borrow))
    budget = window + borrow
    if dur <= budget:
        cue.borrowed = dur - window
        report.borrowed += 1
        return audio

    # 3. Re-synthesize at a faster engine speed, where the backend can.
    supports_speed = getattr(backend, "supports_speed", True)
    if supports_speed:
        needed = dur / budget
        speed = min(needed, cfg.max_engine_speed)
        if speed > 1.01:
            faster = backend.synth(
                cue.text, voice, speed=speed, emotion=cue.emotion
            )
            if faster.size:
                audio, dur = faster, len(faster) / sr
                cue.engine_speed = speed
                report.resynth += 1

        if dur <= budget:
            cue.borrowed = max(0.0, dur - window)
            return audio

    # 4. Pitch-preserving time-stretch, hard-capped. When the backend has no
    # rate control this is the only remaining lever, so it is allowed to work
    # a little harder before giving up and reporting an overrun.
    cap = cfg.max_stretch if supports_speed else cfg.max_stretch_no_speed
    rate = min(dur / budget, cap)
    if rate > 1.01:
        audio = _time_stretch(audio, sr, rate)
        dur = len(audio) / sr
        cue.stretch = rate
        report.stretched += 1

    cue.borrowed = max(0.0, min(dur, budget) - window)
    over = dur - budget
    if over > cfg.overrun_tolerance:
        cue.overrun = over
        report.overrun.append(cue)
    return audio


def render_dialogue_track(
    cues: list[Cue],
    backend: TTSBackend,
    gaps: list[float],
    total_duration: float,
    cfg: FitConfig | None = None,
    progress=None,
) -> tuple[np.ndarray, FitReport]:
    """Fit every cue and lay the clips onto one dialogue bus."""
    cfg = cfg or FitConfig()
    sr = backend.sample_rate
    report = FitReport()

    bus = np.zeros(int(total_duration * sr) + sr, dtype=np.float32)

    for i, cue in enumerate(cues):
        clip = fit_cue(cue, backend, gaps[i], cfg, report)
        if clip.size:
            if cue.gain_db:
                clip = clip * float(10.0 ** (cue.gain_db / 20.0))
            at = int(cue.start * sr)
            end = min(at + clip.size, bus.size)
            # Additive: overlaps are rare after fitting but must not truncate.
            bus[at:end] += clip[: end - at]
        if progress:
            progress(i + 1, len(cues), cue)

    peak = float(np.max(np.abs(bus))) if bus.size else 0.0
    if peak > 0.99:
        bus *= 0.99 / peak
    return bus, report
