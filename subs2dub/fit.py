"""Fit synthesized speech into fixed subtitle windows.

A subtitle cue is a hard [start, end] window; synthesized speech is whatever
length it happens to be. Too long is handled in order of how much each step
degrades the audio:

  1. Nothing      - the line already fits.
  2. Borrow       - run past `end` into the silence before the next cue.
  3. Engine speed - synthesize again faster. Better than post-stretching,
                    because the model re-articulates instead of resampling.
                    Not every engine can.
  4. Time-stretch - pitch-preserving compression. Capped, since past ~15% it
                    audibly wobbles, and again at 1.45 where losing the end of
                    a sentence would be worse.

Too short matters just as much, and for longer went unhandled here. A cue
window is how long the original actor spoke, so a clip that fills half of it
leaves the character silent with their mouth moving. `_fill` lengthens those.

Laying the clips down is its own problem. A line may run into the next one when
the next one belongs to someone else, ducked underneath, because people overlap
in conversation. Running into your own next line is only ever bad timing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import progress as pr
from . import verify
from .cues import speakable
from .model import Cue
from .tts import TTSBackend


@dataclass
class FitConfig:
    max_engine_speed: float = 1.25
    max_stretch: float = 1.15
    max_stretch_no_speed: float = 1.22
    max_stretch_emergency: float = 1.45
    max_borrow: float = 1.50
    borrow_guard: float = 0.12
    overrun_tolerance: float = 0.15
    max_overlap_other: float = 0.45
    duck_overlap_db: float = -3.5
    min_fill: float = 0.88
    max_slowdown: float = 0.78
    attempts: int = 3


@dataclass
class FitReport:
    total: int = 0
    clean: int = 0
    borrowed: int = 0
    resynth: int = 0
    stretched: int = 0
    filled: int = 0
    redrawn: int = 0
    overrun: list[Cue] = field(default_factory=list)
    failed: list[tuple[Cue, str]] = field(default_factory=list)
    problems: list = field(default_factory=list)

    def summary(self) -> str:
        pct = (100.0 * len(self.overrun) / self.total) if self.total else 0.0
        out = (
            f"{self.total} cues: {self.clean} fit as-is, {self.borrowed} borrowed "
            f"gap time, {self.resynth} re-synthesized faster, {self.stretched} "
            f"time-stretched, {self.filled} lengthened to fill, "
            f"{len(self.overrun)} still overrun ({pct:.1f}%)"
        )
        if self.redrawn:
            out += f", {self.redrawn} redrawn after a bad take"
        if self.failed:
            out += f", {len(self.failed)} left silent after repeated failures"
        return out

    def problem_report(self) -> str:
        return verify.summarize(self.problems, self.total)


def _line(cue: Cue, i: int) -> 'pr.Line':
    return pr.Line(index=i, start=cue.start, speaker=cue.speaker or '',
                   text=cue.text)


def _time_stretch(audio: np.ndarray, sr: int, rate: float) -> np.ndarray:
    """rate > 1 makes it shorter. Falls back to the original on failure."""
    if abs(rate - 1.0) < 1e-3 or audio.size == 0:
        return audio
    try:
        import pyrubberband

        return pyrubberband.time_stretch(audio, sr, rate).astype(np.float32)
    except Exception:
        return audio


def _cap(audio: np.ndarray, sr: int, seconds: float) -> np.ndarray:
    """Shorten to `seconds`, cutting at the quietest nearby point.

    Cutting straight at the limit lands mid-word and is very audible. Speech has
    gaps between words; searching backwards a little for the quietest one loses
    a word instead of half of one.
    """
    limit = int(max(0.0, seconds) * sr)
    if limit <= 0 or audio.size <= limit:
        return audio

    frame = max(1, int(0.010 * sr))
    lookback = min(int(0.250 * sr), limit)
    cut = limit
    if lookback > frame:
        window = audio[limit - lookback:limit]
        frames = window[: (len(window) // frame) * frame].reshape(-1, frame)
        if frames.size:
            energy = np.abs(frames).mean(axis=1)
            cut = limit - lookback + (int(np.argmin(energy)) + 1) * frame

    out = audio[:cut].copy()
    fade = min(int(0.020 * sr), out.size)
    if fade:
        out[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
    return out


def _duck_tail(
    audio: np.ndarray, sr: int, from_seconds: float, gain_db: float
) -> np.ndarray:
    """Fade the part running past `from_seconds` down to `gain_db`.

    The overlapping tail sits under the incoming line rather than competing with
    it, which is roughly what a person does when someone starts talking.
    """
    start = int(max(0.0, from_seconds) * sr)
    if start >= audio.size:
        return audio
    out = audio.copy()
    target = float(10.0 ** (gain_db / 20.0))
    ramp = min(int(0.060 * sr), audio.size - start)
    if ramp > 0:
        out[start:start + ramp] *= np.linspace(1.0, target, ramp, dtype=np.float32)
    out[start + ramp:] *= target
    return out


def _fill(
    audio: np.ndarray, sr: int, cue: Cue, text: str,
    cfg: FitConfig, report: FitReport,
) -> np.ndarray:
    """Lengthen a clip that leaves most of its window silent.

    A translation shorter than the line it replaces is normal, but the gap has
    to go somewhere. Spread across the delivery it sounds like an unhurried
    reading; left at the end it sounds like the character stopped mid-scene.
    """
    dur = audio.size / sr
    target = cue.window * cfg.min_fill
    if dur <= 0 or dur >= target:
        return audio

    plausible = len(text) / 12.6 * 0.55
    if dur < plausible:
        return audio

    rate = max(cfg.max_slowdown, dur / target)
    if rate > 0.985:
        return audio
    out = _time_stretch(audio, sr, rate)
    if out.size:
        cue.stretch = rate
        report.filled += 1
        return out
    return audio


def _synth_checked(
    backend: TTSBackend, text: str, voice: str, cue: Cue,
    cfg: FitConfig, report: FitReport,
) -> np.ndarray:
    """Synthesize, and draw again while the result is obviously broken.

    The checks are the cheap, unambiguous ones: a clip far too short for its
    text has dropped words, and a clip that is mostly silence has stalled.
    Neither is worth laying into the track when another draw is available.
    """
    sr = backend.sample_rate
    best = np.zeros(0, dtype=np.float32)

    for attempt in range(max(1, cfg.attempts)):
        try:
            audio = backend.synth(
                text, voice, speed=1.0, emotion=cue.emotion, attempt=attempt
            )
        except TypeError:
            audio = backend.synth(text, voice, speed=1.0, emotion=cue.emotion)
            return audio
        if audio.size > best.size:
            best = audio
        if not _obviously_broken(audio, sr, text):
            return audio
        report.redrawn += 1

    return best


def _obviously_broken(audio: np.ndarray, sr: int, text: str) -> bool:
    if audio.size == 0:
        return True
    duration = audio.size / sr
    expected = len(text) / 12.6
    if expected > 0.4 and duration < expected * 0.5:
        return True
    return verify._internal_silence(audio, sr) > max(0.4, duration * 0.3)


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
    text = speakable(cue.text)

    audio = _synth_checked(backend, text, voice, cue, cfg, report)
    dur = len(audio) / sr
    cue.rendered_dur = dur
    report.total += 1

    window = cue.window
    if dur <= window:
        audio = _fill(audio, sr, cue, text, cfg, report)
        report.clean += 1
        return audio

    borrow = max(0.0, min(gap_after - cfg.borrow_guard, cfg.max_borrow))
    budget = window + borrow
    if dur <= budget:
        cue.borrowed = dur - window
        report.borrowed += 1
        return audio

    supports_speed = getattr(backend, "supports_speed", True)
    if supports_speed:
        needed = dur / budget
        speed = min(needed, cfg.max_engine_speed)
        if speed > 1.01:
            faster = backend.synth(
                text, voice, speed=speed, emotion=cue.emotion
            )
            if faster.size:
                audio, dur = faster, len(faster) / sr
                cue.engine_speed = speed
                report.resynth += 1

        if dur <= budget:
            cue.borrowed = max(0.0, dur - window)
            return audio

    cap = cfg.max_stretch if supports_speed else cfg.max_stretch_no_speed
    if dur / budget > cap:
        cap = min(dur / budget, cfg.max_stretch_emergency)
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
        if progress is not None and hasattr(progress, 'now'):
            progress.now(_line(cue, i))
        try:
            clip = fit_cue(cue, backend, gaps[i], cfg, report)
        except Exception as exc:
            report.failed.append((cue, str(exc).split("\n")[0][:120]))
            clip = np.zeros(0, dtype=np.float32)
        if clip.size:
            if cue.gain_db:
                clip = clip * float(10.0 ** (cue.gain_db / 20.0))

            nxt = cues[i + 1] if i + 1 < len(cues) else None
            free = cue.window + gaps[i] - cfg.borrow_guard
            room = free
            if nxt is not None and nxt.speaker and nxt.speaker != cue.speaker:
                room += cfg.max_overlap_other
            before = clip.size
            clip = _cap(clip, sr, room)
            if clip.size < before:
                cue.truncated = (before - clip.size) / sr

            if clip.size / sr > free > 0:
                clip = _duck_tail(clip, sr, free, cfg.duck_overlap_db)
                cue.overlapped = clip.size / sr - free

            report.problems.extend(verify.check_clip(cue, clip, sr))

            at = int(cue.start * sr)
            end = min(at + clip.size, bus.size)
            bus[at:end] += clip[: end - at]
        if progress is not None:
            if hasattr(progress, 'finish'):
                line = _line(cue, i)
                line.outcome, line.detail = pr.outcome_of(cue)
                progress.finish(line)
            else:
                progress(i + 1, len(cues), cue)

    peak = float(np.max(np.abs(bus))) if bus.size else 0.0
    if peak > 0.99:
        bus *= 0.99 / peak
    return bus, report
