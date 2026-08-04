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
    min_fill: float = 0.85
    max_slowdown: float = 0.94
    attempts: int = 3
    takes: int = 1
    match_register: float = 6.0
    clip_ceiling: float = 0.89


@dataclass
class FitReport:
    total: int = 0
    clean: int = 0
    borrowed: int = 0
    resynth: int = 0
    stretched: int = 0
    filled: int = 0
    redrawn: int = 0
    repitched: int = 0
    register: dict = field(default_factory=dict)
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


def _headroom(audio: np.ndarray, ceiling: float) -> np.ndarray:
    """Hold a clip below full scale before it reaches the bus.

    Intensity transfer and a lively emotion setting both raise level, and a
    clip that already sits at full scale has nowhere to go: it clips on its own
    and again when the background is mixed under it. Scaling the loudest clips
    down loses nothing, since the track is normalised as a whole afterwards.
    """
    if audio.size == 0:
        return audio
    peak = float(np.max(np.abs(audio)))
    if peak <= ceiling or peak <= 0:
        return audio
    return (audio * (ceiling / peak)).astype(np.float32)


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
    """Lengthen a clip that is much shorter than the line it replaces.

    The target is how long the original actor spoke, not how long the subtitle
    is on screen - those differ by seconds, and chasing the window drags the
    delivery out until it no longer matches the picture. Where the original
    left a pause, the dub leaves one too.
    """
    dur = audio.size / sr
    reference = cue.source_speech if cue.source_speech > 0.3 else cue.window
    target = min(reference, cue.window) * cfg.min_fill
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
    """Synthesize, and spend more draws where they buy something.

    Two reasons to draw again. A take can be plainly broken - far too short for
    its text, or mostly silence - and is then not worth laying into the track.
    A take can also simply be worse than another: synthesis is sampled, so
    delivery varies between draws, and one of them sits closer to how the line
    was actually said. Where the original delivery is known, takes are scored
    against it and the closest kept.
    """
    sr = backend.sample_rate
    takes = max(1, cfg.takes)
    tries = max(cfg.attempts, takes)
    best = np.zeros(0, dtype=np.float32)
    best_score = -1e9
    good = 0

    for attempt in range(tries):
        try:
            audio = backend.synth(
                text, voice, speed=1.0, emotion=cue.emotion, attempt=attempt
            )
        except TypeError:
            return backend.synth(text, voice, speed=1.0, emotion=cue.emotion)

        if _obviously_broken(audio, sr, text):
            report.redrawn += 1
            if audio.size > best.size and best_score <= -1e8:
                best = audio
            continue

        good += 1
        score = _score_take(audio, sr, cue)
        if score > best_score:
            best, best_score = audio, score
        if good >= takes:
            break

    return _match_register(best, sr, cue, cfg, report)


def _match_register(
    audio: np.ndarray, sr: int, cue: Cue, cfg: FitConfig, report: FitReport
) -> np.ndarray:
    """Move a cloned voice into the speaker's own register.

    The style vector carries identity but not reliably pitch height, and a clone
    sitting half an octave above the actor does not sound like them however
    expressive it is. Formants are held while shifting, so the voice deepens
    rather than turning into a caricature.

    The correction is per speaker rather than per line: a shift recomputed every
    line would wander with the pitch estimate and make the character's voice
    drift about.
    """
    from . import pitch as pitchmod

    target = cue.speaker_f0 or cue.source_f0
    if cfg.match_register <= 0 or audio.size == 0:
        return audio
    if not pitchmod.plausible(target):
        return audio
    median, _ = _pitch_shape(audio, sr)
    if not pitchmod.plausible(median):
        return audio

    history = report.register.setdefault(cue.speaker or "", [])
    history.append(target / median)
    steps = 12.0 * float(np.log2(np.median(history[-16:])))
    if abs(steps) < 0.75:
        return audio
    steps = max(-cfg.match_register, min(cfg.match_register, steps))

    try:
        import pyrubberband

        shifted = pyrubberband.pitch_shift(
            audio, sr, steps, rbargs={"-F": ""}
        ).astype(np.float32)
    except Exception:
        return audio
    if shifted.size:
        cue.pitch_shift = steps
        report.repitched += 1
        return shifted
    return audio


def _score_take(audio: np.ndarray, sr: int, cue: Cue) -> float:
    """How well a take matches how the line was actually delivered.

    Register and expressive range are what a listener judges: a take an octave
    off the speaker is wrong however clean it is, and a flat one is what gets
    called robotic. Both are read from the original line.
    """
    if cue.source_f0 <= 0:
        return -_internal_silence_seconds(audio, sr)

    median, spread = _pitch_shape(audio, sr)
    if median <= 0:
        return -50.0

    register = abs(12.0 * np.log2(median / cue.source_f0))
    liveliness = min(spread, cue.source_f0_range or spread)
    dead = _internal_silence_seconds(audio, sr)
    return liveliness - 2.0 * register - 3.0 * dead


def _pitch_shape(audio: np.ndarray, sr: int) -> tuple[float, float]:
    try:
        import librosa

        f0, _, _ = librosa.pyin(audio, fmin=75, fmax=400, sr=sr, frame_length=2048)
        v = f0[np.isfinite(f0)]
        if v.size < 8:
            return 0.0, 0.0
        med = float(np.median(v))
        semis = 12.0 * np.log2(v / med)
        return med, float(np.percentile(semis, 90) - np.percentile(semis, 10))
    except Exception:
        return 0.0, 0.0


def _internal_silence_seconds(audio: np.ndarray, sr: int) -> float:
    return verify._internal_silence(audio, sr)


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
            clip = _headroom(clip, cfg.clip_ceiling)

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

            at = int(max(0.0, cue.start + cue.speech_onset) * sr)
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
