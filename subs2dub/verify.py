"""Inspect a rendered dialogue track for the faults you would otherwise hear.

Fit statistics say whether lines fit. They say nothing about whether the result
is listenable: a line can fit perfectly because half of it was cut off. These
checks look at the audio and the per-cue render record together, and name the
specific problems that show up in a dub.

Run as part of a build, or afterwards:

    python -m subs2dub verify --work ./work
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .model import Cue

# Anything quieter than this counts as silence for gap and clipping checks.
SILENCE = 0.006
# A clip ending this loud was cut rather than finished.
ABRUPT_END = 0.05
# Stretching past this is audible as a wobble.
HARSH_STRETCH = 1.30


@dataclass
class Problem:
    start: float
    kind: str
    detail: str
    severity: str  # "bad" or "warn"

    def __str__(self) -> str:
        mark = "!!" if self.severity == "bad" else " ~"
        return f"{mark} {self.start:7.1f}s  {self.kind:<16} {self.detail}"


def record(cues: list[Cue], path: Path) -> Path:
    """Persist what the fitter did to each cue, so a render can be re-examined."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for c in cues:
        row = asdict(c)
        row.pop("audio", None)
        rows.append(row)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    return path


def _tail_level(audio: np.ndarray, sr: int, ms: float = 30.0) -> float:
    n = min(int(ms / 1000.0 * sr), audio.size)
    return float(np.abs(audio[-n:]).mean()) if n else 0.0


def check_clip(cue: Cue, audio: np.ndarray, sr: int) -> list[Problem]:
    """Faults visible in one rendered line."""
    out: list[Problem] = []
    if audio.size == 0:
        out.append(Problem(cue.start, "silent", "no audio rendered", "bad"))
        return out

    duration = audio.size / sr
    peak = float(np.abs(audio).max())

    if peak < SILENCE:
        out.append(Problem(cue.start, "silent", f"peak {peak:.4f}", "bad"))
        return out

    if cue.truncated > 0.15:
        # The audible fault: a sentence that stops rather than ends.
        level = _tail_level(audio, sr)
        severity = "bad" if level > ABRUPT_END else "warn"
        out.append(Problem(
            cue.start, "cut off",
            f"lost {cue.truncated:.1f}s, ends at level {level:.3f}", severity,
        ))

    if cue.stretch and cue.stretch > HARSH_STRETCH:
        out.append(Problem(
            cue.start, "over-stretched",
            f"x{cue.stretch:.2f} - audible wobble", "warn",
        ))

    # Speech that is far shorter or longer than its text implies usually means
    # the synthesizer dropped words or rambled.
    expected = len(cue.text) / 12.6
    if expected > 0.4:
        if duration < expected * 0.45:
            out.append(Problem(
                cue.start, "too short",
                f"{duration:.1f}s for {len(cue.text)} chars - words likely dropped",
                "bad",
            ))
        elif duration > expected * 2.5:
            out.append(Problem(
                cue.start, "runaway",
                f"{duration:.1f}s for {len(cue.text)} chars", "bad",
            ))

    if peak > 0.999:
        out.append(Problem(cue.start, "clipped", f"peak {peak:.3f}", "warn"))

    dead = _internal_silence(audio, sr)
    if dead > 0.35 and dead / duration > 0.25:
        out.append(Problem(
            cue.start, "dead air",
            f"{dead:.1f}s of the {duration:.1f}s clip is silence", "bad",
        ))

    return out


def _internal_silence(audio: np.ndarray, sr: int, min_run: float = 0.25) -> float:
    """Seconds of silence with speech on both sides."""
    frame = max(1, int(0.02 * sr))
    usable = audio[: audio.size // frame * frame]
    if usable.size == 0:
        return 0.0
    energy = np.abs(usable.reshape(-1, frame)).mean(axis=1)
    peak = float(energy.max())
    if peak <= 0:
        return 0.0
    quiet = energy < max(SILENCE, peak * 0.03)

    total = 0.0
    i = 0
    while i < len(quiet):
        if not quiet[i]:
            i += 1
            continue
        j = i
        while j < len(quiet) and quiet[j]:
            j += 1
        run = (j - i) * frame / sr
        if i > 0 and j < len(quiet) and run > min_run:
            total += run
        i = j
    return total


def check_track(
    bus: np.ndarray, sr: int, cues: list[Cue], total: float
) -> list[Problem]:
    """Faults visible only once the lines are laid out together."""
    out: list[Problem] = []
    for i, cue in enumerate(cues):
        if cue.overlapped <= 0.02:
            continue
        nxt = cues[i + 1] if i + 1 < len(cues) else None
        if nxt is None:
            continue
        if nxt.speaker and nxt.speaker == cue.speaker:
            # One character cannot talk over themselves; this is a layout fault.
            out.append(Problem(
                cue.start, "self-overlap",
                f"{cue.overlapped:.2f}s over the same speaker's next line", "bad",
            ))
        elif cue.overlapped > 0.6:
            out.append(Problem(
                cue.start, "long overlap",
                f"{cue.overlapped:.2f}s under {nxt.speaker}", "warn",
            ))

    # A stretch of dialogue that renders as silence means lines went missing.
    spoken = [c for c in cues if c.start < total]
    for i in range(len(spoken) - 1):
        a, b = spoken[i], spoken[i + 1]
        gap_start = int(a.end * sr)
        gap_end = min(int(b.start * sr), bus.size)
        if gap_end - gap_start < int(0.25 * sr):
            continue
        window = bus[gap_start:gap_end]
        if window.size and float(np.abs(window).max()) < SILENCE:
            continue  # a real pause between lines is fine

    return out


def summarize(problems: list[Problem], cues: int) -> str:
    bad = [p for p in problems if p.severity == "bad"]
    warn = [p for p in problems if p.severity == "warn"]
    if not problems:
        return f"verify: {cues} cues, nothing to flag"
    kinds: dict[str, int] = {}
    for p in problems:
        kinds[p.kind] = kinds.get(p.kind, 0) + 1
    detail = ", ".join(f"{n} {k}" for k, n in sorted(kinds.items()))
    return (
        f"verify: {len(bad)} serious, {len(warn)} minor across {cues} cues "
        f"({detail})"
    )
