"""Predict how long a run will take, and learn from the ones that finish.

Every stage has a rate: seconds of wall clock per unit of work. The defaults
below come from an M1 Pro and are only a starting point - each completed run
folds its real timings back into a cache, so estimates converge on whatever the
machine actually does.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

CACHE = Path.home() / ".cache" / "subs2dub" / "timings.json"

DEFAULTS = {
    "separate": 0.12,
    "diarize": 0.55,
    "translate_llm": 4.0,
    "translate_marian": 0.30,
    "references": 0.05,
    "prosody": 0.02,
    "synth_styletts2": 0.60,
    "synth_fish": 3.50,
    "synth_chatterbox": 1.20,
    "synth_kokoro": 0.15,
    "synth_piper": 0.05,
    "convert": 2.50,
    "mux": 0.03,
}

UNITS = {
    "separate": "audio",
    "diarize": "cues",
    "translate_llm": "cues",
    "translate_marian": "cues",
    "references": "cues",
    "prosody": "cues",
    "synth_styletts2": "speech",
    "synth_fish": "speech",
    "synth_chatterbox": "speech",
    "synth_kokoro": "speech",
    "synth_piper": "speech",
    "convert": "speech",
    "mux": "video",
}

LABELS = {
    "separate": "Separate voices from music",
    "diarize": "Identify speakers",
    "translate_llm": "Translate",
    "translate_marian": "Translate",
    "references": "Cut voice references",
    "prosody": "Read delivery",
    "synth_styletts2": "Synthesize speech",
    "synth_fish": "Synthesize speech",
    "synth_chatterbox": "Synthesize speech",
    "synth_kokoro": "Synthesize speech",
    "synth_piper": "Synthesize speech",
    "convert": "Convert voices",
    "mux": "Mux output",
}

STARTUP = {
    "separate": 12.0,
    "diarize": 8.0,
    "translate_llm": 20.0,
    "translate_marian": 10.0,
    "synth_styletts2": 25.0,
    "synth_fish": 30.0,
    "synth_chatterbox": 20.0,
    "synth_kokoro": 6.0,
    "synth_piper": 2.0,
    "convert": 15.0,
}


def _load() -> dict:
    try:
        return json.loads(CACHE.read_text())
    except (OSError, ValueError):
        return {}


def rates() -> dict:
    """Default rates with any learned values applied over the top."""
    out = dict(DEFAULTS)
    for stage, value in _load().items():
        if stage in out and isinstance(value, (int, float)) and value > 0:
            out[stage] = float(value)
    return out


def record(stage: str, seconds: float, units: float, weight: float = 0.4) -> None:
    """Fold one measurement into the cache as a moving average.

    Weighted rather than replaced so a single cold-cache or thermally throttled
    run does not distort later estimates.
    """
    if units <= 0 or seconds <= 0 or stage not in DEFAULTS:
        return
    observed = seconds / units
    data = _load()
    previous = data.get(stage)
    data[stage] = (
        observed if not isinstance(previous, (int, float)) or previous <= 0
        else previous * (1 - weight) + observed * weight
    )
    try:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(data, indent=2, sort_keys=True))
    except OSError:
        pass


def learned() -> set[str]:
    return {k for k in _load() if k in DEFAULTS}


@dataclass
class Plan:
    """What a run intends to do, in the units the rates are expressed in."""

    video_seconds: float
    cues: int
    speech_seconds: float
    stages: list[str] = field(default_factory=list)

    @classmethod
    def from_cues(cls, cues, video_seconds: float, chars_per_sec: float = 12.6):
        """Speech time is bounded by each cue's window; overruns get squeezed."""
        speech = 0.0
        for c in cues:
            spoken = len(c.text) / chars_per_sec
            speech += min(spoken, c.window * 1.25) if c.window else spoken
        return cls(video_seconds=video_seconds, cues=len(cues), speech_seconds=speech)


def breakdown(plan: Plan) -> list[tuple[str, float]]:
    """Per-stage seconds, in pipeline order."""
    r = rates()
    amounts = {
        "audio": plan.video_seconds,
        "cues": float(plan.cues),
        "speech": plan.speech_seconds,
        "video": plan.video_seconds,
    }
    out = []
    for stage in plan.stages:
        if stage not in r:
            continue
        seconds = r[stage] * amounts[UNITS[stage]] + STARTUP.get(stage, 0.0)
        out.append((stage, seconds))
    return out


def total(plan: Plan) -> float:
    return sum(seconds for _, seconds in breakdown(plan))


def human(seconds: float) -> str:
    seconds = max(0.0, seconds)
    if seconds < 90:
        return f"{seconds:.0f}s"
    minutes = seconds / 60
    if minutes < 90:
        return f"{minutes:.0f} min"
    hours = int(minutes // 60)
    return f"{hours}h {int(minutes - hours * 60):02d}m"


class Timer:
    """Time a stage and record it against the units of work it covered."""

    def __init__(self, stage: str, units: float) -> None:
        self.stage = stage
        self.units = units
        self.seconds = 0.0

    def __enter__(self) -> "Timer":
        self._t0 = time.time()
        return self

    def __exit__(self, exc_type, *_) -> bool:
        self.seconds = time.time() - self._t0
        if exc_type is None:
            record(self.stage, self.seconds, self.units)
        return False
