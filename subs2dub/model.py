"""Core data types shared across the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Cue:
    """One unit of dialogue to be spoken.

    Times are in seconds. A Cue starts life as a subtitle event and accumulates
    fields as it moves through the pipeline: `speaker` from diarization,
    `text` possibly rewritten by the adaptation pass, `audio` after synthesis.
    """

    idx: int
    start: float
    end: float
    text: str
    speaker: str | None = None
    voice: str | None = None

    audio: Path | None = None
    rendered_dur: float | None = None
    engine_speed: float = 1.0
    stretch: float = 1.0
    borrowed: float = 0.0
    overrun: float = 0.0
    source_speech: float = 0.0
    overlapped: float = 0.0
    truncated: float = 0.0

    gain_db: float = 0.0
    f0_end_slope: float = 0.0
    intensity: float = 0.0
    punct_fixed: bool = False
    emotion: float = 0.5
    emo_vector: list[float] = field(default_factory=list)

    source_text: str = ""
    merged_from: list[int] = field(default_factory=list)

    @property
    def window(self) -> float:
        """Seconds available in the subtitle cue itself."""
        return self.end - self.start

    @property
    def budget(self) -> float:
        """Seconds available including any time borrowed from the next gap."""
        return self.window + self.borrowed

    def __post_init__(self) -> None:
        if not self.source_text:
            self.source_text = self.text
