"""Turn a subtitle track into clean, non-repeating, correctly timed lines.

Uploaded subtitles are written for a reader: one line, punctuated, timed to the
speech. Automatic captions are not, and a dubbing pipeline that treats them the
same produces nonsense. The formats seen in practice:

  Rolling      Each event repeats the tail of the one before and appends a few
               new words, so the text is a scrolling window. Read literally this
               triples the dialogue, and the dub then races to fit words nobody
               said twice.
  Cumulative   Each event restates the whole line so far, growing to the end.
  Repeated     Consecutive events are byte-identical, holding a line on screen.
  Unpunctuated Nothing ends in a full stop, so every line looks like an
               unfinished sentence and naive merging swallows whole scenes -
               taking both halves of a conversation with it.

Each is detected rather than assumed, because the repairs are destructive: the
rolling repair would strip real repetition out of a properly authored track.

What comes out is a list of events whose text is each said exactly once, timed
to when it was said.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from . import confidence as conf

_WS = re.compile(r"\s+")
_MEMORY = 40


@dataclass
class Event:
    start: float
    end: float
    text: str

    @property
    def window(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class TrackInfo:
    """What kind of subtitle track this is, and what had to be done to it."""

    events: int = 0
    punctuated: bool = True
    rolling: bool = False
    duplicated: int = 0
    removed_words: int = 0
    words: int = 0
    kind: str = "subtitles"
    notes: list[str] = field(default_factory=list)

    def describe(self) -> str:
        out = [f"  {self.kind}: {self.events} lines"]
        for note in self.notes:
            out.append(f"    {note}")
        return "\n".join(out)

    def confidence(self) -> conf.Check:
        """How much to trust this track as a script for dubbing.

        Automatic captions carry no sentence punctuation, so lines merge
        conservatively and questions lose their intonation; a rolling track
        that had to repeat more words than it kept was carried by the repair
        rather than by the track itself.
        """
        remedy = (
            "supply an authored subtitle track with --subs FILE; automatic "
            "captions carry no sentence punctuation, so lines merge "
            "conservatively and questions lose their intonation; "
            "--no-auto-captions refuses them when the source is a URL"
        )

        if self.events == 0:
            return conf.Check(
                stage="subtitles", level=conf.UNRELIABLE,
                detail="no dialogue lines parsed from the track", remedy=remedy,
            )

        if self.rolling and self.removed_words > self.words:
            detail = (
                f"{self.kind}, {self.removed_words} of {self.words} words "
                f"were repeats - the repair carried the track"
            )
            return conf.Check(
                stage="subtitles", level=conf.UNRELIABLE, detail=detail,
                remedy=remedy,
            )

        weak = (
            not self.punctuated
            or self.rolling
            or self.duplicated > self.events * 0.20
        )
        if weak:
            if self.rolling:
                detail = (
                    f"{self.kind}, {self.removed_words} of {self.words} words "
                    f"were repeats"
                )
            elif not self.punctuated:
                detail = f"{self.kind}, no sentence punctuation"
            else:
                detail = f"{self.kind}, {self.duplicated} repeated lines removed"
            return conf.Check(
                stage="subtitles", level=conf.WEAK, detail=detail, remedy=remedy,
            )

        return conf.Check(
            stage="subtitles", level=conf.OK,
            detail=f"{self.kind}, {self.events} lines", remedy=remedy,
        )


def read(path: str | Path) -> list[Event]:
    import pysubs2

    subs = pysubs2.load(str(path))
    events = []
    for ev in subs:
        if ev.is_comment or ev.type != "Dialogue":
            continue
        text = _WS.sub(" ", (ev.plaintext or "").replace("\n", " ")).strip()
        start, end = ev.start / 1000.0, ev.end / 1000.0
        if end > start and text:
            events.append(Event(start, end, text))
    events.sort(key=lambda e: e.start)
    return events


def _words(text: str) -> list[str]:
    return text.lower().split()


def _shared_prefix(a: list[str], b: list[str]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _shared_tail(a: list[str], b: list[str], limit: int = 12) -> int:
    for n in range(min(limit, len(a), len(b)), 0, -1):
        if a[-n:] == b[:n]:
            return n
    return 0


def inspect(events: list[Event]) -> TrackInfo:
    """Classify the track without changing it."""
    info = TrackInfo(events=len(events))
    if not events:
        info.kind = "empty"
        return info

    ended = sum(1 for e in events if e.text.strip()[-1:] in ".!?")
    info.punctuated = ended > max(2, len(events) * 0.15)

    overlaps = identical = 0
    for a, b in zip(events, events[1:]):
        wa, wb = _words(a.text), _words(b.text)
        if not wa or not wb:
            continue
        if wa == wb:
            identical += 1
        if _shared_tail(wa, wb) or _shared_prefix(wa, wb) >= min(3, len(wa)):
            overlaps += 1

    info.duplicated = identical
    info.rolling = len(events) >= 4 and overlaps > len(events) * 0.3

    if info.rolling:
        info.kind = "automatic captions (rolling)"
        info.notes.append("each line repeats the previous one; de-duplicated")
    elif identical > len(events) * 0.2:
        info.kind = "automatic captions"
        info.notes.append(f"{identical} repeated lines removed")
    elif not info.punctuated:
        info.kind = "automatic captions"

    if not info.punctuated:
        info.notes.append(
            "no sentence punctuation, so question intonation is weaker and "
            "lines are merged conservatively"
        )
    return info


def _dedupe(events: list[Event]) -> tuple[list[Event], int]:
    """Keep only the words each event introduces."""
    out: list[Event] = []
    seen: list[str] = []
    dropped = 0

    for ev in events:
        words = ev.text.split()
        lower = _words(ev.text)
        if not words:
            continue

        overlap = _shared_tail(seen, lower)
        if not overlap:
            overlap = _shared_prefix(seen[-len(lower):] if seen else [], lower)
        fresh = words[overlap:]
        dropped += overlap

        seen = (seen + lower[overlap:])[-_MEMORY:]
        if not fresh:
            continue

        text = " ".join(fresh)
        start = max(ev.start, out[-1].end) if out else ev.start
        out.append(Event(min(start, ev.end - 0.05), ev.end, text))

    return out, dropped


def _drop_repeats(events: list[Event]) -> tuple[list[Event], int]:
    out: list[Event] = []
    dropped = 0
    for ev in events:
        if out and _words(out[-1].text) == _words(ev.text):
            out[-1] = Event(out[-1].start, max(out[-1].end, ev.end), out[-1].text)
            dropped += 1
            continue
        out.append(ev)
    return out, dropped


def _untangle(events: list[Event]) -> list[Event]:
    """Stop one event's window from running into the next one's."""
    out: list[Event] = []
    for ev in events:
        if out and ev.start < out[-1].end:
            out[-1] = Event(out[-1].start, max(out[-1].start + 0.05, ev.start),
                            out[-1].text)
        out.append(ev)
    return out


def normalize(events: list[Event]) -> tuple[list[Event], TrackInfo]:
    """Repair whichever faults this track actually has."""
    info = inspect(events)
    if not events:
        return [], info

    working = events
    if info.rolling:
        working, removed = _dedupe(working)
        info.removed_words = removed
        if removed:
            info.notes.append(f"{removed} repeated words removed")
    else:
        working, dropped = _drop_repeats(working)
        if dropped and not info.notes:
            info.notes.append(f"{dropped} repeated lines removed")

    working = _untangle(working)
    info.events = len(working)
    info.words = sum(len(e.text.split()) for e in working)
    return working, info


def merge_policy(info: TrackInfo) -> tuple[float, float]:
    """Merge gap and maximum merged span appropriate to this track.

    A punctuated track can be merged into whole sentences safely, because a full
    stop marks where one ends. Without punctuation every line looks unfinished,
    and merging by that signal alone runs several speakers together into one cue
    - which then gets one voice, and has to be rushed to fit.
    """
    if info.punctuated:
        return 0.30, 12.0
    return 0.24, 5.0
