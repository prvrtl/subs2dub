"""Load subtitle files into speakable cues.

Subtitles are written to be read, not spoken. Three transformations matter:

1. Drop non-dialogue (SFX brackets, song lyrics, credit lines).
2. Split two-speaker cues ("- A" / "- B") so each can get its own voice.
3. Merge cues that are fragments of one sentence, so the TTS engine sees whole
   sentences and produces coherent prosody.
"""

from __future__ import annotations

import re
from pathlib import Path

import pysubs2

from .model import Cue

# Whole-line non-dialogue markers.
_BRACKETED = re.compile(r"^\s*[\[(][^\])]*[\])]\s*$")
_MUSIC = re.compile(r"[♪♫#]")
_CREDIT_JUNK = re.compile(
    r"\b(subtitle[sd]?|subs|sync(?:ed|hronized)?|translat(?:ed|ion)|encoded|ripped)\b"
    r".{0,40}\b(by|from|team|group)\b",
    re.I,
)
_TAGS = re.compile(r"<[^>]+>")
# Inline SFX inside an otherwise-spoken line, e.g. "[SIGHS] I'm fine."
_INLINE_SFX = re.compile(r"[\[(][^\])]{0,40}[\])]")
# A leading speaker-name label: "JOHN: hello" / "MAN ON RADIO: hello"
_SPEAKER_LABEL = re.compile(r"^\s*[A-Z][A-Z0-9 .'#\-]{1,24}:\s*")
_DASH = re.compile(r"^\s*[-–—]\s*")

_TERMINAL = tuple(".!?…\"'”’)]")


def _clean_line(line: str) -> str:
    line = _TAGS.sub("", line)
    line = _INLINE_SFX.sub(" ", line)
    line = _SPEAKER_LABEL.sub("", line)
    return re.sub(r"\s+", " ", line).strip()


def _is_dropped(text: str) -> bool:
    if not text or not re.search(r"[A-Za-z0-9]", text):
        return True
    if _MUSIC.search(text):  # song lyrics: don't dub, and don't voice lyrics
        return True
    if _CREDIT_JUNK.search(text):
        return True
    return bool(_BRACKETED.match(text))


def _split_speakers(lines: list[str]) -> list[str]:
    """A cue like "- Who's there?" / "- It's me." is two speakers sharing a window."""
    dashed = [ln for ln in lines if _DASH.match(ln)]
    if len(dashed) >= 2 and len(dashed) == len(lines):
        return [_DASH.sub("", ln).strip() for ln in lines]
    return [" ".join(_DASH.sub("", ln).strip() for ln in lines)]


def load(
    path: str | Path,
    *,
    merge_gap: float = 0.30,
    max_merge_window: float = 12.0,
) -> list[Cue]:
    """Parse a subtitle file into cleaned, merged Cues (times in seconds)."""
    subs = pysubs2.load(str(path))
    raw: list[Cue] = []

    for ev in subs:
        if ev.is_comment or ev.type != "Dialogue":
            continue
        text = ev.plaintext or ""
        lines = [_clean_line(ln) for ln in text.split("\n")]
        lines = [ln for ln in lines if ln]
        if not lines:
            continue

        parts = _split_speakers(lines)
        parts = [p for p in parts if not _is_dropped(p)]
        if not parts:
            continue

        start, end = ev.start / 1000.0, ev.end / 1000.0
        if end <= start:
            continue

        if len(parts) == 1:
            raw.append(Cue(idx=len(raw), start=start, end=end, text=parts[0]))
        else:
            # Divide the window between speakers in proportion to text length.
            total = sum(len(p) for p in parts) or 1
            cursor = start
            for p in parts:
                share = (end - start) * len(p) / total
                raw.append(
                    Cue(idx=len(raw), start=cursor, end=cursor + share, text=p)
                )
                cursor += share

    return _merge_fragments(raw, merge_gap, max_merge_window)


def _merge_fragments(
    cues: list[Cue], merge_gap: float, max_window: float
) -> list[Cue]:
    """Join consecutive cues that are pieces of one sentence."""
    if not cues:
        return []

    out: list[Cue] = []
    cur = cues[0]
    members = [cur.idx]

    for nxt in cues[1:]:
        gap = nxt.start - cur.end
        unfinished = not cur.text.rstrip().endswith(_TERMINAL)
        continues = nxt.text[:1].islower() or nxt.text.startswith(("and ", "but "))
        fits = (nxt.end - cur.start) <= max_window

        if gap <= merge_gap and (unfinished or continues) and fits:
            cur = Cue(
                idx=cur.idx,
                start=cur.start,
                end=nxt.end,
                text=f"{cur.text} {nxt.text}".strip(),
                source_text=f"{cur.source_text} {nxt.source_text}".strip(),
            )
            members.append(nxt.idx)
            continue

        cur.merged_from = members if len(members) > 1 else []
        out.append(cur)
        cur, members = nxt, [nxt.idx]

    cur.merged_from = members if len(members) > 1 else []
    out.append(cur)

    for i, c in enumerate(out):
        c.idx = i
    return out


def gaps(cues: list[Cue], total_duration: float) -> list[float]:
    """Seconds of silence following each cue, before the next one starts."""
    out = []
    for i, c in enumerate(cues):
        nxt = cues[i + 1].start if i + 1 < len(cues) else total_duration
        out.append(max(0.0, nxt - c.end))
    return out
