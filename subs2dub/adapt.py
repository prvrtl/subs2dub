"""Shorten lines that cannot be made to fit by audio manipulation alone.

When a line still overruns after re-synthesis and time-stretching, the problem
is no longer acoustic: there are too many syllables for the window. Dubbing
studios call the fix adaptation - rewriting the line shorter while preserving
meaning and register. It is a language task, not a signal-processing one.

The fitter exports the offenders with a character budget, the rewrites happen
outside this tool, and the results are applied on the next build.

Rewrites are keyed by cue start time rather than index, so they survive changes
to merge settings or re-parsing of the subtitle file.
"""

from __future__ import annotations

import csv
from pathlib import Path

from .model import Cue

CHARS_PER_SEC = 15.0


def export_overruns(
    overruns: list[Cue], path: Path, chars_per_sec: float = CHARS_PER_SEC
) -> Path:
    """Write the lines that need shortening, with a character budget for each."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(
            ["start", "budget_s", "over_s", "max_chars", "cur_chars", "text", "rewrite"]
        )
        for c in sorted(overruns, key=lambda c: -c.overrun):
            budget = c.budget
            w.writerow([
                f"{c.start:.2f}",
                f"{budget:.2f}",
                f"{c.overrun:.2f}",
                int(budget * chars_per_sec),
                len(c.text),
                c.text,
                "",
            ])
    return path


def load_rewrites(path: Path) -> dict[str, str]:
    """Read back a rewrite file. Blank rewrites are ignored."""
    out: dict[str, str] = {}
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            new = (row.get("rewrite") or "").strip()
            if new:
                out[f"{float(row['start']):.2f}"] = new
    return out


def apply_rewrites(cues: list[Cue], rewrites: dict[str, str]) -> int:
    """Swap in shortened text. Returns how many cues were changed."""
    n = 0
    for c in cues:
        new = rewrites.get(f"{c.start:.2f}")
        if new and new != c.text:
            c.text = new
            n += 1
    return n


def estimate_chars_per_sec(cues: list[Cue]) -> float:
    """Calibrate the budget from cues that actually rendered, not a guess."""
    pairs = [
        (len(c.text), c.rendered_dur)
        for c in cues
        if c.rendered_dur and c.rendered_dur > 0.5 and c.engine_speed == 1.0
    ]
    if len(pairs) < 20:
        return CHARS_PER_SEC
    rates = sorted(ch / d for ch, d in pairs)
    return rates[len(rates) // 2]
