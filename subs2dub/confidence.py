"""Let each stage say how much to trust its own output before synthesis starts.

The recurring failure in this pipeline is not a crash. It is a stage that
produces plausible but wrong output - four speakers found where there are two,
captions merged into run-on sentences, a translated line quietly clipped -
discovered only by listening to a finished render an hour later. A handful of
stages can measure their own reliability at the moment they run: speaker
separation knows its own silhouette score, subtitle parsing knows whether the
track carried punctuation, translation knows how many lines overran their
window. Each states that measurement as a `Check` instead of printing it
inline, so the statements collect into one block read once, right before the
expensive work begins, rather than scrolling past in the log.

`score` is deliberately not normalized to a common scale across stages. A
silhouette of 0.4 and an over-budget share of 0.4 mean opposite things - one
wants to be high, the other low - so folding them into one number would either
lie or need a conversion table nobody would remember. Each stage picks its own
bands instead, and what is shared is the shape of the report, not the scale of
the numbers in it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

OK = "ok"
WEAK = "weak"
UNRELIABLE = "unreliable"

_MARK = {UNRELIABLE: "!!", WEAK: " ~", OK: "ok"}


@dataclass
class Check:
    """One stage's statement of how much to trust what it just produced.

    `score` is the stage's own measure on the stage's own scale - it is
    carried through to the log for a human to read, not for arithmetic across
    checks. Comparing a `speakers` score to a `translation` score is meaningless;
    comparing a `speakers` score across two runs of the same video is not.
    """

    stage: str
    level: str
    detail: str
    remedy: str = ""
    score: float | None = None


def band(value: float, weak: float, bad: float, higher_is_better: bool = True) -> str:
    """Classify `value` into OK, WEAK, or UNRELIABLE against two thresholds.

    `weak` is the boundary between OK and WEAK, `bad` the boundary between WEAK
    and UNRELIABLE. Which side counts as good depends on `higher_is_better`:
    a silhouette score wants to be high, an over-budget share wants to be low.
    """
    if higher_is_better:
        if value < bad:
            return UNRELIABLE
        if value < weak:
            return WEAK
        return OK
    if value > bad:
        return UNRELIABLE
    if value > weak:
        return WEAK
    return OK


def report(checks: list[Check]) -> str:
    """Render the confidence ledger, or "" when there is nothing to flag.

    Silence on a passing check would make it impossible to tell a check that
    passed from a check that never ran, so once anything is non-OK every check
    prints its one-line detail - only a report with nothing to flag at all is
    empty, so a clean run stays quiet.
    """
    bad = [c for c in checks if c.level == UNRELIABLE]
    weak = [c for c in checks if c.level == WEAK]
    if not bad and not weak:
        return ""

    counts = []
    if bad:
        counts.append(f"{len(bad)} unreliable")
    if weak:
        counts.append(f"{len(weak)} weak")
    width = max((len(c.stage) for c in checks), default=8) + 1

    lines = [f"  confidence: {', '.join(counts)} of {len(checks)} checks"]
    for c in checks:
        lines.append(f"    {_MARK[c.level]} {c.stage:<{width}}{c.detail}")
        if c.level != OK and c.remedy:
            lines.append(f"       {'fix':<{width}}{c.remedy}")
    return "\n".join(lines)


def record(checks: list[Check], path: Path) -> Path:
    """Persist the ledger as JSON next to render.json, whatever it says."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(c) for c in checks], indent=1))
    return path


def references(good: dict, poor: dict) -> Check:
    """Whether the clips handed to a cloning engine actually contain speech."""
    total = len(good) + len(poor)
    remedy = (
        "the separated speech carries little voice, which happens when a score "
        "runs under the dialogue; try --no-separate to clone from the mix, or "
        "an engine with preset voices such as --backend piper"
    )
    if total == 0:
        return Check("references", OK, "no cloning needed", remedy)
    if not good:
        return Check(
            "references", UNRELIABLE,
            f"none of the {total} reference clips carry a usable voice", remedy,
        )
    if poor:
        return Check(
            "references", WEAK,
            f"{len(poor)} of {total} reference clips carry little voice", remedy,
        )
    return Check("references", OK, f"{total} clips carry a clear voice", remedy)
