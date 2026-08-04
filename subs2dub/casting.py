"""Assign a distinct TTS voice to each detected speaker.

Two rules do most of the work:

  * Match gender to the original actor, inferred from median F0 of their lines.
    A male character voiced by a female preset is the single most jarring error.
  * Give the busiest characters the best voices. Kokoro's presets vary in
    quality, and leads carry most of the runtime.

Accent is used as a last resort to keep same-gender characters distinguishable:
American presets are handed out first, British ones once those run out.

Every run writes the resulting assignment out as cast.tsv: one row per cue,
with its speaker label and voice. That file can be edited by hand - relabel a
line to a different speaker, swap in a different voice - and fed back with
`--cast`, which skips diarization entirely and applies the edited columns
instead. Rows are matched to cues by start time, the same convention
translate.py and adapt.py already use, so the file survives re-parsing the
same subtitle track.
"""

from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path

from .model import Cue

_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,23}$")

AM_F = ["af_heart", "af_bella", "af_nicole", "af_aoede", "af_kore",
        "af_sarah", "af_nova", "af_sky", "af_alloy", "af_jessica", "af_river"]
AM_M = ["am_michael", "am_fenrir", "am_puck", "am_echo", "am_eric",
        "am_liam", "am_onyx", "am_adam", "am_santa"]
BR_F = ["bf_emma", "bf_isabella", "bf_alice", "bf_lily"]
BR_M = ["bm_george", "bm_fable", "bm_lewis", "bm_daniel"]

FEMALE_POOL = AM_F + BR_F
MALE_POOL = AM_M + BR_M

F0_SPLIT = 158.0


def cast(
    cues: list[Cue],
    f0_by_speaker: dict[int, float],
    *,
    default_voice: str = "af_heart",
    pools: dict[str, list[str]] | None = None,
) -> dict[str, str]:
    """Map speaker id -> voice name, and write `voice` onto each cue.

    `pools` supplies gendered voice lists for the active backend; without it the
    Kokoro English voices are used. Backends for other languages expose their own
    via `voice_pools()`.
    """
    counts = Counter(c.speaker for c in cues if c.speaker)
    if not counts:
        for c in cues:
            c.voice = default_voice
        return {}

    order = [spk for spk, _ in counts.most_common()]

    def gender_of(spk: str) -> str:
        f0 = f0_by_speaker.get(int(spk[1:]))
        if f0 is None:
            return "?"
        return "F" if f0 >= F0_SPLIT else "M"

    genders = {spk: gender_of(spk) for spk in order}

    unknown = [s for s in order if genders[s] == "?"]
    for i, spk in enumerate(unknown):
        genders[spk] = "F" if i % 2 == 0 else "M"

    fem = list((pools or {}).get("F") or FEMALE_POOL)
    mal = list((pools or {}).get("M") or MALE_POOL)
    mapping: dict[str, str] = {}
    for spk in order:
        pool = fem if genders[spk] == "F" else mal
        if not pool:
            pool = list((pools or {}).get(genders[spk]) or
                        (FEMALE_POOL if genders[spk] == "F" else MALE_POOL))
        mapping[spk] = pool.pop(0) if pool else default_voice

    for c in cues:
        c.voice = mapping.get(c.speaker or "", default_voice)
    return mapping


def describe(
    cues: list[Cue], mapping: dict[str, str], f0: dict[int, float]
) -> str:
    counts = Counter(c.speaker for c in cues if c.speaker)
    lines = ["  speaker  lines   F0    voice"]
    for spk, n in counts.most_common():
        hz = f0.get(int(spk[1:]))
        lines.append(
            f"  {spk:>7}  {n:>5}  {hz:>4.0f}  {mapping.get(spk, '?')}"
            if hz
            else f"  {spk:>7}  {n:>5}     -  {mapping.get(spk, '?')}"
        )
    return "\n".join(lines)


def export_cast(cues: list[Cue], path: Path) -> Path:
    """Write the per-cue assignment so it can be edited and fed back with --cast.

    `idx` is there so a human can find a line in the subtitle file; `text` is
    there for the same reason and is not read back by `load_cast` - a rewrite
    belongs to --rewrites, not here. `csv.writer` rather than a hand-rolled
    join keeps this symmetric with `load_cast` under embedded tabs or quotes.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["idx", "start", "speaker", "voice", "text"])
        for c in cues:
            w.writerow([c.idx, f"{c.start:.2f}", c.speaker or "",
                        c.voice or "", c.text])
    return path


def load_cast(path: Path) -> dict[str, tuple[str, str]]:
    """Read an edited cast.tsv back in, keyed on start time like `apply_cast` expects.

    Speaker labels are validated because they end up as filenames under refs/
    (`out_dir / f"{spk}.wav"` in refs.py) and as dict keys throughout fit.py's
    register history - a slash or a blank in an edited spreadsheet cell would
    otherwise write outside the refs directory.
    """
    out: dict[str, tuple[str, str]] = {}
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            try:
                key = f"{float(row['start']):.2f}"
            except (KeyError, TypeError, ValueError):
                continue
            spk = (row.get("speaker") or "").strip()
            voice = (row.get("voice") or "").strip()
            if spk and not _LABEL.match(spk):
                raise ValueError(
                    f"{path}: row start={row.get('start')!r} has an invalid "
                    f"speaker label {spk!r} - labels become filenames, so they "
                    f"must match {_LABEL.pattern}"
                )
            out[key] = (spk, voice)
    return out


def apply_cast(
    cues: list[Cue], table: dict[str, tuple[str, str]]
) -> tuple[int, int]:
    """Apply edited speaker/voice columns onto matching cues.

    A blank cell leaves that column alone, so an editor only has to touch the
    lines they actually want to change. Returns (matched, unmatched).
    """
    matched = unmatched = 0
    for c in cues:
        row = table.get(f"{c.start:.2f}")
        if row is None:
            unmatched += 1
            continue
        matched += 1
        spk, voice = row
        if spk:
            c.speaker = spk
        if voice:
            c.voice = voice
    return matched, unmatched


def describe_cast(cues: list[Cue], f0: dict[str, float]) -> str:
    """Same shape as `describe`, for cues carrying string speaker labels.

    Not a generalization of `describe`: that one is reached from the diarize
    path with an int-keyed F0 dict, and retyping that contract would ripple
    through diarize.py, cli.py and `cast` for no gain here.
    """
    counts = Counter(c.speaker for c in cues if c.speaker)
    voices: dict[str, Counter] = {}
    for c in cues:
        if c.speaker:
            voices.setdefault(c.speaker, Counter())[c.voice or "?"] += 1

    lines = ["  speaker  lines   F0    voice"]
    for spk, n in counts.most_common():
        hz = f0.get(spk)
        vc = voices.get(spk, Counter())
        voice = vc.most_common(1)[0][0] if vc else "?"
        if len(vc) > 1:
            voice += f" (+{len(vc) - 1} more)"
        lines.append(
            f"  {spk:>7}  {n:>5}  {hz:>4.0f}  {voice}"
            if hz
            else f"  {spk:>7}  {n:>5}     -  {voice}"
        )
    return "\n".join(lines)
