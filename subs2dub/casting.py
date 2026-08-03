"""Assign a distinct TTS voice to each detected speaker.

Two rules do most of the work:

  * Match gender to the original actor, inferred from median F0 of their lines.
    A male character voiced by a female preset is the single most jarring error.
  * Give the busiest characters the best voices. Kokoro's presets vary in
    quality, and leads carry most of the runtime.

Accent is used as a last resort to keep same-gender characters distinguishable:
American presets are handed out first, British ones once those run out.
"""

from __future__ import annotations

from collections import Counter

from .model import Cue

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
