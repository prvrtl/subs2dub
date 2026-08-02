"""Copy the original actors' delivery onto the synthesized lines.

Measures how each line was actually delivered and reproduces the parts a TTS
model can be steered on:

  * Intensity      -> per-clip gain, and emotion strength where supported.
  * Terminal pitch -> punctuation. Intonation is driven largely by punctuation,
    so a rising contour becomes a question mark, which becomes rising intonation.
  * Emphasis       -> exclamation marks on high-energy short lines.

Vocal timbre (tension, breathiness) is not transferable this way.

Measurement must run on the isolated vocal stem; on the full mix it measures the
score rather than the performance.
"""

from __future__ import annotations

import re

import numpy as np

from .model import Cue

# Openers that make a line interrogative even when the subtitler dropped the '?'.
_INTERROGATIVE = re.compile(
    r"^\s*(who|what|when|where|why|how|which|whose|whom|"
    r"is|are|was|were|do|does|did|can|could|will|would|should|shall|"
    r"have|has|had|am|may|might|isn't|aren't|don't|didn't|can't|won't)\b",
    re.I,
)
_TAG_QUESTION = re.compile(r",\s*(right|okay|ok|yeah|no|isn't it|aren't you)\s*$", re.I)

_RISE_SEMITONES = 2.0  # terminal rise that reads as a question
_TAIL = 0.45  # seconds of the line used for the terminal contour
_SHOUT_DB = 5.0  # dB above a speaker's median that counts as raised voice


def _f0_track(seg: np.ndarray, sr: int) -> np.ndarray:
    import librosa

    try:
        f0, voiced, _ = librosa.pyin(
            seg, fmin=70, fmax=400, sr=sr, frame_length=1024
        )
    except Exception:
        return np.zeros(0)
    if voiced is None:
        return np.zeros(0)
    f0 = np.where(voiced, f0, np.nan)
    return f0[np.isfinite(f0)]


def _terminal_slope(seg: np.ndarray, sr: int) -> float:
    """Semitones per second across the tail of the utterance."""
    tail = seg[-int(_TAIL * sr):] if seg.size > _TAIL * sr else seg
    f0 = _f0_track(tail, sr)
    if f0.size < 6:
        return 0.0
    semis = 12.0 * np.log2(np.maximum(f0, 1e-6) / max(float(np.median(f0)), 1e-6))
    t = np.linspace(0.0, len(semis) / (sr / 256), len(semis))
    if t[-1] <= 0:
        return 0.0
    slope, _ = np.polyfit(t, semis, 1)
    return float(slope)


def analyze(cues: list[Cue], vocals_wav, progress=None) -> None:
    """Measure intensity and terminal pitch for every cue, in place."""
    import soundfile as sf

    audio, sr = sf.read(str(vocals_wav), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Pass 1: loudness of each line.
    rms = np.full(len(cues), np.nan, dtype=np.float64)
    for i, c in enumerate(cues):
        seg = audio[int(c.start * sr):int(c.end * sr)]
        if seg.size < 0.15 * sr:
            continue
        r = float(np.sqrt(np.mean(seg.astype(np.float64) ** 2)))
        if r > 1e-5:
            rms[i] = 20.0 * np.log10(r)

    # Relative to each speaker's own median, so a naturally quiet actor is not
    # permanently turned down - what matters is deviation from their baseline.
    for spk in {c.speaker for c in cues}:
        idx = [i for i, c in enumerate(cues) if c.speaker == spk and np.isfinite(rms[i])]
        if len(idx) < 3:
            continue
        med = float(np.median(rms[idx]))
        for i in idx:
            cues[i].intensity = float(rms[i] - med)

    # Pass 2: terminal pitch contour, only where it can change the punctuation.
    for i, c in enumerate(cues):
        text = c.text.rstrip()
        if text.endswith(("?", "!")):
            continue
        seg = audio[int(c.start * sr):int(c.end * sr)]
        if seg.size >= 0.3 * sr and float(np.abs(seg).max() or 0) > 5e-3:
            c.f0_end_slope = _terminal_slope(seg, sr)
        if progress and (i + 1) % 100 == 0:
            progress(i + 1, len(cues))


def apply_punctuation(cues: list[Cue], use_pitch: bool = True) -> int:
    """Make the text carry the intonation, since punctuation is Kokoro's lever."""
    n = 0
    for c in cues:
        text = c.text.rstrip()
        if not text or text.endswith(("?", "!")):
            continue

        interrogative = bool(_INTERROGATIVE.match(text)) or bool(_TAG_QUESTION.search(text))
        rising = use_pitch and c.f0_end_slope >= _RISE_SEMITONES

        # Require agreement between wording and delivery before overriding a
        # full stop; intonation does not map identically across languages, so
        # pitch alone would turn statements into questions.
        if interrogative and (rising or not text.endswith((".", ",", "...", "…"))):
            c.text = text.rstrip(".,") + "?"
            c.punct_fixed = True
            n += 1
        elif c.intensity >= _SHOUT_DB and len(text) < 60:
            c.text = text.rstrip(".,") + "!"
            c.punct_fixed = True
            n += 1
    return n


def apply_gain(cues: list[Cue], max_db: float = 5.0, scale: float = 0.6) -> None:
    """Map the actor's deviation from their own norm onto the dub clip."""
    for c in cues:
        c.gain_db = float(np.clip(c.intensity * scale, -max_db, max_db))


def apply_emotion(
    cues: list[Cue], base: float = 0.5, scale: float = 0.045,
    lo: float = 0.25, hi: float = 1.10,
) -> None:
    """Turn measured intensity into an emotion setting for capable backends.

    Loudness relative to the actor's own baseline is a crude but real proxy for
    arousal: a line delivered well above their norm is usually shouted, urgent
    or frightened. Exclamations get a further nudge, questions a slight one.
    Backends without an emotion control ignore this entirely.
    """
    for c in cues:
        v = base + c.intensity * scale
        tail = c.text.rstrip()[-1:]
        if tail == "!":
            v += 0.12
        elif tail == "?":
            v += 0.04
        c.emotion = float(np.clip(v, lo, hi))


def summary(cues: list[Cue]) -> str:
    fixed = sum(1 for c in cues if c.punct_fixed)
    q = sum(1 for c in cues if c.text.rstrip().endswith("?"))
    ex = sum(1 for c in cues if c.text.rstrip().endswith("!"))
    loud = sum(1 for c in cues if c.gain_db >= 2.0)
    soft = sum(1 for c in cues if c.gain_db <= -2.0)
    return (
        f"prosody: {q} questions, {ex} exclamations ({fixed} punctuation marks "
        f"added from delivery), {loud} lines raised, {soft} softened"
    )
