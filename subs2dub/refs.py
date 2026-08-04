"""Cut a voice-cloning reference clip for each character.

Cloning needs ~10-15 seconds of clean speech per speaker. The vocal stem plus
diarization labels provide it: for each character, splice together their longest
and loudest lines.

Reference quality matters more than length - a short clean sample beats a longer
one containing music bleed or a second voice - so cues are ranked and marginal
ones dropped.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from .model import Cue

TARGET_SECONDS = 12.0
MIN_SECONDS = 4.0
_MIN_CUE = 1.0
_GAP = 0.25


def _join(pieces: list[np.ndarray], sr: int, fade: float = 0.02) -> np.ndarray:
    """Concatenate speech segments with a short crossfade at each seam."""
    n = int(fade * sr)
    out = pieces[0]
    for seg in pieces[1:]:
        overlap = min(n, out.size, seg.size)
        if overlap <= 0:
            out = np.concatenate([out, seg])
            continue
        ramp = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
        seam = out[-overlap:] * (1.0 - ramp) + seg[:overlap] * ramp
        out = np.concatenate([out[:-overlap], seam, seg[overlap:]])
    return out


def build_references(
    cues: list[Cue],
    vocals_wav: Path,
    out_dir: Path,
    *,
    target: float = TARGET_SECONDS,
) -> dict[str, Path]:
    """Write one reference clip per speaker. Returns speaker -> wav path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    audio, sr = sf.read(str(vocals_wav), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    by_speaker: dict[str, list[Cue]] = {}
    for c in cues:
        if c.speaker and (c.end - c.start) >= _MIN_CUE:
            by_speaker.setdefault(c.speaker, []).append(c)

    refs: dict[str, Path] = {}
    silence = np.zeros(int(_GAP * sr), dtype=np.float32)

    for spk, group in by_speaker.items():
        scored = []
        for c in group:
            seg = audio[int(c.start * sr):int(c.end * sr)]
            if seg.size < _MIN_CUE * sr:
                continue
            rms = float(np.sqrt(np.mean(seg.astype(np.float64) ** 2)))
            if rms < 1e-4:
                continue
            scored.append((rms * min(seg.size / sr, 6.0), seg, c))

        if not scored:
            continue
        scored.sort(key=lambda x: -x[0])

        pieces, spoken, total = [], [], 0.0
        for _, seg, c in scored:
            pieces.append(seg)
            spoken.append(c.source_text or c.text)
            total += seg.size / sr
            if total >= target:
                break
        if total < MIN_SECONDS:
            continue

        joined = _join(pieces, sr)
        peak = float(np.max(np.abs(joined)) or 0)
        if peak > 0:
            joined = joined * (0.95 / peak)

        path = out_dir / f"{spk}.wav"
        sf.write(path, joined, sr, subtype="PCM_16")
        path.with_suffix(".lab").write_text(
            " ".join(t for t in spoken if t).strip(), encoding="utf-8"
        )
        refs[spk] = path

    return refs


def describe(refs: dict[str, Path]) -> str:
    if not refs:
        return "  no usable reference clips"
    rows = []
    for spk, p in sorted(refs.items()):
        info = sf.info(str(p))
        rows.append(f"  {spk}  {info.duration:5.1f}s  {p.name}")
    return "\n".join(rows)


def line_clips(cues: list[Cue], vocals_wav: Path, out_dir: Path) -> int:
    """Write the original delivery of each line, for per-line prosody transfer.

    A single style vector per character, averaged over a montage of their lines,
    delivers every line identically - which is what a listener hears as robotic.
    The delivery of each line is available in the source, so it is cut here and
    used for that line alone. Identity still comes from the character clip; only
    the prosody comes from here.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    audio, sr = sf.read(str(vocals_wav), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    written = 0
    pending: list = []
    for cue in cues:
        start = max(0.0, cue.start + min(cue.speech_onset, 0.0))
        end = min(len(audio) / sr, max(start + 0.4, cue.end))
        seg = audio[int(start * sr):int(end * sr)]
        if seg.size < int(0.4 * sr):
            continue
        peak = float(np.max(np.abs(seg)) or 0)
        if peak < 5e-3:
            continue
        path = out_dir / f"line{cue.idx:05d}.wav"
        sf.write(path, seg * (0.95 / peak), sr, subtype="PCM_16")
        cue.line_ref = str(path)
        pending.append((cue, seg))
        written += 1

    if pending:
        from . import pitch

        for (cue, _), (median, spread) in zip(
            pending, pitch.shapes([s for _, s in pending], sr)
        ):
            cue.source_f0, cue.source_f0_range = median, spread
    return written


def _pitch_shape(seg: np.ndarray, sr: int) -> tuple[float, float]:
    """Median pitch and expressive range of a line, in Hz and semitones.

    What a synthesized take is judged against: it should sit in the speaker's
    register and move as much as they did.
    """
    try:
        import librosa

        f0, _, _ = librosa.pyin(seg, fmin=75, fmax=400, sr=sr, frame_length=2048)
        v = f0[np.isfinite(f0)]
        if v.size < 8:
            return 0.0, 0.0
        med = float(np.median(v))
        semis = 12.0 * np.log2(v / med)
        return med, float(np.percentile(semis, 90) - np.percentile(semis, 10))
    except Exception:
        return 0.0, 0.0


def voiced_fraction(path: Path, seconds: float = 30.0) -> float:
    """How much of a clip is confidently someone speaking.

    A reference clip is only useful if it contains a voice. On music-led
    material - trailers, montages, anything scored under the dialogue -
    separation returns mostly score, and a cloning model handed that reproduces
    something derived from the music rather than from a person. Ordinary speech
    sits around 0.3 to 0.5 here; music sits near zero.
    """
    try:
        import librosa

        audio, sr = sf.read(str(path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio[: int(seconds * sr)]
        if audio.size < sr:
            return 0.0
        f0, _, prob = librosa.pyin(
            audio, fmin=80, fmax=350, sr=sr, frame_length=2048
        )
        good = np.isfinite(f0) & (prob > 0.5)
        return float(good.mean())
    except Exception:
        return 0.0


def usable(clips: dict, floor: float = 0.004) -> tuple[dict, dict]:
    """Split reference clips into those carrying a voice and those that do not."""
    good, poor = {}, {}
    for speaker, path in clips.items():
        (good if voiced_fraction(path) >= floor else poor)[speaker] = path
    return good, poor
