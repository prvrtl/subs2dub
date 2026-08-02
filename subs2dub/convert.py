"""Transfer the original actors' voices without their accents.

Cloning a foreign-language reference clip transfers the speaker's accent along
with their timbre, since the reference conditions the whole acoustic style.

Voice conversion separates the two: synthesize in native English first, then
convert only the tone colour toward the target speaker. Pronunciation, intonation
and emotion come from the base TTS; identity comes from the reference. Pick an
expressive base model - the converter cannot add expression that was not there.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

# cue.voice carries both halves, separated by this token: preset::reference
VOICE_SEP = "::"


class ToneConverter:
    """Thin wrapper over OpenVoice's ToneColorConverter, with SE caching."""

    def __init__(
        self, ckpt_dir: Path, device: str | None = None, vendor: Path | None = None
    ) -> None:
        self.ckpt_dir = Path(ckpt_dir)
        self.vendor = vendor
        self._model = None
        self._se: dict[str, object] = {}
        self.device = device

    def _load(self):
        if self._model is not None:
            return self._model
        import sys

        if self.vendor and str(self.vendor) not in sys.path:
            sys.path.insert(0, str(self.vendor))

        import torch
        from openvoice.api import ToneColorConverter

        # CPU by default, deliberately. OpenVoice's flow decoder calls a
        # TorchScript-fused op and the JIT graph fuser has no MPS backend, so
        # MPS dies with "Unknown device for graph fuser" partway through the
        # first conversion. CPU measures ~0.4x realtime here, which is fast
        # enough that chasing GPU support is not worth it.
        dev = self.device or "cpu"
        del torch
        try:
            m = ToneColorConverter(str(self.ckpt_dir / "config.json"), device=dev)
            m.load_ckpt(str(self.ckpt_dir / "checkpoint.pth"))
            self.device = dev
        except Exception:
            # Some ops still fall over on MPS; CPU is slower but dependable.
            m = ToneColorConverter(str(self.ckpt_dir / "config.json"), device="cpu")
            m.load_ckpt(str(self.ckpt_dir / "checkpoint.pth"))
            self.device = "cpu"
        self._model = m
        return m

    @property
    def sample_rate(self) -> int:
        return int(self._load().hps.data.sampling_rate)

    def se(self, wav: Path | str):
        """Speaker embedding for a wav file, cached by path."""
        key = str(wav)
        if key not in self._se:
            self._se[key] = self._load().extract_se([key])
        return self._se[key]

    def has_se(self, key: str) -> bool:
        return key in self._se

    def se_for_array(self, audio: np.ndarray, sr: int, key: str):
        """Speaker embedding for in-memory audio (used for the base voice)."""
        if key not in self._se:
            tmp = Path(tempfile.mktemp(suffix=".wav"))
            sf.write(tmp, audio, sr)
            try:
                self._se[key] = self._load().extract_se([str(tmp)])
            finally:
                tmp.unlink(missing_ok=True)
        return self._se[key]

    def convert(
        self, audio: np.ndarray, sr: int, src_se, tgt_se, tau: float = 0.3
    ) -> tuple[np.ndarray, int]:
        model = self._load()
        src = Path(tempfile.mktemp(suffix=".wav"))
        dst = Path(tempfile.mktemp(suffix=".wav"))
        try:
            sf.write(src, audio, sr)
            model.convert(
                audio_src_path=str(src), src_se=src_se, tgt_se=tgt_se,
                output_path=str(dst), tau=tau,
                # Must be non-empty: string_to_bits("") cannot broadcast into
                # the (n, 8) bit array. This is OpenVoice's provenance
                # watermark, so it is left enabled rather than stubbed out.
                message="@subs2dub",
            )
            out, out_sr = sf.read(dst, dtype="float32")
            if out.ndim > 1:
                out = out.mean(axis=1)
            return out, out_sr
        finally:
            src.unlink(missing_ok=True)
            dst.unlink(missing_ok=True)


class VoiceConvertBackend:
    """Wraps a TTS backend and converts its output toward a target speaker.

    `voice` is "<base preset>::<reference wav>". The base preset decides the
    pronunciation; the reference decides who it sounds like.
    """

    def __init__(self, inner, converter: ToneConverter, tau: float = 0.3) -> None:
        self.inner = inner
        self.converter = converter
        self.tau = tau
        self.supports_speed = getattr(inner, "supports_speed", True)
        self.max_speed = getattr(inner, "max_speed", 1.0)
        self.clones = True

    @property
    def sample_rate(self) -> int:
        """The *converter's* rate, not the base TTS rate.

        OpenVoice resamples to 22.05 kHz while Kokoro emits 24 kHz. The mixer
        allocates the dialogue bus from this before the first synth call, so
        reporting the inner rate would place every clip at the wrong offset and
        play the whole dub at the wrong speed.
        """
        return self.converter.sample_rate

    def voices(self) -> list[str]:
        return self.inner.voices()

    @staticmethod
    def pack(preset: str, ref: Path | str) -> str:
        return f"{preset}{VOICE_SEP}{ref}"

    def synth(
        self, text: str, voice: str, speed: float = 1.0, emotion: float = 0.5
    ) -> np.ndarray:
        preset, _, ref = voice.partition(VOICE_SEP)
        audio = self.inner.synth(text, preset, speed=speed, emotion=emotion)
        if not ref or audio.size == 0:
            return audio

        sr = self.inner.sample_rate
        # Characterise each base preset once. Synthesizing the calibration line
        # unconditionally would double the render cost of the entire film.
        key = f"base:{preset}"
        if self.converter.has_se(key):
            src_se = self.converter.se_for_array(None, sr, key)
        else:
            src_se = self.converter.se_for_array(
                self.inner.synth(_CALIBRATION, preset, speed=1.0), sr, key
            )
        tgt_se = self.converter.se(ref)
        out, _ = self.converter.convert(audio, sr, src_se, tgt_se, self.tau)
        return out


# A neutral line used once per preset to characterise the base voice. Content is
# irrelevant beyond covering a decent spread of phonemes.
_CALIBRATION = (
    "The quick brown fox jumps over the lazy dog while the sun sets behind them."
)
