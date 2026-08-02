"""TTS backends behind one interface.

Kokoro is the working backend. The Protocol exists so a slower/higher-quality
engine (or a Ukrainian one) can be dropped in without touching the fitter,
which is where all the real logic lives.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol

import numpy as np
import soundfile as sf


class TTSBackend(Protocol):
    sample_rate: int
    max_speed: float
    # Engines that cannot re-articulate at a requested rate force the fitter to
    # reach for time-stretching earlier. Kokoro can; Chatterbox cannot.
    supports_speed: bool
    # Whether `voice` names a preset or points at a reference clip to clone.
    clones: bool

    def voices(self) -> list[str]: ...

    def synth(
        self,
        text: str,
        voice: str,
        speed: float = 1.0,
        emotion: float = 0.5,
    ) -> np.ndarray: ...


# Kokoro ships ~28 English voices. Prefix encodes accent+gender:
#   a=American, b=British / f=female, m=male
KOKORO_VOICES = [
    "af_heart", "af_bella", "af_nicole", "af_aoede", "af_kore", "af_sarah",
    "af_nova", "af_sky", "af_alloy", "af_jessica", "af_river",
    "am_michael", "am_fenrir", "am_puck", "am_echo", "am_eric",
    "am_liam", "am_onyx", "am_santa", "am_adam",
    "bf_emma", "bf_isabella", "bf_alice", "bf_lily",
    "bm_george", "bm_fable", "bm_lewis", "bm_daniel",
]


def voice_gender(voice: str) -> str:
    return "F" if voice[1] == "f" else "M"


class KokoroTTS:
    """Local Kokoro-82M. Fast enough to re-render the whole film while tuning."""

    sample_rate = 24_000
    max_speed = 1.25  # beyond this Kokoro starts to sound clipped
    supports_speed = True
    clones = False

    def __init__(self, cache_dir: Path | None = None) -> None:
        self._pipelines: dict[str, object] = {}
        self.cache_dir = cache_dir
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def voices(self) -> list[str]:
        return list(KOKORO_VOICES)

    def _pipeline(self, voice: str):
        # Voice prefix selects the G2P language pack: 'a' American, 'b' British.
        lang = voice[0]
        if lang not in self._pipelines:
            from kokoro import KPipeline

            self._pipelines[lang] = KPipeline(
                lang_code=lang, repo_id="hexgrad/Kokoro-82M"
            )
        return self._pipelines[lang]

    def _cache_path(self, text: str, voice: str, speed: float) -> Path | None:
        if not self.cache_dir:
            return None
        key = hashlib.sha1(
            f"kokoro|{voice}|{speed:.4f}|{text}".encode()
        ).hexdigest()[:20]
        return self.cache_dir / f"{key}.wav"

    def synth(
        self, text: str, voice: str, speed: float = 1.0, emotion: float = 0.5
    ) -> np.ndarray:
        del emotion  # Kokoro exposes no emotion control; prosody.py compensates
        cached = self._cache_path(text, voice, speed)
        if cached and cached.exists():
            audio, _ = sf.read(cached, dtype="float32")
            return audio

        chunks = [
            np.asarray(a, dtype=np.float32)
            for _, _, a in self._pipeline(voice)(text, voice=voice, speed=speed)
        ]
        audio = (
            np.concatenate(chunks)
            if chunks
            else np.zeros(0, dtype=np.float32)
        )
        audio = _trim_silence(audio, self.sample_rate)

        if cached is not None:
            # 16-bit keeps the on-disk cache ~215 MB for a feature instead of 430.
            sf.write(cached, audio, self.sample_rate, subtype="PCM_16")
        return audio


def _trim_silence(
    audio: np.ndarray, sr: int, thresh: float = 2e-3, keep_ms: int = 30
) -> np.ndarray:
    """Strip leading/trailing near-silence so cue timing isn't thrown off by it."""
    if audio.size == 0:
        return audio
    loud = np.flatnonzero(np.abs(audio) > thresh)
    if loud.size == 0:
        return audio
    keep = int(sr * keep_ms / 1000)
    lo = max(0, loud[0] - keep)
    hi = min(audio.size, loud[-1] + keep)
    return audio[lo:hi]


class ChatterboxTTS:
    """Resemble AI's Chatterbox: emotion control plus zero-shot voice cloning.

    `voice` is a path to a reference clip rather than a preset name - here that
    is a sample of the original actor cut from the Demucs vocal stem, so each
    character speaks English in something close to their own voice.

    Two things it does not have: a speed parameter (so the fitter must reach
    straight for time-stretching), and Kokoro's throughput. Both are the price
    of the emotion knob.
    """

    sample_rate = 24_000
    max_speed = 1.0
    supports_speed = False
    clones = True

    def __init__(
        self, cache_dir: Path | None = None, device: str | None = None
    ) -> None:
        self.cache_dir = cache_dir
        self._device = device
        self._model = None
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def voices(self) -> list[str]:
        return []  # cloned per speaker; see refs.py

    def _load(self):
        if self._model is None:
            import torch
            from chatterbox.tts import ChatterboxTTS as _CB

            dev = self._device or (
                "mps" if torch.backends.mps.is_available() else "cpu"
            )
            try:
                self._model = _CB.from_pretrained(device=dev)
            except Exception:
                self._model = _CB.from_pretrained(device="cpu")
        return self._model

    def _cache_path(self, text: str, voice: str, emotion: float) -> Path | None:
        if not self.cache_dir:
            return None
        who = Path(voice).name if voice else "builtin"
        key = hashlib.sha1(
            f"cb|{who}|{emotion:.2f}|{text}".encode()
        ).hexdigest()[:20]
        return self.cache_dir / f"{key}.wav"

    def synth(
        self, text: str, voice: str, speed: float = 1.0, emotion: float = 0.5
    ) -> np.ndarray:
        del speed  # no rate control; the fitter compensates by stretching
        cached = self._cache_path(text, voice, emotion)
        if cached and cached.exists():
            audio, _ = sf.read(cached, dtype="float32")
            return audio

        model = self._load()
        # An empty voice means "use the built-in speaker". That matters when
        # Chatterbox is the *base* for voice conversion: cloning a foreign-language
        # reference here would bake the accent in before the converter ever
        # runs, which is exactly what we are trying to avoid.
        kwargs = {"audio_prompt_path": str(voice)} if voice else {}
        wav = model.generate(
            text, exaggeration=float(emotion), cfg_weight=0.5, **kwargs
        )
        audio = np.asarray(wav.squeeze(0).detach().cpu().numpy(), dtype=np.float32)
        audio = _trim_silence(audio, self.sample_rate)

        if cached is not None:
            sf.write(cached, audio, self.sample_rate, subtype="PCM_16")
        return audio


def get_backend(
    name: str, cache_dir: Path | None = None, device: str | None = None
) -> TTSBackend:
    if name == "kokoro":
        return KokoroTTS(cache_dir=cache_dir)
    if name == "chatterbox":
        return ChatterboxTTS(cache_dir=cache_dir, device=device)
    raise ValueError(f"unknown TTS backend: {name!r}")
