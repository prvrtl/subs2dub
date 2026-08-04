"""Speech engines behind one interface.

Which engine to use depends mostly on the language and how much time there is:

  kokoro      English, 28 preset voices, very fast.
  chatterbox  English, clones a voice from a reference clip, expressive.
  piper       Many languages. Fast, and plainly synthetic.
The fitter holds all the timing logic; an engine only has to synthesize a line
and say whether it can do so at a requested speed.
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
    supports_speed: bool
    clones: bool

    def voices(self) -> list[str]: ...

    def synth(
        self,
        text: str,
        voice: str,
        speed: float = 1.0,
        emotion: float = 0.5,
    ) -> np.ndarray: ...


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


LINE_SEP = "||"

_DIGESTS: dict[str, str] = {}


def voice_key(voice: str) -> str:
    """Name a voice for the cache by its contents, not by its filename.

    A reference clip keeps its name from run to run while its contents follow
    the cast: re-assigning lines rebuilds a character's clip from a different
    set of them. Keyed on the name alone, a line that was just re-cast is
    served the previous character's audio, which reads as the edit having
    been ignored.
    """
    if not voice:
        return voice
    parts = []
    for piece in voice.split(LINE_SEP):
        p = Path(piece) if piece else None
        if not piece or p is None or not p.is_file():
            parts.append(piece)
            continue
        st = p.stat()
        cache_key = f"{p}:{st.st_size}:{int(st.st_mtime)}"
        digest = _DIGESTS.get(cache_key)
        if digest is None:
            digest = hashlib.sha1(p.read_bytes()).hexdigest()[:12]
            _DIGESTS[cache_key] = digest
        parts.append(f"{p.name}:{digest}")
    return "|".join(parts)


class KokoroTTS:
    """Local Kokoro-82M. Fast enough to re-render the whole film while tuning."""

    sample_rate = 24_000
    max_speed = 1.25
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
        del emotion
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
        return []

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
        who = voice_key(voice) if voice else "builtin"
        key = hashlib.sha1(
            f"cb|{who}|{emotion:.2f}|{text}".encode()
        ).hexdigest()[:20]
        return self.cache_dir / f"{key}.wav"

    def synth(
        self, text: str, voice: str, speed: float = 1.0, emotion: float = 0.5
    ) -> np.ndarray:
        del speed
        cached = self._cache_path(text, voice, emotion)
        if cached and cached.exists():
            audio, _ = sf.read(cached, dtype="float32")
            return audio

        model = self._load()
        kwargs = {"audio_prompt_path": str(voice)} if voice else {}
        wav = model.generate(
            text, exaggeration=float(emotion), cfg_weight=0.5, **kwargs
        )
        audio = np.asarray(wav.squeeze(0).detach().cpu().numpy(), dtype=np.float32)
        audio = _trim_silence(audio, self.sample_rate)

        if cached is not None:
            sf.write(cached, audio, self.sample_rate, subtype="PCM_16")
        return audio


PIPER_VOICES: dict[str, dict[str, tuple[str, str]]] = {
    "uk": {
        "tetiana": ("uk_UA-tetiana-high", "F"),
        "lada": ("uk_UA-lada-x_low", "F"),
        "mykyta": ("uk_UA-mykyta-high", "M"),
        "oleksa": ("uk_UA-oleksa-high", "M"),
        "ukrainian_tts": ("uk_UA-ukrainian_tts-medium", "F"),
    },
}


class PiperTTS:
    """Piper: small ONNX voices covering languages the larger models omit.

    Covers languages Kokoro and Chatterbox do not support. Quality
    sits below either of them, but `length_scale` gives real rate control, so the
    fitter keeps its re-synthesis lever, and pairing it with voice conversion
    recovers most of the character it lacks on its own.
    """

    sample_rate = 22_050
    max_speed = 1.30
    supports_speed = True
    clones = False

    REPO = "rhasspy/piper-voices"

    def __init__(
        self, lang: str = "uk", cache_dir: Path | None = None,
        model_dir: Path | None = None,
    ) -> None:
        self.lang = lang
        self.cache_dir = cache_dir
        self.model_dir = Path(model_dir or "./models/piper").expanduser()
        self._voices: dict[str, object] = {}
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def voices(self) -> list[str]:
        return list(PIPER_VOICES.get(self.lang, {}))

    def voice_pools(self) -> dict[str, list[str]]:
        pools: dict[str, list[str]] = {"F": [], "M": []}
        for name, (_, gender) in PIPER_VOICES.get(self.lang, {}).items():
            pools[gender].append(name)
        return pools

    def _fetch(self, tag: str) -> Path:
        """Download an .onnx voice and its config from the Piper voice repo."""
        from huggingface_hub import hf_hub_download

        locale, speaker, quality = tag.split("-")
        base = f"{self.lang}/{locale}/{speaker}/{quality}/{tag}"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        onnx = hf_hub_download(
            self.REPO, f"{base}.onnx", local_dir=str(self.model_dir)
        )
        hf_hub_download(self.REPO, f"{base}.onnx.json", local_dir=str(self.model_dir))
        return Path(onnx)

    def _voice(self, name: str):
        if name not in self._voices:
            from piper import PiperVoice

            tag = PIPER_VOICES.get(self.lang, {}).get(name, (name, "F"))[0]
            self._voices[name] = PiperVoice.load(str(self._fetch(tag)))
            self.sample_rate = int(self._voices[name].config.sample_rate)
        return self._voices[name]

    def _cache_path(self, text: str, voice: str, speed: float) -> Path | None:
        if not self.cache_dir:
            return None
        key = hashlib.sha1(
            f"piper|{self.lang}|{voice}|{speed:.4f}|{text}".encode()
        ).hexdigest()[:20]
        return self.cache_dir / f"{key}.wav"

    def synth(
        self, text: str, voice: str, speed: float = 1.0, emotion: float = 0.5
    ) -> np.ndarray:
        del emotion
        cached = self._cache_path(text, voice, speed)
        if cached and cached.exists():
            audio, _ = sf.read(cached, dtype="float32")
            return audio

        from piper import SynthesisConfig

        v = self._voice(voice)
        cfg = SynthesisConfig(length_scale=1.0 / max(speed, 1e-3))

        parts = []
        for chunk in v.synthesize(text, syn_config=cfg):
            arr = getattr(chunk, "audio_float_array", None)
            if arr is None:
                raw = np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16)
                arr = raw.astype(np.float32) / 32768.0
            parts.append(np.asarray(arr, dtype=np.float32))

        audio = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
        audio = _trim_silence(audio, self.sample_rate)

        if cached is not None:
            sf.write(cached, audio, self.sample_rate, subtype="PCM_16")
        return audio


CHARS_PER_SEC = 12.6
RUNAWAY = 2.5


def _pump(stream, replies) -> None:
    """Move worker output into a queue so reads can time out."""
    try:
        for line in stream:
            replies.put(line)
    except (ValueError, OSError):
        pass
    replies.put(None)


class WorkerError(RuntimeError):
    """A synthesis worker failed on one line and may need restarting."""


MAX_PAUSE = 0.22


def compress_pauses(
    audio: np.ndarray, sr: int, max_pause: float = MAX_PAUSE
) -> np.ndarray:
    """Shorten silences inside a clip, leaving the speech itself untouched.

    Unlike time-stretching this changes no pitch and no speaking rate: it only
    removes dead air, so a line that was mostly pause becomes a line that is
    mostly speech at its original tempo.
    """
    if audio.size < int(0.2 * sr):
        return audio

    frame = max(1, int(0.02 * sr))
    usable = audio[: audio.size // frame * frame]
    if usable.size == 0:
        return audio
    energy = np.abs(usable.reshape(-1, frame)).mean(axis=1)
    peak = float(energy.max())
    if peak <= 0:
        return audio
    quiet = energy < max(0.004, peak * 0.03)

    keep_frames = max(1, int(max_pause * sr / frame))
    keep = np.ones(len(quiet), dtype=bool)
    i = 0
    while i < len(quiet):
        if not quiet[i]:
            i += 1
            continue
        j = i
        while j < len(quiet) and quiet[j]:
            j += 1
        run = j - i
        if i > 0 and j < len(quiet) and run > keep_frames:
            half = keep_frames // 2
            keep[i + half:j - (keep_frames - half)] = False
        i = j

    if keep.all():
        return audio
    mask = np.repeat(keep, frame)
    out = usable[mask]
    tail = audio[usable.size:]
    return np.concatenate([out, tail]) if tail.size else out


class _WorkerBackend:
    """Shared plumbing for models that run in their own virtualenv.

    Both large backends here pin torch versions that conflict with the main
    pipeline, so each runs as a subprocess speaking line-delimited JSON. The
    process stays resident because loading the weights costs far more than any
    single line.
    """

    worker = ""
    tag = ""

    def __init__(
        self,
        repo: Path,
        cache_dir: Path | None = None,
        device: str | None = None,
    ) -> None:
        self.repo = Path(repo).expanduser()
        self.cache_dir = cache_dir
        self.device = device
        self._proc = None
        self._log = None
        self._reader = None
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def _config(self) -> dict:
        return {"device": self.device}

    def _start(self):
        if self._proc is not None:
            return self._proc
        import json
        import subprocess

        python = self.repo / ".venv" / "bin" / "python"
        if not python.exists():
            raise SystemExit(
                f"the {self.tag} engine is not installed yet.\n"
                f"  it needs its own checkout, because it pins a torch version\n"
                f"  that conflicts with this one. Install it with:\n\n"
                f"      ./scripts/setup.sh --{self.tag}\n"
            )

        worker = Path(__file__).with_name(self.worker)
        self._log = open(
            (self.cache_dir or self.repo) / f"{self.tag}_worker.log", "w"
        )
        self._proc = subprocess.Popen(
            [str(python), str(worker), json.dumps(self._config())],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._log,
            text=True, cwd=str(self.repo), bufsize=1,
        )
        hello = self._read_reply()
        if not hello.get("ready"):
            raise RuntimeError(
                f"{self.tag} worker failed to start: {hello.get('err')}\n"
                f"{hello.get('trace', '')}\nsee {self._log.name}"
            )
        return self._proc

    def _read_reply(self, timeout: float = 900.0) -> dict:
        """Read one reply, giving up rather than waiting on a dead worker.

        A plain readline() here can block forever. If the worker dies after
        spawning something that inherited its stdout - a model download, say -
        the pipe never reaches EOF, and the run stops with no explanation and
        no way back.
        """
        import json
        import queue
        import threading
        import time

        assert self._proc is not None
        if getattr(self, "_reader", None) is None:
            self._replies: queue.Queue = queue.Queue()
            self._reader = threading.Thread(
                target=_pump, args=(self._proc.stdout, self._replies),
                daemon=True,
            )
            self._reader.start()

        deadline = time.time() + timeout
        while True:
            left = deadline - time.time()
            if left <= 0:
                return {"ok": False, "err": f"worker silent for {timeout:.0f}s"}
            try:
                line = self._replies.get(timeout=min(2.0, left))
            except queue.Empty:
                if self._proc.poll() is not None:
                    return {
                        "ok": False,
                        "err": f"worker exited ({self._proc.returncode}); see "
                               f"{getattr(self._log, 'name', 'the worker log')}",
                    }
                continue
            if line is None:
                return {"ok": False, "err": "worker closed its output"}
            line = line.strip()
            if not line:
                continue
            try:
                return json.loads(line)
            except ValueError:
                continue

    def _job(self, job: dict, restarts: int = 1) -> dict:
        import json

        try:
            proc = self._start()
            proc.stdin.write(json.dumps(job) + "\n")
            proc.stdin.flush()
            reply = self._read_reply()
        except (BrokenPipeError, OSError) as exc:
            reply = {"ok": False, "err": str(exc)}

        if not reply.get("ok") and restarts > 0:
            self.close()
            return self._job(job, restarts - 1)
        return reply

    def _cache_path(
        self, text: str, voice: str, attempt: int = 0
    ) -> Path | None:
        if not self.cache_dir:
            return None
        who = voice_key(voice) if voice else "default"
        key = hashlib.sha1(
            f"{self.tag}|{who}|{attempt}|{text}".encode()
        ).hexdigest()[:20]
        return self.cache_dir / f"{key}.wav"

    def close(self) -> None:
        if self._proc is None:
            return
        import json

        try:
            self._proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
            self._proc.stdin.flush()
            self._proc.wait(timeout=15)
        except Exception:
            self._proc.kill()
        self._proc = None
        self._reader = None


def get_backend(
    name: str, cache_dir: Path | None = None, device: str | None = None,
    lang: str = "en",
) -> TTSBackend:
    if name == "kokoro":
        return KokoroTTS(cache_dir=cache_dir)
    if name == "chatterbox":
        return ChatterboxTTS(cache_dir=cache_dir, device=device)
    if name == "piper":
        return PiperTTS(lang=lang, cache_dir=cache_dir)
    raise ValueError(f"unknown TTS backend: {name!r}")
