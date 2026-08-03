"""Speech engines behind one interface.

Which engine to use depends mostly on the language and how much time there is:

  kokoro      English, 28 preset voices, very fast.
  chatterbox  English, clones a voice from a reference clip, expressive.
  piper       Many languages including Ukrainian. Fast and plainly synthetic.
  styletts2   Ukrainian, clones from a reference clip. Predicts durations in
              one pass, so it has real speed control and cannot stall or run on.
  fish        Ukrainian, clones from a reference clip. Higher sample rate, but
              autoregressive: slower, and prone to both faults above.

The two large engines pin torch versions that conflict with this package, so
they run in their own virtualenvs and are driven over a pipe by _WorkerBackend.

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


# Piper voices, by language. Gender is needed for casting and is not derivable
# from the voice name, so it is recorded here.
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

    Used here for Ukrainian, which Kokoro and Chatterbox do not support. Quality
    sits below either of them, but `length_scale` gives real rate control, so the
    fitter keeps its re-synthesis lever, and pairing it with voice conversion
    recovers most of the character it lacks on its own.
    """

    sample_rate = 22_050  # replaced by the model's own rate once loaded
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
        del emotion  # Piper has no emotion control
        cached = self._cache_path(text, voice, speed)
        if cached and cached.exists():
            audio, _ = sf.read(cached, dtype="float32")
            return audio

        from piper import SynthesisConfig

        v = self._voice(voice)
        # length_scale stretches duration, so it is the reciprocal of speed.
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


# Ukrainian speech, measured from rendered output, and the factor beyond it that
# marks a clip as runaway rather than merely slow.
CHARS_PER_SEC = 12.6
RUNAWAY = 2.5


class WorkerError(RuntimeError):
    """A synthesis worker failed on one line and may need restarting."""


# Longest pause kept inside a line. Speech has real pauses - after a comma, or
# for effect - but an in-context model asked for one short line will happily
# emit seconds of nothing, which reads as the character having stopped talking.
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
        # Leading and trailing silence is handled by the caller's trim; only
        # pauses with speech on both sides are shortened here.
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


class FishSpeechTTS:
    """Fish Speech 1.5 over a pipe, for languages the mainstream models omit.

    Used with a Ukrainian fine-tune. Unlike the VITS-class alternatives it models
    prosody properly and clones voices natively through in-context reference
    audio, so no separate conversion stage is needed.

    It exposes no rate control, so the fitter falls back to time-stretching for
    over-long lines. Getting the translation to fit its window matters more here
    than with a backend that can re-articulate faster.
    """

    sample_rate = 44_100  # corrected from the first reply
    max_speed = 1.0
    supports_speed = False
    clones = True

    def __init__(
        self,
        repo: Path,
        checkpoint: Path,
        cache_dir: Path | None = None,
        device: str | None = None,
        temperature: float = 0.7,
        half: bool = False,
        compile: bool = False,
    ) -> None:
        self.repo = Path(repo).expanduser()
        self.checkpoint = Path(checkpoint).expanduser()
        self.cache_dir = cache_dir
        self.device = device
        self.temperature = temperature
        self.half = half
        self.compile = compile
        self._proc = None
        self._log = None
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def voices(self) -> list[str]:
        return []  # cloned per speaker from reference clips

    def _start(self):
        if self._proc is not None:
            return self._proc
        import json
        import subprocess

        python = self.repo / ".venv" / "bin" / "python"
        if not python.exists():
            raise RuntimeError(f"no Fish Speech venv at {python}")

        decoder = self.checkpoint / "firefly-gan-vq-fsq-8x1024-21hz-generator.pth"
        cfg = {
            "llama": str(self.checkpoint),
            "decoder": str(decoder),
            "device": self.device,
            "half": self.half,
            "compile": self.compile,
        }
        worker = Path(__file__).with_name("fish_worker.py")
        self._log = open(
            (self.cache_dir or self.repo) / "fish_worker.log", "w"
        )
        self._proc = subprocess.Popen(
            [str(python), str(worker), json.dumps(cfg)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._log,
            text=True, cwd=str(self.repo), bufsize=1,
        )
        hello = self._read_reply()
        if not hello.get("ready"):
            raise RuntimeError(
                f"Fish Speech worker failed to start: {hello.get('err')}\n"
                f"{hello.get('trace', '')}\nsee {self._log.name}"
            )
        return self._proc

    def _read_reply(self) -> dict:
        import json

        assert self._proc is not None
        while True:
            line = self._proc.stdout.readline()
            if not line:
                return {"ok": False, "err": "worker exited; see the log"}
            line = line.strip()
            if not line:
                continue
            try:
                return json.loads(line)
            except ValueError:
                continue  # library output that escaped the redirect

    def _cache_path(self, text: str, voice: str, attempt: int = 0) -> Path | None:
        if not self.cache_dir:
            return None
        who = Path(voice).name if voice else "default"
        # The attempt number is part of the key so a redraw is a genuinely new
        # sample rather than the cached clip that was already rejected.
        key = hashlib.sha1(
            f"fish|{who}|{self.temperature:.2f}|{attempt}|{text}".encode()
        ).hexdigest()[:20]
        return self.cache_dir / f"{key}.wav"

    def synth(
        self, text: str, voice: str, speed: float = 1.0, emotion: float = 0.5,
        attempt: int = 0,
    ) -> np.ndarray:
        del speed, emotion  # no rate or emotion parameters
        cached = self._cache_path(text, voice, attempt)
        if cached and cached.exists():
            audio, sr = sf.read(cached, dtype="float32")
            self.sample_rate = sr
            # Applied on read as well as on generation, so clips cached before
            # this existed are cleaned up too. Compressing twice is a no-op.
            return compress_pauses(audio, sr)

        audio = self._generate(
            text, voice, self.temperature + 0.12 * attempt, cached
        )
        # An autoregressive decoder sometimes fails to stop. Capping generation
        # upstream destabilises it, so judge the output instead and redraw at a
        # lower temperature.
        if audio.size / self.sample_rate > RUNAWAY * len(text) / CHARS_PER_SEC:
            second = self._generate(text, voice, 0.3, cached)
            # Keep whichever is shorter: the retry can ramble too, and taking it
            # blindly leaves the runaway in place.
            if second.size and second.size < audio.size:
                audio = second
            elif cached:
                sf.write(cached, audio, self.sample_rate)
        return audio

    def _generate(
        self, text: str, voice: str, temperature: float, cached: Path | None,
        restarts: int = 1,
    ) -> np.ndarray:
        try:
            return self._request(text, voice, temperature, cached)
        except WorkerError:
            # A feature-length render is hours of work, so one bad line must not
            # end it. Restart and retry; the caller drops the line if it fails
            # again.
            if restarts <= 0:
                raise
            self.close()
            return self._generate(text, voice, temperature, cached, restarts - 1)

    def _request(
        self, text: str, voice: str, temperature: float, cached: Path | None
    ) -> np.ndarray:
        import json
        import tempfile

        proc = self._start()
        out = cached or Path(tempfile.mktemp(suffix=".wav"))
        # The worker runs in the Fish Speech checkout, so every path crossing
        # the pipe has to be absolute - a relative one resolves against its cwd.
        job = {
            "text": text,
            "out": str(out.resolve()),
            "temperature": temperature,
        }
        if voice:
            ref = Path(voice).resolve()
            job["ref_audio"] = str(ref)
            # Telling the model what the reference clip says lets it align text
            # to voice instead of guessing, which noticeably steadies the clone.
            lab = ref.with_suffix(".lab")
            if lab.exists():
                job["ref_text"] = lab.read_text(encoding="utf-8").strip()
        proc.stdin.write(json.dumps(job) + "\n")
        proc.stdin.flush()

        reply = self._read_reply()
        if not reply.get("ok"):
            raise WorkerError(
                f"{reply.get('err')}\n{reply.get('trace', '')}".strip()
            )

        audio, sr = sf.read(out, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        self.sample_rate = sr
        return compress_pauses(_trim_silence(audio, sr), sr)

    def close(self) -> None:
        if self._proc is not None:
            try:
                import json

                self._proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
                self._proc.stdin.flush()
                self._proc.wait(timeout=15)
            except Exception:
                self._proc.kill()
            self._proc = None


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
            raise RuntimeError(f"no {self.tag} venv at {python}")

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

    def _read_reply(self) -> dict:
        import json

        assert self._proc is not None
        while True:
            line = self._proc.stdout.readline()
            if not line:
                return {"ok": False, "err": "worker exited; see the log"}
            line = line.strip()
            if not line:
                continue
            try:
                return json.loads(line)
            except ValueError:
                continue  # library output that escaped the redirect

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
        who = Path(voice).name if voice else "default"
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


class StyleTTS2(_WorkerBackend):
    """StyleTTS2 fine-tuned for Ukrainian, driven over a pipe.

    Durations are predicted in one pass rather than sampled token by token, so
    the two failure modes that dog autoregressive synthesis here - stalling into
    silence, and never stopping - do not occur. `speed` re-articulates the line
    instead of resampling it, which gives the fitter back its best lever.

    A voice is cloned from a reference clip through a style vector that carries
    both timbre and delivery.
    """

    sample_rate = 24_000
    max_speed = 1.30  # the model's own limit; past this it distorts
    supports_speed = True
    clones = True

    worker = "styletts_worker.py"
    tag = "styletts2"

    def __init__(
        self,
        repo: Path,
        hf_path: str = "patriotyk/styletts2_ukrainian_multispeaker",
        cache_dir: Path | None = None,
        device: str | None = None,
    ) -> None:
        super().__init__(repo, cache_dir, device)
        self.hf_path = hf_path

    def _config(self) -> dict:
        return {
            "repo": self.hf_path,
            "device": self.device,
            "sample_rate": self.sample_rate,
        }

    def voices(self) -> list[str]:
        return []  # cloned per speaker from reference clips

    def synth(
        self, text: str, voice: str, speed: float = 1.0, emotion: float = 0.5,
        attempt: int = 0,
    ) -> np.ndarray:
        del emotion  # delivery comes from the reference clip's style vector
        speed = float(min(max(speed, 0.7), self.max_speed))
        cached = self._cache_path(f"{speed:.3f}|{text}", voice, attempt)
        if cached and cached.exists():
            audio, sr = sf.read(cached, dtype="float32")
            self.sample_rate = sr
            return audio

        import tempfile

        out = cached or Path(tempfile.mktemp(suffix=".wav"))
        job = {
            "text": text,
            "speed": speed,
            "out": str(out.resolve()),
        }
        if voice:
            job["ref_audio"] = str(Path(voice).resolve())

        reply = self._job(job)
        if not reply.get("ok"):
            raise WorkerError(
                f"{reply.get('err')}\n{reply.get('trace', '')}".strip()
            )

        audio, sr = sf.read(out, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        self.sample_rate = sr
        return _trim_silence(audio, sr)


def get_backend(
    name: str, cache_dir: Path | None = None, device: str | None = None,
    lang: str = "en",
) -> TTSBackend:
    if name == "styletts2":
        return StyleTTS2(
            repo=Path("~/Developer/styletts2-ukrainian").expanduser(),
            cache_dir=cache_dir, device=device,
        )
    if name == "kokoro":
        return KokoroTTS(cache_dir=cache_dir)
    if name == "chatterbox":
        return ChatterboxTTS(cache_dir=cache_dir, device=device)
    if name == "piper":
        return PiperTTS(lang=lang, cache_dir=cache_dir)
    if name == "fish":
        return FishSpeechTTS(
            repo=Path("~/Developer/fish-speech").expanduser(),
            checkpoint=Path("~/Developer/fish-speech/checkpoints/uk").expanduser(),
            cache_dir=cache_dir, device=device,
        )
    raise ValueError(f"unknown TTS backend: {name!r}")
