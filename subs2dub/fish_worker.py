"""Persistent Fish Speech worker. Runs under the fish-speech venv, not ours.

Fish Speech pins its own torch and a large training-oriented dependency set that
conflicts with the main pipeline, so it lives in a separate virtualenv and is
driven over a pipe.

Both stages (text to semantic tokens, then tokens to audio) stay resident, since
loading them costs several seconds and well over a gigabyte. One job per line of
stdin:

    {"text": "...", "ref_audio": "/path/ref.wav", "ref_text": "...",
     "out": "/path/out.wav", "temperature": 0.7}

One JSON reply per job.
"""

from __future__ import annotations

import json
import sys
import traceback

# Model loading logs to stdout, which would corrupt the protocol. Keep a private
# handle to the real stdout and send everything else to stderr.
_CHANNEL = sys.stdout
sys.stdout = sys.stderr


def _emit(obj: dict) -> None:
    _CHANNEL.write(json.dumps(obj) + "\n")
    _CHANNEL.flush()


def _torchaudio_compat(torchaudio, torch) -> None:
    """Bridge the torchaudio API Fish Speech 1.5 was written against.

    Two things moved after torchaudio 2.4: the backend probe was removed, and
    load() was rerouted through TorchCodec, which pins itself to particular
    FFmpeg builds. Reference clips here are plain PCM wav, so soundfile - already
    a dependency - reads them without pulling that coupling in.
    """
    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: ["soundfile"]

    import soundfile as sf

    def load(uri, frame_offset=0, num_frames=-1, normalize=True,
             channels_first=True, format=None, buffer_size=4096, backend=None):
        data, rate = sf.read(
            uri, dtype="float32", always_2d=True,
            start=frame_offset, frames=num_frames if num_frames > 0 else -1,
        )
        waveform = torch.from_numpy(data)  # soundfile gives (frames, channels)
        return (waveform.T.contiguous() if channels_first else waveform), rate

    torchaudio.load = load


def main() -> int:
    cfg = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    try:
        import soundfile as sf
        import torch
        import torchaudio

        _torchaudio_compat(torchaudio, torch)

        from tools.schema import ServeReferenceAudio, ServeTTSRequest
        from tools.server.model_manager import ModelManager

        device = cfg.get("device") or (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        manager = ModelManager(
            mode="tts",
            device=device,
            half=bool(cfg.get("half", False)),
            compile=bool(cfg.get("compile", False)),
            asr_enabled=False,
            llama_checkpoint_path=cfg["llama"],
            decoder_checkpoint_path=cfg["decoder"],
            decoder_config_name=cfg.get("decoder_config", "firefly_gan_vq"),
        )
        engine = manager.tts_inference_engine
    except Exception as exc:
        _emit({"ready": False, "err": str(exc), "trace": traceback.format_exc()[-1800:]})
        return 1

    _emit({"ready": True, "device": device})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            job = json.loads(line)
        except Exception as exc:
            _emit({"ok": False, "err": f"bad job: {exc}"})
            continue
        if job.get("cmd") == "quit":
            return 0

        try:
            refs = []
            if job.get("ref_audio"):
                with open(job["ref_audio"], "rb") as fh:
                    refs.append(
                        ServeReferenceAudio(
                            audio=fh.read(), text=job.get("ref_text", "")
                        )
                    )
            req = ServeTTSRequest(
                text=job["text"],
                references=refs,
                format="wav",
                # A character's reference clip is reused for every line they
                # speak; without this it is decoded and re-encoded each time.
                use_memory_cache="on",
                normalize=False,  # our own normalizer already ran
                temperature=float(job.get("temperature", 0.7)),
                top_p=float(job.get("top_p", 0.7)),
                repetition_penalty=float(job.get("repetition_penalty", 1.2)),
                max_new_tokens=int(job.get("max_new_tokens", 1024)),
            )

            chunks = [
                r.audio[1]
                for r in engine.inference(req)
                if getattr(r, "audio", None) is not None
            ]
            if not chunks:
                _emit({"ok": False, "err": "no audio produced"})
                continue

            import numpy as np

            audio = np.concatenate(chunks)
            rate = engine.decoder_model.spec_transform.sample_rate
            sf.write(job["out"], audio, rate)
            _emit({"ok": True, "out": job["out"], "rate": int(rate)})
        except Exception as exc:
            _emit({
                "ok": False,
                "err": f"{exc}",
                "trace": traceback.format_exc()[-1500:],
            })

    return 0


if __name__ == "__main__":
    sys.exit(main())
