"""Persistent StyleTTS2 worker. Runs under the styletts2-ukrainian venv.

StyleTTS2 pins its own torch and a Ukrainian phonemizer stack that conflicts
with the main pipeline, so it lives in a separate virtualenv and is driven over
a pipe, in the same way as the Fish Speech worker.

Unlike an autoregressive model this one predicts phoneme durations in a single
pass, so a line cannot stall into silence or fail to stop, and `speed` genuinely
re-articulates rather than resampling.

One job per line of stdin:

    {"text": "...", "ref_audio": "/path/ref.wav", "speed": 1.0,
     "out": "/path/out.wav"}

One JSON reply per job.
"""

from __future__ import annotations

import json
import sys
import traceback
import warnings

warnings.filterwarnings("ignore")

_CHANNEL = sys.stdout
sys.stdout = sys.stderr


def _emit(obj: dict) -> None:
    _CHANNEL.write(json.dumps(obj) + "\n")
    _CHANNEL.flush()


def main() -> int:
    cfg = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    try:
        import re
        from unicodedata import normalize

        import soundfile as sf
        import torch
        from ipa_uk import ipa
        from styletts2_inference.models import StyleTTS2
        from ukrainian_word_stress import Stressifier

        device = cfg.get("device") or (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        model = StyleTTS2(
            hf_path=cfg.get("repo", "patriotyk/styletts2_ukrainian_multispeaker"),
            device=device,
        )
        stressify = Stressifier()
        rate = int(cfg.get("sample_rate", 24000))
        styles: dict[str, object] = {}
    except Exception as exc:
        _emit({"ready": False, "err": str(exc), "trace": traceback.format_exc()[-1800:]})
        return 1

    _emit({"ready": True, "device": device, "rate": rate})

    dashes = re.compile(r"[᠆‐‑‒–—―⁻₋−⸺⸻]")

    def prepare(text: str) -> str:
        t = normalize("NFKC", text.replace('"', "").strip())
        t = dashes.sub("-", t)
        t = re.sub(r" - ", ": ", t)
        if t and t[-1] not in ".?!:-":
            t += "."
        return t

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
            ref = job.get("ref_audio") or ""
            if ref and ref not in styles:
                styles[ref] = model.extract_voice_features(ref)
            style = styles.get(ref)
            if style is None:
                _emit({"ok": False, "err": "no reference audio given"})
                continue

            tokens = model.tokenizer.encode(ipa(stressify(prepare(job["text"]))))
            with torch.no_grad():
                wav = model(
                    tokens, speed=float(job.get("speed", 1.0)), s_prev=style
                )
            audio = wav.cpu().numpy()
            sf.write(job["out"], audio, rate)
            _emit({"ok": True, "out": job["out"], "rate": rate})
        except Exception as exc:
            _emit({
                "ok": False,
                "err": f"{exc}",
                "trace": traceback.format_exc()[-1500:],
            })

    return 0


if __name__ == "__main__":
    sys.exit(main())
