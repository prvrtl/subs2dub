# subs2dub

Turn an existing subtitle track into a synced dub. Runs entirely locally — no API
keys, no per-character billing, no account.

Most dubbing pipelines transcribe the source with Whisper and machine-translate
it. If you already have good subtitles, that throws away the best asset you have:
a human translation. `subs2dub` treats the subtitle file as the script.

## What it does

```
subtitles ──► clean & merge ─────┐
                                 ├──► fit to cue windows ──► dialogue bus ──┐
original audio ──► separate ─────┤                                          ├──► mux
                    │  vocals ───┴──► embed ──► cluster ──► cast voices     │
                    └─ music+FX ──────────────────────────────────────────  ┘
```

- Uses the subtitle timings as the script and the sync target
- Removes the original dialogue instead of ducking it, so the dub is not
  competing with the source language
- Detects who speaks each line and gives each character a distinct voice
- Optionally clones each character's voice from the original actor while keeping
  native pronunciation
- Copies each actor's delivery — loudness, emphasis, question intonation
- Keeps the original audio as a second track, so the result is still watchable
  in the source language

## Install

```sh
brew install ffmpeg rubberband espeak-ng      # macOS
# sudo apt install ffmpeg rubberband-cli espeak-ng   # Debian/Ubuntu

git clone https://github.com/prvrtl/subs2dub.git
cd subs2dub
./scripts/setup.sh
```

Python 3.10 or 3.11 (most ML wheels do not ship for 3.12+ yet). Model weights
(~4 GB) download on first run and are cached. Needs ~8 GB free disk.

## Use

```sh
# 90-second preview — use this while tuning, it takes seconds
.venv/bin/python -m subs2dub build movie.mkv -o preview.mkv --clip 60:90

# the whole film, fast preset voices
.venv/bin/python -m subs2dub build movie.mkv -o dubbed.mkv

# the whole film, expressive, in the original cast's voices
.venv/bin/python -m subs2dub build movie.mkv -o dubbed.mkv \
    --backend chatterbox --voice-convert
```

Subtitles are pulled from the container automatically; pass `--subs file.srt` to
override, or `--sub-stream N` to pick a different embedded track.

### Choosing a configuration

Three properties — native pronunciation, the original actor's voice, and
expressive delivery — trade against render time. Figures below are for a
2-hour film on an M1 Pro.

| Configuration | Native | Actor's voice | Emotion | Time |
|---|:---:|:---:|:---:|---|
| `--backend kokoro` | ✓ | ✗ | ✗ | ~8 min |
| `--backend chatterbox` | ✗ accent | ✓ | ✓ | ~6 h |
| `--backend kokoro --voice-convert` | ✓ | ✓ | ✗ | ~30 min |
| `--backend chatterbox --voice-convert` | ✓ | ✓ | ✓ | ~7.5 h |

The last row is the best quality available locally. Expressive models are
roughly 60× slower than Kokoro; that is inherent, not a tuning problem.

**Cloning without `--voice-convert` inherits the source language's accent.**
Chatterbox conditions on the whole acoustic style of the reference clip, so
cloning a foreign-language actor produces English in their accent. Voice
conversion fixes this by synthesizing native English first and transferring only
the tone colour.

Renders are cached by text, voice and emotion, so an interrupted run resumes
where it stopped. Long jobs are safe to kill and restart.

### Options that matter

| Flag | Default | Purpose |
|---|---|---|
| `--clip START:DUR` | — | Render a segment. The most useful flag by far. |
| `--speakers N` | auto | Force a speaker count when detection is wrong. |
| `--merge-sim` | 0.55 | Lower merges more characters together, higher splits more. |
| `--voice-convert` | off | Native pronunciation with the actors' voices. |
| `--convert-tau` | 0.3 | Conversion strength: similarity vs naturalness. |
| `--no-separate` | off | Skip separation. Faster, but the original dialogue stays audible. |
| `--shifts N` | 0 | Extra separation quality passes; each doubles that stage's time. |
| `--duck` | 9.0 | How far the background dips under dialogue. |
| `--max-stretch` | 1.15 | Time-stretch cap. Above ~1.15 it audibly wobbles. |

## How the timing works

A subtitle cue is a fixed `[start, end]` window; TTS output is whatever length it
happens to be. Four levers, applied in order of how much they degrade the audio:

1. **Nothing** — the line already fits. Typically ~75% of lines.
2. **Borrow** — run past `end` into the silence before the next cue.
3. **Engine speed** — re-synthesize faster. Better than post-stretching, because
   the model re-articulates instead of resampling. Not all backends support it.
4. **Time-stretch** — pitch-preserving compression, capped.

Speech starts within ~35 ms of the cue start in practice.

## Adaptation

Lines that survive all four levers are still too long — they have more syllables
than the window allows. That is a writing problem, not an audio one, and it is
what dubbing studios call adaptation.

Those lines are written to `work/overruns.tsv` with a character budget derived
from your measured speech rate. Fill the `rewrite` column with shorter phrasings
and re-run:

```sh
.venv/bin/python -m subs2dub build ... --rewrites work/overruns.tsv
```

An LLM does this well, since it needs meaning and register preserved rather than
signal processing. In practice fewer than 2% of lines need it.

## Checking speaker assignment

`work/cast.tsv` lists every cue with its detected speaker and assigned voice.
Diarization is the least reliable stage — check this before committing to a long
render, and correct it with `--speakers` or `--merge-sim`.

## Known limitations

- Phone calls, shouting and whispering can split one character into two.
- Characters with very few lines may be merged into others.
- Non-dialogue vocals (singing, crowd noise) are removed by separation and do
  not come back; musical numbers play with original audio and no dub.
- Lip sync is approximate. Phoneme-accurate sync needs a video model.
- Emotion is derived from how loudly each line was delivered, which captures
  arousal but not valence — it cannot distinguish frightened from furious.
- OCR artifacts in the source subtitles are pronounced literally.

## Layout

| File | Responsibility |
|---|---|
| `cues.py` | parse, clean, split two-speaker cues, merge sentence fragments |
| `fit.py` | the duration-fitting engine |
| `tts.py` | pluggable synthesis backends |
| `convert.py` | voice conversion wrapper |
| `diarize.py` | speaker embeddings, clustering, pitch |
| `casting.py` | speaker → voice assignment |
| `refs.py` | per-character reference clips for cloning |
| `prosody.py` | delivery measurement and transfer |
| `stems.py` | source separation |
| `mix.py` | ducking, mixing, muxing |
| `adapt.py` | overrun export / rewrite import |

## Credits

Built on [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) (Apache 2.0),
[Chatterbox](https://github.com/resemble-ai/chatterbox) (MIT),
[OpenVoice](https://github.com/myshell-ai/OpenVoice) (MIT),
[Demucs](https://github.com/adefossez/demucs) (MIT),
[SpeechBrain](https://github.com/speechbrain/speechbrain) (Apache 2.0),
rubberband and ffmpeg.

## Legal

This tool processes media you supply. Dubbing copyrighted material may require
permission from the rights holder depending on your jurisdiction and intended
use. Voice cloning of real performers raises separate personality-rights issues
in many places. You are responsible for how you use it.

## License

MIT — see [LICENSE](LICENSE).
