# subs2dub

Generate a synced dub from a video's existing subtitle track.

Most dubbing pipelines transcribe with Whisper and machine-translate the result.
If the video already has subtitles, that discards a human translation and adds
transcription error. `subs2dub` uses the subtitle file as the script and its
timings as the sync target.

Runs locally. One optional translation backend uses the Claude Code CLI;
everything else stays on the machine.

```
subtitles ─► normalize ─► translate ─┐
                                     ├─► fit to cue windows ─► verify ─┐
audio ─► separate ─┬─ vocals ─► diarize ─► clone voices                ├─► mux
                   └─ music + effects ─────────────────────────────────┘
```

## Requirements

- macOS or Linux, Python 3.10 or 3.11
- ffmpeg, rubberband, espeak-ng
- ~8 GB disk for model weights, downloaded on first run
- 16 GB RAM is sufficient; stages run sequentially and release their models

## Install

```sh
brew install ffmpeg rubberband espeak-ng          # macOS
sudo apt install ffmpeg rubberband-cli espeak-ng  # Debian/Ubuntu

git clone https://github.com/prvrtl/subs2dub.git
cd subs2dub
./scripts/setup.sh
./scripts/setup.sh --styletts2                    # Ukrainian speech
```

## Usage

```sh
./bin/subs2dub                        # interactive, with a runtime estimate
./bin/subs2dub movie.mkv
./bin/subs2dub https://youtu.be/VIDEO_ID
```

Interactive mode offers only engines that can speak the chosen language and
estimates runtime before starting.

```sh
# 90-second segment, for tuning
./bin/subs2dub movie.mkv -o preview.mkv --clip 60:90

# English, voices cloned from the cast
./bin/subs2dub movie.mkv -o out.mkv --backend chatterbox

# English source, Ukrainian dub
./bin/subs2dub movie.mkv -o out.mkv \
    --backend styletts2 --target-lang uk --translate-llm
```

Subtitles are read from the container, or via `--subs file.srt` /
`--sub-stream N`. For URLs, uploaded subtitles are preferred over automatic
captions.

## Speech engines

| Engine | Languages | Clones cast | Speed control | 2 h film |
|---|---|:---:|:---:|---|
| `styletts2` | Ukrainian | yes | yes | ~40 min |
| `kokoro` | English | no | yes | ~8 min |
| `chatterbox` | English | yes | no | ~6 h |
| `piper` | many | no | yes | ~4 min |
| `fish` | Ukrainian | yes | no | ~4 h |

`chatterbox` and `fish` are autoregressive: they generate audio one token at a
time, run at roughly 2% of memory bandwidth on Apple silicon, and neither half
precision nor parallel workers change that. They can also stall mid-line or fail
to stop. `styletts2` predicts phoneme durations in a single pass, which makes
both failure modes impossible and accounts for the order-of-magnitude
difference.

`fish` and `styletts2` pin torch versions that conflict with this package, so
each runs in its own virtualenv over a pipe.

## Subtitle normalization

Automatic captions are not subtitles. Two formats break naive processing:

- **Rolling** — each event repeats the tail of the previous one and appends a
  few words, multiplying the dialogue several times over if read literally.
- **Unpunctuated** — nothing ends in a full stop, so every line looks like an
  unfinished sentence and sentence-merging runs whole conversations into single
  cues, which then receive one voice.

`captions.py` classifies the track and applies only the repairs it needs;
de-duplication would strip real repetition out of an authored track. On a
typical auto-captioned video this removes several hundred duplicated words and
yields cues at phrase granularity rather than ten-second blocks.

## Translation

Ukrainian runs roughly 12% longer than the same line in English, so a faithful
translation frequently will not fit the window it must be spoken in. Only an
instructable model can be asked for a shorter rendering, which is why the length
budget belongs to this stage rather than being repaired by time-stretching.

```sh
--target-lang uk --translate-llm --llm-engine claude --llm-model sonnet
--target-lang uk --translate-llm       # local, requires `ollama serve`
--target-lang uk --translate-local     # MarianMT, no context
--target-lang uk --export-translation  # spreadsheet round-trip
```

Before the first line is translated, the whole subtitle track is read once to
establish what the production is, its register, and how recurring names and
domain terms should be rendered. The result is cached to
`work/<source>/context_<lang>.json`, is editable by hand, and sits in the system
prompt — constant for the run, so it stays inside the model server's cached
prefix. Without it, an invented name is rendered differently in every scene.
`--no-context` skips the step.

`--llm-engine claude` shells out to the Claude Code CLI. It is the strongest
option and the only one that transmits anything: subtitle text goes to
Anthropic, and it requires a subscription.

Measured on a 4.5-hour source: MarianMT produced Russian on 18% of lines and
ignored length entirely. Batched LLM translation against per-line budgets runs
at ~1.7 s/line and leaves ~5% of lines over budget.

## Timing

A cue window is a fixed `[start, end]`; synthesized speech is whatever length it
is. Over-long lines are handled in order of damage:

1. Nothing — the line fits
2. Borrow silence before the next cue
3. Re-synthesize faster, where the engine supports it
4. Time-stretch, capped at 1.15, and at 1.45 where the alternative is losing the
   end of a sentence

Under-long lines matter equally. The target is how long the original actor
speaks, measured from the vocal stem, not how long the subtitle is displayed —
those differ by seconds, and chasing the window slows delivery until it no
longer matches the picture.

At layout, a line may run up to 450 ms into the next when the next belongs to a
different character, ducked 3.5 dB underneath. A character overlapping their own
next line is a fault, and is trimmed at the quietest nearby point.

Speech onset lands within ~35 ms of the cue start.

## Verification

Fit statistics do not indicate whether the result is listenable; a line can fit
because half of it was cut. Every render inspects its own audio:

```
26 cues: 15 fit as-is, 11 re-synthesized faster, 9 time-stretched,
         3 lengthened to fill, 0 still overrun (0.0%)
verify: 0 serious, 1 minor across 26 cues (1 over-stretched)
```

Detected: truncation, internal silence, speech far shorter or longer than its
text implies, clipping, and a character overlapping their own next line. Clips
failing these checks are redrawn before reaching the track, up to three
attempts. Per-cue detail is written to `work/<source>/render.json`.

## Caching

Each source gets its own working directory. Every derived artifact records what
it was built from in `provenance.json` and is rebuilt when those inputs change.
Deciding reuse by filename is what caused a clip render's 40-second vocal stem
to be reused by a later full render, silently diarizing most of a film against
nothing.

Synthesized clips are keyed by text and voice, so an interrupted run resumes.

## Options

| Flag | Default | Effect |
|---|---|---|
| `--clip START:DUR` | — | Render a segment |
| `--backend` | `kokoro` | Speech engine |
| `--target-lang` | `en` | Dub language |
| `--translate-llm` | off | Translate to a length budget |
| `--llm-engine` | `ollama` | `claude` for the CLI backend |
| `--original-voices` | `-16` | dB for the original cast under the dub |
| `--speakers N` | auto | Override speaker count |
| `--merge-sim` | 0.55 | Lower merges more characters together |
| `--no-separate` | off | Skip separation |
| `--shifts N` | 0 | Extra separation passes |
| `--duck` | 9.0 | Background dip under dialogue |
| `--max-stretch` | 1.15 | Time-stretch cap |
| `--no-caffeinate` | off | Permit sleep during a render |

## Layout

| Module | Responsibility |
|---|---|
| `source.py` | Resolve a file or URL, fetch subtitles |
| `captions.py` | Classify and repair subtitle tracks |
| `cues.py` | Clean, split, merge into speakable cues |
| `glossary.py` | Read the track once for subject, register, terms |
| `translate.py` | Translation against per-line budgets |
| `stems.py` | Source separation |
| `diarize.py` | Speaker embedding and clustering |
| `casting.py` | Speaker to voice assignment |
| `refs.py` | Per-character reference clips |
| `prosody.py` | Delivery measurement and transfer |
| `tts.py` | Speech engines behind one interface |
| `fit.py` | Duration fitting and track layout |
| `verify.py` | Post-render fault detection |
| `mix.py` | Ducking, mixing, muxing |
| `provenance.py` | Artifact dependency tracking |
| `estimate.py` | Runtime prediction, calibrated from real runs |
| `progress.py` | Live render view |
| `wizard.py` | Interactive setup |

## Limitations

- Phone calls, shouting and whispering can split one character in two;
  characters with very few lines may be merged into others
- Lip sync is approximate; phoneme-accurate sync requires a video model
- Emotion is derived from delivery loudness, capturing arousal but not valence
- Songs are removed by separation and are not dubbed
- Synthesized Ukrainian runs at 10–13 characters per second. A source speaking
  at 18 leaves the translator no option but to condense, and meaning is lost.
  Scripted film sits inside this range; unscripted speech often does not.
- Verification detects structural faults, not whether a delivery sounds right

## Credits

[Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) (Apache 2.0),
[Chatterbox](https://github.com/resemble-ai/chatterbox) (MIT),
[StyleTTS2 Ukrainian](https://huggingface.co/patriotyk/styletts2_ukrainian_multispeaker)
(MIT), [Fish Speech](https://github.com/fishaudio/fish-speech) (CC-BY-NC-SA),
[OpenVoice](https://github.com/myshell-ai/OpenVoice) (MIT),
[Demucs](https://github.com/adefossez/demucs) (MIT),
[SpeechBrain](https://github.com/speechbrain/speechbrain) (Apache 2.0),
[Piper](https://github.com/rhasspy/piper) (MIT), rubberband, ffmpeg, yt-dlp.

## Legal

This tool processes media you supply. Dubbing copyrighted material may require
permission from the rights holder depending on jurisdiction and intended use.
Voice cloning of real performers raises separate personality-rights questions in
many places. You are responsible for how you use it.

## License

MIT — see [LICENSE](LICENSE).
