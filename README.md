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

# voices cloned from the cast (the default)
./bin/subs2dub movie.mkv -o out.mkv

# preset voices instead: forty times faster, no cloning
./bin/subs2dub movie.mkv -o out.mkv --backend kokoro
```

Subtitles are read from the container, or via `--subs file.srt` /
`--sub-stream N`. For URLs, uploaded subtitles are preferred over automatic
captions.

## Speech engines

| Engine | Clones cast | Speed control | 2 h film |
|---|:---:|:---:|---|
| `chatterbox` | yes | no | ~6 h |
| `kokoro` | no | yes | ~8 min |
| `piper` | no | yes | ~4 min |

`chatterbox` is the default. It clones each character from a reference clip cut
from the original, and has an emotion control driven by how loudly each line was
delivered. It is autoregressive, generating audio one token at a time, and on
Apple silicon runs at roughly 2% of memory bandwidth, so it is slow and neither
half precision nor parallel workers change that.

`kokoro` renders about forty times faster with preset voices. Measured on the
same material it is more expressive than a clone, because conditioning on a
reference constrains the model's prosody. Whether that trade is worth it depends
on how much the voices should sound like the cast.

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

## Confidence

Before synthesis starts, four stages that can measure their own reliability
report on it in one block. A clean run says nothing; anything less prints:

```
  confidence: 1 unreliable, 1 weak of 4 checks
    !! speakers     silhouette 0.15 of 1.0 at 4 speakers, 57 of 59 lines in one cluster
       fix          pass --speakers N if you know how many voices there are, or
                     --cast FILE to assign speakers by hand, or --no-diarize for one voice
     ~ subtitles    automatic captions, no sentence punctuation
       fix          supply an authored track with --subs FILE
    ok translation  3 of 59 lines over budget
    ok embeddings   59 of 59 cues embedded
```

| Check | Measures | Fix named in the remedy |
|---|---|---|
| `subtitles` | whether the track is punctuated, and how much a rolling-captions repair had to repeat or drop | `--subs FILE`, `--no-auto-captions` |
| `speakers` | silhouette score of the speaker clustering actually produced, and the largest cluster's share of all cues | `--speakers N`, `--cast FILE`, `--merge-sim`, `--no-diarize` |
| `embeddings` | the share of cues that had usable speech in their window to embed, versus cues that inherited a neighbour's speaker label | `--no-separate`, `--audio-stream N` |
| `translation` | the share of translated lines longer than their window allows, and the worst overrun | `--rewrites`, `--llm-engine claude`, `--max-speed`, `--max-stretch` |

The same block is written to `work/<source>/confidence.json`. No check runs on
the `--no-diarize` path - no speaker decision was made, so there is nothing to
doubt - and none exists for casting gender or delivery: both are inferred from
a fixed pitch split with no calibration, and a confidence number on them would
be invented rather than measured.

## Video speaker detection

`--speakers-from video` anchors speakers from the picture instead of asking
the audio embeddings alone. It needs its own virtualenv - mediapipe and
opencv-python both pull numpy 2, which conflicts with this package's own
numpy pin - installed with `./scripts/setup.sh --vision` (~530 MB: a face
landmark model, an SFace identity model, and the two Python packages). A
worker decodes the video once, tracks faces, and scores which tracked face's
mouth motion best matches the audio during each cue; tracks are then grouped
into characters by face identity, and every cue video labels confidently
becomes an anchor that `diarize()` asks the audio embeddings to agree with,
rather than clustering the embeddings from scratch. This is the case this
exists for: audio embeddings occasionally carry no usable signal at all
(phone-filtered dialogue, heavy processing) and no amount of re-clustering
recovers it, where "which of these two known faces is talking" is often still
answerable.

It defaults to `audio`, so nothing about an existing run changes unless this
is asked for, and it falls back to audio on its own whenever it can't back an
answer with enough evidence: no video stream, the venv or models missing, the
worker crashing, too few cues labelled, the character split too lopsided to
trust, or an identity silhouette too low to mean anything. Every one of those
is a fall back to today's audio-only diarization, never a partial answer.

Tested against the material this was built to fix - a phone-call sketch where
audio diarization puts 57 of 59 lines in one cluster - it did not reliably
solve it. Two concrete faults turned up: the face landmark model occasionally
returns a confident, well-formed face on a hand or a small round prop, which
pollutes identity clustering with a vector that isn't a face at all; and this
clip's identity clusters mixed real, different people rather than cleanly
separating the two callers, in a way a track-level silhouette score alone did
not reveal - only looking at the actual crops did. The safeguards above catch
this specific case and fall back correctly, which was verified directly, but
the underlying face-identity separation should be treated as unproven beyond
that one measurement, not as a solved problem behind a flag.

## Caching

Each source gets its own working directory. Every derived artifact records what
it was built from in `provenance.json` and is rebuilt when those inputs change.
Deciding reuse by filename is what caused a clip render's 40-second vocal stem
to be reused by a later full render, silently diarizing most of a film against
nothing.

Synthesized clips are keyed by text and voice, so an interrupted run resumes.

Every run also writes `work/cast.tsv`, one row per cue with its speaker and
voice. When diarization gets a character wrong - too many speakers, a merged
pair, the wrong gender - edit the `speaker` and/or `voice` column for the
lines that need fixing and re-run with `--cast work/cast.tsv`. This skips
diarization entirely and applies the edited columns, so only the lines that
actually changed re-synthesize; everything else hits the cache. On a
cloning backend (chatterbox, fish, styletts2) the voice column has no effect -
the reference clip built from each speaker's lines is what selects the voice,
so only the speaker column matters there. Re-labelling a character to a new
speaker rebuilds their reference clip from a different set of lines, which
re-renders all of that character's lines, not just the ones that were moved.
Pair `--cast` with `--translations` (or just re-run with `--translate-llm`,
which reuses the previous render's text automatically) so a casting fix does
not re-translate the whole film.

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
| `--speakers-from` | `audio` | `video` anchors speakers from face tracking; needs `--vision` setup |
| `--merge-sim` | 0.55 | Lower merges more characters together |
| `--cast FILE` | — | Apply an edited cast.tsv, skipping diarization |
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
| `vision.py` | Video active speaker detection (worker driver, scoring, identity clustering) |
| `vision_worker.py` | Face tracking and identity extraction; runs in `.venv-vision` |
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
- Verification detects structural faults, not whether a delivery sounds right
- `--speakers-from video` (see above) is experimental: it degrades to
  audio-only whenever it can't back an answer with evidence, but the
  underlying face-identity separation is validated on one clip, not proven
  in general - animation, puppetry, and already-dubbed material all break
  the assumption that mouth motion tracks the audio, and one actor voicing
  several characters isn't split by face identity, since there's one face

## Credits

[Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) (Apache 2.0),
[Chatterbox](https://github.com/resemble-ai/chatterbox) (MIT),
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
