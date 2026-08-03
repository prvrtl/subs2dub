# subs2dub

Turn an existing subtitle track into a synced dub. Runs locally by default — no
API keys, no per-character billing, no account. One optional translation backend
uses the Claude Code CLI; everything else stays on your machine.

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
- Detects who speaks each line and gives each character a distinct voice
- Clones each character's voice from the original actor
- Copies each actor's delivery — loudness, emphasis, question intonation
- Reads the whole subtitle track before translating, so names and specialist
  vocabulary come out consistently across the film
- Fits each line to its window, and lengthens the short ones so a character is
  not left silent while their mouth moves
- Lets a line run into the next when the next belongs to someone else, ducked
  underneath, because conversation overlaps
- Keeps the original cast quietly under the dub, so laughter and reactions
  survive
- Checks its own output and reports what went wrong, rather than trusting that
  a line that fits is a line worth hearing

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

Ukrainian needs one extra checkout, since StyleTTS2 pins its own torch:

```sh
git clone https://huggingface.co/spaces/patriotyk/styletts2-ukrainian \
    ~/Developer/styletts2-ukrainian
cd ~/Developer/styletts2-ukrainian
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

## Use

The simplest way is to run it with no arguments. It asks what you want, shows
how long the job will take, and waits for you to agree before starting:

```sh
.venv/bin/python -m subs2dub
```

Or drive it directly:

```sh
# 90-second preview — use this while tuning, it takes seconds
.venv/bin/python -m subs2dub build movie.mkv -o preview.mkv --clip 60:90

# the whole film, fast preset voices
.venv/bin/python -m subs2dub build movie.mkv -o dubbed.mkv

# the whole film, in the original cast's voices
.venv/bin/python -m subs2dub build movie.mkv -o dubbed.mkv \
    --backend chatterbox

# dub an English video into Ukrainian
.venv/bin/python -m subs2dub build movie.mkv -o dubbed.mkv \
    --backend styletts2 --target-lang uk --translate-llm
```

The machine is kept awake for the duration of a run, and every stage releases
its model before the next one loads. Add `--no-caffeinate` to allow sleep.

Subtitles are pulled from the container automatically; pass `--subs file.srt` to
override, or `--sub-stream N` to pick a different embedded track.

### From a URL

Anything yt-dlp supports can be given directly instead of a path:

```sh
.venv/bin/python -m subs2dub build "https://youtu.be/VIDEO_ID" -o dubbed.mkv
```

Video and subtitles are fetched together. **Uploaded subtitles are preferred over
automatic captions**, and the run reports which it got. That matters more than it
sounds: automatic captions have no sentence-final punctuation, and punctuation is
what drives question intonation and sentence merging — a dub built from them is
noticeably flatter and more chopped. `--no-auto-captions` refuses them outright.

The host's licence field is reported, but treat it as a hint only. It records
what the uploader ticked, not the rights position of the work: Blender's own
Sintel upload reports "standard" despite the film being CC-BY 3.0, and material
tagged Creative Commons is regularly re-uploaded by people who don't hold the
rights. Verify separately.

### Dubbing into another language

The subtitles are the script, so dubbing into a language they aren't already in
needs a translation pass first. Translation here is a timing problem as much as
a language one: Ukrainian runs roughly 12% longer than the same line in English,
so a faithful translation often will not fit the window it has to be spoken in.

```sh
# best: reads the whole subtitle track first, then writes to length
--target-lang uk --translate-llm --llm-engine claude --llm-model sonnet

# same approach, entirely local
--target-lang uk --translate-llm            # needs `ollama serve`

# quick and weak: no context, cannot be told to be brief
--target-lang uk --translate-local

# translate it yourself, with context and budgets supplied
--target-lang uk --export-translation
--target-lang uk --translations work/translate_uk.tsv
```

Before the first line is translated, the whole subtitle file is read once to
work out what the production is, what register it uses, and how its recurring
names and specialist terms should be rendered. That brief is cached in the work
directory and sits in the translator's system prompt, so every batch sees it.
Without it, an invented name comes out three different ways in three scenes.

The result is written to `work/context_<lang>.json` and can be edited by hand.
`--no-context` skips the step.

`--llm-engine claude` shells out to the Claude Code CLI. It is the best of the
options and the only one that sends anything off the machine — subtitle text
goes to Anthropic, and it needs a subscription. `--llm-engine ollama` is the
default and stays local.

For Ukrainian speech use `--backend styletts2`. It clones each character from
the original actor, has real speed control, and renders faster than the audio
plays. `--backend fish` is an alternative at a higher sample rate, but it is
several times slower and prone to long pauses mid-line.

### Choosing a configuration

Figures are for a 2-hour film on an M1 Pro, and the pipeline calibrates against
your own machine as runs complete, so the interactive mode's estimate gets more
accurate the more you use it.

| Configuration | Language | Actor's voice | Time |
|---|---|:---:|---|
| `--backend kokoro` | English | ✗ | ~8 min |
| `--backend chatterbox` | English | ✓ | ~6 h |
| `--backend styletts2` | Ukrainian | ✓ | ~40 min |
| `--backend piper` | many | ✗ | ~4 min |
| `--backend fish` | Ukrainian | ✓ | ~4 h |

Autoregressive engines (`chatterbox`, `fish`) generate audio one token at a
time. That is inherent to them, not a tuning problem: on an M1 Pro they run at
roughly 2% of memory bandwidth, so neither half precision nor parallel workers
help. `styletts2` predicts durations in a single pass instead, which is why it
is an order of magnitude quicker.

Renders are cached by text and voice, so an interrupted run resumes where it
stopped. Long jobs are safe to kill and restart.

### Options that matter

| Flag | Default | Purpose |
|---|---|---|
| `--clip START:DUR` | — | Render a segment. The most useful flag by far. |
| `--translate-llm` | off | Translate with a model that can be told a length. |
| `--llm-engine` | ollama | `claude` for the Claude Code CLI; better, not local. |
| `--original-voices` | -16 | dB for the original cast under the dub. |
| `--no-original-voices` | off | Strip the original speech entirely. |
| `--speakers N` | auto | Force a speaker count when detection is wrong. |
| `--merge-sim` | 0.55 | Lower merges more characters, higher splits more. |
| `--no-separate` | off | Skip separation; the original dialogue stays audible. |
| `--shifts N` | 0 | Extra separation passes; each doubles that stage's time. |
| `--duck` | 9.0 | How far the background dips under dialogue. |
| `--max-stretch` | 1.15 | Time-stretch cap. Above ~1.15 it audibly wobbles. |
| `--no-caffeinate` | off | Allow the machine to sleep mid-render. |

### Checking the result

Fit statistics say whether lines fit. They say nothing about whether the result
is worth listening to — a line can fit perfectly because half of it was cut off.
Every render therefore inspects its own audio and reports what it finds:

```
40 cues: 35 fit as-is, 4 borrowed gap time, 14 re-synthesized faster,
         6 lengthened to fill, 2 still overrun (5.0%)
verify: 1 serious, 5 minor across 40 cues (2 cut off, 1 dead air)
  !!    63.0s  too short   0.2s for 28 chars - words likely dropped
```

It flags lines that were truncated, clips that are mostly silence, speech far
shorter or longer than its text implies, clipping, and any line where one
character talks over their own next line. Per-cue detail is written to
`work/render.json`.

Clips that come back obviously broken are drawn again before they ever reach the
track, up to three attempts.

## How the timing works

A subtitle cue is a fixed `[start, end]` window; synthesized speech is whatever
length it happens to be.

Too long, in order of how much each step degrades the audio:

1. **Nothing** — the line already fits. Typically ~75% of lines.
2. **Borrow** — run past `end` into the silence before the next cue.
3. **Engine speed** — synthesize again faster. Better than post-stretching,
   because the model re-articulates instead of resampling. Not every engine can.
4. **Time-stretch** — pitch-preserving compression, capped at 1.15, and again at
   1.45 where losing the end of a sentence would be worse than a wobble.

Too short matters just as much. A cue window is how long the original actor
spoke, so a clip filling half of it leaves the character silent with their mouth
still moving — heard as choppy rather than as a pause. Lines under 88% of their
window are lengthened to fill it.

Laying the clips down is a separate problem. A line may run up to 450 ms into
the next one when the next belongs to a *different* character, with the
overlapping tail ducked underneath, because people talk over each other. Running
into your own next line is only ever bad timing, and is trimmed at the quietest
nearby point so a word is lost rather than half a syllable.

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
- Non-dialogue vocals (singing, crowd noise) are removed by separation. The
  original cast is mixed back quietly by default, so reactions survive, but a
  musical number will not be dubbed.
- Lip sync is approximate. Phoneme-accurate sync needs a video model.
- Emotion is derived from how loudly each line was delivered, which captures
  arousal but not valence — it cannot distinguish frightened from furious.
- OCR artifacts in the source subtitles are pronounced literally.
- Fast sources are the hard case. Synthesized Ukrainian runs at roughly 10-13
  characters per second; a source speaking at 18 leaves the translator no choice
  but to condense, and meaning is lost. Scripted film sits comfortably inside
  this; unscripted talk often does not.
- The verifier catches structural faults — truncation, dead air, clipping — but
  cannot judge whether a delivery sounds right. Listen before shipping.

## Layout

| File | Responsibility |
|---|---|
| `source.py` | resolve a local file or a URL, fetch subtitles |
| `cues.py` | parse, clean, split two-speaker cues, merge sentence fragments |
| `glossary.py` | read the whole track once: subject, register, recurring terms |
| `translate.py` | translation via LLM, MarianMT, or a spreadsheet round-trip |
| `fit.py` | the duration-fitting engine, and laying clips onto the track |
| `tts.py` | pluggable synthesis backends |
| `verify.py` | inspect the rendered audio and report what is wrong with it |
| `estimate.py` | predict runtime, and learn from completed runs |
| `progress.py` | live view of what is being translated or synthesized |
| `wizard.py` | interactive setup |
| `power.py` | hold off sleep while a render runs |
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
