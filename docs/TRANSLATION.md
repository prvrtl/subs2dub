# Translation contract

Dubbing is not subtitling. A subtitle is read at the viewer's pace; a dub has to
be *said* inside a fixed window while the actor's mouth is moving. That makes
length part of the translation, not an afterthought, and it is the reason this
project does not simply call a machine-translation model and move on.

Any translator plugged into the pipeline satisfies the contract below.

## Input

Per cue:

| Field | Meaning |
|---|---|
| `text` | The source line, already stripped of SFX markers and speaker labels |
| `max_chars` | Characters this cue's window can hold at a natural speaking rate |
| `context` | Up to three lines either side, for register and address forms |

`max_chars` is `window_seconds x chars_per_second` for the target language.
The rate is measured from rendered speech rather than assumed - see
`CHARS_PER_SEC` in `translate.py`. Ukrainian sits at 12.6.

Cues carry `source_text` alongside `text` so a second pass translates the
original rather than compounding an earlier translation.

## Output

One line of target-language text per cue, meeting all of:

1. **Within `max_chars`.** Cut filler and rephrase rather than translating word
   for word. A line that fits needs no time-stretching later, and unstretched
   audio is the single largest factor in a dub sounding natural.
2. **Target language only.** Related languages leak: a small en-uk model
   produced Russian on 18% of lines in testing, which is easy to miss by eye and
   obvious by ear. Output is checked, not trusted.
3. **Spoken register.** Contractions and everyday word order, not literary prose.
4. **Sentence type preserved.** A question stays a question - terminal `?` drives
   the synthesizer's intonation. Same for exclamations.
5. **No decoration.** No quotes, numbering, notes or alternatives; the string is
   fed directly to a speech model, which will happily pronounce stray characters.

## Verification

The pipeline enforces what it can rather than assuming compliance:

- Lines over `max_chars` are retried individually with a tightened limit, then
  reported in `overruns.tsv` if they still do not fit.
- `looks_russian()` flags Cyrillic that is Russian rather than Ukrainian.
- `speakable()` strips anything a phonemizer would read aloud before synthesis.

## Implementations

| Engine | Where it runs | Quality | Notes |
|---|---|---|---|
| `marian` | Local, in-process | Poor for dubbing | Cannot be told a length budget; drifts between related languages |
| `ollama` | Local, GPU | Good | Default. Keeps the tool local and free |
| `claude` | Claude Code CLI | Best | Sends dialogue off the machine; needs a subscription |
| `export` | You | Best | Writes a spreadsheet with budgets and context to fill in by hand |

Local remains the default deliberately. The cloud engine is opt-in because
subtitle text is copyrighted material belonging to someone else, and sending it
to a third party should be a decision rather than a side effect.

## Batching

Prompt processing costs roughly as much as generation, so lines are translated
in groups sharing one system prompt and one scene context. Groups return
numbered lines; anything missing, malformed, over budget or in the wrong
language falls back to a single-line request.

The system prompt is constant for a whole run. Interpolating anything per-line
into it - a character budget, for instance - invalidates the server's cached
prefix and roughly doubles the cost per line.
