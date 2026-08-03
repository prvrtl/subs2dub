"""Work out what a video is about before translating a word of it.

Translating cue by cue, or even in small batches, loses everything that makes a
translation sound like it belongs to the production: character names, invented
places, the vocabulary of whatever the thing is about, and the register the cast
speaks in. Worse, an unremembered name comes out differently in every batch.

So the whole subtitle track is read once, up front, to produce a short brief and
a glossary of recurring terms. Both go into the translator's system prompt,
which is constant for the run and therefore stays inside the model server's
cached prefix - the context costs almost nothing per line.

The result is cached next to the working files; it only changes if the source
does.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from .model import Cue

# Words that start a sentence are capitalized for grammar, not because they are
# names, so candidates are taken from mid-sentence positions only.
_WORD = re.compile(r"\b[A-Z][a-zA-Z'’\-]{2,}\b")
_SENTENCE_START = re.compile(r"(?:^|[.!?]\s+)$")

_COMMON = {
    "The", "This", "That", "There", "Then", "They", "Their", "These", "Those",
    "What", "When", "Where", "Who", "Why", "How", "And", "But", "For", "You",
    "Your", "It's", "I'm", "We're", "Okay", "Yeah", "Yes", "Oh", "Hey", "Well",
    "All", "Are", "Can", "Did", "Do", "Does", "Don't", "Get", "Got", "Have",
    "He", "Her", "Here", "Him", "His", "If", "Just", "Let", "Like", "Look",
    "Make", "Me", "My", "No", "Not", "Now", "One", "Our", "Out", "Right", "She",
    "So", "Some", "Take", "Thank", "Thanks", "Was", "We", "Will", "With",
}


def candidates(cues: list[Cue], top: int = 120) -> list[str]:
    """Recurring capitalized terms, most frequent first."""
    counts: Counter[str] = Counter()
    for cue in cues:
        text = cue.source_text or cue.text
        for match in _WORD.finditer(text):
            if _SENTENCE_START.search(text[:match.start()]):
                continue
            word = match.group()
            if word in _COMMON or word.isupper():
                continue
            counts[word] += 1
    # A term used once is probably not worth pinning down; it is also the case
    # that one-offs are where false positives collect.
    return [w for w, n in counts.most_common(top) if n >= 2]


def sample_lines(cues: list[Cue], n: int = 60) -> list[str]:
    """An evenly spread sample, enough to judge genre and register."""
    if not cues:
        return []
    step = max(1, len(cues) // n)
    picked = [cues[i].source_text or cues[i].text for i in range(0, len(cues), step)]
    return [t for t in picked if len(t) > 15][:n]


_BRIEF = """You are preparing to dub a video into {lang}.

Below is a sample of its subtitles and a list of terms that recur throughout.
Work out what this production is, then decide how its recurring terms should be
rendered in {lang}.

Reply with JSON only, in this exact shape:

{{"summary": "two sentences: what this is, its genre, and the register the \
speakers use",
 "notes": "one or two sentences of guidance for the translator - how formal, \
how to handle names, anything specific to this production",
 "glossary": [{{"term": "original", "target": "{lang} rendering", \
"why": "short reason"}}]}}

Rules for the glossary:
- Include proper nouns: characters, places, organisations, invented things.
- Include vocabulary specific to the subject matter, using the {lang} terms that
  the audience for this kind of material would actually expect. If it is a game,
  a sport or a profession, use that field's established {lang} terminology
  rather than a literal translation.
- Personal names are usually transliterated, not translated. Give the
  transliteration.
- Leave out ordinary words that need no decision.
- At most 60 entries, the most frequent and most consequential.

Recurring terms:
{terms}

Sample of the dialogue:
{lines}"""


def build(
    cues: list[Cue],
    target: str,
    *,
    model: str,
    host: str,
    engine: str,
    chat,
    lang: str,
) -> dict:
    """Read the whole track once and return a brief plus a glossary."""
    terms = candidates(cues)
    lines = sample_lines(cues)
    if not terms and not lines:
        return {}

    prompt = _BRIEF.format(
        lang=lang,
        terms=", ".join(terms) or "(none stood out)",
        lines="\n".join(f"- {t}" for t in lines),
    )
    raw = chat(model, "You reply with JSON and nothing else.", prompt, host, 0.1)
    return _parse(raw)


def _parse(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n|\n```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        data = json.loads(text[start:end + 1])
    except ValueError:
        return {}
    if not isinstance(data, dict):
        return {}
    entries = data.get("glossary")
    data["glossary"] = [
        e for e in entries
        if isinstance(e, dict) and e.get("term") and e.get("target")
    ] if isinstance(entries, list) else []
    return data


def as_prompt(brief: dict) -> str:
    """The part of the system prompt that describes this production."""
    if not brief:
        return ""
    parts = []
    if brief.get("summary"):
        parts.append(f"About this production:\n{brief['summary']}")
    if brief.get("notes"):
        parts.append(brief["notes"])
    glossary = brief.get("glossary") or []
    if glossary:
        lines = "\n".join(
            f"  {e['term']} -> {e['target']}" for e in glossary
        )
        parts.append(
            "Use these renderings consistently, every time the term appears:\n"
            + lines
        )
    return "\n\n".join(parts)


def load(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def save(brief: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(brief, ensure_ascii=False, indent=1))
    return path


def describe(brief: dict) -> str:
    if not brief:
        return "  no context built"
    n = len(brief.get("glossary") or [])
    summary = (brief.get("summary") or "").strip().replace("\n", " ")
    return f"  {summary[:150]}\n  {n} terms pinned down"
