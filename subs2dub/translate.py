"""Translate cue text before synthesis.

Dubbing into a language the subtitles are not already in needs a translation
pass. Four routes:

  * An instruction-following LLM, either a local one through Ollama or Claude
    through its CLI. It sees the surrounding dialogue and the production's
    glossary, and it can be given a character budget.
  * Export the cues with context, translate them elsewhere, import the result.
  * A local MarianMT model, for a quick pass with no external step. It cannot
    be told to be brief, and on en->uk it drifted into Russian on 18% of lines
    in testing, so treat it as a fallback.

Translation is a timing problem as well as a language one. Most languages run
longer than the same line in English, so lines that fitted before
translation may not afterwards. Only an instructable model can be asked for a
shorter rendering, which is why the budget lives in this stage rather than being
repaired by time-stretching later.

What the model is told about the production as a whole - its subject, register
and recurring names - comes from glossary.py and sits in the system prompt.

Translations are keyed by cue start time so they survive re-parsing.
"""

from __future__ import annotations

import csv
import json
import re
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

from . import confidence as conf
from .model import Cue

EXPANSION = {"uk": 1.12, "ru": 1.10, "de": 1.15, "es": 1.20, "fr": 1.18, "pl": 1.10}

MARIAN = {
    "uk": "Helsinki-NLP/opus-mt-en-uk",
    "de": "Helsinki-NLP/opus-mt-en-de",
    "fr": "Helsinki-NLP/opus-mt-en-fr",
    "es": "Helsinki-NLP/opus-mt-en-es",
}


def export_for_translation(
    cues: list[Cue], path: Path, target: str, context: int = 2
) -> Path:
    """Write cues with neighbouring lines so register can be judged in context."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["start", "speaker", "max_chars", "prev", "text", "next", target])
        for i, c in enumerate(cues):
            prev = " | ".join(x.text for x in cues[max(0, i - context):i])
            nxt = " | ".join(x.text for x in cues[i + 1:i + 1 + context])
            budget = int(c.window * 15.0)
            w.writerow([
                f"{c.start:.2f}", c.speaker or "", budget, prev, c.text, nxt, "",
            ])
    return path


def load_translations(path: Path, target: str) -> dict[str, str]:
    out: dict[str, str] = {}
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            new = (row.get(target) or "").strip()
            if new:
                out[f"{float(row['start']):.2f}"] = new
    return out


def apply_translations(cues: list[Cue], table: dict[str, str]) -> int:
    n = 0
    for c in cues:
        new = table.get(f"{c.start:.2f}")
        if new and new != c.text:
            c.text = new
            n += 1
    return n


def translate_local(
    cues: list[Cue], target: str, batch: int = 24, progress=None
) -> int:
    """Translate in-process with MarianMT. Fast, and blind to context."""
    model_id = MARIAN.get(target)
    if not model_id:
        raise SystemExit(
            f"no local translation model for {target!r}; "
            f"use --export-translation and translate externally"
        )
    from transformers import MarianMTModel, MarianTokenizer

    tok = MarianTokenizer.from_pretrained(model_id)
    model = MarianMTModel.from_pretrained(model_id)

    done = 0
    for i in range(0, len(cues), batch):
        group = cues[i:i + batch]
        enc = tok([c.text for c in group], return_tensors="pt", padding=True,
                  truncation=True)
        out = model.generate(**enc, max_new_tokens=256)
        for c, ids in zip(group, out):
            c.text = tok.decode(ids, skip_special_tokens=True)
            done += 1
        if progress:
            progress(min(i + batch, len(cues)), len(cues))
    return done


CHARS_PER_SEC = {"en": 14.8}
DEFAULT_CPS = 14.0

LANGUAGE = {
    "uk": "Ukrainian", "de": "German", "fr": "French",
    "es": "Spanish", "pl": "Polish", "ru": "Russian",
}

_RUSSIAN = re.compile(
    r"[ыэъЫЭЪ]|\b(что|это|этот|здесь|сегодня|очень|тебя|меня|вы|его|её|"
    r"если|когда|где|такой|один|нет|да|только|потому|чтобы)\b",
    re.I,
)


def looks_russian(text: str) -> bool:
    return bool(_RUSSIAN.search(text))


SQUEEZE = 0.97


def budget_for(cue: Cue, target: str) -> int:
    """Characters this cue's window can hold at a natural speaking rate."""
    cps = CHARS_PER_SEC.get(target, DEFAULT_CPS)
    return max(12, int(cue.budget * cps))


_SYSTEM = """You translate film dialogue into {lang} for dubbing.
{brief}
Rules:
- Reply with the {lang} translation ONLY. No quotes, notes, or alternatives.
- Natural spoken {lang}, the way a person actually talks. Not literary, not literal.
- Keep the register, tone and emotional force of the original line.
- A question stays a question; an exclamation stays an exclamation.
- Each line gives a character limit. Stay within it: cut filler and rephrase.
  Meaning matters more than word-by-word fidelity.
- Write {lang} exclusively. Never any other language."""

_USER = """Scene so far:
{context}

Translate this line into {lang}, at most {limit} characters:
{text}"""

_BATCH_USER = """Scene so far:
{context}

Translate each numbered line into {lang}. Reply with the same numbers, one line
each, nothing else. Respect each line's character limit.

{lines}"""

_NUMBERED = re.compile(r"^\s*(\d+)[.):]\s*(.+)$")


def _ollama(model: str, system: str, user: str, host: str, temperature: float) -> str:
    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {"temperature": temperature},
    }).encode()
    req = urllib.request.Request(
        f"{host}/api/chat", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read())["message"]["content"].strip()


def _claude(model: str, system: str, user: str, host: str, temperature: float) -> str:
    """Translate through the Claude Code CLI.

    Signature matches _ollama so the two are interchangeable. `host` and
    `temperature` are unused; the CLI exposes neither.
    """
    del host, temperature
    cmd = ["claude", "-p", user, "--append-system-prompt", system]
    if model and model != "claude":
        cmd += ["--model", model]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except FileNotFoundError:
        raise _Unreachable(
            "claude CLI not found. Install Claude Code, or use --translate-llm "
            "for a local model."
        )
    except subprocess.TimeoutExpired:
        raise _Unreachable("claude CLI timed out")
    if proc.returncode != 0:
        raise _Unreachable(f"claude CLI failed: {proc.stderr.strip()[-300:]}")
    return proc.stdout.strip()


ENGINES = {"ollama": _ollama, "claude": _claude}


def unload(model: str, host: str = "http://localhost:11434") -> None:
    """Evict the model from memory now rather than after the idle timeout.

    Translation is followed immediately by synthesis, and a 12B model left
    resident is several gigabytes the TTS model then has to compete for. On a
    16 GB machine that difference is swapping or not swapping.
    """
    body = json.dumps({"model": model, "messages": [], "keep_alive": 0}).encode()
    req = urllib.request.Request(
        f"{host}/api/chat", data=body, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=30):
            pass
    except Exception:
        pass


def _clean(text: str) -> str:
    """Strip the wrapping a chat model adds even when told not to."""
    text = text.strip().strip('"“”«»').strip()
    text = re.sub(r"^[^:\n]{0,24}:\s*(?=\S)", "", text, count=1)
    return text.split("\n")[0].strip()


def translate_llm(
    cues: list[Cue],
    target: str,
    *,
    model: str = "gemma3:12b",
    host: str = "http://localhost:11434",
    context: int = 3,
    attempts: int = 3,
    batch: int = 6,
    engine: str = "ollama",
    brief: dict | None = None,
    progress=None,
) -> tuple[int, list[Cue]]:
    """Translate every cue to fit inside its own window.

    Returns (translated, still_too_long). Lines are translated in groups, and
    whatever comes back too long or in the wrong language is regrouped and asked
    for again with a tighter limit. Retrying in batches rather than one line at
    a time matters: a request costs about the same whether it carries one line
    or ten.
    """
    chat = ENGINES[engine]
    lang = LANGUAGE.get(target, target)
    brief_text = _brief_block(brief)

    best: dict[int, str] = {}
    remaining = list(enumerate(cues))

    for round_no in range(max(1, attempts)):
        squeeze = SQUEEZE ** (round_no + 1)
        failed: list[tuple[int, Cue]] = []

        for start in range(0, len(remaining), batch):
            group = remaining[start:start + batch]
            try:
                got = _translate_group(
                    group, cues, lang, target, model, host, context, chat,
                    squeeze, brief_text,
                )
            except _Unreachable as exc:
                raise SystemExit(str(exc))

            for i, cue in group:
                out = got.get(i, "")
                limit = budget_for(cue, target)
                if out and not _wrong_language(out, target):
                    if i not in best or len(out) < len(best[i]):
                        best[i] = out
                if not out or len(out) > limit or _wrong_language(out, target):
                    failed.append((i, cue))

            if progress:
                progress(len(best), len(cues))

        remaining = failed
        if not remaining:
            break

    done = 0
    over: list[Cue] = []
    for i, cue in enumerate(cues):
        text = best.get(i)
        if not text:
            continue
        cue.text = text
        done += 1
        if len(text) > budget_for(cue, target):
            over.append(cue)

    if engine == "ollama":
        unload(model, host)
    return done, over


class _Unreachable(RuntimeError):
    pass


def _brief_block(brief: dict | None) -> str:
    """Render the production brief for the system prompt."""
    if not brief:
        return ""
    from .glossary import as_prompt

    body = as_prompt(brief)
    return f"\n{body}\n" if body else ""


def _wrong_language(text: str, target: str) -> bool:
    return target == "uk" and looks_russian(text)


def _translate_group(
    group: list[tuple[int, Cue]], cues: list[Cue], lang: str, target: str,
    model: str, host: str, context: int, ask=None, squeeze: float = 1.0,
    brief: str = "",
) -> dict[int, str]:
    """Translate a run of consecutive lines in one request.

    Returns only the lines that came back cleanly numbered; anything missing or
    malformed is left for the caller to redo individually.
    """
    first, last = group[0][0], group[-1][0]
    before = [c.text for c in cues[max(0, first - context):first]]
    after = [c.source_text or c.text for c in cues[last + 1:last + 1 + context]]
    scene = "\n".join(f"- {t}" for t in before + after) or "- (start of scene)"

    numbered = "\n".join(
        f"{n}. [max {max(12, int(budget_for(cue, target) * squeeze))} chars] "
        f"{cue.source_text or cue.text}"
        for n, (_, cue) in enumerate(group, 1)
    )
    try:
        raw = (ask or _ollama)(
            model,
            _SYSTEM.format(lang=lang, brief=brief),
            _BATCH_USER.format(context=scene, lang=lang, lines=numbered),
            host,
            0.2,
        )
    except (urllib.error.URLError, KeyError, TimeoutError) as exc:
        raise _Unreachable(
            f"translation model unreachable at {host}: {exc}\n"
            f"  start it with: ollama serve   (model: {model})"
        )

    out: dict[int, str] = {}
    for line in raw.splitlines():
        match = _NUMBERED.match(line.strip())
        if not match:
            continue
        n = int(match.group(1))
        if 1 <= n <= len(group):
            text = _clean(match.group(2))
            if text:
                out[group[n - 1][0]] = text
    return out


OVER_BUDGET_UNRELIABLE = 0.25
OVER_BUDGET_WEAK = 0.10


def over_budget(cues: list[Cue], target: str) -> tuple[int, int]:
    """(lines over their character budget, total lines) after translation.

    Works for every route cues can arrive by - LLM, MarianMT, or an imported
    TSV - because it measures the text actually sitting on the cue, not a
    route's own bookkeeping. translate_llm() returns its own `over` list, but
    that only covers its own route.
    """
    over = sum(1 for c in cues if len(c.text) > budget_for(c, target))
    return over, len(cues)


def confidence(cues: list[Cue], target: str) -> conf.Check:
    """How much to trust that the translated lines will fit their windows."""
    over, total = over_budget(cues, target)
    share = over / total if total else 0.0
    level = conf.band(
        share, OVER_BUDGET_WEAK, OVER_BUDGET_UNRELIABLE, higher_is_better=False,
    )
    worst = max(
        (len(c.text) - budget_for(c, target) for c in cues), default=0,
    )
    detail = f"{over} of {total} lines are longer than their window allows"
    if over:
        detail += f", worst overrun {worst} characters"
    remedy = (
        "fill the rewrite column of work/overruns.tsv and re-run with "
        "--rewrites, or use --llm-engine claude for tighter lines; "
        "--max-speed and --max-stretch let the fitter compress more before "
        "it gives up"
    )
    return conf.Check(
        stage="translation", level=level, detail=detail, remedy=remedy,
        score=share,
    )


def expansion_warning(cues: list[Cue], target: str) -> str | None:
    """Flag how many lines are likely to overrun once translated."""
    ratio = EXPANSION.get(target)
    if not ratio:
        return None
    tight = sum(1 for c in cues if len(c.text) * ratio > c.window * 15.0)
    if not tight:
        return None
    pct = 100.0 * tight / max(len(cues), 1)
    return (
        f"note: {target} runs ~{int((ratio - 1) * 100)}% longer than English; "
        f"{tight} lines ({pct:.0f}%) may overrun after translation"
    )
