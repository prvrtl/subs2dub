"""Interactive setup: pick a source and options, see the cost, then run."""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.table import Table

from . import estimate
from . import review
from .source import is_url

CUES_PER_MINUTE = 11.0


@dataclass
class Choice:
    value: str
    title: str
    detail: str
    note: str = ""


BACKENDS = [
    Choice("chatterbox", "Chatterbox (English)",
           "Clones each actor, expressive delivery."),
    Choice("kokoro", "Kokoro (English)",
           "28 preset voices, very fast."),
    Choice("piper", "Piper (many languages)",
           "Preset voices, fastest of all.",
           "noticeably more synthetic than the others"),
]

BACKEND_LANGS = {
    "kokoro": {"en"},
    "chatterbox": {"en"},
    "piper": None,
}


def backends_for(lang: str) -> list[Choice]:
    """Only offer engines that can speak the target language."""
    return [
        c for c in BACKENDS
        if BACKEND_LANGS.get(c.value) is None or lang in BACKEND_LANGS[c.value]
    ]

LANGUAGES = [
    Choice("en", "English", "No translation needed when subtitles are English."),
    Choice("uk", "Ukrainian", "Translated before synthesis."),
    Choice("de", "German", "Translated before synthesis."),
    Choice("fr", "French", "Translated before synthesis."),
    Choice("es", "Spanish", "Translated before synthesis."),
]

TRANSLATORS = [
    Choice("claude", "Claude Code CLI",
           "Best quality. Reads the whole subtitle track first to learn the "
           "production's names and vocabulary, then writes each line to its "
           "length budget.",
           "sends subtitle text to Anthropic; needs a subscription"),
    Choice("llm", "Local LLM (Ollama)",
           "Same approach, run entirely on this machine.",
           "keeps everything local; slower and weaker on rare names"),
    Choice("marian", "MarianMT",
           "Small offline model, one sentence at a time.",
           "fast, but ignores length and drifts between related languages"),
    Choice("export", "Translate it yourself",
           "Writes a spreadsheet with context and budgets, then stops."),
]


def _claude_available() -> bool:
    return shutil.which("claude") is not None


def _menu(console: Console, title: str, options: list[Choice], default: int = 1) -> str:
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column(justify="right", style="bold cyan", no_wrap=True)
    table.add_column()
    for i, opt in enumerate(options, 1):
        text = f"[bold]{opt.title}[/bold]\n[dim]{opt.detail}[/dim]"
        if opt.note:
            text += f"\n[yellow dim]{opt.note}[/yellow dim]"
        table.add_row(f"{i}.", text)
    console.print()
    console.print(f"[bold]{title}[/bold]")
    console.print(table)
    pick = IntPrompt.ask(
        "  choose", default=default,
        choices=[str(i) for i in range(1, len(options) + 1)],
        show_choices=False,
    )
    return options[pick - 1].value


def _probe_source(console: Console, target: str) -> tuple[float, int, str]:
    """Return (duration, subtitle cue count, description)."""
    if is_url(target):
        from .source import probe

        console.print("[dim]  reading video metadata...[/dim]")
        info = probe(target)
        duration = float(info.get("duration") or 0)
        title = (info.get("title") or "")[:60]
        subs = sorted((info.get("subtitles") or {}).keys())
        auto = sorted((info.get("automatic_captions") or {}).keys())
        detail = f"{title}\n  uploaded subtitles: {', '.join(subs) or 'none'}"
        if not subs and auto:
            detail += f"\n  automatic captions: {', '.join(auto[:8])}"
        return duration, 0, detail

    path = Path(target).expanduser()
    if not path.exists():
        raise SystemExit(f"no such file: {path}")
    from .mix import probe_duration

    return probe_duration(path), 0, path.name


def _cue_estimate(duration: float, known: int) -> int:
    return known or max(1, int(duration / 60.0 * CUES_PER_MINUTE))


def _show_estimate(console: Console, plan: estimate.Plan, awake: bool) -> float:
    rows = estimate.breakdown(plan)
    known = estimate.learned()

    table = Table(box=None, padding=(0, 2))
    table.add_column("Stage")
    table.add_column("Time", justify="right")
    table.add_column("", style="dim")
    for stage, seconds in rows:
        table.add_row(
            estimate.LABELS.get(stage, stage),
            estimate.human(seconds),
            "measured here" if stage in known else "estimated",
        )
    total = sum(s for _, s in rows)
    table.add_row("", "", "")
    table.add_row("[bold]Total[/bold]", f"[bold]{estimate.human(total)}[/bold]", "")

    console.print()
    console.print(Panel(table, title="Expected runtime", border_style="cyan"))
    if not known:
        console.print(
            "[dim]  First run on this machine, so these are rough. "
            "Later runs calibrate against measured timings.[/dim]"
        )
    if awake:
        console.print("[dim]  Sleep will be held off while it runs.[/dim]")
    return total


def run(argv: list[str] | None = None) -> int:
    console = Console()
    console.print(Panel.fit(
        "[bold]subs2dub[/bold]\n[dim]Dub a film from its subtitles[/dim]",
        border_style="cyan",
    ))

    target = Prompt.ask("\n[bold]Video file or URL[/bold]").strip().strip("'\"")
    if not target:
        return 1
    duration, cue_count, detail = _probe_source(console, target)
    console.print(f"[dim]  {detail}[/dim]")
    if duration:
        console.print(f"[dim]  duration: {estimate.human(duration)}[/dim]")

    whole = True
    start = length = 0.0
    if duration > 180:
        whole = Confirm.ask("\n[bold]Dub the whole thing?[/bold]", default=True)
        if not whole:
            start = float(IntPrompt.ask("  start at (seconds)", default=0))
            length = float(IntPrompt.ask("  how many seconds", default=120))

    sub_lang = Prompt.ask(
        "\n[bold]Subtitle language to dub from[/bold]", default="en"
    ).strip()
    target_lang = _menu(console, "Dub into", LANGUAGES, default=1)

    translator = ""
    if target_lang != sub_lang:
        options = [
            c for c in TRANSLATORS
            if c.value != "claude" or _claude_available()
        ]
        if len(options) < len(TRANSLATORS):
            console.print(
                "[dim]  (Claude Code CLI not on PATH, so that option is "
                "hidden)[/dim]"
            )
        translator = _menu(console, "How should it translate?", options)

    usable = backends_for(target_lang)
    if len(usable) == 1:
        backend = usable[0].value
        console.print(
            f"\n[bold]Voice engine[/bold]\n  [dim]{usable[0].title} - "
            f"the only one that speaks {target_lang}[/dim]"
        )
    else:
        backend = _menu(console, "Voice engine", usable)

    separate = diarize = True
    if not Confirm.ask(
        "\n[bold]Use the recommended settings?[/bold] [dim](separate the music, "
        "give each character their own voice, keep the original cast quietly "
        "underneath)[/dim]",
        default=True,
    ):
        separate = Confirm.ask("  separate voices from music?", default=True)
        diarize = Confirm.ask("  give each character their own voice?",
                              default=True)

        originals = -16.0 if Confirm.ask(
            "  keep the original cast quietly underneath?", default=True
        ) else None
    else:
        originals = -16.0
    if not separate:
        originals = None

    out = Prompt.ask("\n[bold]Write to[/bold]", default="dubbed.mkv").strip()

    span = length if not whole else duration
    cues = _cue_estimate(span, cue_count)
    plan = estimate.Plan(
        video_seconds=span,
        cues=cues,
        speech_seconds=span * 0.55,
    )
    plan.stages = _stages(backend, translator, separate, diarize)
    _show_estimate(console, plan, awake=True)

    if not Confirm.ask("\n[bold]Start?[/bold]", default=True):
        console.print("[dim]Nothing was run.[/dim]")
        return 0

    preview = None
    if whole and duration:
        preview = review.preview_range(duration)

    args = _to_args(
        target, out, backend, target_lang, sub_lang, translator,
        separate, diarize, None if whole else (start, length), originals,
    )
    from .cli import cmd_build

    if preview is None:
        return cmd_build(args)
    return _preview_then_render(console, args, preview, out, plan)


def _preview_then_render(console, args, preview, out, plan) -> int:
    """Render a minute, let it be judged, and only then commit to the film.

    A feature takes hours, and every question worth asking about a dub - the
    casting, the voices, how present the original should be - can only be
    answered by listening. Getting that wrong after the full render is the
    expensive mistake this avoids.
    """
    import copy

    from . import estimate as est
    from . import review
    from .cli import cmd_build

    start, length = preview
    mix = review.Mix(
        duck=args.duck, original_voices=args.original_voices,
        dialogue_gain=args.dialogue_gain,
        keep_originals=not args.no_original_voices,
    )

    while True:
        trial = copy.deepcopy(args)
        trial.clip = f"{start:.0f}:{length:.0f}"
        trial.out = str(Path(out).with_suffix("")) + ".preview.mkv"
        trial.duck = mix.duck
        trial.original_voices = mix.original_voices
        trial.dialogue_gain = mix.dialogue_gain
        console.print(
            f"\n[bold]Rendering a {length:.0f}-second preview[/bold] "
            f"[dim]from {start / 60:.0f} min in[/dim]"
        )
        code = cmd_build(trial)
        if code != 0:
            return code

        action, mix = review.decide(console, Path(trial.out), mix)
        if action == "quit":
            return 0
        if action == "remix":
            continue
        if action == "cast":
            work = Path(trial.work) if hasattr(trial, "work") else Path("./work")
            review.cast_instructions(console, work / "cast.tsv")
            return 0

        args.duck = mix.duck
        args.original_voices = mix.original_voices
        args.dialogue_gain = mix.dialogue_gain
        if not review.confirm_full(console, est.human(est.total(plan))):
            return 0
        return cmd_build(args)


def _stages(backend: str, translator: str, separate: bool, diarize: bool) -> list[str]:
    stages = []
    if separate:
        stages.append("separate")
    if diarize:
        stages.append("diarize")
    if translator in ("llm", "claude"):
        stages.append("translate_llm")
    elif translator == "marian":
        stages.append("translate_marian")
    if separate:
        stages += ["prosody", "references"]
    stages.append(f"synth_{backend}")
    stages.append("mux")
    return stages


def _to_args(
    target: str, out: str, backend: str, target_lang: str, sub_lang: str,
    translator: str, separate: bool, diarize: bool,
    clip: tuple[float, float] | None, originals: float | None,
) -> argparse.Namespace:
    """Start from the parser's own defaults and override only what was asked.

    Restating every flag here meant a flag added to the parser was simply
    absent on the interactive path, and the run died partway through with an
    AttributeError after the expensive stages had already finished.
    """
    from .cli import defaults_for_build

    args = defaults_for_build()
    args.video = target
    args.out = out
    args.backend = backend
    args.target_lang = target_lang
    args.sub_lang = sub_lang
    args.no_separate = not separate
    args.no_diarize = not diarize
    args.clip = f"{clip[0]:.0f}:{clip[1]:.0f}" if clip else None
    args.translate_llm = translator in ("llm", "claude")
    args.llm_engine = "claude" if translator == "claude" else "ollama"
    args.llm_model = "sonnet" if translator == "claude" else "gemma3:12b"
    args.translate_local = translator == "marian"
    args.export_translation = translator == "export"
    args.original_voices = originals if originals is not None else -16.0
    args.no_original_voices = originals is None
    return args
