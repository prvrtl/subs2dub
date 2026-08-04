"""Hear a minute of the dub, adjust it, then commit to the whole film.

Every decision that matters here - which voices, how present the original cast
should sit, whether the speakers were split correctly - can only be made by
listening. Rendering a feature first and discovering the casting is wrong costs
hours, so a short preview comes first and the run stops there until it is
approved.

Mixing is the cheap half. Levels, ducking and how much of the original cast to
keep are applied by ffmpeg at the end, so changing them re-muxes in seconds
without touching synthesis. Those are offered as a loop; anything that requires
re-synthesis is not.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, FloatPrompt, Prompt
from rich.table import Table

PREVIEW_SECONDS = 60.0


@dataclass
class Mix:
    """The settings that can be changed without synthesizing again."""

    duck: float
    original_voices: float
    dialogue_gain: float
    keep_originals: bool = True

    def describe(self) -> Table:
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column(style="cyan", no_wrap=True)
        table.add_column(justify="right", no_wrap=True)
        table.add_column(style="dim")
        table.add_row("original cast", f"{self.original_voices:+.0f} dB",
                      "louder brings back laughter and reactions")
        table.add_row("music ducking", f"{self.duck:.1f} dB",
                      "lower keeps the score present under the dub")
        table.add_row("dub level", f"{self.dialogue_gain:+.0f} dB",
                      "lower sits the dub back into the scene")
        return table


def preview_range(duration: float, seconds: float = PREVIEW_SECONDS):
    """A minute from the middle, where a film is usually talking."""
    if duration <= seconds * 1.5:
        return None
    return max(0.0, duration * 0.45), seconds


def play(path: Path) -> None:
    if sys.platform != "darwin":
        return
    try:
        subprocess.Popen(
            ["open", str(path)], stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass


def report(console: Console, summary: str, problems: str, checks: str) -> None:
    body = [summary]
    if problems:
        body.append(problems)
    if checks:
        body.append(checks)
    console.print()
    console.print(Panel("\n".join(body), title="What the preview found",
                        border_style="cyan", padding=(0, 1)))


def decide(console: Console, out: Path, mix: Mix) -> tuple[str, Mix]:
    """Ask what to do next. Returns one of: render, remix, cast, quit."""
    console.print()
    console.print(f"[bold]Preview:[/bold] {out}")
    console.print(mix.describe())

    console.print(
        "\n[bold]Next[/bold]\n"
        "  [cyan]1.[/cyan] render the whole film with these settings\n"
        "  [cyan]2.[/cyan] adjust the mix and hear it again  [dim](seconds)[/dim]\n"
        "  [cyan]3.[/cyan] fix who says what, then re-render  [dim](edit cast.tsv)[/dim]\n"
        "  [cyan]4.[/cyan] stop here"
    )
    choice = Prompt.ask("  choose", choices=["1", "2", "3", "4"], default="1")

    if choice == "2":
        console.print("[dim]  enter to keep a value[/dim]")
        mix = Mix(
            original_voices=FloatPrompt.ask(
                "  original cast dB", default=mix.original_voices),
            duck=FloatPrompt.ask("  music ducking dB", default=mix.duck),
            dialogue_gain=FloatPrompt.ask(
                "  dub level dB", default=mix.dialogue_gain),
            keep_originals=mix.keep_originals,
        )
        return "remix", mix

    return {"1": "render", "3": "cast", "4": "quit"}[choice], mix


def cast_instructions(console: Console, cast_tsv: Path) -> None:
    console.print(Panel(
        f"Every line and who the tool thinks says it:\n\n  {cast_tsv}\n\n"
        "Edit the [bold]speaker[/bold] column so each character has one label, "
        "then re-run with:\n\n"
        f"  [cyan]--cast {cast_tsv}[/cyan]\n\n"
        "Lines whose text did not change keep their cached audio, so this "
        "costs seconds rather than a fresh render.",
        title="Fixing the casting", border_style="cyan", padding=(0, 1),
    ))


def confirm_full(console: Console, estimate: str) -> bool:
    console.print()
    return Confirm.ask(
        f"[bold]Render the whole film?[/bold] [dim](about {estimate})[/dim]",
        default=True,
    )
