"""Live view of a render in progress.

A dub takes long enough that "40/280 cues" is not much comfort. What is useful
while it runs is the line being worked on, what the fitter had to do to the
lines before it, and an estimate that comes from this run's measured rate rather
than a guess made before it started.

Falls back to plain lines when the output is not a terminal, so logs stay
readable and nothing is lost to escape codes.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

# How each fitting outcome is shown in the recent-lines list.
MARKS = {
    "clean": ("ok", "green"),
    "borrowed": ("borrowed", "green"),
    "resynth": ("faster", "cyan"),
    "filled": ("lengthened", "cyan"),
    "stretched": ("stretched", "yellow"),
    "overrun": ("overruns", "red"),
    "cut": ("cut off", "red"),
    "silent": ("failed", "red"),
}


@dataclass
class Line:
    index: int
    start: float
    speaker: str
    text: str
    outcome: str = ""
    detail: str = ""


@dataclass
class Reporter:
    """Renders a live panel, or prints plain lines when not a terminal."""

    title: str
    total: int
    keep: int = 4
    recent: list[Line] = field(default_factory=list)
    done: int = 0

    def __post_init__(self) -> None:
        self.console = Console()
        self.live: Live | None = None
        self._started = 0.0
        self.rich = self.console.is_terminal and sys.stdout.isatty()
        self.bar = Progress(
            SpinnerColumn(style="cyan"),
            TextColumn("[bold]{task.description}"),
            BarColumn(bar_width=None, complete_style="cyan", finished_style="green"),
            TaskProgressColumn(),
            TextColumn("[dim]{task.fields[rate]}"),
            TimeRemainingColumn(compact=True),
            console=self.console,
            expand=True,
        )
        self.task = self.bar.add_task(self.title, total=self.total, rate="")

    def __enter__(self) -> "Reporter":
        self._started = time.time()
        if self.rich:
            self.live = Live(
                self._render(), console=self.console,
                refresh_per_second=6, transient=False,
                vertical_overflow="crop",
            )
            self.live.__enter__()
        return self

    def __exit__(self, *exc) -> bool:
        if self.live is not None:
            self.live.update(self._render())
            self.live.__exit__(*exc)
        return False

    def now(self, line: Line) -> None:
        """Announce the line about to be worked on."""
        self.current = line
        self._refresh()

    def finish(self, line: Line) -> None:
        self.done += 1
        self.recent.append(line)
        del self.recent[:-self.keep]
        elapsed = max(time.time() - self._started, 1e-6)
        self.bar.update(
            self.task, completed=self.done,
            rate=f"{self.done / elapsed:.1f}/s",
        )
        if not self.rich:
            mark = MARKS.get(line.outcome, ("", ""))[0]
            suffix = f"  [{mark}]" if mark else ""
            print(f"  {self.done}/{self.total} {line.start:7.1f}s "
                  f"{line.text[:56]}{suffix}", flush=True)
        self._refresh()

    def _refresh(self) -> None:
        if self.live is not None:
            self.live.update(self._render())

    def _render(self):
        # Every row has to fit on one terminal line. A row that wraps makes the
        # panel taller than the frame Live is repainting, and the leftovers end
        # up strewn across the scrollback.
        fixed = len("000.0s") + 4 + 11 + 8  # time, speaker, outcome, padding
        room = max(20, self.console.width - fixed)

        table = Table.grid(padding=(0, 1))
        table.add_column(justify="right", style="dim", no_wrap=True, width=6)
        table.add_column(no_wrap=True, width=4)
        table.add_column(style="dim", no_wrap=True, width=11)
        table.add_column(no_wrap=True, width=room)

        def row(line: Line, active: bool):
            label, colour = MARKS.get(line.outcome, ("", "dim"))
            text = line.text.replace("\n", " ")
            if len(text) > room:
                text = text[:room - 1] + "…"
            table.add_row(
                f"{line.start:.1f}s",
                Text(line.speaker or "-",
                     style="bold magenta" if active else "magenta"),
                Text("..." if active else label, style="cyan" if active else colour),
                Text(text, style="" if active else "dim"),
            )

        for line in self.recent:
            row(line, False)
        current = getattr(self, "current", None)
        if current is not None:
            row(current, True)

        return Panel(
            Group(table, self.bar),
            border_style="cyan",
            padding=(0, 1),
        )


def outcome_of(cue) -> tuple[str, str]:
    """Classify what the fitter did to a cue, for display."""
    if getattr(cue, "truncated", 0) > 0.15:
        return "cut", f"lost {cue.truncated:.1f}s"
    if cue.overrun:
        return "overrun", f"+{cue.overrun:.1f}s"
    if cue.stretch and cue.stretch < 1.0:
        return "filled", f"x{cue.stretch:.2f}"
    if cue.stretch and cue.stretch > 1.0:
        return "stretched", f"x{cue.stretch:.2f}"
    if cue.engine_speed and cue.engine_speed > 1.0:
        return "resynth", f"x{cue.engine_speed:.2f}"
    if cue.borrowed:
        return "borrowed", f"+{cue.borrowed:.1f}s"
    return "clean", ""
