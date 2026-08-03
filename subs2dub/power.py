"""Keep the machine awake for the length of a run."""

from __future__ import annotations

import os
import subprocess
import sys


class Awake:
    """Suppress idle, display and disk sleep until the block exits.

    caffeinate is given our pid to watch, so it exits with us even if the run
    crashes or is killed.
    """

    def __init__(self, enabled: bool = True) -> None:
        self.active = False
        self._proc: subprocess.Popen | None = None
        self._enabled = enabled and sys.platform == "darwin"

    def __enter__(self) -> "Awake":
        if not self._enabled:
            return self
        try:
            self._proc = subprocess.Popen(
                ["caffeinate", "-dims", "-w", str(os.getpid())],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self.active = True
        except (FileNotFoundError, OSError):
            self._proc = None
        return self

    def __exit__(self, *exc) -> bool:
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self.active = False
        return False
