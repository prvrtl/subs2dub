"""Track what each derived file was made from.

Every expensive stage writes something to the working directory, and every later
run wants to reuse it. Deciding that by filename is what produced the worst
faults in this tool: a clip render left a forty-second vocal stem behind, the
next full render reused it, and half the film was diarized against silence. No
error, no warning, just a dub with one voice where there were two.

A file is reusable only if the things it was made from have not changed. That is
recorded here rather than inferred, so a stale artifact is a rebuild instead of a
silent wrong answer.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

MANIFEST = "provenance.json"


def _fingerprint(value) -> str:
    if isinstance(value, Path):
        try:
            st = value.stat()
        except OSError:
            return "missing"
        return f"{st.st_size}:{int(st.st_mtime)}"
    return str(value)


def _key(inputs: dict) -> str:
    payload = json.dumps(
        {k: _fingerprint(v) for k, v in sorted(inputs.items())},
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


class Provenance:
    """The record of what produced each file in a working directory."""

    def __init__(self, work: Path) -> None:
        self.path = Path(work) / MANIFEST
        try:
            self.entries: dict = json.loads(self.path.read_text())
        except (OSError, ValueError):
            self.entries = {}

    def fresh(self, artifact: Path, **inputs) -> bool:
        """Whether `artifact` exists and still matches the inputs given."""
        if not Path(artifact).exists():
            return False
        recorded = self.entries.get(str(artifact))
        return bool(recorded) and recorded.get("key") == _key(inputs)

    def record(self, artifact: Path, **inputs) -> None:
        self.entries[str(artifact)] = {
            "key": _key(inputs),
            "inputs": {k: _fingerprint(v) for k, v in sorted(inputs.items())},
        }
        self._save()

    def forget(self, artifact: Path) -> None:
        if self.entries.pop(str(artifact), None) is not None:
            self._save()

    def stale(self) -> list[str]:
        """Recorded artifacts whose files have since disappeared."""
        return [name for name in self.entries if not Path(name).exists()]

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.entries, indent=1, sort_keys=True))
        except OSError:
            pass


def reuse(work: Path, artifact: Path, build, **inputs) -> Path:
    """Return `artifact`, rebuilding it unless it already matches `inputs`.

    The point of routing every cached stage through one function is that adding
    a stage cannot reintroduce the fault: there is no path that reuses a file
    without stating what it depends on.
    """
    prov = Provenance(work)
    artifact = Path(artifact)
    if prov.fresh(artifact, **inputs):
        return artifact
    result = build()
    out = Path(result) if result is not None else artifact
    if out.exists():
        prov.record(out, **inputs)
    return out
