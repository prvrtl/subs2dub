"""Resolve an input that may be a local file or a video URL.

Anything yt-dlp supports can be used as a source. Subtitles are fetched at the
same time, preferring human-uploaded tracks over automatic captions.

That preference matters more here than it might elsewhere. Automatic captions
arrive without sentence-final punctuation, and punctuation is what drives
question intonation and sentence merging downstream - a dub built from auto
captions is noticeably flatter and more chopped than one built from uploaded
subtitles.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

_URL = re.compile(r"^https?://", re.I)

_OPEN = {"creativeCommons", "cc-by"}


def _worth_ranging(section: tuple[float, float], duration: float) -> bool:
    """Only fetch a range when it saves a worthwhile amount of downloading."""
    if duration <= 0:
        return False
    start, dur = section
    return duration > 600 and (start + dur) < duration * 0.9 and dur < duration / 3


def is_url(s: str) -> bool:
    return bool(_URL.match(str(s)))


@dataclass
class Source:
    video: Path
    subs: Path | None = None
    title: str = ""
    license: str = ""
    auto_captions: bool = False
    trimmed: bool = False
    languages: list[str] = field(default_factory=list)

    @property
    def openly_licensed(self) -> bool:
        return self.license in _OPEN


def _yt_dlp() -> str:
    for candidate in ("yt-dlp", str(Path.home() / ".local/bin/yt-dlp")):
        try:
            subprocess.run([candidate, "--version"], capture_output=True, check=True)
            return candidate
        except Exception:
            continue
    raise SystemExit("yt-dlp not found. pip install yt-dlp")


def probe(url: str) -> dict:
    """Metadata without downloading, so licence and captions can be checked."""
    proc = subprocess.run(
        [_yt_dlp(), "-J", "--no-warnings", "--skip-download", url],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise SystemExit(f"could not read {url}\n{proc.stderr.strip()[-300:]}")
    return json.loads(proc.stdout)


def fetch(
    url: str,
    work: Path,
    *,
    sub_lang: str = "en",
    allow_auto_captions: bool = True,
    max_height: int = 0,
    section: tuple[float, float] | None = None,
) -> Source:
    """Download the video and its subtitles into `work`.

    Defaults to the best available quality. Video is copied through to the
    output untouched, so source quality carries over directly, and the best
    audio stream measurably helps separation and speaker embedding. Pass
    `max_height` to cap it when disk is tight.
    """
    work.mkdir(parents=True, exist_ok=True)
    info = probe(url)

    for old in list(work.glob("source.*")) + list(work.glob("source*.srt")):
        old.unlink(missing_ok=True)

    manual = set((info.get("subtitles") or {}).keys())
    auto = set((info.get("automatic_captions") or {}).keys())

    def pick(available: set[str]) -> str | None:
        if sub_lang in available:
            return sub_lang
        for code in sorted(available):
            if code.split("-")[0] == sub_lang:
                return code
        return None

    chosen, is_auto = pick(manual), False
    if chosen is None and allow_auto_captions:
        chosen, is_auto = pick(auto), True
    if chosen is None:
        raise SystemExit(
            f"no '{sub_lang}' subtitles on this video.\n"
            f"  uploaded: {', '.join(sorted(manual)) or 'none'}\n"
            f"  automatic: {', '.join(sorted(auto)[:12]) or 'none'}"
        )

    out = work / "source.%(ext)s"
    fmt = (
        f"bestvideo[height<={max_height}]+bestaudio/best[height<={max_height}]"
        if max_height
        else "bestvideo*+bestaudio/best"
    )
    cmd = [
        _yt_dlp(), "--no-warnings",
        "-f", fmt,
        "--merge-output-format", "mkv",
        "--write-auto-subs" if is_auto else "--write-subs",
        "--sub-langs", chosen, "--convert-subs", "srt",
        "-o", str(out), url,
    ]
    duration = float(info.get("duration") or 0)
    ranged = bool(section) and _worth_ranging(section, duration)
    trimmed = False

    if ranged:
        start, dur = section
        proc = subprocess.run(
            cmd[:1] + [
                "--download-sections", f"*{start:.2f}-{start + dur:.2f}",
                "--force-keyframes-at-cuts",
            ] + cmd[1:],
            capture_output=True, text=True,
        )
        trimmed = proc.returncode == 0
        if not trimmed:
            print("  ranged download refused by the host; fetching the whole "
                  "video instead")
            for stale in list(work.glob("source.*")):
                stale.unlink(missing_ok=True)

    if not trimmed:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise SystemExit(
                f"download failed\n{proc.stderr.strip()[-400:]}\n\n"
                f"  if this is a 403, try: {_yt_dlp()} -U"
            )

    video = next(iter(sorted(work.glob("source.mkv"))), None) or next(
        iter(sorted(work.glob("source.*"))), None
    )
    subs = next(iter(sorted(work.glob("source*.srt"))), None)
    if video is None:
        raise SystemExit("download produced no video file")

    return Source(
        video=video, subs=subs, trimmed=trimmed,
        title=info.get("title", ""),
        license=info.get("license", "") or "",
        auto_captions=is_auto,
        languages=sorted(manual),
    )


def describe(src: Source) -> str:
    lines = [f'  title:   {src.title[:64]}']
    lines.append(
        f'  licence: {src.license or "not declared"} '
        f'(as tagged by the uploader - verify separately)'
    )
    lines.append(
        "  subs:    automatic captions"
        if src.auto_captions
        else "  subs:    uploaded by the author"
    )
    if src.auto_captions:
        lines.append(
            "           note: automatic captions carry no sentence punctuation,\n"
            "           which weakens question intonation and sentence merging"
        )
    return "\n".join(lines)
