"""Command line entry point.

    python -m subs2dub build movie.mkv -o out.mkv --clip 60:90

`--clip` is the one you'll use most: it renders a short segment so you can hear
timing decisions in seconds instead of re-rendering a two-hour film.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from . import adapt, casting
from . import cues as cuemod
from .diarize import diarize
from .fit import FitConfig, render_dialogue_track
from .mix import extract_clip, mux, probe_duration, write_dialogue_wav
from . import stems
from .model import Cue
from .tts import get_backend


def extract_subs(video: Path, out: Path, stream: int = 0) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(video),
         "-map", f"0:s:{stream}", "-c:s", "srt", str(out)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0 or not out.exists():
        raise SystemExit(
            f"could not extract subtitle stream {stream} from {video.name}.\n"
            f"{proc.stderr.strip()[-400:]}"
        )
    return out


def slice_cues(cues: list[Cue], start: float, duration: float) -> list[Cue]:
    """Keep cues inside a window and rebase their times to zero."""
    end = start + duration
    out: list[Cue] = []
    for c in cues:
        if c.end <= start or c.start >= end:
            continue
        # A cue straddling the boundary would get an artificially short window
        # and report a phantom overrun. Drop it rather than dub half a line.
        if c.start < start or c.end > end:
            kept = min(c.end, end) - max(c.start, start)
            if kept < 0.8 * c.window:
                continue
        out.append(
            Cue(
                idx=len(out),
                start=max(0.0, c.start - start),
                end=min(duration, c.end - start),
                text=c.text,
                speaker=c.speaker,
                voice=c.voice,
                source_text=c.source_text,
            )
        )
    return out


def cmd_build(args: argparse.Namespace) -> int:
    video = Path(args.video).expanduser()
    if not video.exists():
        raise SystemExit(f"no such file: {video}")

    work = Path(args.work).expanduser()
    work.mkdir(parents=True, exist_ok=True)

    srt = (
        Path(args.subs).expanduser()
        if args.subs
        else extract_subs(video, work / "subs.srt", args.sub_stream)
    )
    print(f"subtitles: {srt}")

    cues = cuemod.load(srt, merge_gap=args.merge_gap)
    print(f"parsed {len(cues)} speakable cues")

    source = video
    if args.clip:
        start, dur = (float(x) for x in args.clip.split(":"))
        source = extract_clip(video, work / "clip.mkv", start, dur)
        cues = slice_cues(cues, start, dur)
        total = dur
        print(f"clip mode: {start:.0f}s +{dur:.0f}s -> {len(cues)} cues")
    else:
        total = probe_duration(video)

    if not cues:
        raise SystemExit("no cues in range - nothing to dub")

    vocals = bed = None
    if not args.no_separate:
        print("separating vocals from music/effects...")
        t_sep = time.time()
        src_wav = stems.extract_wav(
            source, work / f"{source.stem}_src.wav", args.audio_stream
        )
        vocals, bed = stems.separate(src_wav, work, shifts=args.shifts)
        print(f"  stems ready in {time.time() - t_sep:.0f}s")

    if args.no_diarize:
        for c in cues:
            c.voice = args.voice
    else:
        print("identifying speakers...")

        def dprog(done: int, n: int) -> None:
            print(f"  embedded {done}/{n} cues", end="\r", flush=True)

        f0 = diarize(
            cues, source, work,
            n_speakers=args.speakers,
            max_speakers=args.max_speakers,
            merge_sim=args.merge_sim,
            audio_stream=args.audio_stream,
            vocals=vocals,
            progress=dprog,
        )
        mapping = casting.cast(cues, f0, default_voice=args.voice)
        print(f"\nfound {len(mapping)} speakers")
        print(casting.describe(cues, mapping, f0))

        # So the assignment can actually be checked by ear/eye and hand-fixed.
        cast_tsv = work / "cast.tsv"
        with cast_tsv.open("w") as fh:
            fh.write("idx\tstart\tspeaker\tvoice\ttext\n")
            for c in cues:
                fh.write(
                    f"{c.idx}\t{c.start:.2f}\t{c.speaker}\t{c.voice}\t{c.text}\n"
                )
        print(f"per-cue assignment: {cast_tsv}")

    # Prosody must run before synthesis: it rewrites punctuation, which is the
    # only lever Kokoro exposes on intonation.
    if not args.no_prosody and vocals is not None:
        from . import prosody
        from .diarize import to_mono16k

        vw = work / "vocals16k.wav"
        if not vw.exists():
            to_mono16k(vocals, vw)

        def pprog(done: int, n: int) -> None:
            print(f"  analyzing delivery {done}/{n}", end="\r", flush=True)

        prosody.analyze(cues, vw, progress=pprog)
        prosody.apply_punctuation(cues, use_pitch=not args.no_pitch_questions)
        prosody.apply_gain(cues, max_db=args.max_gain)
        prosody.apply_emotion(cues)
        print(f"\n{prosody.summary(cues)}")
    elif not args.no_prosody:
        print("prosody: skipped (needs the vocal stem; --no-separate was used)")

    if args.rewrites:
        rw = Path(args.rewrites).expanduser()
        if rw.exists():
            n = adapt.apply_rewrites(cues, adapt.load_rewrites(rw))
            print(f"applied {n} shortened lines from {rw}")

    backend = get_backend(args.backend, cache_dir=work / "cache")

    # Voice conversion: keep the base voice's native English, take only the
    # actor's timbre. Wraps the backend so fitting and mixing are unchanged.
    if args.voice_convert:
        if vocals is None:
            raise SystemExit("--voice-convert needs the vocal stem; drop --no-separate")
        from . import refs as refsmod
        from .convert import ToneConverter, VoiceConvertBackend

        clips = refsmod.build_references(cues, vocals, work / "refs")
        if not clips:
            raise SystemExit("could not cut any usable reference clips")

        conv = ToneConverter(
            ckpt_dir=Path(args.ov_ckpt).expanduser(),
            vendor=Path(__file__).resolve().parent.parent / "vendor",
        )
        presets = backend.voices()
        lead = max(clips, key=lambda s: sum(1 for x in cues if x.speaker == s))
        for c in cues:
            ref = clips.get(c.speaker or "", clips[lead])
            # Preset backends (Kokoro) keep a per-speaker voice so pitch range
            # roughly matches. Cloning backends used as a base get the empty
            # string, meaning their built-in speaker: expression comes from
            # them, identity comes from the converter.
            if presets:
                preset = c.voice if c.voice in presets else args.voice
            else:
                preset = ""
            c.voice = VoiceConvertBackend.pack(preset, ref)
        backend = VoiceConvertBackend(backend, conv, tau=args.convert_tau)
        print(f"voice conversion: {len(clips)} target voices, tau={args.convert_tau}")
        print(refsmod.describe(clips))

    # Cloning backends want a reference clip per character rather than a preset.
    elif getattr(backend, "clones", False):
        if vocals is None:
            raise SystemExit(
                f"--backend {args.backend} clones voices and needs the vocal "
                "stem; drop --no-separate"
            )
        from . import refs as refsmod

        clips = refsmod.build_references(cues, vocals, work / "refs")
        if not clips:
            raise SystemExit("could not cut any usable reference clips")
        print(f"cloning {len(clips)} voices from the original cast:")
        print(refsmod.describe(clips))
        missing = 0
        for c in cues:
            ref = clips.get(c.speaker or "")
            if ref is None:  # too few lines to cut a reference from
                ref = clips[max(clips, key=lambda s: sum(
                    1 for x in cues if x.speaker == s))]
                missing += 1
            c.voice = str(ref)
        if missing:
            print(f"  ({missing} cues fell back to the lead voice)")
    gaps = cuemod.gaps(cues, total)

    t0 = time.time()

    def progress(done: int, n: int, cue: Cue) -> None:
        if done % 25 == 0 or done == n:
            el = time.time() - t0
            rate = done / el if el else 0
            eta = (n - done) / rate if rate else 0
            print(
                f"  {done}/{n} cues  {rate:.1f}/s  eta {eta/60:.1f}m",
                end="\r", flush=True,
            )

    bus, report = render_dialogue_track(
        cues, backend, gaps, total,
        cfg=FitConfig(
            max_engine_speed=args.max_speed,
            max_stretch=args.max_stretch,
            max_borrow=args.max_borrow,
        ),
        progress=progress,
    )
    print()
    print(report.summary())

    dlg = write_dialogue_wav(bus, backend.sample_rate, work / "dialogue.wav")
    print(f"dialogue track: {dlg} ({dlg.stat().st_size/1e6:.0f} MB)")

    if report.overrun:
        cps = adapt.estimate_chars_per_sec(cues)
        path = adapt.export_overruns(report.overrun, work / "overruns.tsv", cps)
        print(
            f"{len(report.overrun)} lines still overrun -> {path}\n"
            f"  (speech rate measured at {cps:.1f} chars/sec; fill the 'rewrite' "
            f"column and re-run with --rewrites {path})"
        )
        for c in sorted(report.overrun, key=lambda c: -c.overrun)[:args.report_n]:
            print(f"  +{c.overrun:4.1f}s @{c.start:7.1f}s  {len(c.text):3d} chars")

    out = Path(args.out).expanduser()
    mux(
        source, dlg, out,
        bed=bed,
        audio_stream=args.audio_stream,
        duck_db=args.duck,
        dialogue_gain_db=args.dialogue_gain,
        keep_original=not args.drop_original,
    )
    print(f"wrote {out} ({out.stat().st_size/1e6:.0f} MB) in {time.time()-t0:.0f}s")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="subs2dub")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="render a dub from a subtitle track")
    b.add_argument("video")
    b.add_argument("-o", "--out", default="dubbed.mkv")
    b.add_argument("-s", "--subs", help="SRT/ASS file (default: extract embedded)")
    b.add_argument("--sub-stream", type=int, default=0)
    b.add_argument("--work", default="./work")
    b.add_argument("--backend", default="kokoro")
    b.add_argument("--voice", default="af_heart",
                   help="voice used when diarization is off or a speaker is unknown")
    b.add_argument("--no-diarize", action="store_true",
                   help="one voice for everything")
    b.add_argument("--speakers", type=int,
                   help="force an exact speaker count instead of auto")
    b.add_argument("--max-speakers", type=int, default=20,
                   help="upper bound when auto-detecting speaker count")
    b.add_argument("--merge-sim", type=float, default=0.55,
                   help="centroid similarity above which two clusters are merged "
                        "into one character; raise for more speakers")
    b.add_argument("--audio-stream", type=int, default=0)
    b.add_argument("--no-separate", action="store_true",
                   help="skip Demucs; original dialogue stays audible under the dub")
    b.add_argument("--shifts", type=int, default=0,
                   help="Demucs quality passes; each one doubles separation time")
    b.add_argument("--clip", help="START:DURATION in seconds, e.g. 60:90")
    b.add_argument("--merge-gap", type=float, default=0.30)
    b.add_argument("--max-speed", type=float, default=1.25)
    b.add_argument("--max-stretch", type=float, default=1.15)
    b.add_argument("--max-borrow", type=float, default=1.50)
    b.add_argument("--duck", type=float, default=9.0)
    b.add_argument("--dialogue-gain", type=float, default=2.0)
    b.add_argument("--drop-original", action="store_true")
    b.add_argument("--report-n", type=int, default=8)
    b.add_argument("--rewrites",
                   help="TSV of shortened lines produced from overruns.tsv")
    b.add_argument("--voice-convert", action="store_true",
                   help="synthesize native English, then convert the timbre to "
                        "each original actor (keeps pronunciation, takes voice)")
    b.add_argument("--ov-ckpt", default="./checkpoints_ov/converter",
                   help="OpenVoice tone-colour converter checkpoint directory")
    b.add_argument("--convert-tau", type=float, default=0.3,
                   help="conversion strength; higher moves further toward the actor")
    b.add_argument("--no-prosody", action="store_true",
                   help="don't copy delivery from the original actors")
    b.add_argument("--no-pitch-questions", action="store_true",
                   help="use wording alone for questions, ignore actor intonation")
    b.add_argument("--max-gain", type=float, default=5.0,
                   help="dB ceiling for intensity transfer")
    b.set_defaults(func=cmd_build)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
