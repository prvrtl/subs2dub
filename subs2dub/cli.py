"""Command line entry point.

    python -m subs2dub                       interactive, with a time estimate
    python -m subs2dub build movie.mkv -o out.mkv --clip 60:90

`--clip` is the flag to reach for while tuning: it renders a short segment so
decisions can be heard in seconds instead of re-rendering a two-hour film.

The stages run strictly in sequence, and each releases its model before the next
one loads. On a 16 GB machine that is the difference between running and
swapping, since separation, speaker embedding, translation and synthesis each
want several gigabytes.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
import time
from pathlib import Path

from . import adapt, casting, estimate, source as srcmod, translate as trmod
from . import glossary as glossmod
from . import progress as prog
from . import verify
from . import cues as cuemod
from .diarize import diarize
from .fit import FitConfig, render_dialogue_track
from .mix import extract_clip, mux, probe_duration, write_dialogue_wav
from .power import Awake
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
    with Awake(not getattr(args, "no_caffeinate", False)) as awake:
        if awake.active:
            print("holding off sleep for the duration of this run")
        try:
            return _build(args)
        except KeyboardInterrupt:
            print("\nstopped. Re-run the same command to carry on; finished "
                  "lines are cached.")
            return 130


def output_for(requested: str, fetched, target_lang: str) -> Path:
    """Where to write the dub.

    A download is identified by its title, not by whatever the working file was
    called, and it belongs with the user's other downloads rather than in the
    checkout.
    """
    out = Path(requested).expanduser()
    if requested != "dubbed.mkv" or fetched is None or not fetched.title:
        return out
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", fetched.title).strip(" .")
    name = re.sub(r"\s+", " ", name)[:120] or "dub"
    downloads = Path.home() / "Downloads"
    folder = downloads if downloads.is_dir() else Path.cwd()
    return folder / f"{name}.{target_lang}.mkv"


def work_dir_for(source: str, requested: str) -> Path:
    """Give each source its own working directory.

    Everything cached in here - the download, the stems, the glossary, the
    synthesized clips - belongs to one video. Sharing a directory between two
    videos means the second run can pick up the first one's files.
    """
    base = Path(requested).expanduser()
    if base.name != "work" or base.parent != Path("."):
        return base
    name = Path(source).stem if not srcmod.is_url(source) else source
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", name).strip("-").lower()[:40]
    digest = hashlib.sha1(source.encode()).hexdigest()[:8]
    return base / f"{slug or 'video'}-{digest}"


def _build(args: argparse.Namespace) -> int:
    work = work_dir_for(args.video, args.work)
    work.mkdir(parents=True, exist_ok=True)
    print(f"working directory: {work}")

    clip_range = None
    if args.clip:
        _s, _d = (float(x) for x in args.clip.split(":"))
        clip_range = (_s, _d)

    fetched = None
    if srcmod.is_url(args.video):
        print(f"fetching {args.video}")
        fetched = srcmod.fetch(
            args.video, work, sub_lang=args.sub_lang,
            allow_auto_captions=not args.no_auto_captions,
            max_height=args.max_height,
            section=clip_range,
        )
        print(srcmod.describe(fetched))
        if not fetched.openly_licensed:
            print("  reminder: this video is not openly licensed. Dubbing it may\n"
                  "           require the rights holder's permission to share.")
        video = fetched.video
    else:
        video = Path(args.video).expanduser()
        if not video.exists():
            raise SystemExit(f"no such file: {video}")

    srt = (
        Path(args.subs).expanduser() if args.subs
        else (fetched.subs if fetched and fetched.subs
              else extract_subs(video, work / "subs.srt", args.sub_stream))
    )
    print(f"subtitles: {srt}")

    track: list = []
    cues = cuemod.load(srt, info=track)
    all_cues = cues
    if track:
        print(track[0].describe())
    print(f"parsed {len(cues)} speakable cues")

    source = video
    if clip_range:
        start, dur = clip_range
        if fetched is not None and fetched.trimmed:
            dur = min(dur, probe_duration(video))
        else:
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
        with estimate.Timer("separate", total) as t_sep:
            src_wav = stems.extract_wav(
                source, work / f"{source.stem}_src.wav", args.audio_stream
            )
            vocals, bed = stems.separate(src_wav, work, shifts=args.shifts)
        print(f"  stems ready in {t_sep.seconds:.0f}s")

    backend = get_backend(args.backend, cache_dir=work / "cache",
                          lang=args.target_lang)

    if args.no_diarize:
        for c in cues:
            c.voice = args.voice
    else:
        print("identifying speakers...")

        def dprog(done: int, n: int) -> None:
            print(f"  embedded {done}/{n} cues", end="\r", flush=True)

        with estimate.Timer("diarize", len(cues)):
            f0 = diarize(
                cues, source, work,
                n_speakers=args.speakers,
                max_speakers=args.max_speakers,
                merge_sim=args.merge_sim,
                audio_stream=args.audio_stream,
                vocals=vocals,
                progress=dprog,
            )
        mapping = casting.cast(
            cues, f0, default_voice=args.voice,
            pools=getattr(backend, 'voice_pools', lambda: None)(),
        )
        print(f"\nfound {len(mapping)} speakers")
        print(casting.describe(cues, mapping, f0))

        cast_tsv = work / "cast.tsv"
        with cast_tsv.open("w") as fh:
            fh.write("idx\tstart\tspeaker\tvoice\ttext\n")
            for c in cues:
                fh.write(
                    f"{c.idx}\t{c.start:.2f}\t{c.speaker}\t{c.voice}\t{c.text}\n"
                )
        print(f"per-cue assignment: {cast_tsv}")

    if args.target_lang != "en":
        warn = trmod.expansion_warning(cues, args.target_lang)
        if warn:
            print(warn)
        if args.translations:
            tp = Path(args.translations).expanduser()
            n = trmod.apply_translations(
                cues, trmod.load_translations(tp, args.target_lang))
            print(f"applied {n} translated lines from {tp}")
        elif args.export_translation:
            out_tsv = trmod.export_for_translation(
                cues, work / f"translate_{args.target_lang}.tsv", args.target_lang)
            raise SystemExit(
                f"wrote {out_tsv}\n"
                f"fill the '{args.target_lang}' column, then re-run with "
                f"--translations {out_tsv}")
        elif args.translate_llm:
            brief = {}
            if not args.no_context:
                gpath = work / f"context_{args.target_lang}.json"
                brief = glossmod.load(gpath)
                if not brief:
                    print("reading the whole subtitle track for context...")
                    brief = glossmod.build(
                        all_cues, args.target_lang,
                        model=args.llm_model, host=args.llm_host,
                        engine=args.llm_engine,
                        chat=trmod.ENGINES[args.llm_engine],
                        lang=trmod.LANGUAGE.get(args.target_lang, args.target_lang),
                    )
                    if brief:
                        glossmod.save(brief, gpath)
                print(glossmod.describe(brief))
            print(
                f"translating en -> {args.target_lang} with {args.llm_model} "
                f"(against each line's character budget)..."
            )

            reporter = prog.Reporter("Translating", len(cues))

            def tprog(done: int, n: int) -> None:
                while reporter.done < done and reporter.done < len(cues):
                    c = cues[reporter.done]
                    reporter.finish(prog.Line(
                        index=c.idx, start=c.start, speaker=c.speaker or "",
                        text=c.text, outcome="clean",
                    ))

            with estimate.Timer("translate_llm", len(cues)), reporter:
                n, over = trmod.translate_llm(
                    cues, args.target_lang,
                    model=args.llm_model, host=args.llm_host,
                    engine=args.llm_engine, batch=args.llm_batch,
                    brief=brief, progress=tprog,
                )
            print(f"\n  {n} lines translated, {len(over)} still over budget")
        elif args.translate_local:
            print(f"translating en -> {args.target_lang} locally...")

            def tprog(done: int, n: int) -> None:
                print(f"  {done}/{n}", end="\r", flush=True)

            trmod.translate_local(cues, args.target_lang, progress=tprog)
            print()
        else:
            raise SystemExit(
                f"--target-lang {args.target_lang} needs a translation source: "
                "--export-translation, --translations FILE, or --translate-local")

    if not args.no_prosody and vocals is not None:
        from . import prosody
        from .diarize import to_mono16k
        from .provenance import reuse

        vw = vocals.with_name("vocals16k.wav")
        reuse(work, vw, lambda: to_mono16k(vocals, vw), source=vocals)

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
            if presets:
                preset = c.voice if c.voice in presets else args.voice
            else:
                preset = ""
            c.voice = VoiceConvertBackend.pack(preset, ref)
        backend = VoiceConvertBackend(backend, conv, tau=args.convert_tau)
        print(f"voice conversion: {len(clips)} target voices, tau={args.convert_tau}")
        print(refsmod.describe(clips))

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
            if ref is None:
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

    speech = sum(c.window for c in cues)
    with estimate.Timer(f"synth_{args.backend}", speech), \
            prog.Reporter("Synthesizing", len(cues)) as reporter:
        bus, report = render_dialogue_track(
            cues, backend, gaps, total,
            cfg=FitConfig(
                max_engine_speed=args.max_speed,
                max_stretch=args.max_stretch,
                max_borrow=args.max_borrow,
            ),
            progress=reporter,
        )
    print()
    print(report.summary())
    for cue, why in report.failed[:args.report_n]:
        print(f"  silent @{cue.start:7.1f}s  {why}")

    report.problems.extend(
        verify.check_track(bus, backend.sample_rate, cues, total)
    )
    print(report.problem_report())
    serious = [p for p in report.problems if p.severity == "bad"]
    for p in sorted(serious, key=lambda p: p.start)[:args.report_n]:
        print(f"  {p}")
    verify.record(cues, work / "render.json")

    dlg = write_dialogue_wav(bus, backend.sample_rate, work / "dialogue.wav")
    closer = getattr(backend, "close", None)
    if closer:
        closer()
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

    out = output_for(args.out, fetched, args.target_lang)
    out.parent.mkdir(parents=True, exist_ok=True)
    mux(
        source, dlg, out,
        bed=bed,
        vocals=None if args.no_original_voices else vocals,
        voices_db=args.original_voices,
        lang=args.target_lang,
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
    b.add_argument("--backend", default="kokoro",
                   choices=("kokoro", "chatterbox", "piper", "fish", "styletts2"),
                   help="speech engine; styletts2 for Ukrainian, kokoro or "
                        "chatterbox for English")
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
    b.add_argument("--original-voices", type=float, default=-16.0,
                   help="level in dB for the original cast under the dub, so "
                        "laughter and reactions stay audible (0 = same as dub)")
    b.add_argument("--no-original-voices", action="store_true",
                   help="remove the original speech entirely")
    b.add_argument("--report-n", type=int, default=8)
    b.add_argument("--rewrites",
                   help="TSV of shortened lines produced from overruns.tsv")
    b.add_argument("--target-lang", default="en",
                   help="dub into this language (e.g. uk); needs a translation source")
    b.add_argument("--translate-llm", action="store_true",
                   help="translate with a local instruction-following LLM via "
                        "Ollama, fitting each line to its cue window")
    b.add_argument("--llm-model", default="gemma3:12b",
                   help="model for --translate-llm; with --llm-engine claude "
                        "this is a Claude model name such as 'sonnet'")
    b.add_argument("--llm-engine", default="ollama", choices=("ollama", "claude"),
                   help="ollama keeps everything local; claude uses the Claude "
                        "Code CLI, which is better but sends dialogue off the "
                        "machine and needs a subscription")
    b.add_argument("--no-context", action="store_true",
                   help="skip reading the whole subtitle track for terms "
                        "and register before translating")
    b.add_argument("--llm-batch", type=int, default=6,
                   help="lines per translation request")
    b.add_argument("--llm-host", default="http://localhost:11434")
    b.add_argument("--translate-local", action="store_true",
                   help="translate in-process with MarianMT (fast, no context)")
    b.add_argument("--export-translation", action="store_true",
                   help="write a TSV to translate elsewhere, then stop")
    b.add_argument("--translations",
                   help="TSV of translated lines to apply")
    b.add_argument("--sub-lang", default="en",
                   help="subtitle language to fetch when the source is a URL")
    b.add_argument("--no-auto-captions", action="store_true",
                   help="refuse automatic captions; they lack sentence punctuation")
    b.add_argument("--max-height", type=int, default=0,
                   help="cap video height when downloading (0 = best available)")
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
    b.add_argument("--no-caffeinate", action="store_true",
                   help="allow the machine to sleep during a run")
    b.set_defaults(func=cmd_build)

    w = sub.add_parser("wizard", help="interactive setup with a runtime estimate")
    w.set_defaults(func=lambda _a: wizard_run())

    if not argv and len(sys.argv) == 1:
        return wizard_run()

    args = p.parse_args(argv)
    return args.func(args)


def wizard_run() -> int:
    from .wizard import run

    try:
        return run()
    except (KeyboardInterrupt, EOFError):
        print()
        return 130


if __name__ == "__main__":
    sys.exit(main())
