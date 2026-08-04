"""Video active speaker detection: which visible face is talking, per cue.

Speaker diarization asks the audio "how many voices are there, and which cue
goes with which" - on material where the embeddings carry no usable signal
(phone-filtered audio, one actor doing several voices) that question has no
good answer no matter how it is asked; a forced --speakers 2 can still put 57
of 59 lines in one cluster. Video asks a different, often easier, question:
is this particular face's mouth moving in sync with the audio right now. A
worker in its own virtualenv (mediapipe and opencv-python both need numpy 2,
which conflicts with this package's numpy<2 pin) tracks faces across one
ordered decode pass and writes a per-face-per-frame record to a .npz; this
module reads that with numpy alone, scores which track is speaking during
each cue, groups tracks into characters by face identity, and hands diarize()
an "anchor" speaker id for every cue video could answer confidently.

Anchored cues are then a *reconciliation* problem for diarize(), not a
clustering one: instead of asking the audio embeddings how many groups exist,
it asks which of the already-known characters a given cue's embedding sits
closest to. That is the entire point of routing through video first - it is
answerable at a silhouette that blind clustering cannot use at all.

Off-screen speech is the failure mode this cannot rule out by construction: a
listener whose mouth happens to move while someone off-camera talks looks
identical to a speaker, from mouth motion alone. Requiring that motion to also
correlate with the audio envelope is the main defence. The degradation ladder
in speakers() is the backstop: every rung returns None and falls back to
audio-only diarization rather than handing back a partial or low-confidence
answer, because an anchoring built on too little evidence is worse than no
anchoring at all.

Two things this does not solve, worth stating plainly rather than glossing
over: one actor voicing several characters is not separated by face identity,
since there is only one face; and animation, puppetry, or already-dubbed
material breaks the core assumption that mouth motion tracks the audio.
"""

from __future__ import annotations

import json
import queue
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .model import Cue
from .provenance import reuse

SAMPLE_FPS = 12.5
DECODE_W, DECODE_H = 960, 540
MAX_FACES = 3

MIN_TRACK_FRAMES = 3
MIN_FACE_SIZE = 0.03
SCORE_FLOOR = 0.35
WINNER_MARGIN = 0.15
MERGE_SIM = 0.40
NEAREST_MARGIN = 0.05

MIN_LABEL_SHARE = 0.25
MIN_CHARACTERS = 2
MIN_IDENTITY_SILHOUETTE = 0.10
MAX_ANCHOR_SHARE = 0.80

ROOT = Path(__file__).resolve().parent.parent
VENV = ROOT / ".venv-vision"
MODELS = ROOT / "models" / "vision"
LANDMARKER = MODELS / "face_landmarker.task"
SFACE = MODELS / "face_recognition_sface_2021dec.onnx"


class VisionUnavailable(RuntimeError):
    """Video speaker detection could not run; audio diarization carries on alone."""


@dataclass
class VisionLabels:
    """Per-cue anchor speaker ids from video, and how much of the cast they cover.

    `anchors` has one entry per cue: a small non-negative integer naming a
    character video is confident spoke that line, or -1 where video found no
    confident answer. diarize() treats -1 exactly like a cue with no usable
    audio embedding - it fills in from context rather than leaving it unlabelled.
    """

    anchors: np.ndarray
    n_characters: int
    labelled: int
    silhouette: float


def has_video_stream(video: Path) -> bool:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries",
         "stream=index", "-of", "csv=p=0", str(video)],
        capture_output=True, text=True,
    )
    return bool(proc.stdout.strip())


def installed() -> bool:
    return (VENV / "bin" / "python").exists() and LANDMARKER.exists() and SFACE.exists()


def _pump(stream, replies: "queue.Queue") -> None:
    try:
        for line in stream:
            replies.put(line)
    except (ValueError, OSError):
        pass
    replies.put(None)


class _Worker:
    """Drives vision_worker.py over a pipe, in the shape of the TTS workers.

    The one deliberate difference: nothing here is fatal. A missing venv, a
    missing model file, or a dead process all raise VisionUnavailable instead
    of SystemExit, because unlike a TTS engine this stage is an accuracy
    improvement over audio-only diarization, not something a run depends on.
    """

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self._proc = None
        self._log = None
        self._reader = None
        self._replies: "queue.Queue" = queue.Queue()

    def _start(self):
        if self._proc is not None:
            return self._proc
        if not (VENV / "bin" / "python").exists():
            raise VisionUnavailable(
                "the vision worker is not installed; run ./scripts/setup.sh --vision"
            )
        if not (LANDMARKER.exists() and SFACE.exists()):
            raise VisionUnavailable(
                f"vision models missing under {MODELS}; run ./scripts/setup.sh --vision"
            )

        cfg = {
            "landmarker": str(LANDMARKER), "sface": str(SFACE),
            "width": DECODE_W, "height": DECODE_H, "fps": SAMPLE_FPS,
            "max_faces": MAX_FACES,
        }
        worker = Path(__file__).with_name("vision_worker.py")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._log = open(self.cache_dir / "vision_worker.log", "w")
        self._proc = subprocess.Popen(
            [str(VENV / "bin" / "python"), str(worker), json.dumps(cfg)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._log,
            text=True, bufsize=1,
        )
        self._reader = threading.Thread(
            target=_pump, args=(self._proc.stdout, self._replies), daemon=True,
        )
        self._reader.start()

        hello = self._read_reply(timeout=120.0)
        if not hello.get("ready"):
            raise VisionUnavailable(
                f"vision worker failed to start: {hello.get('err')}; "
                f"see {self._log.name}"
            )
        return self._proc

    def _read_reply(self, timeout: float = 900.0, on_progress=None) -> dict:
        """Read one non-progress reply, skipping {"progress": ...} lines.

        Otherwise the same shape as the TTS workers' reply reader: a plain
        readline() can block forever on a worker that dies after spawning
        something that inherited its stdout - a model download, say.
        """
        assert self._proc is not None
        deadline = time.time() + timeout
        while True:
            left = deadline - time.time()
            if left <= 0:
                return {"ok": False, "err": f"worker silent for {timeout:.0f}s"}
            try:
                line = self._replies.get(timeout=min(2.0, left))
            except queue.Empty:
                if self._proc.poll() is not None:
                    return {
                        "ok": False,
                        "err": f"worker exited ({self._proc.returncode}); see "
                               f"{getattr(self._log, 'name', 'the worker log')}",
                    }
                continue
            if line is None:
                return {"ok": False, "err": "worker closed its output"}
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            if "progress" in obj and "ok" not in obj and "ready" not in obj:
                if on_progress:
                    on_progress(obj.get("progress", 0), obj.get("total", 0))
                continue
            return obj

    def scan(self, video: Path, windows: list, out: Path, progress=None) -> Path:
        try:
            proc = self._start()
            job = {"cmd": "scan", "video": str(video), "windows": windows,
                   "out": str(out)}
            proc.stdin.write(json.dumps(job) + "\n")
            proc.stdin.flush()
        except VisionUnavailable:
            raise
        except (BrokenPipeError, OSError) as exc:
            raise VisionUnavailable(f"vision worker pipe failed: {exc}")

        reply = self._read_reply(timeout=3600.0, on_progress=progress)
        if not reply.get("ok"):
            raise VisionUnavailable(
                f"vision scan failed: {reply.get('err')}; see "
                f"{getattr(self._log, 'name', 'the worker log')}"
            )
        return Path(reply["out"])

    def close(self) -> None:
        if self._proc is None:
            return
        try:
            self._proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
            self._proc.stdin.flush()
            self._proc.wait(timeout=15)
        except Exception:
            self._proc.kill()
        self._proc = None


def scan_faces(video: Path, cues: list[Cue], work: Path, progress=None) -> Path:
    """Run (or reuse) the worker's face scan, cached through provenance.reuse.

    Raises VisionUnavailable on any failure; the caller decides whether that
    means falling back quietly or telling the user why.
    """
    out = work / "faces.npz"
    windows = [(c.start, c.end) for c in cues]

    def build():
        worker = _Worker(work / "cache")
        try:
            worker.scan(video, windows, out, progress=progress)
        finally:
            worker.close()
        return out

    return reuse(
        work, out, build, source=video, fps=SAMPLE_FPS, width=DECODE_W,
        height=DECODE_H, landmarker=LANDMARKER, sface=SFACE, windows=windows,
    )


def load_faces(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def _audio_envelope(wav: Path, times: np.ndarray, hop: float = 0.05) -> np.ndarray:
    """RMS envelope of `wav` in `hop`-second frames, resampled onto `times`."""
    import soundfile as sf

    audio, sr = sf.read(wav, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    frame = max(1, int(hop * sr))
    n = audio.size // frame
    if n == 0 or times.size == 0:
        return np.zeros_like(times)
    env = np.sqrt(np.mean(audio[: n * frame].reshape(n, frame).astype(np.float64) ** 2, axis=1))
    env_times = (np.arange(n) + 0.5) * hop
    return np.interp(times, env_times, env, left=float(env[0]), right=float(env[-1]))


def _track_vector(faces: dict, track_id: int) -> np.ndarray:
    m = faces["track_id"] == track_id
    v = np.median(faces["sface"][m], axis=0)
    return v / (np.linalg.norm(v) + 1e-9)


def _characters(faces: dict) -> dict[int, int]:
    """Group tracks into characters by face identity.

    Over-clusters first and merges back by centroid cosine similarity, the
    same shape as diarize._merge_close: an over-split track is repaired here,
    a track fused with the wrong person is not, so every threshold leans
    towards leaving tracks apart.

    Tracks shorter than MIN_TRACK_FRAMES are dropped before clustering rather
    than merged in: a one- or two-frame track is a stray detection, never has
    enough frames inside a cue window to win one in _score_cues, and only
    adds noise to the character count and identity silhouette reported to
    the user.
    """
    counts = np.bincount(faces["track_id"])
    ids = [int(i) for i in np.unique(faces["track_id"])
           if counts[i] >= MIN_TRACK_FRAMES]
    if not ids:
        return {}
    if len(ids) == 1:
        return {ids[0]: 0}

    X = np.stack([_track_vector(faces, tid) for tid in ids])
    from sklearn.cluster import AgglomerativeClustering

    lab = AgglomerativeClustering(
        n_clusters=None, distance_threshold=1.0 - MERGE_SIM,
        metric="cosine", linkage="average",
    ).fit_predict(X)
    return {tid: int(c) for tid, c in zip(ids, lab)}


def _merge_to(faces: dict, tracks: dict[int, int], n: int) -> dict[int, int]:
    """Collapse characters down to `n` by repeatedly merging the closest pair.

    Used only when --speakers forces a count lower than what identity
    clustering found on its own; the audio side of a forced count still gets
    to split the remainder in diarize(), this just stops video from
    over-supplying characters it was never asked for.
    """
    ids = [int(i) for i in np.unique(faces["track_id"])]
    vecs = {tid: _track_vector(faces, tid) for tid in ids}
    tracks = dict(tracks)
    chars = sorted(set(tracks.values()))

    while len(chars) > n:
        cents = {}
        for c in chars:
            members = [vecs[tid] for tid, cc in tracks.items() if cc == c]
            cent = np.mean(members, axis=0)
            cents[c] = cent / (np.linalg.norm(cent) + 1e-9)
        best = None
        for i, a in enumerate(chars):
            for b in chars[i + 1:]:
                sim = float(np.dot(cents[a], cents[b]))
                if best is None or sim > best[0]:
                    best = (sim, a, b)
        _, a, b = best
        tracks = {tid: (a if c == b else c) for tid, c in tracks.items()}
        chars = sorted(set(tracks.values()))
    return tracks


def _identity_silhouette(faces: dict, tracks: dict[int, int]) -> float:
    """Silhouette of the character split actually produced, over real tracks only.

    Scored only over the tracks _characters() kept, not every raw track id in
    the scan - the ones it dropped (fewer than MIN_TRACK_FRAMES frames) are
    single stray detections with no real identity signal, and folding them in
    as an ad hoc extra "cluster" distorts the score rather than measuring it.
    """
    from sklearn.metrics import silhouette_score

    ids = sorted(tracks)
    if len(ids) < 2:
        return 0.0
    X = np.stack([_track_vector(faces, tid) for tid in ids])
    labs = np.array([tracks[tid] for tid in ids])
    if len(set(labs.tolist())) < 2:
        return 0.0
    try:
        return float(silhouette_score(X, labs, metric="cosine"))
    except Exception:
        return 0.0


def _detrend(x: np.ndarray, win: int = 7) -> np.ndarray:
    """Subtract a moving median so a permanently open or closed mouth scores zero."""
    if x.size == 0:
        return x
    half = max(1, win // 2)
    out = np.empty_like(x)
    for i in range(x.size):
        lo, hi = max(0, i - half), min(x.size, i + half + 1)
        out[i] = x[i] - np.median(x[lo:hi])
    return out


def _track_baselines(faces: dict, cues: list[Cue]) -> dict[int, float]:
    """Per-track detrended mouth-motion std outside any cue window.

    A track that is naturally jittery (a wide shot, a shaky camera) needs its
    own reference for "not speaking" - a single fixed threshold either misses
    the calm speakers or fires on every twitchy listener.
    """
    t = faces["t"]
    in_any_cue = np.zeros(t.size, dtype=bool)
    for c in cues:
        in_any_cue |= (t >= c.start) & (t <= c.end)

    baselines: dict[int, float] = {}
    for tid in np.unique(faces["track_id"]):
        m = (faces["track_id"] == tid) & ~in_any_cue
        vals = faces["mouth"][m]
        if vals.size >= 4:
            baselines[int(tid)] = max(float(np.std(_detrend(vals))), 0.02)
        else:
            baselines[int(tid)] = 0.05
    return baselines


def _score_cues(
    cues: list[Cue], faces: dict, env: np.ndarray | None,
) -> list[int | None]:
    """Pick the winning track per cue, or None where no track is convincing.

    Score is mouth motion (relative to that track's own quiet baseline)
    scaled by (0.5 + 0.5 * max(audio correlation, 0)). The correlation term is
    what makes this active *speaker* detection rather than mouth-movement
    detection - without it a listener who happens to be chewing would win as
    readily as the person actually talking.
    """
    t, track_id, mouth, face_size = (
        faces["t"], faces["track_id"], faces["mouth"], faces["face_size"]
    )
    baselines = _track_baselines(faces, cues)

    winners: list[int | None] = []
    for c in cues:
        m_cue = (t >= c.start) & (t <= c.end)
        scores: dict[int, float] = {}
        for tid in np.unique(track_id[m_cue]):
            m = m_cue & (track_id == tid)
            if m.sum() < MIN_TRACK_FRAMES or face_size[m].mean() < MIN_FACE_SIZE:
                continue
            series = mouth[m]
            motion = float(np.std(_detrend(series))) / (baselines[int(tid)] * 3.0 + 1e-6)
            motion = min(motion, 1.0)

            corr = 0.0
            if env is not None and series.size >= 3:
                e = env[m]
                if np.std(series) > 1e-6 and np.std(e) > 1e-6:
                    c_ = float(np.corrcoef(series, e)[0, 1])
                    corr = c_ if np.isfinite(c_) else 0.0

            scores[int(tid)] = motion * (0.5 + 0.5 * max(corr, 0.0))

        if not scores:
            winners.append(None)
            continue
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        top_tid, top = ranked[0]
        runner = ranked[1][1] if len(ranked) > 1 else 0.0
        winners.append(top_tid if top >= SCORE_FLOOR and top - runner >= WINNER_MARGIN else None)

    return winners


def speakers(
    cues: list[Cue], video: Path, work: Path, wav: Path | None = None,
    n_speakers: int | None = None, progress=None,
) -> VisionLabels | None:
    """Label cues by which visible face is talking, or None if video can't answer.

    Every rung of the fallback ladder below returns None rather than a
    partial answer: an anchoring built on one character, or on a handful of
    confident cues, is worse than leaving diarize() to cluster the audio on
    its own, which is designed to handle exactly that case alone.
    """
    if not has_video_stream(video):
        return None
    if not installed():
        print("  video speaker detection: .venv-vision not installed "
              "(run ./scripts/setup.sh --vision); using audio only")
        return None

    try:
        npz = scan_faces(video, cues, work, progress=progress)
    except VisionUnavailable as exc:
        print(f"\n  video speaker detection unavailable: {exc}")
        return None

    faces = load_faces(npz)
    if faces["t"].size == 0:
        print("\n  video speaker detection: no faces found; using audio only")
        return None

    tracks = _characters(faces)
    if len(set(tracks.values())) < MIN_CHARACTERS:
        print("\n  video speaker detection: fewer than 2 characters found; "
              "using audio only")
        return None

    if n_speakers and len(set(tracks.values())) > n_speakers:
        tracks = _merge_to(faces, tracks, n_speakers)

    identity_sil = _identity_silhouette(faces, tracks)
    if identity_sil < MIN_IDENTITY_SILHOUETTE:
        print(f"\n  video speaker detection: identity silhouette "
              f"{identity_sil:.2f} too low to trust the character split "
              "(faces found don't separate cleanly by who they are); "
              "using audio only")
        return None

    env = _audio_envelope(wav, faces["t"]) if wav is not None else None
    winners = _score_cues(cues, faces, env)
    raw_anchors = np.array(
        [tracks[tid] if tid is not None else -1 for tid in winners], dtype=int
    )

    labelled = int((raw_anchors >= 0).sum())
    if labelled < MIN_LABEL_SHARE * len(cues):
        print(f"\n  video speaker detection: only {labelled} of {len(cues)} "
              "cues labelled; using audio only")
        return None

    _, counts = np.unique(raw_anchors[raw_anchors >= 0], return_counts=True)
    if counts.max() / labelled > MAX_ANCHOR_SHARE:
        print(f"\n  video speaker detection: {counts.max()} of {labelled} "
              "labelled cues went to one character; the split is too "
              "lopsided to trust, using audio only")
        return None

    used = sorted(set(v for v in raw_anchors.tolist() if v >= 0))
    remap = {old: new for new, old in enumerate(used)}
    anchors = np.array(
        [remap[a] if a >= 0 else -1 for a in raw_anchors.tolist()], dtype=int
    )

    return VisionLabels(
        anchors=anchors, n_characters=len(used), labelled=labelled,
        silhouette=_identity_silhouette(faces, tracks),
    )
