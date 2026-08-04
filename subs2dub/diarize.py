"""Assign a speaker to each cue.

Subtitle timings already establish *when* each line is spoken, so unlike general
diarization this only has to determine *who*: embed the audio under each cue and
cluster the embeddings. This also avoids a gated pyannote dependency.

Three things drive accuracy: embed the isolated vocal stem rather than the mix
(music dominates the embedding otherwise), average several sub-windows per cue,
and over-cluster before merging back by centroid similarity.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from . import confidence as conf
from .model import Cue
from .provenance import reuse

EMBED_SR = 16_000
_MIN_SEG = 0.45
_WIN = 2.0
_EMB_DIM = 192

SEPARABLE = 0.40
UNSEPARABLE = 0.20
LOPSIDED = 0.90
SKEWED = 0.75
EMBEDDED_OK = 0.85
EMBEDDED_BAD = 0.60


def _current(derived: Path, source: Path) -> bool:
    """Whether a cached conversion still matches the file it came from.

    Caching by filename alone silently reuses another run's audio: a clip
    render leaves a short stem behind, and the next full render embeds
    against it, so every cue past the clip gets no audio at all.
    """
    try:
        return (derived.exists()
                and derived.stat().st_mtime >= source.stat().st_mtime)
    except OSError:
        return False


def extract_audio(video: Path, out: Path, stream: int = 0) -> Path:
    """Mono 16 kHz WAV - what the embedder wants."""
    out.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(video),
         "-map", f"0:a:{stream}", "-ac", "1", "-ar", str(EMBED_SR),
         "-c:a", "pcm_s16le", str(out)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"audio extract failed: {proc.stderr[-300:]}")
    return out


def to_mono16k(src: Path, out: Path) -> Path:
    """Downmix any stem to what the embedder and pitch tracker expect."""
    out.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(src),
         "-ac", "1", "-ar", str(EMBED_SR), "-c:a", "pcm_s16le", str(out)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"downmix failed: {proc.stderr[-300:]}")
    return out


def _load_encoder(cache: Path):
    from speechbrain.inference.speaker import EncoderClassifier

    return EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(cache / "ecapa"),
        run_opts={"device": "cpu"},
    )


def _windows(start: float, end: float) -> list[tuple[float, float]]:
    """Sub-windows to embed and average over."""
    dur = end - start
    pad = min(0.10, dur * 0.15)
    s, e = start + pad, end - pad
    if e - s <= 0:
        return []
    if e - s <= 1.2:
        return [(s, e)]
    w = min(_WIN, e - s)
    return [(s, s + w), ((s + e - w) / 2, (s + e + w) / 2), (e - w, e)]


def embed_cues(
    cues: list[Cue], wav: Path, cache: Path, progress=None
) -> tuple[np.ndarray, np.ndarray]:
    """Return (embeddings, valid_mask). Invalid rows are zero."""
    import soundfile as sf
    import torch

    encoder = _load_encoder(cache)
    audio, sr = sf.read(wav, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    embs = np.zeros((len(cues), _EMB_DIM), dtype=np.float32)
    valid = np.zeros(len(cues), dtype=bool)

    for i, c in enumerate(cues):
        vecs = []
        for ws, we in _windows(c.start, c.end):
            seg = audio[max(0, int(ws * sr)):min(len(audio), int(we * sr))]
            if seg.size < _MIN_SEG * sr or float(np.abs(seg).max() or 0) < 1e-3:
                continue
            with torch.no_grad():
                v = encoder.encode_batch(
                    torch.from_numpy(seg).unsqueeze(0)
                ).squeeze().cpu().numpy()
            vecs.append(v / (np.linalg.norm(v) + 1e-9))

        if vecs:
            m = np.mean(vecs, axis=0)
            embs[i] = m / (np.linalg.norm(m) + 1e-9)
            valid[i] = True

        if progress and (i + 1) % 25 == 0:
            progress(i + 1, len(cues))

    del encoder
    _release(torch)

    return embs, valid


def _release(torch) -> None:
    """Return cached allocator blocks to the OS."""
    import gc

    gc.collect()
    for backend in (getattr(torch, "mps", None), getattr(torch, "cuda", None)):
        empty = getattr(backend, "empty_cache", None)
        if empty is not None:
            try:
                empty()
            except Exception:
                pass


def _refine(X: np.ndarray, labels: np.ndarray, iters: int = 5) -> np.ndarray:
    """Spherical k-means style refinement: reassign to nearest centroid."""
    for _ in range(iters):
        ids = np.unique(labels)
        cents = np.stack([
            X[labels == k].mean(axis=0) if np.any(labels == k) else X[0]
            for k in ids
        ])
        cents /= np.linalg.norm(cents, axis=1, keepdims=True) + 1e-9
        new = ids[np.argmax(X @ cents.T, axis=1)]
        if np.array_equal(new, labels):
            break
        labels = new
    return labels


def _centroids(X: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ids = np.unique(labels)
    c = np.stack([X[labels == k].mean(axis=0) for k in ids])
    return c / (np.linalg.norm(c, axis=1, keepdims=True) + 1e-9), ids


def _merge_close(
    X: np.ndarray, labels: np.ndarray, sim: float, min_size: int,
    target: int = 0,
) -> np.ndarray:
    """Collapse clusters that are the same voice.

    The initial clustering deliberately over-segments, because splitting one
    character in two is recoverable and lumping two together is not. Merging
    back stops at whichever comes first: the count the separation scores
    actually support, or a pair of centroids too dissimilar to be one person.

    An absolute similarity threshold alone is not enough. How close two
    recordings of the same voice sit depends on the material - short noisy cues
    from a conversation sit far lower than long clean ones - so a fixed
    threshold either splits everyone or merges everyone.
    """
    labels = labels.copy()
    while len(np.unique(labels)) > 1:
        cents, ids = _centroids(X, labels)
        S = cents @ cents.T
        np.fill_diagonal(S, -1.0)
        i, j = np.unravel_index(np.argmax(S), S.shape)
        if S[i, j] < sim and len(ids) <= max(target, 1):
            break
        labels[labels == ids[j]] = ids[i]

    while len(np.unique(labels)) > 1:
        cents, ids = _centroids(X, labels)
        sizes = np.array([(labels == k).sum() for k in ids])
        small = int(np.argmin(sizes))
        if sizes[small] >= min_size:
            break
        S = cents @ cents.T
        np.fill_diagonal(S, -1.0)
        labels[labels == ids[small]] = ids[int(np.argmax(S[small]))]
    return labels


def separability(X: np.ndarray, k: int) -> float:
    """How cleanly the embeddings fall into k groups, 0 to 1.

    Below roughly 0.2 the voices are not separable from the audio at all -
    telephone filtering, shouting and heavy processing all flatten the
    differences the embedder relies on. The count chosen in that regime is
    arbitrary, so it is worth saying so rather than presenting it as a finding.
    """
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score

    if len(X) <= k or k < 2:
        return 0.0
    try:
        lab = AgglomerativeClustering(
            n_clusters=k, metric="cosine", linkage="average"
        ).fit_predict(X)
        return float(silhouette_score(X, lab, metric="cosine"))
    except Exception:
        return 0.0


def label_quality(X: np.ndarray, labels: np.ndarray) -> float:
    """Silhouette score of the clustering actually produced, 0 to 1.

    separability() re-clusters from scratch, so it scores a hypothetical
    partition; this scores the labels cast.tsv will actually carry, which is
    the number a listener discovers the hard way when a split goes wrong.
    """
    from sklearn.metrics import silhouette_score

    if len(set(labels.tolist())) < 2:
        return 0.0
    try:
        return float(silhouette_score(X, labels, metric="cosine"))
    except Exception:
        return 0.0


def largest_share(labels: np.ndarray) -> float:
    """Fraction of all cues carried by the single largest speaker cluster.

    Silhouette alone misses the forced-speaker-count failure: a 57/2 split can
    still score acceptably while being useless, and 57 of 59 lines in one
    cluster is the plainer signal that the split did not happen.
    """
    if len(labels) == 0:
        return 0.0
    _, counts = np.unique(labels, return_counts=True)
    return float(counts.max() / len(labels))


def _auto_k(X: np.ndarray, kmax: int) -> int:
    """Pick a speaker count by silhouette score over cosine distance."""
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score

    kmax = int(max(2, min(kmax, len(X) // 3)))
    if kmax < 2:
        return 1

    best_k, best_s = 2, -1.0
    for k in range(2, kmax + 1):
        lab = AgglomerativeClustering(
            n_clusters=k, metric="cosine", linkage="average"
        ).fit_predict(X)
        if len(np.unique(lab)) < 2:
            continue
        try:
            s = silhouette_score(X, lab, metric="cosine")
        except Exception:
            continue
        if s > best_s:
            best_k, best_s = k, s
    return best_k


def cluster_speakers(
    embs: np.ndarray,
    valid: np.ndarray,
    n_speakers: int | None = None,
    max_speakers: int = 20,
    merge_sim: float = 0.55,
    min_lines: int = 2,
) -> np.ndarray:
    """Cluster embeddings into speaker ids. Invalid cues inherit a neighbour.

    Deliberately over-clusters first, then merges back by centroid similarity.
    Splitting one character in two is easy to fix that way; a greedy pass that
    lumps two characters together is not recoverable.
    """
    from sklearn.cluster import AgglomerativeClustering

    labels = np.full(len(embs), -1, dtype=int)
    idx = np.flatnonzero(valid)
    if idx.size == 0:
        return np.zeros(len(embs), dtype=int)
    if idx.size <= 2:
        labels[idx] = 0
    else:
        X = embs[idx]
        if n_speakers:
            k = max(1, min(n_speakers, len(X)))
        else:
            target = _auto_k(X, max_speakers)
            k = max(2, min(target + 2, max_speakers, len(X) - 1))
        lab = (
            np.zeros(len(X), dtype=int)
            if k <= 1
            else AgglomerativeClustering(
                n_clusters=k, metric="cosine", linkage="average"
            ).fit_predict(X)
        )
        lab = _refine(X, lab)
        if not n_speakers:
            lab = _merge_close(X, lab, merge_sim, min_lines, target)
            lab = _refine(X, lab)
        labels[idx] = lab

    for i in np.flatnonzero(labels < 0):
        near = idx[np.argmin(np.abs(idx - i))]
        labels[i] = labels[near]

    remap = {old: new for new, old in enumerate(sorted(set(labels.tolist())))}
    return np.array([remap[v] for v in labels], dtype=int)


def cue_f0(cues: list[Cue], wav: Path) -> np.ndarray:
    """Median pitch for every cue, NaN where none could be measured."""
    import librosa
    import soundfile as sf

    audio, sr = sf.read(wav, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    out = np.full(len(cues), np.nan, dtype=float)
    for i, c in enumerate(cues):
        seg = audio[int(c.start * sr):int(min(c.end, c.start + 4.0) * sr)]
        if seg.size < 0.4 * sr or float(np.abs(seg).max() or 0) < 5e-3:
            continue
        try:
            f0, voiced, _ = librosa.pyin(
                seg, fmin=75, fmax=400, sr=sr, frame_length=1024
            )
            f0 = f0[np.isfinite(f0) & voiced] if voiced is not None else f0
            f0 = f0[np.isfinite(f0)]
            if f0.size >= 8:
                out[i] = float(np.median(f0))
        except Exception:
            continue
    return out


def cluster_by_pitch(
    labels: np.ndarray, f0: np.ndarray, n_speakers: int | None = None,
    apart: float = 90.0,
) -> np.ndarray:
    """Group cues by pitch when the voice embeddings cannot separate them.

    Embeddings fail on processed audio, and on material where one performer
    plays several characters they are answering the wrong question: the voices
    really are the same person. Pitch is a far weaker identity signal in
    general, but when a scene holds a low voice and a high one it separates
    them outright where the embeddings do not separate them at all.

    Only used when the embeddings have already been measured as unreliable, and
    only when the pitches genuinely fall into two groups far apart. Otherwise
    the existing labels stand.
    """
    usable = np.flatnonzero(np.isfinite(f0))
    if usable.size < 6:
        return labels
    values = np.sort(f0[usable])

    best, cut = 0.0, None
    for k in range(2, values.size - 1):
        low, high = values[:k], values[k:]
        gap = float(np.median(high) - np.median(low))
        spread = max((low.std() + high.std()) / 2.0, 15.0)
        score = gap / spread
        if gap >= apart and score > best:
            best, cut = score, (values[k - 1] + values[k]) / 2.0

    if cut is None or best < 1.2:
        return labels
    if n_speakers and n_speakers < 2:
        return labels

    out = labels.copy()
    out[usable] = (f0[usable] > cut).astype(int)
    return out


def split_by_pitch(
    labels: np.ndarray, f0: np.ndarray, low: float = 140.0, high: float = 178.0,
    min_side: int = 2,
) -> np.ndarray:
    """Separate a cluster that holds both a man and a woman.

    Embeddings occasionally merge two voices, and when those voices differ in
    gender it is the worst error available: half the lines are delivered in the
    wrong register. Pitch is measured independently and settles that particular
    question reliably.

    Deliberately narrow. Only clusters with members well clear of the boundary
    on both sides are split, because within one register pitch says little that
    the embeddings do not say better - one person ranges widely across a scene,
    and two people of the same gender can sit almost on top of each other.
    """
    labels = labels.copy()
    for lab in sorted(set(labels.tolist())):
        idx = np.flatnonzero((labels == lab) & np.isfinite(f0))
        if idx.size < min_side * 2:
            continue
        below = idx[f0[idx] < low]
        above = idx[f0[idx] > high]
        if below.size < min_side or above.size < min_side:
            continue
        if below.size + above.size < idx.size * 0.7:
            continue
        boundary = (low + high) / 2.0
        moving = idx[f0[idx] > boundary]
        if moving.size and moving.size < idx.size:
            labels[moving] = max(labels) + 1
    return labels


def median_f0(
    cues: list[Cue], wav: Path, labels, per_speaker: int = 15
) -> dict:
    """Median pitch per speaker, keyed by whatever label the cues carry.

    Meaningful only on an isolated vocal stem. `diarize()` passes integer
    cluster ids; the --cast path passes the string speaker labels cues already
    hold, so this needs no coercion of its own.
    """
    import librosa
    import soundfile as sf

    audio, sr = sf.read(wav, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    pitches: dict = {}
    for c, lab in zip(cues, labels):
        if len(pitches.get(lab, [])) >= per_speaker:
            continue
        seg = audio[int(c.start * sr):int(c.end * sr)]
        if seg.size < 0.4 * sr or float(np.abs(seg).max() or 0) < 5e-3:
            continue
        try:
            f0, voiced, _ = librosa.pyin(
                seg, fmin=75, fmax=400, sr=sr, frame_length=1024
            )
            f0 = f0[np.isfinite(f0) & voiced] if voiced is not None else f0
            f0 = f0[np.isfinite(f0)]
            if f0.size >= 8:
                pitches.setdefault(lab, []).append(float(np.median(f0)))
        except Exception:
            continue
    return {k: float(np.median(v)) for k, v in pitches.items() if v}


def analysis_wav(
    video: Path, work: Path, audio_stream: int = 0, vocals: Path | None = None,
) -> Path:
    """The 16 kHz mono wav that embedding and pitch tracking both work from.

    The vocal stem is preferred whenever it exists, because pitch measured
    over music describes the scene rather than the speaker.
    """
    if vocals is not None:
        wav = vocals.with_name("vocals16k.wav")
        return reuse(work, wav, lambda: to_mono16k(vocals, wav), source=vocals)
    wav = work / f"{video.stem}-{audio_stream}-16k.wav"
    return reuse(work, wav, lambda: extract_audio(video, wav, audio_stream),
                  source=video, stream=audio_stream)


def _assign_from_anchors(
    embs: np.ndarray, valid: np.ndarray, anchors: np.ndarray,
) -> np.ndarray:
    """Label every cue from a small set of video-anchored characters.

    Anchored cues keep their video label outright. Every other cue is asked
    which anchored character's audio centroid it sits closest to - a nearest-
    of-two-known-things question, not a how-many-groups-are-there one, and it
    stays answerable on audio that has no usable structure of its own for
    blind clustering. A cue with no usable embedding, or with no centroid
    close enough to trust, falls back to whichever labelled cue sits nearest
    in time - the same rule cluster_speakers() uses for invalid rows.
    """
    labels = np.full(len(anchors), -1, dtype=int)
    anchored = np.flatnonzero(anchors >= 0)
    labels[anchored] = anchors[anchored]

    centroids: dict[int, np.ndarray] = {}
    for cid in sorted(set(anchors[anchored].tolist())):
        idx = anchored[(anchors[anchored] == cid) & valid[anchored]]
        if idx.size:
            v = embs[idx].mean(axis=0)
            centroids[cid] = v / (np.linalg.norm(v) + 1e-9)

    for i in np.flatnonzero(labels < 0):
        if valid[i] and centroids:
            ranked = sorted(
                ((cid, float(embs[i] @ c)) for cid, c in centroids.items()),
                key=lambda kv: -kv[1],
            )
            best_cid, best = ranked[0]
            runner = ranked[1][1] if len(ranked) > 1 else -1.0
            if len(ranked) == 1 or best - runner >= 0.05:
                labels[i] = best_cid

    still_open = np.flatnonzero(labels < 0)
    if still_open.size:
        known = np.flatnonzero(labels >= 0)
        for i in still_open:
            labels[i] = (
                labels[known[np.argmin(np.abs(known - i))]] if known.size else 0
            )

    remap = {old: new for new, old in enumerate(sorted(set(labels.tolist())))}
    return np.array([remap[v] for v in labels], dtype=int)


def diarize(
    cues: list[Cue],
    video: Path,
    work: Path,
    *,
    n_speakers: int | None = None,
    max_speakers: int = 20,
    merge_sim: float = 0.55,
    min_lines: int = 2,
    audio_stream: int = 0,
    vocals: Path | None = None,
    progress=None,
    checks: list | None = None,
    anchors: np.ndarray | None = None,
) -> dict[int, float]:
    """Label every cue with a speaker id. Returns median F0 per speaker.

    Pass `vocals` (a Demucs stem) whenever available - clustering the mix is
    what produced the six-speakers-for-two-people failure. Pass `checks` to
    have this append its confidence in the speaker split and in the embedding
    coverage; a forced --speakers count is checked too, because forcing a
    count does not guarantee the audio actually supports it.

    Pass `anchors` (from vision.speakers()) when video has already answered
    who is speaking for some cues: instead of clustering the audio from
    scratch, every other cue is asked which anchored character its embedding
    sits closest to. split_by_pitch is skipped in that case, since video has
    already answered the question it exists to answer. Every other path here
    is unchanged when `anchors` is None.
    """
    wav = analysis_wav(video, work, audio_stream, vocals)

    embs, valid = embed_cues(cues, wav, work / "models", progress=progress)
    if anchors is not None and (anchors >= 0).any():
        labels = _assign_from_anchors(embs, valid, anchors)
    else:
        labels = cluster_speakers(
            embs, valid, n_speakers, max_speakers, merge_sim, min_lines
        )
        f0 = cue_f0(cues, wav)
        if separability(embs[valid], len(set(labels[valid].tolist()))) < 0.20:
            labels = cluster_by_pitch(labels, f0, n_speakers)
        if not n_speakers:
            labels = split_by_pitch(labels, f0)
    for c, lab in zip(cues, labels):
        c.speaker = f"S{int(lab):02d}"

    if checks is not None:
        checks.append(_speaker_check(embs, valid, labels, n_speakers))
        checks.append(_embedding_check(valid, vocals))

    return {int(k): v for k, v in median_f0(cues, wav, labels).items()}


def _speaker_check(
    embs: np.ndarray, valid: np.ndarray, labels: np.ndarray, n_speakers: int | None,
) -> conf.Check:
    """How much to trust the speaker split just produced."""
    quality = label_quality(embs[valid], labels[valid])
    share = largest_share(labels)
    n_labels = len(set(labels.tolist()))

    if quality < UNSEPARABLE or (share > LOPSIDED and n_labels > 1):
        level = conf.UNRELIABLE
    elif quality < SEPARABLE or share > SKEWED:
        level = conf.WEAK
    else:
        level = conf.OK

    detail = (
        f"silhouette {quality:.2f} of 1.0 at {n_labels} speakers, "
        f"{int(round(share * len(labels)))} of {len(labels)} lines in one cluster"
    )
    if n_speakers is None:
        remedy = (
            "pass --speakers N if you know how many voices there are, or "
            "--cast FILE to assign speakers by hand, or --no-diarize for one voice"
        )
    else:
        remedy = (
            "--speakers N did not separate them; edit cast.tsv and pass it back "
            "with --cast FILE, raise --merge-sim above 0.55, or use --no-diarize "
            "for one voice"
        )
    return conf.Check(
        stage="speakers", level=level, detail=detail, remedy=remedy, score=quality,
    )


def _embedding_check(valid: np.ndarray, vocals: Path | None) -> conf.Check:
    """How much of the speaker labelling rests on a measured embedding.

    Cues with no usable embedding inherit a neighbour's label silently in
    cluster_speakers(), so this share is exactly the share of speaker labels
    that were guessed rather than measured.
    """
    score = float(valid.mean()) if len(valid) else 0.0
    level = conf.band(score, EMBEDDED_OK, EMBEDDED_BAD)
    detail = f"{int(valid.sum())} of {len(valid)} cues had usable speech in their window"
    remedy = (
        "drop --no-separate so cues are embedded from the isolated vocal stem"
        if vocals is None else
        "check --audio-stream N - these cues had no measurable speech where the "
        "subtitles say there was some"
    )
    return conf.Check(
        stage="embeddings", level=level, detail=detail, remedy=remedy, score=score,
    )
