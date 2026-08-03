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

from .model import Cue

EMBED_SR = 16_000
_MIN_SEG = 0.45
_WIN = 2.0
_EMB_DIM = 192


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
    X: np.ndarray, labels: np.ndarray, sim: float, min_size: int
) -> np.ndarray:
    """Collapse clusters that are the same voice.

    Silhouette-style scores reward splitting, so the initial clustering
    over-segments. Speaker verification gives a principled way back: two
    clusters whose centroids are more similar than `sim` are one person.
    Tiny clusters are then absorbed - a character with one line is far more
    likely to be a misassignment than a real role.
    """
    labels = labels.copy()
    while len(np.unique(labels)) > 1:
        cents, ids = _centroids(X, labels)
        S = cents @ cents.T
        np.fill_diagonal(S, -1.0)
        i, j = np.unravel_index(np.argmax(S), S.shape)
        if S[i, j] < sim:
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
            k = max(_auto_k(X, max_speakers), min(max_speakers, len(X) // 4))
            k = max(2, min(k, len(X) - 1))
        lab = (
            np.zeros(len(X), dtype=int)
            if k <= 1
            else AgglomerativeClustering(
                n_clusters=k, metric="cosine", linkage="average"
            ).fit_predict(X)
        )
        lab = _refine(X, lab)
        if not n_speakers:
            lab = _merge_close(X, lab, merge_sim, min_lines)
            lab = _refine(X, lab)
        labels[idx] = lab

    for i in np.flatnonzero(labels < 0):
        near = idx[np.argmin(np.abs(idx - i))]
        labels[i] = labels[near]

    remap = {old: new for new, old in enumerate(sorted(set(labels.tolist())))}
    return np.array([remap[v] for v in labels], dtype=int)


def median_f0(
    cues: list[Cue], wav: Path, labels: np.ndarray, per_speaker: int = 15
) -> dict[int, float]:
    """Median pitch per speaker. Meaningful only on an isolated vocal stem."""
    import librosa
    import soundfile as sf

    audio, sr = sf.read(wav, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    pitches: dict[int, list[float]] = {}
    for c, lab in zip(cues, labels):
        lab = int(lab)
        if len(pitches.get(lab, [])) >= per_speaker:
            continue
        seg = audio[int(c.start * sr):int(c.end * sr)]
        if seg.size < 0.4 * sr or float(np.abs(seg).max() or 0) < 5e-3:
            continue
        try:
            f0, voiced, _ = librosa.pyin(
                seg, fmin=60, fmax=400, sr=sr, frame_length=1024
            )
            f0 = f0[np.isfinite(f0) & voiced] if voiced is not None else f0
            f0 = f0[np.isfinite(f0)]
            if f0.size >= 8:
                pitches.setdefault(lab, []).append(float(np.median(f0)))
        except Exception:
            continue
    return {k: float(np.median(v)) for k, v in pitches.items() if v}


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
) -> dict[int, float]:
    """Label every cue with a speaker id. Returns median F0 per speaker.

    Pass `vocals` (a Demucs stem) whenever available - clustering the mix is
    what produced the six-speakers-for-two-people failure.
    """
    if vocals is not None:
        wav = vocals.with_name("vocals16k.wav")
        if not _current(wav, vocals):
            to_mono16k(vocals, wav)
    else:
        wav = work / f"{video.stem}-{audio_stream}-16k.wav"
        if not _current(wav, video):
            extract_audio(video, wav, audio_stream)

    embs, valid = embed_cues(cues, wav, work / "models", progress=progress)
    labels = cluster_speakers(
        embs, valid, n_speakers, max_speakers, merge_sim, min_lines
    )
    for c, lab in zip(cues, labels):
        c.speaker = f"S{int(lab):02d}"
    return {int(k): v for k, v in median_f0(cues, wav, labels).items()}
