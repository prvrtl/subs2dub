"""Face tracking and identity worker. Runs under the .venv-vision virtualenv.

mediapipe and opencv-python both pull numpy 2.x, which conflicts with the
numpy<2 pin the main pipeline needs for its own dependencies, so this stays a
separate venv driven over a pipe, in the same shape as the other
Speech workers.

One job per line of stdin:

    {"cmd": "scan", "video": "/path/in.mkv", "windows": [[12.4, 15.1], ...],
     "out": "/path/faces.npz"}

`windows` are the cue start/end times cast.tsv would carry; frames inside one,
padded by WINDOW_PAD, get full-cost inference on every sampled frame, frames
outside get it on every SPARSE_STRIDE'th sampled frame. That keeps identity
coverage and per-track mouth baselines everywhere without paying decode-and-
infer cost for a whole film at full density.

The decode is one ordered ffmpeg pass at a fixed sample rate rather than a seek
per cue: FaceLandmarker in VIDEO mode requires monotonically increasing
timestamps, long-GOP footage makes per-cue seeking both slow and inaccurate,
and decode is cheap next to inference anyway.

Output is a single .npz, one row per (sampled frame, detected face), written
with allow_pickle=False - that flag is the entire boundary between this venv's
numpy 2 and the main venv's numpy 1.
"""

from __future__ import annotations

import json
import subprocess
import sys
import traceback

_CHANNEL = sys.stdout
sys.stdout = sys.stderr

WINDOW_PAD = 0.25
SPARSE_STRIDE = 6
IOU_MATCH = 0.30
IDENTITY_MATCH = 0.20
TRACK_GAP = 0.40
SHOT_DIFF = 18.0

RIGHT_EYE, LEFT_EYE, NOSE, MOUTH_R, MOUTH_L = 473, 468, 1, 291, 61
LIP_UPPER, LIP_LOWER = 13, 14


def _emit(obj: dict) -> None:
    _CHANNEL.write(json.dumps(obj) + "\n")
    _CHANNEL.flush()


def _probe_duration(video: str) -> float:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", video],
        capture_output=True, text=True,
    )
    try:
        return float(proc.stdout.strip())
    except ValueError:
        return 0.0


def _frame_source(video: str, w: int, h: int, fps: float):
    """Yield RGB frames as (H, W, 3) uint8 arrays, one ordered decode pass."""
    import numpy as np

    nbytes = w * h * 3
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", video,
         "-vf", f"fps={fps},scale={w}:{h}:force_original_aspect_ratio=decrease,"
                f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2",
         "-pix_fmt", "rgb24", "-f", "rawvideo", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    try:
        while True:
            buf = proc.stdout.read(nbytes)
            if len(buf) < nbytes:
                break
            yield np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)
    finally:
        proc.stdout.close()
        proc.wait(timeout=10)


def _iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class _Tracker:
    """Assigns a track id to each detection, breaking on gaps and cuts.

    mediapipe hands back faces in an unspecified order with no track ids of
    its own, so continuity is entirely reconstructed here from bounding-box
    overlap between consecutive processed frames, with identity similarity as
    both a hard gate and the tiebreak when two active tracks are both
    plausible by IoU alone. Every threshold below is biased towards breaking
    a track rather than bridging one: an over-split track is repaired later
    by identity clustering in vision.py, but a track that bridges a shot cut
    fuses two different people and cannot be recovered downstream.

    IoU alone is not enough of a gate: a cut between two similarly framed
    shots of different people - a close-up of one phone caller followed by a
    same-sized close-up of the other, say - lands a new face in almost the
    same box as the old one despite being someone else entirely, and the
    frame-difference shot detector below is not reliable enough to catch
    every such cut. Requiring the SFace vectors to agree at least loosely
    before continuing a track is what actually stops that fusion; IoU alone
    was measured to bridge exactly this case.
    """

    def __init__(self) -> None:
        self._next_id = 0
        self._active: dict[int, dict] = {}

    def reset(self) -> None:
        self._active = {}

    def assign(self, bbox, t: float, vec) -> int:
        import numpy as np

        candidates = []
        for tid, tr in self._active.items():
            if t - tr["t"] > TRACK_GAP:
                continue
            iou = _iou(bbox, tr["bbox"])
            if iou < IOU_MATCH:
                continue
            sim = float(np.dot(vec, tr["vec"]))
            if sim < IDENTITY_MATCH:
                continue
            candidates.append((tid, iou, sim))

        tid = None
        if candidates:
            best_iou = max(iou for _, iou, _ in candidates)
            near = [c for c in candidates if c[1] >= best_iou - 0.05]
            tid = max(near, key=lambda c: c[2])[0]

        if tid is None:
            tid = self._next_id
            self._next_id += 1

        self._active[tid] = {"bbox": bbox, "t": t, "vec": vec}
        return tid


def _load_models(cfg: dict):
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision as mpv
    import cv2

    landmarker = mpv.FaceLandmarker.create_from_options(mpv.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=cfg["landmarker"]),
        running_mode=mpv.RunningMode.VIDEO,
        num_faces=int(cfg.get("max_faces", 3)),
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        min_face_detection_confidence=0.6,
        min_face_presence_confidence=0.6,
    ))
    sface = cv2.FaceRecognizerSF.create(cfg["sface"], "")
    return mp, landmarker, sface, cv2


def _detect(mp, landmarker, frame, ts_ms: int):
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    return landmarker.detect_for_video(img, ts_ms)


def _scan(job: dict, cfg: dict, mp, landmarker, sface, cv2) -> dict:
    import numpy as np

    video = job["video"]
    windows = [(float(s) - WINDOW_PAD, float(e) + WINDOW_PAD)
               for s, e in job.get("windows", [])]
    w = int(cfg.get("width", 960))
    h = int(cfg.get("height", 540))
    fps = float(cfg.get("fps", 12.5))

    duration = _probe_duration(video)
    total = max(1, int(duration * fps) + 1)

    tracker = _Tracker()
    prev_gray = None
    frames_seen = 0

    rows: dict[str, list] = {
        "track_id": [], "t": [], "bbox": [], "face_size": [],
        "frontality": [], "sface": [], "mouth": [],
    }

    for i, frame in enumerate(_frame_source(video, w, h, fps)):
        frames_seen = i + 1
        t = i / fps
        in_window = any(lo <= t <= hi for lo, hi in windows)
        do_infer = in_window or (i % SPARSE_STRIDE == 0)

        small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), (32, 18))
        if prev_gray is not None:
            diff = float(np.abs(small.astype(np.float32)
                                 - prev_gray.astype(np.float32)).mean())
            if diff > SHOT_DIFF:
                tracker.reset()
        prev_gray = small

        if do_infer:
            ts_ms = int(round(t * 1000.0))
            res = _detect(mp, landmarker, frame, ts_ms)
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            for face_i, marks in enumerate(res.face_landmarks):
                xs = np.array([p.x for p in marks]) * w
                ys = np.array([p.y for p in marks]) * h
                x0, y0, x1, y1 = float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())
                if x1 <= x0 or y1 <= y0:
                    continue

                pts = [x0, y0, x1 - x0, y1 - y0]
                for j in (RIGHT_EYE, LEFT_EYE, NOSE, MOUTH_R, MOUTH_L):
                    pts += [float(xs[j]), float(ys[j])]
                det = np.array([pts + [1.0]], dtype=np.float32)
                try:
                    crop = sface.alignCrop(bgr, det)
                    vec = sface.feature(crop).flatten().astype(np.float32)
                except cv2.error:
                    continue
                vec = vec / (np.linalg.norm(vec) + 1e-9)

                eye_dist = float(np.hypot(xs[RIGHT_EYE] - xs[LEFT_EYE],
                                           ys[RIGHT_EYE] - ys[LEFT_EYE]))
                gap = float(ys[LIP_LOWER] - ys[LIP_UPPER])
                gap_norm = gap / eye_dist if eye_dist > 1e-6 else 0.0

                blend = {b.category_name: b.score
                         for b in (res.face_blendshapes[face_i]
                                   if face_i < len(res.face_blendshapes) else [])}
                openness = (blend.get("jawOpen", 0.0) + blend.get("mouthFunnel", 0.0)
                            + blend.get("mouthPucker", 0.0) - blend.get("mouthClose", 0.0))
                mouth = 0.5 * min(max(gap_norm / 0.15, 0.0), 1.5) + 0.5 * min(max(openness, 0.0), 1.5)

                frontality = 0.5
                if face_i < len(res.facial_transformation_matrixes or []):
                    m = res.facial_transformation_matrixes[face_i]
                    frontality = float(min(max(m[2][2], 0.0), 1.0))

                bbox_n = (x0 / w, y0 / h, x1 / w, y1 / h)
                face_size = float(np.sqrt(max(0.0, (x1 - x0) * (y1 - y0))) / np.sqrt(w * h))
                tid = tracker.assign(bbox_n, t, vec)

                rows["track_id"].append(tid)
                rows["t"].append(t)
                rows["bbox"].append(bbox_n)
                rows["face_size"].append(face_size)
                rows["frontality"].append(frontality)
                rows["sface"].append(vec)
                rows["mouth"].append(mouth)

        if i % 50 == 0:
            _emit({"progress": i, "total": total})

    n = len(rows["t"])
    np_kwargs = {
        "track_id": np.array(rows["track_id"], dtype=np.int32),
        "t": np.array(rows["t"], dtype=np.float32),
        "bbox": np.array(rows["bbox"], dtype=np.float32).reshape(n, 4),
        "face_size": np.array(rows["face_size"], dtype=np.float32),
        "frontality": np.array(rows["frontality"], dtype=np.float32),
        "sface": np.array(rows["sface"], dtype=np.float32).reshape(n, 128),
        "mouth": np.array(rows["mouth"], dtype=np.float32),
        "fps": np.array([fps], dtype=np.float32),
    }
    np.savez(job["out"], allow_pickle=False, **np_kwargs)
    return {"ok": True, "out": job["out"], "rows": n, "frames": frames_seen}


def main() -> int:
    cfg = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    try:
        mp, landmarker, sface, cv2 = _load_models(cfg)
    except Exception as exc:
        _emit({"ready": False, "err": str(exc), "trace": traceback.format_exc()[-1800:]})
        return 1

    _emit({"ready": True})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            job = json.loads(line)
        except Exception as exc:
            _emit({"ok": False, "err": f"bad job: {exc}"})
            continue
        if job.get("cmd") == "quit":
            return 0

        try:
            _emit(_scan(job, cfg, mp, landmarker, sface, cv2))
        except Exception as exc:
            _emit({
                "ok": False,
                "err": f"{exc}",
                "trace": traceback.format_exc()[-1800:],
            })

    return 0


if __name__ == "__main__":
    sys.exit(main())
