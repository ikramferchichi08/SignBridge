"""Extract normalized MediaPipe Holistic landmark sequences from the WLASL videos.

Usage (PowerShell, from the project root, venv active):

    # quick test on 10 videos
    python src/extract_landmarks.py --limit 10 --workers 2

    # full extraction (resumable: existing files are skipped)
    python src/extract_landmarks.py --workers 4

    # Big Data benchmark: same 40 videos with 1, 2, 4 workers
    python src/extract_landmarks.py --benchmark 1 2 4 --limit 40

For every video listed in data/raw/wlasl/metadata.csv the script
  1. runs MediaPipe Holistic on every frame (frames are downscaled to --max-side px;
     landmarks are normalized so this does not change them),
  2. keeps 25 upper-body pose points, 21 + 21 hand points and a face subset (lips + eyebrows),
  3. normalizes each frame: origin = shoulder midpoint, scale = median shoulder width,
  4. fills missing detections by interpolation and resamples to --seq-len frames,
  5. saves data/landmarks/wlasl/<sign>/<video_id>.npz  (x: [T, F] float32, mask: [T, 4] bool),
  6. builds data/landmarks/wlasl/dataset_T<seq_len>.npz with all sequences, ready for training.

mask columns: pose, left_hand, right_hand, face (True = detected in the source frame).
Missing parts are filled with 0 in x, so always look at the mask.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import yaml

# --- feature layout ---------------------------------------------------------
N_POSE, N_HAND = 25, 21  # pose landmarks 0-24 = head, torso, arms, hips
POSE = slice(0, N_POSE)
LEFT_HAND = slice(N_POSE, N_POSE + N_HAND)
RIGHT_HAND = slice(N_POSE + N_HAND, N_POSE + 2 * N_HAND)
FACE_START = N_POSE + 2 * N_HAND
L_SHOULDER, R_SHOULDER = 11, 12
PARTS = ("pose", "left_hand", "right_hand", "face")


def part_slices(n_face: int) -> list[slice]:
    return [POSE, LEFT_HAND, RIGHT_HAND, slice(FACE_START, FACE_START + n_face)]


def get_face_indices(mode: str) -> list[int]:
    """Face-mesh points kept as features: lips and eyebrows (60 points)."""
    if mode == "none":
        return []
    import mediapipe as mp

    fm = mp.solutions.face_mesh
    edges = (
        set(fm.FACEMESH_LIPS)
        | set(fm.FACEMESH_LEFT_EYEBROW)
        | set(fm.FACEMESH_RIGHT_EYEBROW)
    )
    return sorted({i for edge in edges for i in edge})


# --- pure numpy part (unit-tested, no MediaPipe needed) ---------------------
def result_to_frame(result, w: int, h: int, face_idx: list[int]):
    """Convert one Holistic result to (points [P, 3] with NaN if missing, present [4])."""
    n_points = FACE_START + len(face_idx)
    pts = np.full((n_points, 3), np.nan, dtype=np.float32)
    present = np.zeros(4, dtype=bool)

    def to_array(landmarks) -> np.ndarray:
        # x, y in pixels (keeps the true aspect ratio), z on the same scale as x
        return np.array(
            [[lm.x * w, lm.y * h, lm.z * w] for lm in landmarks.landmark],
            dtype=np.float32,
        )

    if result.pose_landmarks:
        pts[POSE] = to_array(result.pose_landmarks)[:N_POSE]
        present[0] = True
    if result.left_hand_landmarks:
        pts[LEFT_HAND] = to_array(result.left_hand_landmarks)
        present[1] = True
    if result.right_hand_landmarks:
        pts[RIGHT_HAND] = to_array(result.right_hand_landmarks)
        present[2] = True
    if face_idx and result.face_landmarks:
        pts[FACE_START:] = to_array(result.face_landmarks)[face_idx]
        present[3] = True
    return pts, present


def build_sequence(pts: np.ndarray, present: np.ndarray, seq_len: int):
    """Normalize and resample one clip.

    pts:     [N, P, 3] raw landmarks (NaN where a part is missing)
    present: [N, 4]    which parts were detected in each frame
    returns: x [seq_len, P * 3] float32, mask [seq_len, 4] bool
    """
    n_frames, n_points, _ = pts.shape
    n_face = n_points - FACE_START
    if n_frames < 2:
        raise ValueError("need at least 2 frames")
    if not present[:, 0].any():
        raise ValueError("no pose detected in the whole clip")

    # 1) normalization anchors: shoulder midpoint (per frame) and shoulder width (median)
    left, right = pts[:, L_SHOULDER, :], pts[:, R_SHOULDER, :]
    center = (left + right) / 2.0
    width = np.linalg.norm((left - right)[:, :2], axis=1)
    valid = present[:, 0] & np.isfinite(width) & (width > 1e-6)
    if not valid.any():
        raise ValueError("shoulders are degenerate")
    median_center = np.median(center[valid], axis=0)
    scale = float(np.median(width[valid]))
    center = np.where(valid[:, None], center, median_center)  # frames without pose
    normalized = (pts - center[:, None, :]) / scale

    # 2) temporal resampling: interpolate each part over the frames where it was seen
    positions = np.linspace(0, n_frames - 1, seq_len)
    nearest = np.rint(positions).astype(int)
    out = np.zeros((seq_len, n_points, 3), dtype=np.float32)
    mask = np.zeros((seq_len, 4), dtype=bool)
    frame_ids = np.arange(n_frames)
    for k, sl in enumerate(part_slices(n_face)):
        seen = present[:, k]
        mask[:, k] = seen[nearest]
        if not seen.any() or sl.stop == sl.start:
            continue  # part never detected: stays 0, mask False
        block = normalized[seen][:, sl, :].reshape(int(seen.sum()), -1)
        filled = np.empty((seq_len, block.shape[1]), dtype=np.float32)
        for col in range(block.shape[1]):
            filled[:, col] = np.interp(positions, frame_ids[seen], block[:, col])
        out[:, sl, :] = filled.reshape(seq_len, sl.stop - sl.start, 3)
    return out.reshape(seq_len, -1), mask


# --- video processing (runs in worker processes) ----------------------------
_CFG: dict = {}


def _init_worker(cfg: dict) -> None:
    cv2.setNumThreads(1)
    _CFG.update(cfg)


def extract_one(job: dict) -> dict:
    import mediapipe as mp

    cfg = _CFG
    started = time.perf_counter()
    rec = {
        "video_id": job["video_id"], "label": job["label"], "status": "ok",
        "detail": "", "n_frames": 0, "pose_rate": 0.0, "left_rate": 0.0,
        "right_rate": 0.0, "any_hand_rate": 0.0, "seconds": 0.0,
    }
    try:
        cap = cv2.VideoCapture(job["src"])
        if not cap.isOpened():
            raise RuntimeError("cannot open video")
        pts_list, present_list = [], []
        with mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=cfg["complexity"],
            smooth_landmarks=True,
            refine_face_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as model:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                h, w = frame.shape[:2]
                factor = cfg["max_side"] / max(h, w)
                if factor < 1:
                    frame = cv2.resize(frame, None, fx=factor, fy=factor,
                                       interpolation=cv2.INTER_AREA)
                    h, w = frame.shape[:2]
                result = model.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                pts, present = result_to_frame(result, w, h, cfg["face_idx"])
                pts_list.append(pts)
                present_list.append(present)
        cap.release()

        pts_all, present_all = np.stack(pts_list), np.stack(present_list)
        rec["n_frames"] = len(pts_list)
        rec["pose_rate"] = float(present_all[:, 0].mean())
        rec["left_rate"] = float(present_all[:, 1].mean())
        rec["right_rate"] = float(present_all[:, 2].mean())
        rec["any_hand_rate"] = float((present_all[:, 1] | present_all[:, 2]).mean())

        x, mask = build_sequence(pts_all, present_all, cfg["seq_len"])
        dst = Path(job["dst"])
        dst.parent.mkdir(parents=True, exist_ok=True)
        np.savez(dst, x=x, mask=mask)
    except Exception as exc:  # noqa: BLE001 - log every failure, keep going
        rec["status"] = "failed"
        rec["detail"] = str(exc).splitlines()[0][:150] if str(exc) else type(exc).__name__
    rec["seconds"] = round(time.perf_counter() - started, 3)
    return rec


def run_jobs(jobs: list[dict], workers: int, cfg: dict):
    """Run extraction jobs; return (results, elapsed seconds)."""
    started = time.perf_counter()
    results: list[dict] = []
    if not jobs:
        return results, 0.0
    with Pool(processes=workers, initializer=_init_worker, initargs=(cfg,)) as pool:
        for i, rec in enumerate(pool.imap_unordered(extract_one, jobs), 1):
            results.append(rec)
            if i % 25 == 0 or i == len(jobs):
                ok = sum(r["status"] == "ok" for r in results)
                print(f"  {i}/{len(jobs)} processed, {ok} ok", flush=True)
    return results, time.perf_counter() - started


# --- dataset assembly -------------------------------------------------------
def build_dataset(rows: list[dict], out_dir: Path, seq_len: int) -> Path | None:
    xs, masks, keep = [], [], []
    for row in rows:
        f = out_dir / row["label"] / f"{Path(row['video_path']).stem}.npz"
        if f.exists():
            with np.load(f) as data:
                xs.append(data["x"])
                masks.append(data["mask"])
            keep.append(row)
    if not keep:
        return None
    classes = sorted({r["label"] for r in keep})
    class_to_id = {c: i for i, c in enumerate(classes)}
    (out_dir / "classes.json").write_text(json.dumps(classes, indent=1), encoding="utf-8")
    path = out_dir / f"dataset_T{seq_len}.npz"
    np.savez_compressed(
        path,
        x=np.stack(xs).astype(np.float32),
        mask=np.stack(masks),
        y=np.array([class_to_id[r["label"]] for r in keep], dtype=np.int64),
        signer_id=np.array([int(r["signer_id"]) for r in keep], dtype=np.int64),
        video_id=np.array([Path(r["video_path"]).stem for r in keep]),
        original_split=np.array([r["original_split"] for r in keep]),
    )
    return path


def load_defaults(params_path: Path) -> dict:
    if params_path.exists():
        return yaml.safe_load(params_path.read_text(encoding="utf-8")) or {}
    return {}


def main() -> int:
    params = load_defaults(Path("configs/params.yaml"))
    ext = params.get("extraction", {})
    cpu = __import__("os").cpu_count() or 2

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--metadata", type=Path, default=Path("data/raw/wlasl/metadata.csv"))
    ap.add_argument("--out", type=Path, default=Path("data/landmarks/wlasl"))
    ap.add_argument("--seq-len", type=int, default=params.get("sequence_length", 32))
    ap.add_argument("--workers", type=int, default=min(4, cpu))
    ap.add_argument("--max-side", type=int, default=ext.get("max_side", 640))
    ap.add_argument("--complexity", type=int, choices=(0, 1, 2),
                    default=ext.get("model_complexity", 1))
    ap.add_argument("--face", choices=("subset", "none"), default=ext.get("face", "subset"))
    ap.add_argument("--limit", type=int, default=0, help="process only N videos (test)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--benchmark", type=int, nargs="+",
                    help="worker counts to compare, e.g. --benchmark 1 2 4")
    ap.add_argument("--log", type=Path, default=Path("reports/landmark_extraction_log.csv"))
    args = ap.parse_args()

    with args.metadata.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    face_idx = get_face_indices(args.face)
    cfg = {"seq_len": args.seq_len, "max_side": args.max_side,
           "complexity": args.complexity, "face_idx": face_idx}
    n_points = FACE_START + len(face_idx)
    print(f"{len(rows)} videos | {n_points} landmarks/frame | "
          f"{n_points * 3} features | sequence length {args.seq_len}")

    def make_jobs(base: Path, subset: list[dict], skip_existing: bool) -> list[dict]:
        jobs = []
        for r in subset:
            vid = Path(r["video_path"]).stem
            dst = base / r["label"] / f"{vid}.npz"
            if skip_existing and dst.exists():
                continue
            jobs.append({"src": r["video_path"], "label": r["label"],
                         "video_id": vid, "dst": str(dst)})
        return jobs

    # ---- benchmark mode ------------------------------------------------------
    if args.benchmark:
        subset = rows[: args.limit] if args.limit else rows[:40]
        table, base_time = [], None
        for n in args.benchmark:
            tmp = args.out.parent / "_bench" / f"w{n}"
            shutil.rmtree(tmp, ignore_errors=True)
            print(f"\nBenchmark: {n} worker(s), {len(subset)} videos")
            results, secs = run_jobs(make_jobs(tmp, subset, False), n, cfg)
            ok = sum(r["status"] == "ok" for r in results)
            base_time = base_time or secs
            table.append({"workers": n, "videos": ok, "seconds": round(secs, 1),
                          "videos_per_sec": round(ok / secs, 3) if secs else 0,
                          "speedup": round(base_time / secs, 2) if secs else 0,
                          "efficiency": round(base_time / secs / n, 2) if secs else 0})
            shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(args.out.parent / "_bench", ignore_errors=True)
        out_csv = Path("reports/extraction_benchmark.csv")
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=list(table[0]))
            wr.writeheader()
            wr.writerows(table)
        print("\nworkers  videos  seconds  videos/s  speedup  efficiency")
        for t in table:
            print(f"{t['workers']:>7}  {t['videos']:>6}  {t['seconds']:>7}  "
                  f"{t['videos_per_sec']:>8}  {t['speedup']:>7}  {t['efficiency']:>10}")
        print(f"saved to {out_csv}")
        return 0

    # ---- normal mode ---------------------------------------------------------
    subset = rows[: args.limit] if args.limit else rows
    jobs = make_jobs(args.out, subset, not args.overwrite)
    print(f"to process: {len(jobs)} (already done: {len(subset) - len(jobs)}) "
          f"with {args.workers} worker(s)")
    args.out.mkdir(parents=True, exist_ok=True)
    results, secs = run_jobs(jobs, args.workers, cfg)

    if results:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        with args.log.open("w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=list(results[0]))
            wr.writeheader()
            wr.writerows(results)
        ok = [r for r in results if r["status"] == "ok"]
        print(f"\n{len(ok)} ok, {len(results) - len(ok)} failed in {secs:.0f}s "
              f"({len(results) / secs:.2f} videos/s) | log: {args.log}")
        if ok:
            for key, name in (("pose_rate", "pose"), ("any_hand_rate", "at least one hand"),
                              ("left_rate", "left hand"), ("right_rate", "right hand")):
                print(f"  frames with {name}: {np.mean([r[key] for r in ok]):.0%}")

    (args.out / "feature_spec.json").write_text(json.dumps({
        "seq_len": args.seq_len, "n_points": n_points, "n_features": n_points * 3,
        "layout": {"pose": [0, N_POSE], "left_hand": [N_POSE, N_POSE + N_HAND],
                   "right_hand": [N_POSE + N_HAND, FACE_START],
                   "face": [FACE_START, n_points]},
        "face_mesh_indices": face_idx,
        "normalization": "origin = shoulder midpoint; scale = median shoulder width; "
                         "x,y in pixels, z on x scale; missing parts = 0 (see mask)",
    }, indent=1), encoding="utf-8")

    dataset = build_dataset(rows, args.out, args.seq_len)
    if dataset:
        with np.load(dataset) as d:
            print(f"\ndataset: {dataset}  x={d['x'].shape}  signers={len(set(d['signer_id']))}  "
                  f"signs={len(set(d['y']))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
