"""Evaluate the final model on clips recorded by the team (never used for training).

Folder layout (one sign per clip; the folder name is the true sign):

    data/custom_clips/<sign>/<signer>_<number>.mp4      e.g. data/custom_clips/before/ikram_1.mp4

Usage (PowerShell, from the project root, venv active):

    python -m src.eval_custom_clips
    python -m src.eval_custom_clips --checkpoint models\\transformer_final\\final.pt --no-trim

Every clip goes through the same steps as the training data: MediaPipe Holistic on all
frames -> normalization and resampling to 32 frames -> the model. By default each clip is
trimmed to the frames where a hand is visible (+ a few frames of margin), because the
training clips are cut tightly around the sign.

Writes in reports/custom_eval/: predictions.csv (one row per clip), summary.json, and
custom_features_T32.npz (features of every clip, for drift monitoring later).

The accuracy of a few dozen clips is noisy; the script prints a 95% confidence interval.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.metrics import f1_score

from src.extract_landmarks import result_to_frame
from src.live_demo import features_from_clip, load_checkpoint

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


# ------------------------------------------------------------ pure functions
def trim_to_hands(pts: np.ndarray, present: np.ndarray, pad: int = 3):
    """Keep the frames from the first to the last frame with a hand, plus `pad` frames."""
    hands = np.flatnonzero(present[:, 1] | present[:, 2])
    if len(hands) < 2:
        return pts, present
    start, end = max(hands[0] - pad, 0), min(hands[-1] + pad + 1, len(pts))
    if end - start < 2:
        return pts, present
    return pts[start:end], present[start:end]


def wilson_interval(successes: int, n: int, z: float = 1.96):
    """95% confidence interval of a proportion."""
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def summarize(y_true, probs: np.ndarray) -> dict:
    y_true = np.asarray(y_true)
    pred = probs.argmax(axis=1)
    k = min(5, probs.shape[1])
    top5 = np.argsort(-probs, axis=1)[:, :k]
    hits1 = int((pred == y_true).sum())
    hits5 = int(sum(y_true[i] in top5[i] for i in range(len(y_true))))
    low, high = wilson_interval(hits1, len(y_true))
    return {
        "n_clips": int(len(y_true)),
        "top1": hits1 / len(y_true),
        "top1_ci95": [low, high],
        "top5": hits5 / len(y_true),
        "macro_f1": float(f1_score(y_true, pred, labels=np.unique(y_true), average="macro",
                                   zero_division=0)),
    }


# ----------------------------------------------------------------- video I/O
def video_to_landmarks(path: Path, face_idx: list[int], max_side: int, complexity: int):
    """Landmarks of every frame: (pts [N, P, 3], present [N, 4]) or None if unreadable."""
    import mediapipe as mp

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    pts_list, present_list = [], []
    with mp.solutions.holistic.Holistic(
        static_image_mode=False, model_complexity=complexity, smooth_landmarks=True,
        refine_face_landmarks=False, min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as tracker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            factor = max_side / max(h, w)
            if factor < 1:
                frame = cv2.resize(frame, None, fx=factor, fy=factor,
                                   interpolation=cv2.INTER_AREA)
                h, w = frame.shape[:2]
            result = tracker.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            pts, present = result_to_frame(result, w, h, face_idx)
            pts_list.append(pts)
            present_list.append(present)
    cap.release()
    if len(pts_list) < 2:
        return None
    return np.stack(pts_list), np.stack(present_list)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--checkpoint", type=Path,
                    default=Path("models/transformer_final/final.pt"))
    ap.add_argument("--clips", type=Path, default=Path("data/custom_clips"))
    ap.add_argument("--out", type=Path, default=Path("reports/custom_eval"))
    ap.add_argument("--no-trim", action="store_true", help="use the whole clip")
    ap.add_argument("--pad", type=int, default=3, help="frames kept around the hand activity")
    ap.add_argument("--max-side", type=int, default=640)
    ap.add_argument("--complexity", type=int, choices=(0, 1, 2), default=1)
    args = ap.parse_args()

    model, ckpt = load_checkpoint(args.checkpoint)
    classes = list(ckpt["classes"])
    class_id = {c: i for i, c in enumerate(classes)}
    face_idx = list(ckpt["face_idx"])

    files = sorted(p for p in args.clips.rglob("*") if p.suffix.lower() in VIDEO_EXTENSIONS)
    unknown = sorted({p.parent.name for p in files if p.parent.name not in class_id})
    if unknown:
        print(f"skipped folders that are not in the vocabulary: {', '.join(unknown)}")
    files = [p for p in files if p.parent.name in class_id]
    if not files:
        print(f"no clips found in {args.clips} (expected {args.clips}\\<sign>\\<signer>_<n>.mp4)")
        return 1
    print(f"{len(files)} clips, {len({p.parent.name for p in files})} signs")

    rows, feats, y_true, probs_all = [], [], [], []
    for path in files:
        data = video_to_landmarks(path, face_idx, args.max_side, args.complexity)
        if data is None:
            print(f"  cannot read {path.name}, skipped")
            continue
        pts, present = data
        hand_rate = float((present[:, 1] | present[:, 2]).mean())
        n_frames = len(pts)
        if not args.no_trim:
            pts, present = trim_to_hands(pts, present, args.pad)
        try:
            features = features_from_clip(pts, present, ckpt)
        except ValueError as exc:
            print(f"  {path.name}: {exc}, skipped")
            continue
        with torch.no_grad():
            logits = model(torch.tensor(features, dtype=torch.float32).unsqueeze(0))
        probs = torch.softmax(logits, dim=1)[0].numpy()
        true = class_id[path.parent.name]
        pred = int(probs.argmax())
        signer = path.stem.split("_")[0] if "_" in path.stem else "unknown"
        rows.append({
            "clip": str(path), "signer": signer, "true": classes[true], "pred": classes[pred],
            "confidence": round(float(probs[pred]), 3),
            "correct": pred == true,
            "in_top5": true in np.argsort(-probs)[:5],
            "hand_rate": round(hand_rate, 2), "frames": n_frames, "frames_used": len(pts),
        })
        feats.append(features)
        y_true.append(true)
        probs_all.append(probs)
        print(f"  {path.parent.name:>12} -> {classes[pred]:<12} {probs[pred]:.2f} "
              f"{'ok' if pred == true else 'WRONG'}  ({path.name})")

    if not rows:
        print("no clip could be processed")
        return 1
    summary = summarize(y_true, np.stack(probs_all))
    by_signer = defaultdict(list)
    for r in rows:
        by_signer[r["signer"]].append(r["correct"])
    summary["top1_by_signer"] = {k: float(np.mean(v)) for k, v in by_signer.items()}
    summary["trimmed_to_hands"] = not args.no_trim

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    np.savez_compressed(args.out / "custom_features_T32.npz", x=np.stack(feats).astype(np.float32),
                        y=np.array(y_true), clip=np.array([r["clip"] for r in rows]))

    low, high = summary["top1_ci95"]
    print(f"\n=== custom clips ({summary['n_clips']} clips) ===")
    print(f"top-1    {summary['top1']:.1%}   (95% interval {low:.1%} to {high:.1%})")
    print(f"top-5    {summary['top5']:.1%}")
    print(f"macro-F1 {summary['macro_f1']:.3f}")
    for signer, acc in summary["top1_by_signer"].items():
        print(f"  signer {signer}: {acc:.1%} ({len(by_signer[signer])} clips)")
    print(f"saved to {args.out}")
    print("Reference: cross-validation on unseen WLASL signers gave 31.5% top-1.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
