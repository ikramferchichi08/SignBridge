"""Record test clips from the webcam, one sign per clip.

Usage (PowerShell, from the project root, venv active):

    python -m src.record_clips --sign before --signer ikram --count 5

Controls: SPACE starts and stops a recording, q quits.
Clips are saved as data/custom_clips/<sign>/<signer>_<number>.mp4 (not mirrored).

How to record a good clip
  - Good light, face the camera, upper body and both hands in view.
  - Start with your hands down, press SPACE, do the sign once (about 1-3 s), lower your
    hands, press SPACE again.
  - Record the exact ASL sign of the word, with different takes (distance, speed).
  - Only record people who agreed to it, and do not publish the clips.

The preview is mirrored to feel natural, but the saved video is not mirrored: the model was
trained on unmirrored videos.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2


def next_index(folder: Path, signer: str) -> int:
    existing = [p for p in folder.glob(f"{signer}_*.mp4")]
    return len(existing) + 1


def save_clip(frames: list, duration: float, path: Path) -> float:
    fps = min(max(len(frames) / duration, 10.0), 60.0) if duration > 0 else 30.0
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame in frames:
        writer.write(frame)
    writer.release()
    return fps


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--sign", required=True, help="the word being signed (folder name)")
    ap.add_argument("--signer", required=True, help="who signs, e.g. ikram or malak")
    ap.add_argument("--count", type=int, default=5, help="number of clips to record")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("data/custom_clips"))
    ap.add_argument("--classes", type=Path, default=Path("data/landmarks/wlasl/classes.json"))
    args = ap.parse_args()

    if args.classes.exists():
        vocabulary = json.loads(args.classes.read_text(encoding="utf-8"))
        if args.sign not in vocabulary:
            raise SystemExit(f"'{args.sign}' is not one of the 100 signs. See {args.classes}")

    folder = args.out / args.sign
    folder.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("cannot open the camera")

    saved, recording, frames, started = 0, False, [], 0.0
    font = cv2.FONT_HERSHEY_SIMPLEX
    while saved < args.count:
        ok, frame = cap.read()
        if not ok:
            break
        if recording:
            frames.append(frame.copy())
        view = cv2.flip(frame, 1)
        status = "RECORDING - SPACE to stop" if recording else "SPACE to start"
        color = (0, 0, 255) if recording else (0, 255, 0)
        cv2.putText(view, f"{args.sign} | {args.signer} | clip {saved + 1}/{args.count}",
                    (10, 28), font, 0.7, (255, 255, 255), 2)
        cv2.putText(view, status, (10, 58), font, 0.7, color, 2)
        cv2.imshow("record clips", view)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            if not recording:
                recording, frames, started = True, [], time.perf_counter()
            else:
                recording = False
                duration = time.perf_counter() - started
                if len(frames) < 10:
                    print("clip too short, discarded")
                    continue
                path = folder / f"{args.signer}_{next_index(folder, args.signer)}.mp4"
                fps = save_clip(frames, duration, path)
                saved += 1
                print(f"saved {path} ({len(frames)} frames, {duration:.1f}s, {fps:.0f} fps)")
    cap.release()
    cv2.destroyAllWindows()
    print(f"{saved} clip(s) saved in {folder}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
