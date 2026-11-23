"""Live sign recognition demo: sliding window + temporal smoothing + running transcript.

Usage (PowerShell, from the project root, venv active):

    # 1) train the final model once (on all clips) and save the checkpoint
    python -m src.train_gru --final --model transformer --hand-local --run-name transformer_final

    # 2) webcam (q = quit, c = clear the transcript)
    python -m src.live_demo --checkpoint models\\transformer_final\\final.pt

    # 3) a video file, without a window (prints the transcript and speed statistics)
    python -m src.live_demo --checkpoint models\\transformer_final\\final.pt ^
        --video path\\to\\clip.mp4 --no-display

How it works
  webcam frame -> MediaPipe Holistic landmarks -> sliding buffer of the last --window-seconds
  seconds -> every --stride frames: if hands are active, normalize + resample to 32 frames,
  run the model -> WordSmoother (average, confidence threshold, hold, cooldown) -> transcript.
  The window is measured in seconds, not frames, so a slow laptop (few FPS) still shows the
  model about the same amount of motion as in the training clips (1-3 s per sign).

Limits (be open about them)
  - The model was trained on short clips that contain exactly one sign. In a live stream the
    window also contains rest frames and parts of other signs, so accuracy is lower than the
    cross-validation result.
  - "Is somebody signing?" is a simple heuristic: enough frames with a detected hand.
  - Only the 100 signs of the vocabulary can be recognized (see classes.json). The model has
    no "unknown" class: random gestures still produce one of the 100 words.
"""
from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from src.extract_landmarks import build_sequence, result_to_frame
from src.smoothing import WordSmoother
from src.train_gru import assemble_features, build_model, hand_local_features


def load_checkpoint(path: Path):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    h = ckpt["hparams"]
    spec = SimpleNamespace(model=ckpt["model"], hidden=h["hidden"], layers=h["layers"],
                           heads=h["heads"], dropout=h["dropout"],
                           bidirectional=h["bidirectional"])
    model = build_model(spec, ckpt["in_dim"], ckpt["n_classes"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def features_from_clip(pts: np.ndarray, present: np.ndarray, ckpt: dict) -> np.ndarray:
    """Raw landmarks of a window -> standardized model input [seq_len, features].

    Uses the same functions as training (build_sequence, hand_local_features,
    assemble_features), so live and training features match.
    """
    x, mask = build_sequence(pts, present, int(ckpt["seq_len"]))
    if ckpt["hand_local"]:
        n_frames, n_points = x.shape[0], pts.shape[1]
        x = hand_local_features(x.reshape(1, n_frames, n_points, 3))[0].reshape(n_frames, -1)
    feats = assemble_features(x, mask, bool(ckpt["delta"]))
    return (feats - np.asarray(ckpt["mu"])) / np.asarray(ckpt["sd"])


class Recognizer:
    """Keeps the frames of the last `window_seconds` and predicts class probabilities."""

    def __init__(self, model, ckpt: dict, window_seconds: float, min_frames: int,
                 min_hand_frac: float):
        self.model, self.ckpt = model, ckpt
        self.window_seconds = window_seconds
        self.frames: deque = deque()  # (time, landmarks, detection flags)
        self.min_frames, self.min_hand_frac = min_frames, min_hand_frac

    def push(self, t: float, pts: np.ndarray, present: np.ndarray) -> None:
        self.frames.append((t, pts, present))
        while t - self.frames[0][0] > self.window_seconds:
            self.frames.popleft()

    @torch.no_grad()
    def predict(self):
        """Class probabilities, or None when nobody seems to be signing."""
        if len(self.frames) < self.min_frames:
            return None
        present = np.stack([f[2] for f in self.frames])
        if (present[:, 1] | present[:, 2]).mean() < self.min_hand_frac:
            return None  # hands down / out of view
        pts = np.stack([f[1] for f in self.frames])
        try:
            feats = features_from_clip(pts, present, self.ckpt)
        except ValueError:  # no pose in the window
            return None
        x = torch.tensor(feats, dtype=torch.float32).unsqueeze(0)
        return torch.softmax(self.model(x), dim=1)[0].numpy()


def draw_overlay(frame, label: str, confidence: float, transcript: list[str], fps: float):
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.rectangle(frame, (0, 0), (w, 66), (0, 0, 0), -1)
    text = f"Sign: {label} ({confidence:.0%})" if label else "Sign: ..."
    cv2.putText(frame, text, (10, 26), font, 0.75, (255, 255, 255), 2)
    cv2.putText(frame, "Transcript: " + " ".join(transcript[-6:]), (10, 54), font, 0.65,
                (0, 255, 255), 2)
    cv2.putText(frame, f"{fps:.0f} FPS", (w - 110, 26), font, 0.65, (0, 255, 0), 2)
    cv2.putText(frame, "Assistive demo - not a substitute for interpreters", (10, h - 12),
                font, 0.5, (200, 200, 200), 1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--checkpoint", type=Path,
                    default=Path("models/transformer_final/final.pt"))
    ap.add_argument("--video", help="video file instead of the webcam")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--window-seconds", type=float, default=1.8,
                    help="length of the sliding window in seconds")
    ap.add_argument("--stride", type=int, default=5, help="predict every N frames")
    ap.add_argument("--min-frames", type=int, default=8)
    ap.add_argument("--min-hand-frac", type=float, default=0.3,
                    help="share of window frames with a hand needed to predict")
    ap.add_argument("--avg", type=int, default=5, help="predictions averaged")
    ap.add_argument("--hold", type=int, default=3, help="consecutive agreeing predictions")
    ap.add_argument("--threshold", type=float, default=0.35, help="minimum confidence")
    ap.add_argument("--cooldown", type=float, default=1.5, help="seconds between words")
    ap.add_argument("--max-side", type=int, default=640)
    ap.add_argument("--complexity", type=int, choices=(0, 1, 2), default=1)
    ap.add_argument("--no-display", action="store_true")
    ap.add_argument("--save-transcript", type=Path)
    args = ap.parse_args()

    import mediapipe as mp  # imported here so that the unit tests do not need it

    model, ckpt = load_checkpoint(args.checkpoint)
    classes, face_idx = ckpt["classes"], list(ckpt["face_idx"])
    recognizer = Recognizer(model, ckpt, args.window_seconds, args.min_frames,
                            args.min_hand_frac)
    smoother = WordSmoother(args.avg, args.hold, args.threshold, args.cooldown)

    cap = cv2.VideoCapture(args.video if args.video else args.camera)
    if not cap.isOpened():
        raise SystemExit("cannot open the video source")
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    holistic = mp.solutions.holistic
    drawing = mp.solutions.drawing_utils

    transcript: list[str] = []
    label, confidence = "", 0.0
    frame_idx, inference_ms = 0, []
    started = time.perf_counter()
    try:
        with holistic.Holistic(static_image_mode=False, model_complexity=args.complexity,
                               smooth_landmarks=True, refine_face_landmarks=False,
                               min_detection_confidence=0.5,
                               min_tracking_confidence=0.5) as tracker:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                h, w = frame.shape[:2]
                factor = args.max_side / max(h, w)
                if factor < 1:
                    frame = cv2.resize(frame, None, fx=factor, fy=factor,
                                       interpolation=cv2.INTER_AREA)
                    h, w = frame.shape[:2]
                result = tracker.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                pts, present = result_to_frame(result, w, h, face_idx)
                frame_idx += 1
                now = frame_idx / source_fps if args.video else time.perf_counter()
                recognizer.push(now, pts, present)

                if frame_idx % args.stride == 0:
                    t0 = time.perf_counter()
                    probs = recognizer.predict()
                    if probs is not None:
                        inference_ms.append((time.perf_counter() - t0) * 1000)
                    top, confidence, accepted = smoother.update(probs, now)
                    label = classes[top] if top is not None else ""
                    if accepted is not None:
                        transcript.append(classes[accepted])
                        stamp = now if args.video else now - started
                        print(f"[{stamp:6.1f}s] {classes[accepted]}", flush=True)

                if not args.no_display:
                    for lms, conn in ((result.pose_landmarks, holistic.POSE_CONNECTIONS),
                                      (result.left_hand_landmarks, holistic.HAND_CONNECTIONS),
                                      (result.right_hand_landmarks, holistic.HAND_CONNECTIONS)):
                        drawing.draw_landmarks(frame, lms, conn)
                    elapsed = time.perf_counter() - started
                    draw_overlay(frame, label, confidence, transcript, frame_idx / elapsed)
                    try:
                        cv2.imshow("SignBridge", frame)
                    except cv2.error:
                        raise SystemExit(
                            "This OpenCV build cannot open windows (headless build installed).\n"
                            "Fix: python -m pip uninstall -y opencv-python opencv-python-headless "
                            "opencv-contrib-python opencv-contrib-python-headless, then "
                            "python -m pip install opencv-contrib-python==4.11.0.86 --no-deps.\n"
                            "Or run with --no-display."
                        )
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    if key == ord("c"):
                        transcript.clear()
    except KeyboardInterrupt:
        print("\nstopped with Ctrl+C")
    cap.release()
    if not args.no_display:
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass

    elapsed = time.perf_counter() - started
    print(f"\n{frame_idx} frames in {elapsed:.1f}s -> {frame_idx / elapsed:.1f} FPS "
          "(includes tracking, prediction and display)")
    if inference_ms:
        print(f"model + features per prediction: {np.median(inference_ms):.1f} ms (median of "
              f"{len(inference_ms)} predictions)")
    print("transcript:", " ".join(transcript) if transcript else "(no word accepted)")
    if args.save_transcript:
        args.save_transcript.write_text(" ".join(transcript), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
