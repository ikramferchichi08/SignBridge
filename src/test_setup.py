"""Verify the SignBridge environment and optionally process video."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import mlflow
import torch
import mediapipe as mp

from src.utils.mlflow_utils import start_configured_run

drawing_utils = mp.solutions.drawing_utils
holistic = mp.solutions.holistic


def print_versions() -> None:
    print(f"Python: {sys.version.split()[0]}")
    print(f"torch: {torch.__version__}")
    print(f"mediapipe: {__import__('mediapipe').__version__}")
    print(f"opencv: {cv2.__version__}")
    print(f"mlflow: {mlflow.__version__}")
    print(f"dvc: {__import__('dvc').__version__}")


def log_dummy_run() -> None:
    with start_configured_run() as run:
        mlflow.log_param("setup_check", "true")
        mlflow.log_param("setup_version", "step-1")
        for step, score in enumerate((0.25, 0.5, 0.75), start=1):
            mlflow.log_metric("setup_score", score, step=step)
        artifact = Path("setup_artifact.txt")
        artifact.write_text("SignBridge setup verification\n", encoding="utf-8")
        mlflow.log_artifact(str(artifact))
        artifact.unlink()
        print(f"MLflow dummy run: {run.info.run_id}")


def process_video(source: str | int, display: bool, max_frames: int | None) -> None:
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    frame_count = 0
    detected_hands = 0
    detected_pose = 0
    started = time.perf_counter()
    with holistic.Holistic(static_image_mode=False) as model:
        while max_frames is None or frame_count < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            frame_count += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = model.process(rgb)
            if result.left_hand_landmarks or result.right_hand_landmarks:
                detected_hands += 1
            if result.pose_landmarks:
                detected_pose += 1
            if display:
                if result.pose_landmarks:
                    drawing_utils.draw_landmarks(
                        frame, result.pose_landmarks, holistic.POSE_CONNECTIONS
                    )
                for hand in (result.left_hand_landmarks, result.right_hand_landmarks):
                    if hand:
                        drawing_utils.draw_landmarks(frame, hand, holistic.HAND_CONNECTIONS)
                cv2.imshow("SignBridge setup test", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    capture.release()
    cv2.destroyAllWindows()
    elapsed = time.perf_counter() - started
    fps = frame_count / elapsed if elapsed else 0.0
    print(
        f"Processed {frame_count} frames; hands detected in {detected_hands}; "
        f"pose detected in {detected_pose}; average FPS: {fps:.2f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, help="Path to a video file.")
    parser.add_argument("--no-display", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print_versions()
    log_dummy_run()
    source: str | int = str(args.video) if args.video else 0
    try:
        process_video(source, display=not args.no_display, max_frames=30 if args.no_display else None)
    except RuntimeError as error:
        print(f"Video check skipped: {error}", file=sys.stderr)
        return 0 if args.video is None else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
