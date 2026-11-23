import numpy as np

from src.live_demo import Recognizer

N_POINTS = 71


def frame(hands: bool):
    pts = np.zeros((N_POINTS, 3), dtype=np.float32)
    present = np.array([True, hands, hands, True])
    return pts, present


def test_window_keeps_only_the_last_seconds():
    rec = Recognizer(model=None, ckpt={}, window_seconds=1.0, min_frames=2, min_hand_frac=0.3)
    for i in range(21):  # 0.0 s ... 2.0 s
        rec.push(i / 10, *frame(hands=True))
    assert len(rec.frames) == 11  # frames from 1.0 s to 2.0 s


def test_no_prediction_when_hands_are_down_or_too_few_frames():
    rec = Recognizer(model=None, ckpt={}, window_seconds=1.0, min_frames=5, min_hand_frac=0.3)
    for i in range(3):
        rec.push(i / 10, *frame(hands=True))
    assert rec.predict() is None  # fewer than min_frames
    rec = Recognizer(model=None, ckpt={}, window_seconds=1.0, min_frames=5, min_hand_frac=0.3)
    for i in range(10):
        rec.push(i / 10, *frame(hands=False))
    assert rec.predict() is None  # nobody is signing
