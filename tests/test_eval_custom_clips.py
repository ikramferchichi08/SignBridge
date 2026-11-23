import numpy as np

from src.eval_custom_clips import summarize, trim_to_hands, wilson_interval


def make(n=20, hands=(5, 14)):
    pts = np.zeros((n, 10, 3), dtype=np.float32)
    present = np.zeros((n, 4), dtype=bool)
    present[:, 0] = True
    present[hands[0]:hands[1] + 1, 2] = True  # right hand from frame 5 to 14
    return pts, present


def test_trim_keeps_hand_frames_plus_margin():
    pts, present = make()
    out_pts, out_present = trim_to_hands(pts, present, pad=3)
    assert len(out_pts) == 16  # frames 2..17
    assert out_present[3:-3, 2].all()


def test_trim_is_limited_by_clip_borders_and_keeps_clips_without_hands():
    pts, present = make(hands=(0, 19))
    assert len(trim_to_hands(pts, present, pad=3)[0]) == 20
    pts, present = make()
    present[:, 1:3] = False
    assert len(trim_to_hands(pts, present)[0]) == 20


def test_summarize_perfect_and_partial():
    probs = np.eye(4)[[0, 1, 2, 3]] * 0.9 + 0.025
    perfect = summarize([0, 1, 2, 3], probs)
    assert perfect["top1"] == 1.0 and perfect["macro_f1"] == 1.0
    half = summarize([0, 1, 3, 2], probs)
    assert half["top1"] == 0.5 and half["top5"] == 1.0
    assert half["top1_ci95"][0] < 0.5 < half["top1_ci95"][1]


def test_wilson_interval_is_wide_for_few_clips():
    low, high = wilson_interval(5, 10)
    assert 0.2 < low < 0.3 and 0.7 < high < 0.8
    low, high = wilson_interval(500, 1000)
    assert high - low < 0.07
