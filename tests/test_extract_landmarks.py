from types import SimpleNamespace

import numpy as np
import pytest

from src.extract_landmarks import (
    FACE_START,
    L_SHOULDER,
    R_SHOULDER,
    build_sequence,
    result_to_frame,
)

N_FACE = 3
N_POINTS = FACE_START + N_FACE  # 70


def make_clip(n_frames=20, seed=0):
    """Random clip with all parts present and shoulders 100 px apart."""
    rng = np.random.default_rng(seed)
    pts = rng.normal(0, 50, size=(n_frames, N_POINTS, 3)).astype(np.float32)
    pts[:, L_SHOULDER, :] = [300, 200, 0]
    pts[:, R_SHOULDER, :] = [200, 200, 0]
    present = np.ones((n_frames, 4), dtype=bool)
    return pts, present


def test_output_shapes():
    pts, present = make_clip(20)
    x, mask = build_sequence(pts, present, seq_len=32)
    assert x.shape == (32, N_POINTS * 3)
    assert mask.shape == (32, 4)
    assert x.dtype == np.float32
    assert mask.all()


def test_invariant_to_translation_and_scale():
    pts, present = make_clip(20)
    x1, _ = build_sequence(pts, present, 32)
    moved = pts * 2.5 + np.array([120.0, -40.0, 5.0], dtype=np.float32)
    x2, _ = build_sequence(moved, present, 32)
    assert np.allclose(x1, x2, atol=1e-3)


def test_shoulders_are_normalized():
    pts, present = make_clip(20)
    x, _ = build_sequence(pts, present, 32)
    frames = x.reshape(32, N_POINTS, 3)
    width = np.linalg.norm(frames[:, L_SHOULDER, :2] - frames[:, R_SHOULDER, :2], axis=1)
    assert np.allclose(width, 1.0, atol=1e-4)  # shoulder width = 1 after scaling


def test_missing_hand_is_zero_with_false_mask():
    pts, present = make_clip(20)
    present[:, 2] = False  # right hand never seen
    pts[:, 46:67, :] = np.nan
    x, mask = build_sequence(pts, present, 32)
    frames = x.reshape(32, N_POINTS, 3)
    assert not mask[:, 2].any()
    assert np.all(frames[:, 46:67, :] == 0)
    assert not np.isnan(x).any()


def test_gap_in_hand_is_interpolated():
    pts, present = make_clip(21)
    pts[:, 25:46, :] = 0.0
    for t in range(21):  # left hand drifts linearly in x
        pts[t, 25:46, 0] = 300 + 10 * t
    present[5:15, 1] = False
    pts[5:15, 25:46, :] = np.nan
    x, mask = build_sequence(pts, present, 21)
    frames = x.reshape(21, N_POINTS, 3)
    assert not np.isnan(x).any()
    # in the gap the hand continues along the line, so x keeps increasing
    gap_x = frames[5:15, 25, 0]
    assert np.all(np.diff(gap_x) > 0)
    assert not mask[5:15, 1].any()


def test_no_pose_raises():
    pts, present = make_clip(10)
    present[:, 0] = False
    with pytest.raises(ValueError):
        build_sequence(pts, present, 32)


def test_result_to_frame_scales_to_pixels():
    def lms(n, x, y, z):
        return SimpleNamespace(
            landmark=[SimpleNamespace(x=x, y=y, z=z) for _ in range(n)]
        )

    result = SimpleNamespace(
        pose_landmarks=lms(33, 0.5, 0.25, 0.1),
        left_hand_landmarks=None,
        right_hand_landmarks=lms(21, 0.1, 0.2, 0.0),
        face_landmarks=lms(468, 0.3, 0.3, 0.0),
    )
    pts, present = result_to_frame(result, w=640, h=480, face_idx=[1, 2, 3])
    assert pts.shape == (N_POINTS, 3)
    assert present.tolist() == [True, False, True, True]
    assert np.allclose(pts[0], [320, 120, 64])
    assert np.isnan(pts[25:46]).all()  # left hand missing
