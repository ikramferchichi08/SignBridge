import json

import numpy as np
import torch

from src.train_gru import (
    FACE_START,
    LEFT_HAND,
    RIGHT_HAND,
    CNNLSTMClassifier,
    GRUClassifier,
    TransformerClassifier,
    augment,
    dominant_is_left,
    fold_metrics,
    hand_energy,
    hand_local_features,
    load_data,
    make_folds,
    mirror_clips,
    signer_weights,
    standardize,
)

N_SIGNS, N_SIGNERS, PER_SIGN, T, F = 6, 12, 8, 32, 10


def make_npz(tmp_path, n_feat=F):
    rng = np.random.default_rng(0)
    n = N_SIGNS * PER_SIGN
    y = np.repeat(np.arange(N_SIGNS), PER_SIGN)
    signer = np.array([(i * 5 + j) % N_SIGNERS for i in range(N_SIGNS) for j in range(PER_SIGN)])
    mask = np.ones((n, T, 4), dtype=bool)
    mask[0, :, 1:3] = False  # clip 0 has no hand at all
    np.savez(
        tmp_path / "dataset_T32.npz",
        x=rng.normal(size=(n, T, n_feat)).astype(np.float32),
        mask=mask,
        y=y,
        signer_id=signer,
        video_id=np.array([f"v{i}" for i in range(n)]),
        original_split=np.array(["train"] * n),
    )
    (tmp_path / "classes.json").write_text(json.dumps([f"s{i}" for i in range(N_SIGNS)]))
    return tmp_path / "dataset_T32.npz"


def test_load_data_drops_clips_without_hands(tmp_path):
    data = load_data(make_npz(tmp_path), min_hand_rate=0.2, delta=False)
    assert data["n_dropped"] == 1
    assert data["X"].shape == (N_SIGNS * PER_SIGN - 1, T, F + 4)  # + 4 mask flags
    delta = load_data(make_npz(tmp_path), min_hand_rate=0.2, delta=True)
    assert delta["X"].shape[-1] == 2 * F + 4


def test_folds_never_share_signers(tmp_path):
    data = load_data(make_npz(tmp_path), 0.2, False)
    folds = make_folds(data["y"], data["signer"], n_folds=4, seed=1)
    assert len(folds) == 4
    all_test = np.concatenate([te for _, te in folds])
    assert sorted(all_test) == list(range(len(data["y"])))  # every clip tested once
    for tr, te in folds:
        assert not set(data["signer"][tr]) & set(data["signer"][te])


def test_signer_weights_downweight_frequent_signers():
    signers = np.array([1] * 9 + [2])
    w = signer_weights(signers)
    assert abs(w.mean() - 1.0) < 1e-6
    assert w[0] < w[-1]  # signer 1 has many videos, so a smaller weight


def test_standardize_uses_train_statistics():
    rng = np.random.default_rng(0)
    train = rng.normal(5, 2, size=(20, T, F)).astype(np.float32)
    test = rng.normal(5, 2, size=(5, T, F)).astype(np.float32)
    tr, te = standardize(train, test)
    assert abs(tr.reshape(-1, F).mean()) < 1e-4
    assert te.shape == test.shape


def test_fold_metrics_perfect_and_wrong():
    y = np.array([0, 1, 2, 2])
    logits = np.eye(3)[y] * 5
    perfect = fold_metrics(logits, y)
    assert perfect["top1"] == 1.0 and perfect["macro_f1"] == 1.0
    wrong = fold_metrics(np.roll(logits, 1, axis=1), y)
    assert wrong["top1"] == 0.0


def test_model_and_augment_shapes():
    x = torch.randn(4, T, F + 4)
    for bidirectional in (False, True):
        model = GRUClassifier(F + 4, N_SIGNS, hidden=16, layers=2, bidirectional=bidirectional)
        assert model(x).shape == (4, N_SIGNS)
    assert augment(x, noise=0.05, max_shift=3, frame_drop=0.1).shape == x.shape


# ------------------------------------------------ canonical hand / hand-local
N_FACE = 4
P = FACE_START + N_FACE


def make_points(n=2, t=8, seed=0):
    rng = np.random.default_rng(seed)
    pts = rng.normal(size=(n, t, P, 3)).astype(np.float32)
    return pts, np.ones((n, t, 4), dtype=bool)


def test_mirror_twice_is_identity():
    pts, mask = make_points(3)
    mask[1, :, 1] = False  # one clip without a left hand
    pts[1, :, LEFT_HAND] = 0
    face_perm = np.array([1, 0, 3, 2])
    everything = np.ones(3, dtype=bool)
    once, mask1 = mirror_clips(pts, mask, face_perm, everything)
    twice, mask2 = mirror_clips(once, mask1, face_perm, everything)
    assert not np.array_equal(once, pts)
    assert np.array_equal(twice, pts)
    assert np.array_equal(mask2, mask)
    assert not mask1[1, :, 2].any()  # the missing left hand became a missing right hand


def test_dominant_hand_detection_and_mirroring():
    pts, mask = make_points(2)
    pts[:, :, LEFT_HAND] = 0.0
    pts[:, :, RIGHT_HAND] = 0.0
    for t in range(8):
        pts[0, t, LEFT_HAND, 0] += 0.2 * t   # clip 0: left hand moves
        pts[1, t, RIGHT_HAND, 0] += 0.2 * t  # clip 1: right hand moves
    select = dominant_is_left(pts, mask)
    assert select.tolist() == [True, False]
    mirrored, new_mask = mirror_clips(pts, mask, np.arange(N_FACE), select)
    energy_l, _ = hand_energy(mirrored, new_mask, 1, LEFT_HAND)
    energy_r, _ = hand_energy(mirrored, new_mask, 2, RIGHT_HAND)
    assert np.all(energy_r > energy_l)  # now the right hand is dominant in both clips


def test_hand_local_is_invariant_to_hand_position_and_size():
    pts, _ = make_points(1)
    moved = pts.copy()
    moved[:, :, RIGHT_HAND, :] = pts[:, :, RIGHT_HAND, :] * 3.0 + np.array([5.0, -2.0, 1.0])
    a = hand_local_features(pts)[:, :, RIGHT_HAND, :]
    b = hand_local_features(moved)[:, :, RIGHT_HAND, :]
    assert np.allclose(a[:, :, 1:], b[:, :, 1:], atol=1e-3)  # shape is the same
    assert np.allclose(b[:, :, 0], moved[:, :, RIGHT_HAND.start, :])  # wrist keeps its position


def test_hand_local_keeps_missing_hand_at_zero():
    pts, _ = make_points(1)
    pts[:, :, LEFT_HAND] = 0.0
    out = hand_local_features(pts)
    assert np.all(out[:, :, LEFT_HAND] == 0)


def test_load_data_with_canonical_hand_and_hand_local(tmp_path):
    n_feat = P * 3
    path = make_npz(tmp_path, n_feat=n_feat)
    plain = load_data(path, 0.2, False)
    flagged = load_data(path, 0.2, False, canonical_hand=True, hand_local=True)
    assert flagged["X"].shape == plain["X"].shape == (N_SIGNS * PER_SIGN - 1, T, n_feat + 4)
    assert 0 <= flagged["n_mirrored"] <= len(flagged["y"])
    assert not np.isnan(flagged["X"]).any()


def test_transformer_and_cnn_lstm_shapes():
    x = torch.randn(4, T, F + 4)
    transformer = TransformerClassifier(F + 4, N_SIGNS, d_model=16, layers=2, heads=4)
    assert transformer(x).shape == (4, N_SIGNS)
    for bidirectional in (False, True):
        cnn_lstm = CNNLSTMClassifier(F + 4, N_SIGNS, hidden=16, bidirectional=bidirectional)
        assert cnn_lstm(x).shape == (4, N_SIGNS)
    transformer.eval()
    assert transformer(x[:1]).shape == (1, N_SIGNS)  # batch of one (latency measurement)
