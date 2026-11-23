import json

import numpy as np

from src.extract_landmarks import FACE_START, build_sequence
from src.live_demo import features_from_clip
from src.train_gru import load_data

N_POINTS = FACE_START + 4
T = 32


def make_clip(rng):
    pts = rng.normal(0, 50, size=(20, N_POINTS, 3)).astype(np.float32)
    pts[:, 11] = [300, 200, 0]
    pts[:, 12] = [200, 200, 0]
    present = np.ones((20, 4), dtype=bool)
    present[:, 2] = rng.random(20) > 0.3  # right hand missing in some frames
    pts[~present[:, 2], 46:67, :] = np.nan
    return pts, present


def test_live_features_match_training_features(tmp_path):
    rng = np.random.default_rng(0)
    clips = [make_clip(rng) for _ in range(3)]
    built = [build_sequence(pts, present, T) for pts, present in clips]
    np.savez(
        tmp_path / "dataset_T32.npz",
        x=np.stack([b[0] for b in built]),
        mask=np.stack([b[1] for b in built]),
        y=np.zeros(3, dtype=np.int64),
        signer_id=np.arange(3),
        video_id=np.array(["a", "b", "c"]),
        original_split=np.array(["train"] * 3),
    )
    (tmp_path / "classes.json").write_text(json.dumps(["s0"]))

    data = load_data(tmp_path / "dataset_T32.npz", 0.2, delta=True, hand_local=True)
    n_features = data["X"].shape[-1]
    ckpt = {"seq_len": T, "hand_local": True, "delta": True,
            "mu": np.zeros(n_features), "sd": np.ones(n_features)}
    for i, (pts, present) in enumerate(clips):
        live = features_from_clip(pts, present, ckpt)
        assert live.shape == (T, n_features)
        assert np.allclose(live, data["X"][i], atol=1e-5)
