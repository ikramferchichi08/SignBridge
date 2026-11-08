"""GRU / Transformer / CNN+LSTM on WLASL landmark sequences, evaluated on unseen signers.

Usage (PowerShell, from the project root, venv active):

    # quick test: 1 fold, 5 epochs
    python -m src.train_gru --limit-folds 1 --epochs 5

    # full baseline: 5 signer-grouped folds, every fold logged to MLflow
    python -m src.train_gru

    # variants to compare later
    python -m src.train_gru --bidirectional --run-name gru_bidirectional
    python -m src.train_gru --delta --run-name gru_delta
    python -m src.train_gru --hand-local --run-name gru_handlocal
    python -m src.train_gru --canonical-hand --run-name gru_canonical
    python -m src.train_gru --canonical-hand --hand-local --bidirectional --delta \
        --run-name gru_all

    # other model families, same folds and metrics
    python -m src.train_gru --model transformer --hand-local --run-name transformer_handlocal
    python -m src.train_gru --model cnn_lstm --hand-local --run-name cnn_lstm_handlocal

Protocol
- Input: data/landmarks/wlasl/dataset_T32.npz (made by src/extract_landmarks.py).
- Clips with fewer than --min-hand-rate of frames with a detected hand are dropped.
- Features per frame: landmark coordinates + the 4 detection-mask flags (+ optional deltas).
- --canonical-hand: mirror clips whose LEFT hand moves more, so the dominant hand is always
  the right one (left-handed signers then look like right-handed ones).
- --hand-local: describe each hand by its shape (points relative to the wrist, divided by hand
  size) and keep the wrist position (relative to the shoulders) as the position feature.
- Split: StratifiedGroupKFold, group = signer_id, stratified by sign. A signer is never in
  both the training and the test part of a fold.
- No early stopping or tuning on the test fold: every fold trains for a fixed number of epochs.
- Training loss is weighted so that signers with many videos do not dominate.
- Reported per fold and as mean/std: top-1, top-5, macro-F1, latency, model size.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn

import mlflow
from src.utils.mlflow_utils import configure_mlflow, load_params


# ---------------------------------------------------------- feature options
# Layout of one frame (see feature_spec.json): 25 pose, 21 left hand, 21 right hand, face points.
N_POSE, N_HAND = 25, 21
LEFT_HAND = slice(N_POSE, N_POSE + N_HAND)
RIGHT_HAND = slice(N_POSE + N_HAND, N_POSE + 2 * N_HAND)
FACE_START = N_POSE + 2 * N_HAND
POSE_MIRROR_PAIRS = [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12),
                     (13, 14), (15, 16), (17, 18), (19, 20), (21, 22), (23, 24)]


def pose_permutation() -> np.ndarray:
    perm = np.arange(N_POSE)
    for a, b in POSE_MIRROR_PAIRS:
        perm[a], perm[b] = b, a
    return perm


def face_permutation(pts: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Left/right partner of every face point, found from the mean face shape."""
    face = pts[:, :, FACE_START:, :]
    seen = mask[:, :, 3]
    if face.shape[2] == 0 or not seen.any():
        return np.arange(face.shape[2])
    template = face[seen].mean(axis=0)  # [K, 3]
    mirrored = template * np.array([-1.0, 1.0, 1.0])
    dist = np.linalg.norm(mirrored[:, None, :2] - template[None, :, :2], axis=2)
    return dist.argmin(axis=1)


def hand_energy(pts: np.ndarray, mask: np.ndarray, slot: int, hand: slice):
    """Total frame-to-frame movement of a hand, and the number of frames it was seen."""
    seen = mask[:, :, slot]
    both = seen[:, 1:] & seen[:, :-1]
    step = np.linalg.norm(pts[:, 1:, hand, :2] - pts[:, :-1, hand, :2], axis=3).mean(axis=2)
    return (step * both).sum(axis=1), seen.sum(axis=1)


def dominant_is_left(pts: np.ndarray, mask: np.ndarray, ratio: float = 1.25,
                     min_frames: int = 3) -> np.ndarray:
    energy_l, frames_l = hand_energy(pts, mask, 1, LEFT_HAND)
    energy_r, frames_r = hand_energy(pts, mask, 2, RIGHT_HAND)
    left_only = (frames_l >= min_frames) & (frames_r < min_frames)
    both = (frames_l >= min_frames) & (frames_r >= min_frames)
    return left_only | (both & (energy_l > ratio * energy_r))


def mirror_clips(pts: np.ndarray, mask: np.ndarray, face_perm: np.ndarray,
                 select: np.ndarray):
    """Mirror the selected clips left <-> right (x flipped, hands/pose/face points swapped)."""
    pts, mask = pts.copy(), mask.copy()
    if not select.any():
        return pts, mask
    src = pts[select]
    out = np.empty_like(src)
    out[:, :, :N_POSE] = src[:, :, pose_permutation()]
    out[:, :, LEFT_HAND] = src[:, :, RIGHT_HAND]
    out[:, :, RIGHT_HAND] = src[:, :, LEFT_HAND]
    out[:, :, FACE_START:] = src[:, :, FACE_START:][:, :, face_perm]
    out[..., 0] *= -1.0
    pts[select] = out
    swapped = mask[select]
    swapped[:, :, [1, 2]] = swapped[:, :, [2, 1]]
    mask[select] = swapped
    return pts, mask


def hand_local_features(pts: np.ndarray) -> np.ndarray:
    """Hand shape relative to the wrist and hand size; point 0 keeps the wrist position."""
    out = pts.copy()
    for hand in (LEFT_HAND, RIGHT_HAND):
        block = pts[:, :, hand, :]  # [N, T, 21, 3]
        wrist = block[:, :, :1, :]
        size = np.linalg.norm(block[:, :, 9, :2] - block[:, :, 0, :2], axis=2)
        size = np.where(size > 1e-4, size, 1.0)
        local = (block - wrist) / size[:, :, None, None]
        local[:, :, 0, :] = wrist[:, :, 0, :]
        present = (np.abs(block).sum(axis=(2, 3)) > 0)[:, :, None, None]
        out[:, :, hand, :] = np.where(present, local, block)  # missing hands stay 0
    return out.astype(np.float32)


# --------------------------------------------------------------------- data
def load_data(path: Path, min_hand_rate: float, delta: bool,
              canonical_hand: bool = False, hand_local: bool = False) -> dict:
    with np.load(path) as d:
        x, mask = d["x"], d["mask"]
        y, signer, video_id = d["y"], d["signer_id"], d["video_id"]
    hand_rate = (mask[:, :, 1] | mask[:, :, 2]).mean(axis=1)
    keep = hand_rate >= min_hand_rate
    n_mirrored = 0
    if canonical_hand or hand_local:
        pts = x.reshape(x.shape[0], x.shape[1], -1, 3)
        if canonical_hand:
            select = dominant_is_left(pts, mask)
            n_mirrored = int((select & keep).sum())
            pts, mask = mirror_clips(pts, mask, face_permutation(pts, mask), select)
        if hand_local:
            pts = hand_local_features(pts)
        x = pts.reshape(x.shape).astype(np.float32)
    parts = [x]
    if delta:
        parts.append(np.diff(x, axis=1, prepend=x[:, :1]))
    parts.append(mask.astype(np.float32))
    feats = np.concatenate(parts, axis=2).astype(np.float32)
    classes = json.loads((path.parent / "classes.json").read_text(encoding="utf-8"))
    return {
        "X": feats[keep], "y": y[keep], "signer": signer[keep],
        "video_id": video_id[keep], "classes": classes,
        "n_total": len(y), "n_dropped": int((~keep).sum()), "n_mirrored": n_mirrored,
    }


def make_folds(y: np.ndarray, signer: np.ndarray, n_folds: int, seed: int):
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []
    for train_idx, test_idx in splitter.split(np.zeros(len(y)), y, groups=signer):
        overlap = set(signer[train_idx]) & set(signer[test_idx])
        assert not overlap, f"signer leak between train and test: {sorted(overlap)[:5]}"
        folds.append((train_idx, test_idx))
    return folds


def signer_weights(signers: np.ndarray) -> np.ndarray:
    """Weight 1/sqrt(videos of the signer), normalized to mean 1."""
    _, inverse, counts = np.unique(signers, return_inverse=True, return_counts=True)
    w = 1.0 / np.sqrt(counts[inverse])
    return (w / w.mean()).astype(np.float32)


def standardize(train: np.ndarray, *others: np.ndarray):
    flat = train.reshape(-1, train.shape[-1])
    mu, sd = flat.mean(axis=0), flat.std(axis=0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return [(a - mu) / sd for a in (train, *others)]


# -------------------------------------------------------------------- model
class GRUClassifier(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, hidden: int = 128, layers: int = 2,
                 dropout: float = 0.3, bidirectional: bool = False):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout)
        )
        self.gru = nn.GRU(hidden, hidden, num_layers=layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0,
                          bidirectional=bidirectional)
        out = hidden * (2 if bidirectional else 1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(out * 2, n_classes))

    def forward(self, x):
        h, _ = self.gru(self.proj(x))
        pooled = torch.cat([h.mean(dim=1), h.max(dim=1).values], dim=1)
        return self.head(pooled)


class TransformerClassifier(nn.Module):
    """Small Transformer encoder over the 32 frames (learned positions, mean+max pooling)."""

    def __init__(self, in_dim: int, n_classes: int, d_model: int = 128, layers: int = 2,
                 heads: int = 4, dropout: float = 0.3, max_len: int = 64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout)
        )
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=heads, dim_feedforward=2 * d_model, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers,
                                             enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model * 2, n_classes))

    def forward(self, x):
        h = self.proj(x) + self.pos[:, : x.shape[1]]
        h = self.norm(self.encoder(h))
        return self.head(torch.cat([h.mean(dim=1), h.max(dim=1).values], dim=1))


class CNNLSTMClassifier(nn.Module):
    """1D temporal convolutions over the landmark sequence, then an LSTM."""

    def __init__(self, in_dim: int, n_classes: int, hidden: int = 128, dropout: float = 0.3,
                 bidirectional: bool = False):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout)
        )
        self.conv = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2), nn.BatchNorm1d(hidden),
            nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2), nn.BatchNorm1d(hidden),
            nn.GELU(),
        )
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True, bidirectional=bidirectional)
        out = hidden * (2 if bidirectional else 1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(out * 2, n_classes))

    def forward(self, x):
        h = self.proj(x).transpose(1, 2)           # [B, hidden, T]
        h = self.conv(h).transpose(1, 2)           # [B, T, hidden]
        h, _ = self.lstm(h)
        return self.head(torch.cat([h.mean(dim=1), h.max(dim=1).values], dim=1))


def build_model(args, in_dim: int, n_classes: int):
    if args.model == "transformer":
        return TransformerClassifier(in_dim, n_classes, args.hidden, args.layers,
                                     args.heads, args.dropout)
    if args.model == "cnn_lstm":
        return CNNLSTMClassifier(in_dim, n_classes, args.hidden, args.dropout,
                                 args.bidirectional)
    return GRUClassifier(in_dim, n_classes, args.hidden, args.layers, args.dropout,
                         args.bidirectional)


def augment(x, noise: float, max_shift: int, frame_drop: float):
    """Light augmentation in standardized feature space (training only)."""
    if noise > 0:
        x = x + noise * torch.randn_like(x)
    if max_shift > 0:
        s = int(np.random.randint(-max_shift, max_shift + 1))
        if s > 0:
            x = torch.cat([x[:, :1].expand(-1, s, -1), x[:, :-s]], dim=1)
        elif s < 0:
            s = -s
            x = torch.cat([x[:, s:], x[:, -1:].expand(-1, s, -1)], dim=1)
    if frame_drop > 0:
        keep = (torch.rand(x.shape[0], x.shape[1], 1, device=x.device) > frame_drop)
        x = x * keep
    return x


def train_fold(Xtr, ytr, wtr, n_classes: int, args, device, seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_model(args, Xtr.shape[-1], n_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(reduction="none", label_smoothing=args.label_smoothing)
    n = Xtr.shape[0]
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        total = 0.0
        for i in range(0, n, args.batch_size):
            idx = perm[i:i + args.batch_size]
            xb = augment(Xtr[idx], args.noise, args.max_shift, args.frame_drop)
            loss = (criterion(model(xb), ytr[idx]) * wtr[idx]).mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(idx)
        scheduler.step()
        mlflow.log_metric("train_loss", total / n, step=epoch)
    return model


@torch.no_grad()
def predict(model, X, device, batch_size: int = 128):
    model.eval()
    out = [model(X[i:i + batch_size]).cpu() for i in range(0, X.shape[0], batch_size)]
    return torch.cat(out).numpy()


@torch.no_grad()
def measure_latency_ms(model, sample, runs: int = 100) -> float:
    """Median CPU latency for one sequence (batch size 1)."""
    cpu_model = copy.deepcopy(model).cpu().eval()
    x = sample[:1].cpu()
    for _ in range(20):
        cpu_model(x)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        cpu_model(x)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


# ------------------------------------------------------------------ metrics
def fold_metrics(logits: np.ndarray, y: np.ndarray) -> dict:
    pred = logits.argmax(axis=1)
    k = min(5, logits.shape[1])
    top5 = np.argsort(-logits, axis=1)[:, :k]
    present = np.unique(y)  # macro-F1 over the signs that occur in this test fold
    return {
        "top1": float((pred == y).mean()),
        "top5": float(np.mean([y[i] in top5[i] for i in range(len(y))])),
        "macro_f1": float(f1_score(y, pred, labels=present, average="macro", zero_division=0)),
        "n_test": int(len(y)),
        "n_test_signs": int(len(present)),
    }


def save_confusion(y_true, y_pred, n_classes: int, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(n_classes)).astype(float)
    cm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="viridis", vmin=0, vmax=1)
    ax.set_xlabel("predicted sign")
    ax.set_ylabel("true sign")
    ax.set_title("Pooled out-of-fold confusion matrix (row-normalized)")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------- main
def main() -> int:
    params = load_params()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/landmarks/wlasl/dataset_T32.npz"))
    ap.add_argument("--min-hand-rate", type=float, default=0.2)
    ap.add_argument("--folds", type=int, default=params.get("dataset", {}).get("n_folds", 5))
    ap.add_argument("--limit-folds", type=int, default=0, help="run only the first N folds")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--model", choices=("gru", "transformer", "cnn_lstm"), default="gru")
    ap.add_argument("--heads", type=int, default=4, help="attention heads (transformer)")
    ap.add_argument("--bidirectional", action="store_true")
    ap.add_argument("--delta", action="store_true", help="add frame-to-frame differences")
    ap.add_argument("--canonical-hand", action="store_true",
                    help="mirror clips so the dominant (more moving) hand is the right one")
    ap.add_argument("--hand-local", action="store_true",
                    help="hand shape relative to the wrist + wrist position")
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--noise", type=float, default=0.05)
    ap.add_argument("--max-shift", type=int, default=3)
    ap.add_argument("--frame-drop", type=float, default=0.1)
    ap.add_argument("--no-signer-weights", action="store_true")
    ap.add_argument("--seed", type=int, default=params.get("seed", 42))
    ap.add_argument("--run-name", default="gru_baseline")
    ap.add_argument("--models-dir", type=Path, default=Path("models"))
    ap.add_argument("--reports-dir", type=Path, default=Path("reports"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_data(args.data, args.min_hand_rate, args.delta,
                     args.canonical_hand, args.hand_local)
    X, y, signer, vid = data["X"], data["y"], data["signer"], data["video_id"]
    classes = data["classes"]
    n_classes = len(classes)
    print(f"device: {device} | clips kept: {len(y)} of {data['n_total']} "
          f"(dropped {data['n_dropped']} with < {args.min_hand_rate:.0%} hand frames) | "
          f"signs: {n_classes} | signers: {len(set(signer))} | features: {X.shape[-1]}")
    if args.canonical_hand:
        print(f"mirrored {data['n_mirrored']} clips ({data['n_mirrored'] / len(y):.0%}) "
              "so the dominant hand is the right one")

    folds = make_folds(y, signer, args.folds, args.seed)
    if args.limit_folds:
        folds = folds[: args.limit_folds]

    model_dir = args.models_dir / args.run_name
    report_dir = args.reports_dir / args.run_name
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    configure_mlflow(params)
    all_true, all_pred, all_vid, all_fold = [], [], [], []
    results = []
    with mlflow.start_run(run_name=args.run_name):
        mlflow.set_tags({"model": args.model, "split": "stratified_group_kfold"})
        mlflow.log_params({
            **{k: v for k, v in vars(args).items() if k not in ("data", "models_dir", "reports_dir")},
            "n_clips": len(y), "n_dropped_low_hands": data["n_dropped"],
            "n_signers": len(set(signer)), "n_classes": n_classes,
            "n_features": X.shape[-1], "device": str(device),
            "n_mirrored": data["n_mirrored"],
        })

        for k, (tr, te) in enumerate(folds):
            with mlflow.start_run(run_name=f"fold{k}", nested=True):
                Xtr, Xte = standardize(X[tr], X[te])
                Xtr_t = torch.tensor(Xtr, device=device)
                Xte_t = torch.tensor(Xte, device=device)
                ytr_t = torch.tensor(y[tr], dtype=torch.long, device=device)
                weights = (np.ones(len(tr), dtype=np.float32) if args.no_signer_weights
                           else signer_weights(signer[tr]))
                wtr_t = torch.tensor(weights, device=device)

                t0 = time.perf_counter()
                model = train_fold(Xtr_t, ytr_t, wtr_t, n_classes, args, device,
                                   args.seed + k)
                train_seconds = time.perf_counter() - t0

                logits = predict(model, Xte_t, device)
                m = fold_metrics(logits, y[te])
                path = model_dir / f"fold{k}.pt"
                torch.save(model.state_dict(), path)
                m["params"] = int(sum(p.numel() for p in model.parameters()))
                m["model_mb"] = round(path.stat().st_size / 1e6, 3)
                m["latency_ms"] = round(measure_latency_ms(model, Xte_t), 3)
                m["train_seconds"] = round(train_seconds, 1)
                mlflow.log_metrics({k_: v for k_, v in m.items()
                                    if isinstance(v, (int, float))})
                mlflow.log_param("test_signers", len(set(signer[te])))
                results.append(m)
                all_true.append(y[te])
                all_pred.append(logits.argmax(axis=1))
                all_vid.append(vid[te])
                all_fold.append(np.full(len(te), k))
                print(f"fold {k}: top1 {m['top1']:.3f} | top5 {m['top5']:.3f} | "
                      f"macro-F1 {m['macro_f1']:.3f} | {m['n_test']} test clips, "
                      f"{len(set(signer[te]))} unseen signers | {m['latency_ms']} ms | "
                      f"{train_seconds:.0f}s")

        summary = {}
        for key in ("top1", "top5", "macro_f1", "latency_ms", "model_mb"):
            vals = np.array([r[key] for r in results])
            summary[f"mean_{key}"] = float(vals.mean())
            summary[f"std_{key}"] = float(vals.std())
        summary["params"] = results[0]["params"]
        mlflow.log_metrics(summary)

        y_true, y_pred = np.concatenate(all_true), np.concatenate(all_pred)
        per_class = f1_score(y_true, y_pred, labels=np.arange(n_classes),
                             average=None, zero_division=0)
        with (report_dir / "per_class_f1.csv").open("w", encoding="utf-8") as f:
            f.write("sign,f1\n")
            for name, score in sorted(zip(classes, per_class), key=lambda t: t[1]):
                f.write(f"{name},{score:.4f}\n")
        with (report_dir / "oof_predictions.csv").open("w", encoding="utf-8") as f:
            f.write("video_id,fold,true,pred\n")
            for v, fo, t, p in zip(np.concatenate(all_vid), np.concatenate(all_fold),
                                   y_true, y_pred):
                f.write(f"{v},{fo},{classes[t]},{classes[p]}\n")
        save_confusion(y_true, y_pred, n_classes, report_dir / "confusion_matrix.png")
        (report_dir / "metrics.json").write_text(
            json.dumps({"folds": results, "summary": summary}, indent=1), encoding="utf-8")
        mlflow.log_artifacts(str(report_dir))
        mlflow.log_artifacts(str(model_dir), artifact_path="models")

    print("\n=== " + args.run_name + " (unseen-signer cross-validation) ===")
    print(f"top-1    {summary['mean_top1']:.3f} +/- {summary['std_top1']:.3f}")
    print(f"top-5    {summary['mean_top5']:.3f} +/- {summary['std_top5']:.3f}")
    print(f"macro-F1 {summary['mean_macro_f1']:.3f} +/- {summary['std_macro_f1']:.3f}")
    print(f"latency  {summary['mean_latency_ms']:.2f} ms/sequence (CPU, batch 1) | "
          f"{summary['params']:,} parameters | {summary['mean_model_mb']:.2f} MB")
    print(f"reports: {report_dir} | models: {model_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
