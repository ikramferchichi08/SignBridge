from pathlib import Path
import json

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles


ROOT = Path(__file__).resolve().parents[1]
app = FastAPI(title="SignBridge API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"status": "ok", "project": "SignBridge"}


@app.get("/api/overview")
def overview():
    metadata = pd.read_csv(ROOT / "data" / "raw" / "wlasl" / "metadata.csv")
    comparison = pd.read_csv(ROOT / "reports" / "model_comparison.csv")
    comparison = comparison.sort_values("top1_mean", ascending=False)
    best = comparison.iloc[0]
    checkpoints = [str(p.relative_to(ROOT)).replace("\\", "/")
                   for p in (ROOT / "models").glob("**/final.pt")]
    return {
        "videos": int(len(metadata)),
        "signs": int(metadata["label"].nunique()),
        "signers": int(metadata["signer_id"].nunique()),
        "best_model": best.to_dict(),
        "final_checkpoints": checkpoints,
        "workflow": [
            {"title": "Dataset WLASL", "detail": "1 102 vidéos / 100 signes", "status": "done"},
            {"title": "Landmarks MediaPipe", "detail": "Corps, mains et visage", "status": "done"},
            {"title": "Préparation des séquences", "detail": "32 frames par vidéo", "status": "done"},
            {"title": "Modèles temporels", "detail": "GRU, Transformer, CNN-LSTM", "status": "done"},
            {"title": "Évaluation", "detail": "Validation croisée séparée par signer", "status": "done"},
            {"title": "Application live", "detail": "Webcam, lissage et transcription", "status": "planned"},
        ],
    }


@app.get("/api/dataset")
def dataset(limit: int = 500, search: str = ""):
    metadata = pd.read_csv(ROOT / "data" / "raw" / "wlasl" / "metadata.csv")
    if search:
        metadata = metadata[metadata["label"].str.contains(search, case=False, na=False)]

    benchmark = pd.read_csv(ROOT / "reports" / "extraction_benchmark.csv")
    sign_counts = (
        metadata["label"].value_counts()
        .rename_axis("sign")
        .reset_index(name="videos")
        .head(100)
        .to_dict(orient="records")
    )
    rows = metadata.head(min(limit, 1000)).fillna("").to_dict(orient="records")
    return {
        "videos": int(len(metadata)),
        "signs": int(metadata["label"].nunique()),
        "signers": int(metadata["signer_id"].nunique()),
        "duration_mean": round(float(metadata["duration_sec"].mean()), 2),
        "duration_min": round(float(metadata["duration_sec"].min()), 2),
        "duration_max": round(float(metadata["duration_sec"].max()), 2),
        "sign_counts": sign_counts,
        "benchmark": benchmark.to_dict(orient="records"),
        "rows": rows,
    }


@app.get("/api/experiments")
def experiments():
    comparison = pd.read_csv(ROOT / "reports" / "model_comparison.csv")
    comparison = comparison.sort_values("top1_mean", ascending=False)
    comparison = comparison.fillna("")
    return {"runs": comparison.to_dict(orient="records")}


frontend_dist = ROOT / "app" / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
