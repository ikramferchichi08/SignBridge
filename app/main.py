import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app import auth


ROOT = Path(__file__).resolve().parents[1]


@asynccontextmanager
async def lifespan(_app: FastAPI):
    auth.init_auth_store()
    yield


app = FastAPI(title="SignBridge API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AccountCredentials(BaseModel):
    email: str = Field(max_length=254)
    password: str = Field(max_length=128)


class SignHistoryEntry(BaseModel):
    label: str = Field(min_length=1, max_length=100)
    confidence: float = Field(ge=0, le=1)


def public_user(user: dict) -> dict:
    return {"id": user["id"], "email": user["email"], "role": user["role"]}


@app.get("/api/health")
def health():
    return {"status": "ok", "project": "SignBridge"}


@app.post("/api/auth/register", status_code=201)
def register(credentials: AccountCredentials, response: Response):
    email = auth.normalize_email(credentials.email)
    if len(credentials.password) < 12:
        raise HTTPException(
            status_code=422, detail="Password must be at least 12 characters."
        )

    with auth.database() as connection:
        try:
            cursor = connection.execute(
                """
                INSERT INTO users (email, password_hash, role, created_at)
                VALUES (?, ?, 'user', ?)
                """,
                (email, auth.hash_password(credentials.password), auth.utc_now()),
            )
        except sqlite3.IntegrityError as error:
            if "UNIQUE constraint failed: users.email" in str(error):
                raise HTTPException(
                    status_code=409, detail="An account with this email already exists."
                ) from error
            raise
        user_id = cursor.lastrowid
        if user_id is None:
            raise RuntimeError("Account creation did not return a user ID.")
        user = {"id": user_id, "email": email, "role": "user"}
        auth.record_event(connection, user["id"], "register")
        token = auth.new_session(connection, user["id"])
    auth.set_session_cookie(response, token)
    return {"user": public_user(user)}


@app.post("/api/auth/login")
def login(credentials: AccountCredentials, response: Response):
    email = auth.normalize_email(credentials.email)
    with auth.database() as connection:
        user = connection.execute(
            "SELECT id, email, password_hash, role FROM users WHERE email = ?",
            (email,),
        ).fetchone()
        if user is None or not auth.verify_password(
            credentials.password, user["password_hash"]
        ):
            raise HTTPException(status_code=401, detail="Invalid email or password.")
        auth.record_event(connection, user["id"], "login")
        token = auth.new_session(connection, user["id"])
        public = {"id": user["id"], "email": user["email"], "role": user["role"]}
    auth.set_session_cookie(response, token)
    return {"user": public}


@app.post("/api/auth/logout")
def logout(
    request: Request,
    response: Response,
    user: dict = Depends(auth.current_user),
):
    token = request.cookies[auth.SESSION_COOKIE]
    with auth.database() as connection:
        auth.record_event(connection, user["id"], "logout")
        connection.execute(
            "DELETE FROM sessions WHERE token_hash = ?", (auth.token_digest(token),)
        )
    auth.clear_session_cookie(response)
    return {"status": "ok"}


@app.get("/api/auth/me")
def get_me(user: dict = Depends(auth.current_user)):
    return {"user": public_user(user)}


@app.get("/api/history")
def get_history(
    limit: int = Query(default=1000, ge=1, le=5000),
    user: dict = Depends(auth.current_user),
):
    with auth.database() as connection:
        if user["role"] == "admin":
            events = connection.execute(
                """
                SELECT history_events.id, history_events.event_type, history_events.label,
                       history_events.confidence, history_events.created_at, users.email
                FROM history_events JOIN users ON users.id = history_events.user_id
                ORDER BY history_events.id DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
            scope = "all"
        else:
            events = connection.execute(
                """
                SELECT history_events.id, history_events.event_type, history_events.label,
                       history_events.confidence, history_events.created_at, users.email
                FROM history_events JOIN users ON users.id = history_events.user_id
                WHERE history_events.user_id = ?
                ORDER BY history_events.id DESC LIMIT ?
                """,
                (user["id"], limit),
            ).fetchall()
            scope = "self"
    return {"scope": scope, "events": [dict(event) for event in events]}


@app.post("/api/history/signs", status_code=201)
def add_sign_to_history(
    entry: SignHistoryEntry,
    user: dict = Depends(auth.current_user),
):
    label = entry.label.strip()
    if not label:
        raise HTTPException(status_code=422, detail="Sign label cannot be empty.")
    with auth.database() as connection:
        auth.record_event(
            connection,
            user["id"],
            "sign",
            label=label,
            confidence=entry.confidence,
        )
    return {"status": "recorded"}


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

    benchmark_path = ROOT / "reports" / "extraction_benchmark.csv"
    benchmark = (
        pd.read_csv(benchmark_path)
        if benchmark_path.exists()
        else pd.DataFrame(columns=["workers", "videos_per_sec"])
    )
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
