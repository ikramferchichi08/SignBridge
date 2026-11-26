from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Generator

from fastapi import HTTPException, Request


ROOT = Path(__file__).resolve().parents[1]
SESSION_COOKIE = "signbridge_session"
SESSION_TTL_SECONDS = 60 * 60 * 24 * 14
PASSWORD_ITERATIONS = 600_000
EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def database_path() -> Path:
    return Path(os.environ.get("SIGNBRIDGE_DB_PATH", ROOT / "data" / "signbridge.sqlite3"))


@contextmanager
def database() -> Generator[sqlite3.Connection, None, None]:
    path = database_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def init_auth_store() -> None:
    with database() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL COLLATE NOCASE UNIQUE,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'user' CHECK (role IN ('user', 'admin')),
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sessions (
                token_hash TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                expires_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS history_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                event_type TEXT NOT NULL CHECK (
                    event_type IN ('register', 'login', 'logout', 'sign')
                ),
                label TEXT,
                confidence REAL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS history_events_user_created
                ON history_events(user_id, created_at DESC);
            """
        )
        connection.execute("DELETE FROM sessions WHERE expires_at <= ?", (utc_now(),))
        admin_email = os.environ.get("SIGNBRIDGE_ADMIN_EMAIL", "").strip().lower()
        admin_password = os.environ.get("SIGNBRIDGE_ADMIN_PASSWORD", "")
        if bool(admin_email) != bool(admin_password):
            raise RuntimeError(
                "Set both SIGNBRIDGE_ADMIN_EMAIL and SIGNBRIDGE_ADMIN_PASSWORD to bootstrap an admin."
            )
        if admin_email:
            if not EMAIL_PATTERN.fullmatch(admin_email) or len(admin_password) < 12:
                raise RuntimeError("Admin bootstrap credentials are invalid.")
            admin = connection.execute(
                "SELECT id, password_hash, role FROM users WHERE email = ? COLLATE NOCASE",
                (admin_email,),
            ).fetchone()
            if admin is None:
                cursor = connection.execute(
                    """
                    INSERT INTO users (email, password_hash, role, created_at)
                    VALUES (?, ?, 'admin', ?)
                    """,
                    (admin_email, hash_password(admin_password), utc_now()),
                )
                if cursor.lastrowid is None:
                    raise RuntimeError("Admin account creation did not return a user ID.")
                record_event(connection, cursor.lastrowid, "register")
            elif admin["role"] != "admin":
                if not verify_password(admin_password, admin["password_hash"]):
                    raise RuntimeError(
                        "Configured admin password does not match the stored admin account."
                    )
                connection.execute(
                    "UPDATE users SET role = 'admin' WHERE id = ?", (admin["id"],)
                )


def normalize_email(email: str) -> str:
    value = email.strip().lower()
    if len(value) > 254 or not EMAIL_PATTERN.fullmatch(value):
        raise HTTPException(status_code=422, detail="Enter a valid email address.")
    return value


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, PASSWORD_ITERATIONS
    )
    return f"{PASSWORD_ITERATIONS}${salt.hex()}${digest.hex()}"


def verify_password(password: str, encoded_hash: str) -> bool:
    try:
        iterations_text, salt_text, digest_text = encoded_hash.split("$")
        iterations = int(iterations_text)
        salt = bytes.fromhex(salt_text)
        expected = bytes.fromhex(digest_text)
    except (ValueError, TypeError):
        return False
    actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return hmac.compare_digest(actual, expected)


def record_event(
    connection: sqlite3.Connection,
    user_id: int,
    event_type: str,
    label: str | None = None,
    confidence: float | None = None,
) -> None:
    connection.execute(
        """
        INSERT INTO history_events (user_id, event_type, label, confidence, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (user_id, event_type, label, confidence, utc_now()),
    )


def new_session(connection: sqlite3.Connection, user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    expires_at = (
        datetime.now(timezone.utc) + timedelta(seconds=SESSION_TTL_SECONDS)
    ).isoformat(timespec="seconds")
    connection.execute(
        "INSERT INTO sessions (token_hash, user_id, expires_at) VALUES (?, ?, ?)",
        (token_digest(token), user_id, expires_at),
    )
    return token


def token_digest(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def set_session_cookie(response, token: str) -> None:
    response.set_cookie(
        SESSION_COOKIE,
        token,
        max_age=SESSION_TTL_SECONDS,
        httponly=True,
        secure=os.environ.get("SIGNBRIDGE_COOKIE_SECURE", "").lower() == "true",
        samesite="strict",
        path="/",
    )


def clear_session_cookie(response) -> None:
    response.delete_cookie(
        SESSION_COOKIE,
        httponly=True,
        secure=os.environ.get("SIGNBRIDGE_COOKIE_SECURE", "").lower() == "true",
        samesite="strict",
        path="/",
    )


def current_user(request: Request) -> dict:
    token = request.cookies.get(SESSION_COOKIE)
    if not token:
        raise HTTPException(status_code=401, detail="Sign in to continue.")
    with database() as connection:
        row = connection.execute(
            """
            SELECT users.id, users.email, users.role, sessions.expires_at
            FROM sessions JOIN users ON users.id = sessions.user_id
            WHERE sessions.token_hash = ?
            """,
            (token_digest(token),),
        ).fetchone()
    if row is None or row["expires_at"] <= utc_now():
        raise HTTPException(status_code=401, detail="Your session has expired.")
    return {"id": row["id"], "email": row["email"], "role": row["role"]}
