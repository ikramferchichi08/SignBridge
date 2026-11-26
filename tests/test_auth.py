import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def clients(tmp_path, monkeypatch):
    monkeypatch.setenv("SIGNBRIDGE_DB_PATH", str(tmp_path / "signbridge.sqlite3"))
    monkeypatch.setenv("SIGNBRIDGE_ADMIN_EMAIL", "admin@example.com")
    monkeypatch.setenv("SIGNBRIDGE_ADMIN_PASSWORD", "configured-admin-password")
    admin = TestClient(app)
    user = TestClient(app)
    with admin, user:
        yield admin, user


def register(client: TestClient, email: str) -> dict:
    response = client.post(
        "/api/auth/register",
        json={"email": email, "password": "a-secure-password-123"},
    )
    assert response.status_code == 201
    return response.json()["user"]


def test_registration_login_and_logout_are_recorded(clients):
    _, client = clients
    user = register(client, "person@example.com")
    assert user["role"] == "user"
    assert client.get("/api/auth/me").json()["user"]["email"] == "person@example.com"

    client.post("/api/auth/logout")
    assert client.get("/api/auth/me").status_code == 401
    assert client.get("/api/history").status_code == 401
    assert client.post(
        "/api/history/signs",
        json={"label": "hello", "confidence": 0.9},
    ).status_code == 401
    assert client.post(
        "/api/auth/login",
        json={"email": "person@example.com", "password": "a-secure-password-123"},
    ).status_code == 200

    events = client.get("/api/history").json()["events"]
    assert [event["event_type"] for event in events] == ["login", "logout", "register"]


def test_users_see_private_history_and_admins_see_all(clients):
    admin, user = clients
    admin_login = admin.post(
        "/api/auth/login",
        json={
            "email": "admin@example.com",
            "password": "configured-admin-password",
        },
    )
    assert admin_login.status_code == 200
    admin_user = admin_login.json()["user"]
    user_account = register(user, "person@example.com")
    assert admin_user["role"] == "admin"
    assert user_account["role"] == "user"

    admin.post("/api/history/signs", json={"label": "hello", "confidence": 0.9})
    user.post("/api/history/signs", json={"label": "thanks", "confidence": 0.8})

    own_history = user.get("/api/history").json()
    assert own_history["scope"] == "self"
    assert {event["email"] for event in own_history["events"]} == {"person@example.com"}
    assert {event["label"] for event in own_history["events"] if event["label"]} == {"thanks"}

    all_history = admin.get("/api/history").json()
    assert all_history["scope"] == "all"
    assert {event["email"] for event in all_history["events"]} == {
        "admin@example.com",
        "person@example.com",
    }


def test_registration_rejects_weak_password_and_duplicate_email(clients):
    _, client = clients
    weak = client.post(
        "/api/auth/register",
        json={"email": "person@example.com", "password": "short"},
    )
    assert weak.status_code == 422

    register(client, "person@example.com")
    duplicate = client.post(
        "/api/auth/register",
        json={"email": "PERSON@example.com", "password": "a-secure-password-123"},
    )
    assert duplicate.status_code == 409
