import hashlib
import sqlite3
import time

from fastapi.testclient import TestClient
import pytest

from backend.main import create_app
from backend.schemas import AnalysisResult
from tests.helpers import clean_analyzer, wait_job

HEADERS = {"X-UI-Request": "1"}
USER = {"name": "Тест интерфейса", "email": "reader@example.test", "password": "local-test-password-123"}


@pytest.fixture
def app(tmp_path):
    return create_app(
        clean_analyzer,
        api_key="test-key", upload_dir=tmp_path / "uploads", auth_db=tmp_path / "accounts.sqlite3",
    )


def test_registration_login_logout_and_protected_app(app):
    with TestClient(app) as client:
        assert client.get("/", follow_redirects=False).headers["location"] == "/login"
        assert client.get("/auth/api/me").status_code == 401
        assert client.get("/login").status_code == 200
        assert client.get("/register").status_code == 200
        registration = client.post("/auth/api/register", headers=HEADERS, json=USER)
        assert registration.status_code == 201
        assert registration.json()["user"]["name"] == USER["name"]
        assert set(registration.json()["user"]) == {"id", "name", "email"}
        assert "httponly" in registration.headers["set-cookie"].lower()
        assert "samesite=strict" in registration.headers["set-cookie"].lower()
        old_token = client.cookies.get("ui_session")
        assert client.get("/").status_code == 200
        assert client.get("/login", follow_redirects=False).status_code == 303
        assert client.post("/auth/api/logout", headers=HEADERS).status_code == 200
        assert client.get("/auth/api/me").status_code == 401
        # Replaying a logged-out cookie cannot restore access.
        assert client.get("/auth/api/me", headers={"Cookie": f"ui_session={old_token}"}).status_code == 401
        login = client.post("/auth/api/login", headers=HEADERS, json=USER)
        assert login.status_code == 200
        assert client.cookies.get("ui_session") != old_token
        assert client.get("/auth/api/me").json()["user"]["email"] == USER["email"]


def test_credentials_are_hashed_and_survive_restart(tmp_path):
    def build():
        return create_app(api_key="test-key", upload_dir=tmp_path / "uploads", auth_db=tmp_path / "accounts.sqlite3")
    with TestClient(build()) as first:
        assert first.post("/auth/api/register", headers=HEADERS, json=USER).status_code == 201
        token = first.cookies.get("ui_session")
    with sqlite3.connect(tmp_path / "accounts.sqlite3") as db:
        salt, digest = db.execute("SELECT salt, password_hash FROM users").fetchone()
        assert salt and digest != USER["password"]
        assert USER["password"].encode() not in (tmp_path / "accounts.sqlite3").read_bytes()
        assert db.execute("SELECT token_hash FROM sessions").fetchone()[0] == hashlib.sha256(token.encode()).hexdigest()
    with TestClient(build()) as second:
        second.cookies.set("ui_session", token)
        assert second.get("/auth/api/me").status_code == 200


def test_wrong_password_and_unknown_email_are_equivalent(app):
    with TestClient(app) as client:
        client.post("/auth/api/register", headers=HEADERS, json=USER)
        wrong = client.post("/auth/api/login", headers=HEADERS, json={**USER, "password": "incorrect-password-123"})
        unknown = client.post("/auth/api/login", headers=HEADERS, json={**USER, "email": "nobody@example.test"})
        assert wrong.status_code == unknown.status_code == 401
        assert wrong.json() == unknown.json()


def test_duplicate_email_is_case_insensitive(app):
    with TestClient(app) as client:
        assert client.post("/auth/api/register", headers=HEADERS, json=USER).status_code == 201
        assert client.post("/auth/api/register", headers=HEADERS, json={**USER, "email": USER["email"].upper()}).status_code == 409


@pytest.mark.parametrize("patch", [{"email": "invalid"}, {"password": "short"}, {"name": "  "}])
def test_invalid_registration_is_rejected(app, patch):
    with TestClient(app) as client:
        assert client.post("/auth/api/register", headers=HEADERS, json={**USER, **patch}).status_code == 422


def test_auth_mutations_need_custom_header(app):
    with TestClient(app) as client:
        for route in ["register", "login", "logout"]:
            assert client.post(f"/auth/api/{route}", json=USER).status_code == 403


def test_expired_session_is_rejected(app, tmp_path):
    with TestClient(app) as client:
        client.post("/auth/api/register", headers=HEADERS, json=USER)
        with sqlite3.connect(tmp_path / "accounts.sqlite3") as db:
            db.execute("UPDATE sessions SET expires_at=?", (int(time.time()) - 1,))
        assert client.get("/auth/api/me").status_code == 401


def test_users_cannot_read_each_others_jobs(app, mov_bytes):
    with TestClient(app) as first, TestClient(app) as second:
        first.post("/auth/api/register", headers=HEADERS, json=USER)
        second.post("/auth/api/register", headers=HEADERS, json={**USER, "email": "second@example.test"})
        job = first.post("/app/api/v1/analyze", headers=HEADERS, files={"file": ("test.mov", mov_bytes)}).json()["job_id"]
        assert first.get(f"/app/api/v1/jobs/{job}").status_code == 200
        assert second.get(f"/app/api/v1/jobs/{job}").status_code == 404
        assert second.get(f"/api/v1/jobs/{job}").status_code == 401
        assert second.get(f"/api/v1/jobs/{job}", headers={"X-API-Key": "test-key"}).status_code == 200


def test_auth_attempts_are_rate_limited(app):
    with TestClient(app) as client:
        # Invalid requests count as attempts before expensive password hashing.
        for _ in range(10):
            assert client.post("/auth/api/login", headers=HEADERS, json={}).status_code == 422
        assert client.post("/auth/api/login", headers=HEADERS, json={}).status_code == 429


def test_https_sets_secure_cookie(app):
    with TestClient(app, base_url="https://testserver") as client:
        response = client.post("/auth/api/register", headers=HEADERS, json=USER)
        assert "secure" in response.headers["set-cookie"].lower()
