from fastapi.testclient import TestClient
import pytest

from backend.main import create_app
from backend.schemas import AnalysisResult
from tests.helpers import clean_analyzer, broken_analyzer, wait_job

KEY = {"X-API-Key": "test-key"}


@pytest.fixture
def client(tmp_path):
    upload_dir = tmp_path / "uploads"
    app = create_app(
        analyzer=clean_analyzer,
        api_key="test-key", upload_dir=upload_dir, auth_db=tmp_path / "accounts.sqlite3", max_upload_bytes=1024 * 1024,
    )
    with TestClient(app) as test_client:
        yield test_client
    assert not list(upload_dir.iterdir())


def test_requires_key_on_both_routes(client):
    assert client.post("/api/v1/analyze", files={"file": ("a.mov", b"test")}).status_code == 401
    assert client.get("/api/v1/jobs/unknown").status_code == 401
    assert client.get("/api/v1/jobs/unknown", headers={"X-API-Key": "wrong"}).status_code == 401


def test_upload_then_get_result(client, mov_bytes):
    response = client.post("/api/v1/analyze", headers=KEY, files={"file": ("a.mov", mov_bytes)})
    assert response.status_code == 202
    job_id = response.json()["job_id"]
    result = wait_job(client, job_id, KEY)
    assert result == {
        "job_id": job_id, "status": "done", "verdict": "no_violation", "violations": []
    }


@pytest.mark.parametrize("name,content,status", [
    ("a.mp4", b"test", 415), ("a.mov", b"", 400), ("a.mov", b"x" * (1024 * 1024 + 1), 413),
    ("a.mov", b"not a video", 415),
], ids=["wrong-extension", "empty", "too-large", "not-video"])
def test_rejects_invalid_uploads(client, name, content, status):
    response = client.post("/api/v1/analyze", headers=KEY, files={"file": (name, content)})
    assert response.status_code == status


def test_unknown_job(client):
    assert client.get("/api/v1/jobs/unknown", headers=KEY).status_code == 404


def test_unconfigured_model_fails_explicitly(tmp_path, mov_bytes, monkeypatch):
    monkeypatch.setenv("ANALYZER_CALLABLE", "")
    with TestClient(create_app(api_key="test-key", upload_dir=tmp_path / "uploads", auth_db=tmp_path / "accounts.sqlite3")) as client:
        response = client.post("/api/v1/analyze", headers=KEY, files={"file": ("a.mov", mov_bytes)})
        assert response.status_code == 503
        assert "job_id" not in response.json()
        assert client.get('/health').json()["analyzer_configured"] is False
    assert not list((tmp_path / "uploads").iterdir())


def test_inconsistent_model_result_is_rejected():
    with pytest.raises(ValueError):
        AnalysisResult(verdict="violation", violations=[])


def test_model_error_does_not_become_clean_verdict(tmp_path, mov_bytes):
    with TestClient(create_app(broken_analyzer, api_key="test-key", upload_dir=tmp_path / "uploads", auth_db=tmp_path / "accounts.sqlite3")) as client:
        job = client.post("/api/v1/analyze", headers=KEY, files={"file": ("a.mov", mov_bytes)}).json()
        result = wait_job(client, job['job_id'], KEY)
        assert result["status"] == "failed"
        assert "verdict" not in result
        assert "internal details" not in result["error"]
    assert not list((tmp_path / "uploads").iterdir())


def test_home_serves_ui_without_exposing_jury_key(client):
    response = client.post("/auth/api/register", headers={"X-UI-Request": "1"}, json={
        "name": "API Test", "email": "api@example.test", "password": "local-test-password-123",
    })
    assert response.status_code == 201
    assert "HttpOnly" in response.headers["set-cookie"]
    page = client.get("/")
    assert page.status_code == 200
    assert "Каждый кадр" in page.text
    assert "test-key" not in page.text
    assert client.get("/assets/app.js").status_code == 200
    assert client.get("/assets/bee/front.png").status_code == 200
    assert client.get("/.env").status_code == 404


def test_ui_upload_uses_session_and_csrf_header(client, mov_bytes):
    route = "/app/api/v1/analyze"
    files = {"file": ("a.mov", mov_bytes)}
    assert client.post(route, files=files).status_code == 401
    client.post("/auth/api/register", headers={"X-UI-Request": "1"}, json={
        "name": "UI Test", "email": "ui@example.test", "password": "local-test-password-123",
    })
    assert client.post(route, files=files).status_code == 403
    response = client.post(route, files=files, headers={"X-UI-Request": "1"})
    assert response.status_code == 202
    job_id = response.json()["job_id"]
    assert wait_job(client, job_id, route="/app/api/v1/jobs/")["status"] == "done"
    # A browser session cannot bypass jury API authentication.
    assert client.get(f"/api/v1/jobs/{job_id}").status_code == 401
