"""The release check must fail on analyzer errors and malformed results."""
import argparse

import httpx
import pytest

from scripts.check_api import CheckFailed, run_checks, validate_url


def transport(final_job=None, unavailable=False):
    def handle(request):
        path = request.url.path
        if path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if path.startswith("/docs"):
            return httpx.Response(200, text="local asset")
        if request.headers.get("X-API-Key") != "synthetic-test-key":
            return httpx.Response(401)
        if path.startswith("/api/v1/jobs/"):
            if path != "/api/v1/jobs/accepted-job":
                return httpx.Response(404)
            return httpx.Response(200, json=final_job or {
                "job_id": "accepted-job", "status": "done", "verdict": "no_violation", "violations": [],
            })
        body = request.read()
        if b'filename="empty.mov"' in body:
            return httpx.Response(400)
        if b'filename="invalid.mov"' in body or b'filename="invalid.txt"' in body:
            return httpx.Response(415)
        return httpx.Response(503 if unavailable else 202, json={"job_id": "accepted-job"})
    return httpx.MockTransport(handle)


def test_release_check_requires_successful_analysis():
    with httpx.Client(base_url="https://example.test", transport=transport()) as client:
        assert "202 → done" in run_checks(client, "synthetic-test-key")[-1]


@pytest.mark.parametrize("job", [
    {"job_id": "accepted-job", "status": "failed", "error": "private details"},
    {"job_id": "other-job", "status": "done", "verdict": "no_violation", "violations": []},
    {"job_id": "accepted-job", "status": "done", "verdict": "violation", "violations": []},
])
def test_release_check_rejects_failure_or_invalid_result(job):
    with httpx.Client(base_url="https://example.test", transport=transport(final_job=job)) as client:
        with pytest.raises((CheckFailed, ValueError)):
            run_checks(client, "synthetic-test-key")


def test_unavailable_mode_is_explicit():
    with httpx.Client(base_url="https://example.test", transport=transport(unavailable=True)) as client:
        assert "inference not checked" in run_checks(client, "synthetic-test-key", expect_unavailable=True)[-1]


@pytest.mark.parametrize("url", ["http://example.test", "https://user:secret@example.test", "https://example.test/?key=secret"])
def test_key_is_never_sent_to_insecure_remote_or_embedded_url(url):
    with pytest.raises(argparse.ArgumentTypeError):
        validate_url(url)
