"""Check a deployed API using generated pixels only; never accept dataset files."""
import argparse
import io
import ipaddress
import os
from pathlib import Path
import sys
import time
from urllib.parse import urlsplit
from uuid import uuid4

import av
import httpx
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.schemas import AnalysisResult, JobCreated, JobResponse


class CheckFailed(RuntimeError):
    pass


def validate_url(value: str) -> str:
    url = urlsplit(value)
    if not url.hostname or url.username or url.password or url.query or url.fragment or url.path not in ("", "/"):
        raise argparse.ArgumentTypeError("Use a server origin without credentials, path or query")
    try:
        loopback = ipaddress.ip_address(url.hostname).is_loopback
    except ValueError:
        loopback = url.hostname == "localhost"
    if url.scheme != "https" and not (url.scheme == "http" and loopback):
        raise argparse.ArgumentTypeError("HTTPS is required except for localhost")
    return value.rstrip("/")


def synthetic_mov() -> bytes:
    output = io.BytesIO()
    with av.open(output, "w", format="mov") as container:
        stream = container.add_stream("mpeg4", rate=8)
        stream.width, stream.height, stream.pix_fmt = 64, 64, "yuv420p"
        for i in range(32):
            frame = av.VideoFrame.from_image(Image.new("RGB", (64, 64), (i * 7, 60, 80)))
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return output.getvalue()


def run_checks(client: httpx.Client, api_key: str, *, budget: float = 300, expect_unavailable: bool = False) -> list[str]:
    passed = []
    deadline = time.monotonic() + budget
    headers = {"X-API-Key": api_key}

    def request(method, path, expected, **kwargs):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise CheckFailed("Overall check deadline exceeded")
        response = client.request(method, path, timeout=min(35, remaining), **kwargs)
        if response.status_code != expected:
            # Never include response bodies, credentials, frames or scene descriptions.
            raise CheckFailed(f"{method} {path}: expected HTTP {expected}, got {response.status_code}")
        return response

    health = request("GET", "/health", 200).json()
    if health.get("status") != "ok":
        raise CheckFailed("Health status is not ok")
    passed.append("health")
    request("GET", "/docs", 200)
    request("GET", "/docs-assets/swagger-ui-bundle.js", 200)
    passed.append("API documentation and local JS")
    missing_job = "/api/v1/jobs/" + uuid4().hex
    request("GET", missing_job, 401)
    request("GET", missing_job, 401, headers={"X-API-Key": "incorrect-" + uuid4().hex})
    request("GET", missing_job, 404, headers=headers)
    for auth in ({}, {"X-API-Key": "incorrect-" + uuid4().hex}):
        request("POST", "/api/v1/analyze", 401, headers=auth,
                files={"file": ("synthetic.mov", b"synthetic")})
    passed.append("both endpoints reject missing/incorrect keys; unknown job is 404")
    for name, content, expected in (("empty.mov", b"", 400), ("invalid.mov", b"not a movie", 415), ("invalid.txt", b"text", 415)):
        request("POST", "/api/v1/analyze", expected, headers=headers, files={"file": (name, content)})
    passed.append("empty, damaged and unsupported uploads")
    movie = synthetic_mov()
    response = request("POST", "/api/v1/analyze", 503 if expect_unavailable else 202,
                       headers=headers, files={"file": ("synthetic.mov", movie, "video/quicktime")})
    if expect_unavailable:
        passed.append("disabled analyzer explicitly returns 503 (inference not checked)")
        return passed
    created = JobCreated.model_validate(response.json())
    if not created.job_id:
        raise CheckFailed("Empty job_id")
    while True:
        job = JobResponse.model_validate(request("GET", "/api/v1/jobs/" + created.job_id, 200, headers=headers).json())
        if job.job_id != created.job_id:
            raise CheckFailed("Job identity changed")
        if job.status == "failed":
            raise CheckFailed("Synthetic analysis failed; inspect the local server configuration")
        if job.status == "done":
            AnalysisResult.model_validate({"verdict": job.verdict, "violations": job.violations})
            passed.append("synthetic video: 202 → done; verdict, events and JPEG schema validated")
            return passed
        if deadline - time.monotonic() <= 0:
            raise CheckFailed("Analysis deadline exceeded")
        time.sleep(min(.5, max(0, deadline - time.monotonic())))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", type=validate_url, default="http://127.0.0.1:8000")
    parser.add_argument("--expect-unavailable", action="store_true", help="Expect 503 for an intentionally disabled model")
    args = parser.parse_args()
    key = os.environ.get("API_KEY", "")
    if not key:
        parser.error("Set API_KEY in the environment; do not pass it as a command-line argument")
    try:
        # No environment proxies, redirects or real video inputs.
        with httpx.Client(base_url=args.base_url, trust_env=False, follow_redirects=False) as client:
            passed = run_checks(client, key, expect_unavailable=args.expect_unavailable)
        for check in passed:
            print("PASS:", check)
        print("Synthetic integration check only; recognition accuracy was not measured.")
        return 0
    except (CheckFailed, httpx.HTTPError, ValueError):
        print("FAIL: API check did not complete successfully. Check URL, key, server status and model configuration.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
