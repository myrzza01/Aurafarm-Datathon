"""Check the public jury/browser routes using artificial video only.

Credentials stay in memory and the ignored .env file; the saved report contains
statuses/timings, never keys, passwords, frames, or source video.
"""
import base64
import io
import json
import math
from pathlib import Path
import secrets
import time

import av
import httpx
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent.parent


def local_key():
    key = None
    for line in (ROOT / ".env").read_text(encoding="utf-8-sig").splitlines():
        name, separator, value = line.partition("=")
        if separator and name.strip() == "API_KEY":
            key = value.strip().strip("\"'")
    if not key:
        raise RuntimeError("The local ignored .env has no API_KEY")
    return key


def artificial_mov():
    output = io.BytesIO()
    with av.open(output, "w", format="mov") as container:
        stream = container.add_stream("mpeg4", rate=30)
        stream.width, stream.height, stream.pix_fmt = 160, 96, "yuv420p"
        for index in range(300):
            image = Image.new("RGB", (160, 96), (18, 35, 70))
            draw = ImageDraw.Draw(image)
            draw.text((8, 8), "SYNTHETIC TEST", fill=(240, 240, 240))
            left = 8 + index % 100
            draw.rectangle((left, 42, left + 25, 67), fill=(50, 160, 190))
            frame = av.VideoFrame.from_image(image)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return output.getvalue()


def check_result(result):
    assert result["status"] == "done", result.get("error", "The job did not finish")
    events = result["violations"]
    assert result["verdict"] in {"violation", "no_violation"}
    assert (result["verdict"] == "violation") == bool(events)
    for event in events:
        assert event["type"].strip() and event["description"].strip()
        assert math.isfinite(event["timecode_sec"]) and 0 <= event["timecode_sec"] <= 10
        frames = event["evidence_frames"]
        assert 1 <= len(frames) <= 3
        assert event["rule_source"]["status"] == "provided"
        assert event["rule_source"]["document"].strip()
        for frame in frames:
            assert math.isfinite(frame["timestamp_sec"]) and 0 <= frame["timestamp_sec"] <= 10
            with Image.open(io.BytesIO(base64.b64decode(frame["image_base64"], validate=True))) as image:
                assert image.format == "JPEG"
                image.load()


def analyze(client, body, *, browser, key):
    started = time.monotonic()
    route = "/app/api/v1/" if browser else "/api/v1/"
    headers = {"X-UI-Request": "1"} if browser else {"X-API-Key": key}
    response = client.post(route + "analyze", headers=headers, files={
        "file": ("synthetic-public-check.mov", body, "video/quicktime"),
    })
    assert response.status_code == 202, f"Upload returned HTTP {response.status_code}"
    job_id = response.json()["job_id"]
    while time.monotonic() - started < 295:
        response = client.get(route + "jobs/" + job_id, headers=headers)
        response.raise_for_status()
        result = response.json()
        assert result["status"] in {"processing", "done", "failed"}
        if result["status"] != "processing":
            check_result(result)
            return {
                "upload_http": 202, "status": result["status"], "verdict": result["verdict"],
                "events": len(result["violations"]), "elapsed_seconds": round(time.monotonic() - started, 3),
            }
        time.sleep(.5)
    raise TimeoutError("Public upload/polling exceeded the case's time limit")


def main():
    state = json.loads((ROOT / "data/public-run.json").read_text(encoding="utf-8-sig"))
    url = state["url"]
    assert url.startswith("https://") and url.endswith(".trycloudflare.com")
    key = local_key()
    report = {"url": url, "synthetic_video_only": True}
    with httpx.Client(base_url=url, follow_redirects=False, timeout=45, trust_env=False) as client:
        health = client.get("/health")
        health.raise_for_status()
        assert health.json()["analyzer_configured"] is True
        assert health.json()["analyzer_kind"] == "semantic"
        report["health_http"] = health.status_code
        home = client.get("/")
        assert home.status_code == 303 and home.headers["location"] == "/login"
        assert client.get("/login").status_code == 200
        assert client.get("/app/api/v1/jobs/unknown").status_code == 401
        assert client.get("/api/v1/jobs/unknown").status_code == 401
        assert client.get("/api/v1/jobs/unknown", headers={"X-API-Key": key}).status_code == 404
        report["unauthenticated_ui_http"] = 401
        report["jury_without_key_http"] = 401
        credentials = {
            "name": "Synthetic public check", "email": "public-check-" + secrets.token_hex(8) + "@example.test",
            "password": secrets.token_urlsafe(32),
        }
        headers = {"X-UI-Request": "1"}
        registration = client.post("/auth/api/register", headers=headers, json=credentials)
        assert registration.status_code == 201
        cookie_flags = registration.headers["set-cookie"].casefold()
        assert "secure" in cookie_flags and "httponly" in cookie_flags and "samesite=strict" in cookie_flags
        assert client.get("/").status_code == 200
        assert client.post("/auth/api/logout", headers=headers).status_code == 200
        assert client.get("/auth/api/me").status_code == 401
        login = client.post("/auth/api/login", headers=headers, json=credentials)
        assert login.status_code == 200
        assert client.get("/auth/api/me").status_code == 200
        assert client.get("/api/v1/jobs/unknown").status_code == 401
        report["registration_http"] = registration.status_code
        report["login_http"] = login.status_code
        report["secure_cookie"] = True
        video = artificial_mov()
        report["synthetic_video_bytes"] = len(video)
        report["jury_api"] = analyze(client, video, browser=False, key=key)
        report["browser_api"] = analyze(client, video, browser=True, key=key)
        assert client.post("/auth/api/logout", headers=headers).status_code == 200
    destination = ROOT / "data/integration-checks/public-http.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
