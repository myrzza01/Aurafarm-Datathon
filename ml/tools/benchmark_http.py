"""Measure the real local team's asynchronous HTTP API without logging its content.

This client never starts an inference service of its own. It accepts only loopback,
RFC1918 IPv4 or ULA IPv6 literal addresses (plus localhost), bypasses proxies, and
never follows redirects. Response images/descriptions and API keys are not saved.
"""
from __future__ import annotations

import argparse
import http.client
import ipaddress
import json
import os
from pathlib import Path
import socket
import ssl
import subprocess
import time
from urllib.parse import quote, urlsplit
import uuid


class BenchmarkError(RuntimeError):
    """A deliberately content-free error, safe for aggregate reports."""


def local_endpoint(url):
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or parsed.username or parsed.password:
        raise ValueError("Use http(s) with no credentials in the URL")
    if parsed.query or parsed.fragment or not parsed.hostname:
        raise ValueError("Base URL must have a host and no query or fragment")
    host = parsed.hostname
    if host == "localhost":
        host = "127.0.0.1"
    else:
        try:
            address = ipaddress.ip_address(host)
        except ValueError as exc:
            raise ValueError("Use localhost or a numeric local/LAN address") from exc
        networks = [ipaddress.ip_network(value) for value in
                    ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "fc00::/7")]
        if not address.is_loopback and not any(address in network for network in networks
                                               if address.version == network.version):
            raise ValueError("Public and non-LAN endpoints are prohibited")
    return parsed.scheme, host, parsed.port or (443 if parsed.scheme == "https" else 80), parsed.path.rstrip("/")


def connection(endpoint, timeout):
    scheme, host, port, _ = endpoint
    if scheme == "https":
        return http.client.HTTPSConnection(host, port, timeout=timeout, context=ssl.create_default_context())
    return http.client.HTTPConnection(host, port, timeout=timeout)


def read_json_response(conn, deadline):
    response = conn.getresponse()
    if not 200 <= response.status < 300:
        # Do not log response text, which can contain scenes, filenames or keys.
        raise BenchmarkError(f"HTTP status {response.status}")
    pieces, size = [], 0
    while True:
        # read1 performs at most one buffered/raw read; enforce the shared deadline.
        if conn.sock is not None:
            conn.sock.settimeout(remaining(deadline))
        chunk = response.read1(65536)
        remaining(deadline)
        if not chunk:
            break
        size += len(chunk)
        if size > 64 * 1024 * 1024:
            raise BenchmarkError("HTTP response exceeds 64 MiB")
        pieces.append(chunk)
    body = b"".join(pieces)
    try:
        result = json.loads(body)
    except (ValueError, UnicodeError) as exc:
        raise BenchmarkError("HTTP response is not JSON") from exc
    if not isinstance(result, dict):
        raise BenchmarkError("Expected a JSON object")
    return result


def remaining(deadline):
    left = deadline - time.perf_counter()
    if left <= 0:
        raise TimeoutError("Full request time budget exceeded")
    return left


def upload(endpoint, video, api_key, deadline):
    boundary = "cashwatch_" + uuid.uuid4().hex
    suffix = video.suffix.lower()
    if not suffix.isascii() or not suffix.replace(".", "").isalnum():
        suffix = ".video"
    header = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
              f"filename=\"input{suffix}\"\r\nContent-Type: application/octet-stream\r\n\r\n").encode("ascii")
    footer = f"\r\n--{boundary}--\r\n".encode("ascii")
    conn = connection(endpoint, remaining(deadline))
    try:
        conn.putrequest("POST", endpoint[3] + "/api/v1/analyze")
        conn.putheader("X-API-Key", api_key)
        conn.putheader("Content-Type", f"multipart/form-data; boundary={boundary}")
        conn.putheader("Content-Length", len(header) + video.stat().st_size + len(footer))
        conn.endheaders()
        conn.send(header)
        with video.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                conn.sock.settimeout(remaining(deadline))
                conn.send(chunk)
        conn.sock.settimeout(remaining(deadline))
        conn.send(footer)
        return read_json_response(conn, deadline)
    finally:
        conn.close()


def get_job(endpoint, job_id, api_key, deadline):
    conn = connection(endpoint, remaining(deadline))
    try:
        conn.request("GET", endpoint[3] + "/api/v1/jobs/" + quote(job_id, safe=""),
                     headers={"X-API-Key": api_key})
        return read_json_response(conn, deadline)
    finally:
        conn.close()


def check_payload(body):
    """Only check shape; this deliberately makes no claim about detection quality."""
    result = body.get("result", body)
    if not isinstance(result, dict) or result.get("verdict") not in {"violation", "no_violation"}:
        raise BenchmarkError("Completed response lacks a valid verdict")
    events = result.get("violations")
    if not isinstance(events, list):
        raise BenchmarkError("Completed response lacks violations array")
    evidence_count = 0
    for event in events:
        if not isinstance(event, dict) or not {"type", "timecode_sec", "description", "evidence_frames"} <= event.keys():
            raise BenchmarkError("Event shape does not match the required API")
        frames = event["evidence_frames"]
        if not isinstance(frames, list) or not 1 <= len(frames) <= 3:
            raise BenchmarkError("Expected 1-3 evidence frames per event")
        if any(not isinstance(frame, dict) or not {"timestamp_sec", "image_base64"} <= frame.keys() for frame in frames):
            raise BenchmarkError("Evidence frame shape is invalid")
        evidence_count += len(frames)
    if (result["verdict"] == "violation") != bool(events):
        raise BenchmarkError("Verdict and event count disagree")
    return {"events": len(events), "evidence_frames": evidence_count, "api_shape_passed": True}


def benchmark_request(endpoint, video, api_key, budget_sec=300.0, poll_sec=0.2):
    started = time.perf_counter()
    deadline = started + budget_sec
    submitted = upload(endpoint, video, api_key, deadline)
    accepted = time.perf_counter()
    job_id = submitted.get("job_id")
    if not isinstance(job_id, str) or not job_id or len(job_id) > 512:
        raise BenchmarkError("POST response lacks a string job_id")
    polls = 0
    while True:
        body = get_job(endpoint, job_id, api_key, deadline)
        polls += 1
        status = body.get("status")
        if status == "done":
            aggregate = check_payload(body)
            break
        if status == "failed":
            raise BenchmarkError("Backend job failed")
        if status != "processing":
            raise BenchmarkError("Unknown backend job status")
        time.sleep(min(poll_sec, remaining(deadline)))
    finished = time.perf_counter()
    if finished > deadline:
        raise TimeoutError("Full request time budget exceeded")
    return {"full_http_request_sec": finished - started,
            "upload_and_accept_sec": accepted - started,
            "poll_until_done_sec": finished - accepted, "polls": polls,
            "poll_interval_sec": poll_sec, "within_300_sec": finished - started <= 300, **aggregate}


def wait_ready(endpoint, process, timeout):
    deadline = time.perf_counter() + timeout
    while True:
        if process.poll() is not None:
            raise BenchmarkError("Started backend exited before opening its port")
        try:
            with socket.create_connection((endpoint[1], endpoint[2]), timeout=min(1.0, remaining(deadline))):
                return
        except OSError:
            time.sleep(min(0.1, remaining(deadline)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="Local backend URL; public hosts are rejected")
    parser.add_argument("--video", required=True)
    parser.add_argument("--api-key-env", default="CASHWATCH_API_KEY")
    parser.add_argument("--output", default="artifacts/http_benchmark.json")
    parser.add_argument("--warm-runs", type=int, default=3)
    parser.add_argument("--poll-sec", type=float, default=0.2)
    parser.add_argument("--timeout-sec", type=float, default=300.0)
    parser.add_argument("--start-command-file", help="Local JSON {argv: [...], cwd: ...}; foreground backend, no shell")
    parser.add_argument("--startup-timeout-sec", type=float, default=120.0)
    args = parser.parse_args()
    endpoint = local_endpoint(args.url)
    video = Path(args.video).resolve(strict=True)
    api_key = os.environ.get(args.api_key_env)
    if not api_key or "\r" in api_key or "\n" in api_key:
        parser.error("Set a valid API key in the selected environment variable")
    if args.warm_runs < 0 or args.poll_sec <= 0 or args.timeout_sec <= 0 or args.startup_timeout_sec <= 0:
        parser.error("Invalid run count or timeout")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "source": "real_backend_http", "measurements": [],
              "video_bytes": video.stat().st_size, "cold_start_measured": False,
              "content_quality_validated": False,
              "note": "API shape checks do not validate event content; polling adds bounded sampling delay."}
    process = None
    try:
        if args.start_command_file:
            if endpoint[1] not in {"127.0.0.1", "::1"}:
                raise ValueError("Starting a process is supported only for a loopback backend")
            # A pre-existing listener would make a claimed cold start invalid.
            try:
                with socket.create_connection((endpoint[1], endpoint[2]), timeout=0.5):
                    raise BenchmarkError("Port is already in use; a cold start cannot be asserted")
            except (ConnectionRefusedError, TimeoutError):
                pass
            launch = json.loads(Path(args.start_command_file).read_text(encoding="utf-8-sig"))
            argv = launch.get("argv")
            if not isinstance(argv, list) or not argv or not all(isinstance(value, str) for value in argv):
                raise ValueError("Start command file must contain a nonempty argv string array")
            started = time.perf_counter()
            process = subprocess.Popen(argv, cwd=launch.get("cwd"), shell=False,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            wait_ready(endpoint, process, args.startup_timeout_sec)
            report["startup_to_listening_sec"] = time.perf_counter() - started
            report["cold_start_measured"] = True
        for index in range(args.warm_runs + 1):
            measurement = benchmark_request(endpoint, video, api_key, args.timeout_sec, args.poll_sec)
            measurement["run"] = index
            measurement["kind"] = ("first_after_process_start" if process else "first_observed_request") if index == 0 else "warm"
            if index == 0 and process:
                measurement["startup_plus_first_request_sec"] = time.perf_counter() - started
                measurement["startup_plus_request_within_300_sec"] = measurement["startup_plus_first_request_sec"] <= 300
            report["measurements"].append(measurement)
        report["status"] = "passed"
    except Exception as exc:
        report["status"] = "failed"
        report["error_type"] = type(exc).__name__
        if isinstance(exc, BenchmarkError):
            report["reason"] = str(exc)
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "cold_start_measured": report["cold_start_measured"],
                      "completed_requests": len(report["measurements"])}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
