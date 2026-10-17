"""Synthetic transport checks only: they do not measure the team's real backend."""
import importlib.util
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("benchmark_http", ROOT / "tools" / "benchmark_http.py")
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


class Handler(BaseHTTPRequestHandler):
    calls = 0
    fail_job = False
    uploaded = b""

    def log_message(self, *args):
        pass

    def respond(self, body, status=200):
        payload = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self):
        if self.headers.get("X-API-Key") != "synthetic-test-key":
            return self.respond({}, 403)
        if self.path != "/api/v1/analyze":
            return self.respond({}, 404)
        type(self).uploaded = self.rfile.read(int(self.headers["Content-Length"]))
        type(self).calls = 0
        self.respond({"job_id": "synthetic-id"}, 202)

    def do_GET(self):
        if self.headers.get("X-API-Key") != "synthetic-test-key":
            return self.respond({}, 403)
        if self.path != "/api/v1/jobs/synthetic-id":
            return self.respond({}, 404)
        type(self).calls += 1
        if self.fail_job:
            return self.respond({"status": "failed", "error": "private server details must not be logged"})
        if self.calls == 1:
            return self.respond({"status": "processing"})
        self.respond({"status": "done", "verdict": "no_violation", "violations": []})


class HttpBenchmarkTests(unittest.TestCase):
    def test_rejects_public_hosts_and_credentials(self):
        for url in ["https://example.com", "http://1.1.1.1", "http://100.64.0.1", "http://user:pass@127.0.0.1"]:
            with self.subTest(url=url), self.assertRaises(ValueError):
                benchmark.local_endpoint(url)
        self.assertEqual(benchmark.local_endpoint("http://localhost:8080")[1], "127.0.0.1")
        self.assertEqual(benchmark.local_endpoint("http://192.168.1.20")[1], "192.168.1.20")

    def test_multipart_polling_and_explicit_failure(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as folder:
                video = Path(folder) / "synthetic.bin"
                video.write_bytes(b"SYNTHETIC_TRANSPORT_BYTES_NOT_A_VIDEO")
                endpoint = benchmark.local_endpoint(f"http://127.0.0.1:{server.server_port}")
                result = benchmark.benchmark_request(endpoint, video, "synthetic-test-key", 5, 0.01)
                self.assertTrue(result["api_shape_passed"])
                self.assertEqual(result["polls"], 2)
                self.assertIn(b'name="file"', Handler.uploaded)
                self.assertIn(video.read_bytes(), Handler.uploaded)
                self.assertNotIn(video.name.encode(), Handler.uploaded)
                Handler.fail_job = True
                with self.assertRaisesRegex(benchmark.BenchmarkError, "Backend job failed"):
                    benchmark.benchmark_request(endpoint, video, "synthetic-test-key", 5, 0.01)
                Handler.fail_job = False
        finally:
            Handler.fail_job = False
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_cold_process_and_aggregate_only_report(self):
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            video = folder / "synthetic.bin"
            video.write_bytes(b"SYNTHETIC_TRANSPORT_BYTES_NOT_A_VIDEO")
            command = folder / "command.json"
            command.write_text(json.dumps({"argv": [sys.executable, str(Path(__file__).resolve()), "--serve", str(port)]}), encoding="utf-8")
            output = folder / "report.json"
            env = dict(os.environ, CASHWATCH_API_KEY="synthetic-test-key")
            completed = subprocess.run([sys.executable, str(ROOT / "tools" / "benchmark_http.py"),
                "--url", f"http://127.0.0.1:{port}", "--video", str(video),
                "--start-command-file", str(command), "--output", str(output),
                "--warm-runs", "1", "--poll-sec", "0.01"], env=env, capture_output=True, text=True, timeout=30)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            report = json.loads(output.read_text())
            self.assertTrue(report["cold_start_measured"])
            self.assertEqual(len(report["measurements"]), 2)
            self.assertGreater(report["measurements"][0]["startup_plus_first_request_sec"],
                               report["measurements"][0]["full_http_request_sec"])
            self.assertFalse(report["content_quality_validated"])
            self.assertNotIn("synthetic-test-key", output.read_text())
            self.assertNotIn("SYNTHETIC_TRANSPORT_BYTES", output.read_text())
            with socket.socket() as closed:
                closed.settimeout(1)
                self.assertNotEqual(closed.connect_ex(("127.0.0.1", port)), 0)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--serve":
        ThreadingHTTPServer(("127.0.0.1", int(sys.argv[2])), Handler).serve_forever()
    else:
        unittest.main()
