"""Worker for the Windows Firewall experiment. No Python socket blocking is used."""
from __future__ import annotations

import argparse
import ctypes
import ipaddress
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time


def process_image():
    if os.name != "nt":
        raise RuntimeError("This verifier requires Windows")
    buffer = ctypes.create_unicode_buffer(32768)
    if not ctypes.windll.kernel32.GetModuleFileNameW(None, buffer, len(buffer)):
        raise OSError("Cannot determine the native Python executable")
    return buffer.value


def probe(ip, port):
    # A TCP handshake only: no HTTP, credentials, media, paths or scene data.
    ipaddress.ip_address(ip)
    started = time.perf_counter()
    try:
        with socket.create_connection((ip, port), timeout=4.0):
            return {"connected": True, "elapsed_sec": time.perf_counter() - started}
    except OSError as exc:
        return {"connected": False, "elapsed_sec": time.perf_counter() - started,
                "error_type": type(exc).__name__, "errno": exc.errno,
                "winerror": getattr(exc, "winerror", None)}


def infer(bundle, video):
    sys.path.insert(0, str(Path(bundle).resolve()))
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    network_events = []

    def audit(event, args):
        if event in {"socket.connect", "socket.getaddrinfo", "socket.sendto"}:
            network_events.append(event)

    sys.addaudithook(audit)  # Observe only; the OS firewall supplies enforcement.
    started = time.perf_counter()
    from cashwatch.inference import VideoAnalyzer
    if "cashwatch.data" in sys.modules or "cashwatch.train" in sys.modules:
        raise RuntimeError("Unexpected training module import")
    analyzer = VideoAnalyzer(Path(bundle) / "weights" / "detector.pt")
    loaded = time.perf_counter()
    result, diagnostics = analyzer.analyze(video, timeout_sec=240.0)
    return {"passed": True, "cold_import_and_model_load_sec": loaded - started,
            "cold_local_inference_sec": time.perf_counter() - started,
            "events": len(result["violations"]), "python_network_audit_events": len(network_events),
            "python_socket_guard_used": False, "training_modules_imported": False,
            "diagnostics": {key: value for key, value in diagnostics.items()
                            if key not in {"intervals", "limitation"}},
            "content_quality_validated": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["executables", "probe", "infer", "infer-child"])
    parser.add_argument("--ip", default="1.1.1.1")
    parser.add_argument("--port", type=int, default=443)
    parser.add_argument("--bundle")
    parser.add_argument("--video")
    args = parser.parse_args()
    try:
        if args.mode == "executables":
            result = {"paths": sorted({str(Path(path).resolve()) for path in
                       [sys.executable, sys._base_executable, process_image()]})}
        elif args.mode == "probe":
            result = probe(args.ip, args.port)
        elif args.mode == "infer-child":
            result = infer(args.bundle, args.video)
        else:
            started = time.perf_counter()
            completed = subprocess.run([sys.executable, "-I", str(Path(__file__).resolve()),
                "infer-child", "--bundle", args.bundle, "--video", args.video],
                capture_output=True, text=True, timeout=300, check=False)
            if completed.returncode:
                result = {"passed": False, "error_type": "InferenceChildFailed", "exit_code": completed.returncode}
            else:
                result = json.loads(completed.stdout)
                result["separate_process_wall_sec"] = time.perf_counter() - started
        print(json.dumps(result))
        return 0 if result.get("passed", True) else 1
    except Exception as exc:
        print(json.dumps({"passed": False, "error_type": type(exc).__name__}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
