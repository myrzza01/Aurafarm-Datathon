import time
import os
import json
import subprocess
import sys
from backend.platform import FileLock
from backend.schemas import AnalysisResult


def clean_analyzer(path):
    return AnalysisResult(verdict="no_violation", violations=[])


def broken_analyzer(path):
    raise RuntimeError("private internal details")


def slow_analyzer(path):
    path.with_suffix(".pid").write_text(str(os.getpid()))
    time.sleep(60)
    return clean_analyzer(path)


def descendant_analyzer(path):
    marker = path.with_suffix(".child.pid")
    child = subprocess.Popen([
        sys.executable, "-c",
        "import os,sys,time; from pathlib import Path; "
        "stream=open(sys.argv[1], 'rb'); Path(sys.argv[2]).write_text(str(os.getpid())); time.sleep(60)",
        str(path), str(marker),
    ])
    deadline = time.monotonic() + 10
    while not marker.exists() and time.monotonic() < deadline:
        if child.poll() is not None:
            raise RuntimeError("The descendant exited before opening the video")
        time.sleep(.01)
    if not marker.exists():
        raise RuntimeError("The descendant did not open the video")
    if path.name.startswith("slow"):
        time.sleep(60)
    return clean_analyzer(path)


def unicode_analyzer(path):
    import base64
    import io
    from PIL import Image
    stream = io.BytesIO()
    Image.new("RGB", (8, 8)).save(stream, format="JPEG")
    return {
        "verdict": "violation", "violations": [{
            "type": "Тестовый тип", "timecode_sec": 0,
            "description": "Проверка кириллицы и қазақша мәтін",
            "evidence_frames": [{"timestamp_sec": 0, "image_base64": base64.b64encode(stream.getvalue()).decode()}],
            "rule_source": {"document": "Синтетический источник для теста кодировки"},
        }],
    }


def process_is_running(pid):
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel.WaitForSingleObject.restype = wintypes.DWORD
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x00100000, False, pid)
        if not handle:
            return False
        try:
            return kernel.WaitForSingleObject(handle, 0) == 258
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # A terminated orphan can briefly remain as a zombie until the OS reaps it.
    from pathlib import Path
    state = Path(f"/proc/{pid}/stat")
    try:
        if state.exists() and state.read_text().split(")", 1)[1].split()[0] == "Z":
            return False
    except FileNotFoundError:
        return False
    return True


def counted_analyzer(path):
    counter = path.parent / "concurrency.json"
    with FileLock(path.parent / "counter.lock"):
        data = json.loads(counter.read_text()) if counter.exists() else {"active": 0, "peak": 0}
        data["active"] += 1
        data["peak"] = max(data["peak"], data["active"])
        counter.write_text(json.dumps(data))
    time.sleep(.2)
    with FileLock(path.parent / "counter.lock"):
        data = json.loads(counter.read_text())
        data["active"] -= 1
        counter.write_text(json.dumps(data))
    return clean_analyzer(path)


def wait_job(client, job_id, headers=None, timeout=15, route="/api/v1/jobs/"):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(route + job_id, headers=headers)
        assert response.status_code == 200, response.text
        job = response.json()
        if job["status"] != "processing":
            return job
        time.sleep(.05)
    raise AssertionError("Job did not finish within test budget")
