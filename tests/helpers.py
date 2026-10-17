import time
import os
import json
import fcntl
from backend.schemas import AnalysisResult


def clean_analyzer(path):
    return AnalysisResult(verdict="no_violation", violations=[])


def broken_analyzer(path):
    raise RuntimeError("private internal details")


def slow_analyzer(path):
    path.with_suffix(".pid").write_text(str(os.getpid()))
    time.sleep(60)
    return clean_analyzer(path)


def counted_analyzer(path):
    counter = path.parent / "concurrency.json"
    with (path.parent / "counter.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = json.loads(counter.read_text()) if counter.exists() else {"active": 0, "peak": 0}
        data["active"] += 1
        data["peak"] = max(data["peak"], data["active"])
        counter.write_text(json.dumps(data))
    time.sleep(.2)
    with (path.parent / "counter.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
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
