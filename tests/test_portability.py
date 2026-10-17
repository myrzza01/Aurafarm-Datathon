import asyncio
import multiprocessing
from pathlib import Path
import time

from fastapi.testclient import TestClient
import httpx
import pytest

from backend.main import create_app
from backend.jobs import JobRunner, JobStore, JobTimeout
from backend.platform import FileLock
from tests.helpers import clean_analyzer, descendant_analyzer, process_is_running, unicode_analyzer, wait_job


def _contend_for_lock(path, result):
    try:
        with FileLock(Path(path), blocking=False):
            result.send(True)
    except BlockingIOError:
        result.send(False)
    finally:
        result.close()


def _lock_holder(path, ready):
    with FileLock(Path(path)):
        ready.set()
        time.sleep(60)


def _coordinator(path):
    runner = JobRunner(JobStore(Path(path)), descendant_analyzer, 1, 86400)
    runner.start()
    time.sleep(60)


def test_lock_is_interprocess_and_released_after_owner_dies(tmp_path):
    context = multiprocessing.get_context("spawn")
    path = str(tmp_path / "shared.lock")
    ready = context.Event()
    owner = context.Process(target=_lock_holder, args=(path, ready))
    owner.start()
    try:
        assert ready.wait(10)
        parent, child = context.Pipe(duplex=False)
        contender = context.Process(target=_contend_for_lock, args=(path, child))
        contender.start()
        child.close()
        try:
            assert parent.poll(10)
            assert parent.recv() is False
            contender.join(10)
            assert contender.exitcode == 0
        finally:
            if contender.is_alive():
                contender.kill()
                contender.join()
            contender.close()
            parent.close()
    finally:
        owner.kill()
        owner.join()
        owner.close()
    with FileLock(Path(path), blocking=False):
        pass


@pytest.mark.parametrize("name", ["slow.mov", "quick.mov"])
def test_analyzer_descendants_are_stopped_on_timeout_or_completion(tmp_path, name):
    store = JobStore(tmp_path / "jobs.sqlite3")
    path = tmp_path / name
    path.write_bytes(b"Analyzer fixture does not decode video")
    job = store.submit(path, None, 4, 1)
    runner = JobRunner(store, descendant_analyzer, 1, 86400)
    runner.start()
    try:
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and store.get(job)["state"] not in {"done", "failed"}:
            time.sleep(.05)
        row = store.get(job)
        assert row["state"] == ("failed" if name.startswith("slow") else "done")
        marker = path.with_suffix(".child.pid")
        assert marker.exists(), "The analyzer must really launch a descendant"
        child_pid = int(marker.read_text(encoding="utf-8"))
        deadline = time.monotonic() + 3
        while process_is_running(child_pid) and time.monotonic() < deadline:
            time.sleep(.05)
        assert not process_is_running(child_pid)
    finally:
        runner.close()


def test_descendants_do_not_survive_abrupt_coordinator_exit(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite3")
    path = tmp_path / "slow.mov"
    path.write_bytes(b"Analyzer fixture")
    store.submit(path, None, 30, 1)
    process = multiprocessing.get_context("spawn").Process(target=_coordinator, args=(str(store.path),))
    process.start()
    try:
        marker = path.with_suffix(".child.pid")
        deadline = time.monotonic() + 15
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(.05)
        assert marker.exists()
        child_pid = int(marker.read_text(encoding="utf-8"))
        assert process_is_running(child_pid)
        process.kill()
        process.join(5)
        deadline = time.monotonic() + 5
        while process_is_running(child_pid) and time.monotonic() < deadline:
            time.sleep(.05)
        assert not process_is_running(child_pid)
    finally:
        if process.is_alive():
            process.kill()
            process.join()
        process.close()


def test_unicode_model_result_roundtrips_through_worker_and_sqlite(tmp_path, mov_bytes):
    app = create_app(unicode_analyzer, api_key="key", upload_dir=tmp_path / "uploads")
    headers = {"X-API-Key": "key"}
    with TestClient(app) as client:
        response = client.post("/api/v1/analyze", headers=headers, files={"file": ("test.mov", mov_bytes)})
        assert response.status_code == 202
        result = wait_job(client, response.json()["job_id"], headers)
        assert result["status"] == "done"
        event = result["violations"][0]
        assert event["type"] == "Тестовый тип"
        assert event["description"] == "Проверка кириллицы и қазақша мәтін"


def test_slow_multipart_upload_consumes_analysis_budget_and_closes_spool(tmp_path, monkeypatch):
    from starlette import formparsers
    files = []
    original = formparsers.SpooledTemporaryFile

    def capture_file(*args, **kwargs):
        result = original(*args, **kwargs)
        files.append(result)
        return result

    monkeypatch.setattr(formparsers, "SpooledTemporaryFile", capture_file)
    app = create_app(clean_analyzer, api_key="key", upload_dir=tmp_path / "uploads", analysis_timeout=.15)

    async def run():
        async def body():
            yield b'--test\r\nContent-Disposition: form-data; name="file"; filename="slow.mov"\r\nContent-Type: video/quicktime\r\n\r\n'
            yield b"partial video bytes"
            await asyncio.sleep(1)
            yield b"\r\n--test--\r\n"
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            return await client.post("/api/v1/analyze", headers={
                "X-API-Key": "key", "Content-Type": "multipart/form-data; boundary=test",
            }, content=body())

    started = time.monotonic()
    response = asyncio.run(run())
    assert response.status_code == 408, response.text
    assert time.monotonic() - started < .8
    assert files and all(stream.closed for stream in files)
    assert not list((tmp_path / "uploads").iterdir())
    with JobStore(tmp_path / "jobs.sqlite3").connect() as db:
        assert db.execute("SELECT count(*) FROM jobs").fetchone()[0] == 0


def test_completed_upload_does_not_reset_queue_deadline(tmp_path, monkeypatch):
    validation_timeouts = []
    monkeypatch.setattr("backend.main.validate_video", lambda path, timeout: validation_timeouts.append(timeout))
    app = create_app(clean_analyzer, api_key="key", upload_dir=tmp_path / "uploads", analysis_timeout=.8)

    async def run():
        async def body():
            yield b'--test\r\nContent-Disposition: form-data; name="file"; filename="test.mov"\r\n\r\n'
            await asyncio.sleep(.25)
            yield b"fixture bytes\r\n--test--\r\n"
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            return await client.post("/api/v1/analyze", headers={
                "X-API-Key": "key", "Content-Type": "multipart/form-data; boundary=test",
            }, content=body())

    started = time.time()
    response = asyncio.run(run())
    assert response.status_code == 202, response.text
    assert 0 < validation_timeouts[0] < .65
    store = JobStore(tmp_path / "jobs.sqlite3")
    row = store.get(response.json()["job_id"])
    assert row["deadline"] <= started + .9
    store.finish(row["job_id"], error="Test cleanup")


def test_database_lock_cannot_extend_submission_deadline(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite3")
    path = tmp_path / "fixture.mov"
    path.write_bytes(b"fixture")
    db = store.connect()
    try:
        db.execute("BEGIN IMMEDIATE")
        started = time.monotonic()
        with pytest.raises(JobTimeout):
            store.submit(path, None, .15, 1)
        assert time.monotonic() - started < .8
    finally:
        db.rollback()
        db.close()
