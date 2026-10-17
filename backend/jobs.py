"""Durable jobs and a bounded, process-isolated local analysis queue."""
from contextlib import closing
import json
import multiprocessing
import os
from pathlib import Path
import sqlite3
import threading
import time
from uuid import uuid4

from backend.analyzer import AnalyzerNotConfigured
from backend.schemas import AnalysisResult, JobResponse, MAX_RESULT_BYTES
from backend.platform import FileLock, ParentMonitor, ProcessTree, kill_current_tree, prepare_child


class QueueFull(RuntimeError):
    pass


class JobTimeout(RuntimeError):
    pass


class JobStore:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self.connect()) as db, db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY, owner TEXT, state TEXT NOT NULL,
                    created REAL NOT NULL, updated REAL NOT NULL, deadline REAL NOT NULL,
                    upload_path TEXT, result_json TEXT, error TEXT
                );
                CREATE INDEX IF NOT EXISTS jobs_state_created ON jobs(state, created);
            """)
        path.chmod(0o600)

    def connect(self, timeout: float = 10):
        db = sqlite3.connect(self.path, timeout=timeout)
        db.row_factory = sqlite3.Row
        return db

    def submit(self, path: Path, owner: str | None, timeout: float, capacity: int) -> str:
        now = time.time()
        deadline = now + timeout
        job_id = uuid4().hex
        try:
            with closing(self.connect(timeout=min(10, max(.001, timeout)))) as db, db:
                db.execute("BEGIN IMMEDIATE")
                if time.time() >= deadline:
                    raise JobTimeout()
                count = db.execute("SELECT count(*) FROM jobs WHERE state IN ('queued','running')").fetchone()[0]
                if count >= capacity:
                    raise QueueFull()
                db.execute("INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?,?)", (
                    job_id, owner, "queued", now, now, deadline, str(path.resolve()), None, None,
                ))
        except sqlite3.OperationalError as exc:
            if time.time() >= deadline:
                raise JobTimeout() from exc
            raise
        return job_id

    def get(self, job_id: str):
        with closing(self.connect()) as db:
            row = db.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        return dict(row) if row else None

    @staticmethod
    def response(row: dict) -> JobResponse:
        return JobResponse(
            job_id=row["job_id"], status=row["state"] if row["state"] in {"done", "failed"} else "processing",
            error=row["error"], **(json.loads(row["result_json"]) if row["result_json"] else {}),
        )

    def claim(self):
        with closing(self.connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM jobs WHERE state='queued' AND deadline>? ORDER BY created LIMIT 1", (time.time(),)).fetchone()
            if row:
                db.execute("UPDATE jobs SET state='running',updated=? WHERE job_id=?", (time.time(), row["job_id"]))
                return dict(row)
        return None

    def finish(self, job_id: str, *, result: str | None = None, error: str | None = None):
        row = self.get(job_id)
        with closing(self.connect()) as db, db:
            db.execute(
                "UPDATE jobs SET state=?,result_json=?,error=?,updated=?,upload_path=NULL WHERE job_id=? AND state IN ('queued','running')",
                ("failed" if error else "done", result, error, time.time(), job_id),
            )
        if row and row["upload_path"]:
            Path(row["upload_path"]).unlink(missing_ok=True)

    def recover(self):
        # Only the scheduler holding the exclusive filesystem lock calls this.
        with closing(self.connect()) as db:
            rows = db.execute("SELECT job_id FROM jobs WHERE state='running'").fetchall()
        for row in rows:
            self.finish(row["job_id"], error="Обработка прервана перезапуском сервера. Загрузите видео заново.")

    def active_paths(self):
        with closing(self.connect()) as db:
            rows = db.execute("SELECT upload_path FROM jobs WHERE state IN ('queued','running')").fetchall()
        return {row[0] for row in rows if row[0]}

    def maintain(self, retention: float):
        now = time.time()
        with closing(self.connect()) as db:
            expired = db.execute("SELECT job_id FROM jobs WHERE state='queued' AND deadline<=?", (now,)).fetchall()
        for row in expired:
            self.finish(row["job_id"], error="Время ожидания и обработки превысило допустимый лимит")
        with closing(self.connect()) as db, db:
            db.execute("DELETE FROM jobs WHERE state IN ('done','failed') AND updated<?", (now - retention,))
            db.execute("""DELETE FROM jobs WHERE job_id IN (
                SELECT job_id FROM jobs WHERE state IN ('done','failed') ORDER BY updated DESC LIMIT -1 OFFSET 200
            )""")


def _analyze_child(analyzer, video_path: str, output_path: str, deadline: float, parent_pid: int, gate, job_name):
    prepare_child()
    def watchdog():
        # Also stop an orphaned model after an ungraceful API-process exit.
        with ParentMonitor(parent_pid) as parent:
            while time.time() < deadline and parent.alive():
                time.sleep(.1)
        kill_current_tree(job_name)
    threading.Thread(target=watchdog, daemon=True).start()
    # On Windows the parent must assign the process to its Job Object first.
    gate.wait()
    try:
        result = AnalysisResult.model_validate(analyzer(Path(video_path)))
        payload = result.model_dump_json()
        if len(payload.encode()) > MAX_RESULT_BYTES:
            raise ValueError("Result too large")
        body = {"result": payload}
    except AnalyzerNotConfigured:
        body = {"error": "Локальный обработчик модели недоступен. Проверьте ANALYZER_CALLABLE."}
    except Exception:
        # Do not log model payloads, validation inputs, frames or credentials.
        body = {"error": "Не удалось обработать видео или модель вернула некорректный результат"}
    path = Path(output_path)
    with path.open("x", encoding="utf-8") as stream:
        path.chmod(0o600)
        json.dump(body, stream, ensure_ascii=False)


class JobRunner:
    def __init__(self, store: JobStore, analyzer, workers: int, retention: float, upload_dir: Path | None = None):
        self.store, self.analyzer = store, analyzer
        self.workers, self.retention = workers, retention
        self.upload_dir = upload_dir
        self.stop = threading.Event()
        self.thread = None
        self.users = 0
        self.lifecycle_lock = threading.Lock()
        self.work = store.path.parent / "job-work"
        self.work.mkdir(parents=True, exist_ok=True)
        self.work.chmod(0o700)

    def start(self):
        with self.lifecycle_lock:
            self.users += 1
            if self.users == 1:
                self.stop.clear()
                self.thread = threading.Thread(target=self._coordinate, daemon=True, name="forte-jobs")
                self.thread.start()

    def close(self):
        with self.lifecycle_lock:
            self.users -= 1
            if self.users == 0:
                self.stop.set()
                if self.thread:
                    self.thread.join()

    def _coordinate(self):
        # A single leader coordinates all API workers sharing this local data directory.
        while not self.stop.is_set():
            try:
                with FileLock(self.store.path.parent / "jobs.lock", blocking=False):
                    self._run()
                return
            except BlockingIOError:
                self.stop.wait(.1)

    def _run(self):
        self.store.recover()
        for path in self.work.glob("*.json"):
            path.unlink(missing_ok=True)
        context = multiprocessing.get_context("spawn")
        active = {}
        next_cleanup = 0
        try:
            while not self.stop.is_set():
                self.store.maintain(self.retention)
                if self.upload_dir is not None and time.monotonic() >= next_cleanup:
                    known = self.store.active_paths()
                    for path in self.upload_dir.glob("tmp*.mov"):
                        try:
                            if str(path.resolve()) not in known and path.stat().st_mtime < time.time() - 600:
                                path.unlink(missing_ok=True)
                        except FileNotFoundError:
                            pass
                    next_cleanup = time.monotonic() + 60
                for job_id, (row, process, output, tree) in list(active.items()):
                    expired = time.time() >= row["deadline"]
                    if not expired and process.is_alive():
                        continue
                    self._stop_process(process, tree)
                    error = "Время ожидания и обработки превысило допустимый лимит" if expired else None
                    payload = None
                    if not error:
                        try:
                            if output.stat().st_size > MAX_RESULT_BYTES * 2:
                                raise ValueError()
                            data = json.loads(output.read_text(encoding="utf-8"))
                            payload, error = data.get("result"), data.get("error")
                            if not payload and not error:
                                raise ValueError()
                        except (OSError, ValueError, TypeError):
                            error = "Процесс анализа завершился без результата"
                    self.store.finish(job_id, result=payload, error=error)
                    output.unlink(missing_ok=True)
                    process.close()
                    del active[job_id]
                while len(active) < self.workers and not self.stop.is_set():
                    row = self.store.claim()
                    if not row:
                        break
                    output = self.work / f'{row["job_id"]}.json'
                    tree = None
                    process = None
                    try:
                        tree = ProcessTree()
                        gate = context.Event()
                        process = context.Process(target=_analyze_child, args=(
                            self.analyzer, row["upload_path"], str(output), row["deadline"], os.getpid(), gate, tree.name,
                        ), daemon=True)
                        process.start()
                        tree.attach(process)
                        gate.set()
                    except Exception:
                        if process is not None:
                            if process.pid is not None:
                                self._stop_process(process, tree)
                            process.close()
                        if tree is not None:
                            tree.close()
                        self.store.finish(row["job_id"], error="Не удалось запустить процесс анализа")
                        continue
                    active[row["job_id"]] = (row, process, output, tree)
                self.stop.wait(.05)
        finally:
            for job_id, (_, process, output, tree) in active.items():
                self._stop_process(process, tree)
                process.close()
                self.store.finish(job_id, error="Обработка прервана остановкой сервера. Загрузите видео заново.")
                output.unlink(missing_ok=True)

    @staticmethod
    def _stop_process(process, tree):
        # Kill any subprocesses created by a local model, even if its entrypoint exited.
        try:
            tree.terminate(process)
        finally:
            tree.close()
