import logging
import os
import secrets
import sqlite3
import tempfile
import time
from threading import Lock
from pathlib import Path
from typing import Annotated, Callable
from uuid import uuid4

from fastapi import BackgroundTasks, Cookie, Depends, FastAPI, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles

from backend.analyzer import AnalyzerNotConfigured, analyze_video
from backend.auth import AuthStore, LoginData, RegisterData, SESSION_SECONDS
from backend.schemas import AnalysisResult, JobCreated, JobResponse

logger = logging.getLogger(__name__)
key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def create_app(
    analyzer: Callable[[Path], AnalysisResult] = analyze_video,
    *,
    api_key: str | None = None,
    upload_dir: Path | None = None,
    auth_db: Path | None = None,
    max_upload_bytes: int = 100 * 1024 * 1024,
) -> FastAPI:
    expected_key = api_key or os.environ.get("API_KEY")
    if not expected_key:
        raise RuntimeError("Set API_KEY before starting the server")
    storage = upload_dir or Path("data/uploads")
    storage.mkdir(parents=True, exist_ok=True)
    jobs: dict[str, JobResponse] = {}
    job_owners: dict[str, str | None] = {}
    app = FastAPI(title="ForteBank Video Review API", version="0.1.0")
    frontend = Path(__file__).resolve().parent.parent / "frontend"
    accounts = AuthStore(auth_db or storage.parent / "auth.sqlite3")
    attempts: dict[str, list[float]] = {}
    attempts_lock = Lock()
    app.mount("/assets", StaticFiles(directory=frontend / "assets"), name="assets")

    def require_key(key: Annotated[str | None, Depends(key_header)]):
        if key is None or not secrets.compare_digest(key.encode(), expected_key.encode()):
            raise HTTPException(status_code=401, detail="Invalid API key")

    def require_ui_header(request: Request):
        if request.headers.get("X-UI-Request") != "1":
            raise HTTPException(status_code=403, detail="Invalid UI request")

    def auth_rate_limit(request: Request):
        require_ui_header(request)
        address = request.client.host if request.client else "local"
        now = time.monotonic()
        with attempts_lock:
            for key in list(attempts):
                attempts[key] = [stamp for stamp in attempts[key] if now - stamp < 60]
                if not attempts[key]:
                    del attempts[key]
            recent = attempts.setdefault(address, [])
            if len(recent) >= 10:
                raise HTTPException(status_code=429, detail="Слишком много попыток. Подождите минуту.", headers={"Retry-After": "60"})
            recent.append(now)

    def require_ui_session(
        request: Request, ui_session: Annotated[str | None, Cookie()] = None,
    ):
        user = accounts.user_for_session(ui_session)
        if not user:
            raise HTTPException(status_code=401, detail="Войдите в аккаунт")
        if request.method == "POST":
            require_ui_header(request)
        request.state.user = user
        return user

    def set_session(response: Response, request: Request, user_id: str):
        accounts.revoke(request.cookies.get("ui_session"))
        response.set_cookie(
            "ui_session", accounts.create_session(user_id), httponly=True,
            samesite="strict", secure=request.url.scheme == "https", max_age=SESSION_SECONDS,
        )
        response.headers["Cache-Control"] = "no-store"

    @app.get("/login", include_in_schema=False)
    @app.get("/register", include_in_schema=False)
    def auth_page(request: Request):
        if accounts.user_for_session(request.cookies.get("ui_session")):
            return RedirectResponse("/", status_code=303)
        return FileResponse(frontend / "auth.html", headers={"Cache-Control": "no-store"})

    @app.post("/auth/api/register", status_code=201, include_in_schema=False, dependencies=[Depends(auth_rate_limit)])
    def register(data: RegisterData, request: Request, response: Response):
        try:
            user = accounts.register(data)
        except sqlite3.IntegrityError:
            raise HTTPException(status_code=409, detail="Не удалось создать аккаунт с этим email. Попробуйте войти.")
        set_session(response, request, user["id"])
        return {"user": user}

    @app.post("/auth/api/login", include_in_schema=False, dependencies=[Depends(auth_rate_limit)])
    def login(data: LoginData, request: Request, response: Response):
        user = accounts.authenticate(data)
        if not user:
            raise HTTPException(status_code=401, detail="Неверный email или пароль")
        set_session(response, request, user["id"])
        return {"user": user}

    @app.get("/auth/api/me", include_in_schema=False)
    def me(response: Response, user: Annotated[dict, Depends(require_ui_session)]):
        response.headers["Cache-Control"] = "no-store"
        return {"user": user}

    @app.post("/auth/api/logout", include_in_schema=False, dependencies=[Depends(require_ui_header)])
    def logout(request: Request, response: Response):
        accounts.revoke(request.cookies.get("ui_session"))
        response.delete_cookie("ui_session", httponly=True, samesite="strict", secure=request.url.scheme == "https")
        response.headers["Cache-Control"] = "no-store"
        return {"ok": True}

    @app.get("/", include_in_schema=False)
    def home(request: Request):
        if not accounts.user_for_session(request.cookies.get("ui_session")):
            return RedirectResponse("/login", status_code=303)
        response = FileResponse(frontend / "index.html", headers={"Cache-Control": "no-store"})
        return response

    def process(job_id: str, video_path: Path):
        try:
            result = AnalysisResult.model_validate(analyzer(video_path))
            jobs[job_id] = JobResponse(
                job_id=job_id, status="done", **result.model_dump()
            )
        except AnalyzerNotConfigured as exc:
            jobs[job_id] = JobResponse(job_id=job_id, status="failed", error=str(exc))
        except Exception:
            logger.exception("Analysis failed for job %s", job_id)
            jobs[job_id] = JobResponse(
                job_id=job_id, status="failed", error="Не удалось обработать видео"
            )
        finally:
            video_path.unlink(missing_ok=True)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post(
        "/app/api/v1/analyze", response_model=JobCreated, status_code=202,
        dependencies=[Depends(require_ui_session)], include_in_schema=False,
    )
    @app.post(
        "/api/v1/analyze", response_model=JobCreated, status_code=202,
        dependencies=[Depends(require_key)],
    )
    def analyze(file: UploadFile, background_tasks: BackgroundTasks, request: Request):
        path = None
        try:
            if Path(file.filename or "").suffix.lower() != ".mov":
                raise HTTPException(status_code=415, detail="Expected a .mov file")
            size = 0
            with tempfile.NamedTemporaryFile(dir=storage, suffix=".mov", delete=False) as out:
                path = Path(out.name)
                while chunk := file.file.read(1024 * 1024):
                    size += len(chunk)
                    if size > max_upload_bytes:
                        raise HTTPException(status_code=413, detail="File exceeds upload limit")
                    out.write(chunk)
            if size == 0:
                raise HTTPException(status_code=400, detail="Empty file")
            job_id = uuid4().hex
            jobs[job_id] = JobResponse(job_id=job_id, status="processing")
            job_owners[job_id] = getattr(request.state, "user", {}).get("id")
            background_tasks.add_task(process, job_id, path)
            return JobCreated(job_id=job_id)
        except Exception:
            if path is not None:
                path.unlink(missing_ok=True)
            raise
        finally:
            file.file.close()

    @app.get(
        "/app/api/v1/jobs/{job_id}", response_model=JobResponse,
        response_model_exclude_none=True, dependencies=[Depends(require_ui_session)],
        include_in_schema=False,
    )
    @app.get(
        "/api/v1/jobs/{job_id}", response_model=JobResponse,
        response_model_exclude_none=True, dependencies=[Depends(require_key)],
    )
    def get_job(job_id: str, request: Request, response: Response):
        response.headers["Cache-Control"] = "no-store"
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        user = getattr(request.state, "user", None)
        if user and job_owners.get(job_id) != user["id"]:
            raise HTTPException(status_code=404, detail="Job not found")
        return jobs[job_id]

    return app
