from contextlib import asynccontextmanager
import os
import secrets
import sqlite3
import tempfile
import time
from threading import Lock
from pathlib import Path
from typing import Annotated, Callable

from fastapi import Cookie, Depends, FastAPI, HTTPException, Request, Response, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import FileResponse, RedirectResponse, JSONResponse
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles

from backend.analyzer import analyze_video, analyzer_configured, analyzer_spec, DEFAULT_ANALYZER
from backend.auth import AuthStore, LoginData, RegisterData, SESSION_SECONDS
from backend.jobs import JobStore, JobRunner, QueueFull, JobTimeout
from backend.media import InvalidVideo, validate_video
from backend.deadline import AnalysisDeadlineMiddleware
from backend.platform import FileLock
from backend.schemas import AnalysisResult, JobCreated, JobResponse

key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def create_app(
    analyzer: Callable[[Path], AnalysisResult] = analyze_video,
    *,
    api_key: str | None = None,
    upload_dir: Path | None = None,
    auth_db: Path | None = None,
    jobs_db: Path | None = None,
    max_upload_bytes: int = 100 * 1024 * 1024,
    analysis_timeout: float | None = None,
    max_workers: int | None = None,
    queue_capacity: int | None = None,
    result_retention: float | None = None,
) -> FastAPI:
    expected_key = api_key or os.environ.get("API_KEY")
    if not expected_key:
        raise RuntimeError("Set API_KEY before starting the server")
    storage = upload_dir or Path("data/uploads")
    storage.mkdir(parents=True, exist_ok=True)
    timeout = analysis_timeout if analysis_timeout is not None else float(os.environ.get("ANALYSIS_TIMEOUT_SECONDS", "270"))
    workers = max_workers if max_workers is not None else int(os.environ.get("ANALYSIS_WORKERS", "1"))
    capacity = queue_capacity if queue_capacity is not None else int(os.environ.get("ANALYSIS_QUEUE_CAPACITY", "4"))
    retention = result_retention if result_retention is not None else float(os.environ.get("RESULT_RETENTION_SECONDS", "86400"))
    if not (0 < timeout <= 290 and 1 <= workers <= 4 and workers <= capacity <= 100 and retention > 0):
        raise ValueError("Invalid job limits: timeout <= 290s, workers 1..4, capacity workers..100, retention > 0")
    jobs = JobStore(jobs_db or storage.parent / "jobs.sqlite3")
    runner = JobRunner(jobs, analyzer, workers, retention, upload_dir=storage)

    @asynccontextmanager
    async def lifespan(app):
        runner.start()
        try:
            yield
        finally:
            runner.close()

    app = FastAPI(title="ForteBank Video Review API", version="0.3.0", lifespan=lifespan, docs_url=None, redoc_url=None)
    app.add_middleware(AnalysisDeadlineMiddleware, timeout=timeout)
    frontend = Path(__file__).resolve().parent.parent / "frontend"
    accounts = AuthStore(auth_db or storage.parent / "auth.sqlite3")
    attempts: dict[str, list[float]] = {}
    attempts_lock = Lock()
    app.mount("/assets", StaticFiles(directory=frontend / "assets"), name="assets")
    app.mount("/docs-assets", StaticFiles(directory=Path(__file__).parent / "static" / "swagger"), name="docs-assets")

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        # Never include Pydantic input/ctx: these may contain a password or model payload.
        errors = [{"loc": list(error["loc"]), "type": error["type"], "msg": "Некорректное значение поля"} for error in exc.errors()]
        return JSONResponse(status_code=422, content={"detail": errors})

    @app.get("/docs", include_in_schema=False)
    def docs():
        return get_swagger_ui_html(
            openapi_url="/openapi.json", title=f"{app.title} — Swagger UI",
            swagger_js_url="/docs-assets/swagger-ui-bundle.js",
            swagger_css_url="/docs-assets/swagger-ui.css",
            swagger_favicon_url="/assets/favicon.svg",
            swagger_ui_parameters={"validatorUrl": None},
        )

    def model_configured():
        return analyzer is not analyze_video or analyzer_configured()

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

    @app.get("/health")
    def health():
        from backend.cashwatch_adapter import analyzer_kind
        return {"status": "ok", "analyzer_configured": model_configured(),
                "analyzer_kind": analyzer_kind() if analyzer is analyze_video and analyzer_spec() == DEFAULT_ANALYZER else "custom"}

    @app.post(
        "/app/api/v1/analyze", response_model=JobCreated, status_code=202,
        dependencies=[Depends(require_ui_session)], include_in_schema=False,
    )
    @app.post(
        "/api/v1/analyze", response_model=JobCreated, status_code=202,
        dependencies=[Depends(require_key)],
    )
    def analyze(file: UploadFile, request: Request):
        path = None
        deadline = getattr(request.state, "analysis_deadline", time.monotonic() + timeout)

        def remaining_time():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise HTTPException(status_code=408, detail="Передача и обработка файла превысили допустимое время")
            return remaining

        try:
            remaining_time()
            if Path(file.filename or "").suffix.lower() != ".mov":
                raise HTTPException(status_code=415, detail="Expected a .mov file")
            size = 0
            with tempfile.NamedTemporaryFile(dir=storage, suffix=".mov", delete=False) as out:
                path = Path(out.name)
                while chunk := file.file.read(1024 * 1024):
                    remaining_time()
                    size += len(chunk)
                    if size > max_upload_bytes:
                        raise HTTPException(status_code=413, detail="File exceeds upload limit")
                    out.write(chunk)
            if size == 0:
                raise HTTPException(status_code=400, detail="Empty file")
            # One validation process per data directory, shared across API workers.
            try:
                with FileLock(storage.parent / "video-validation.lock", blocking=False):
                    try:
                        validate_video(path, timeout=min(20, remaining_time()))
                    except InvalidVideo as exc:
                        remaining_time()
                        raise HTTPException(status_code=415, detail=str(exc)) from exc
            except BlockingIOError:
                raise HTTPException(status_code=429, detail="Проверяется другой файл. Повторите загрузку позже.", headers={"Retry-After": "2"})
            if not model_configured():
                raise HTTPException(status_code=503, detail="Локальный анализатор пока не подключён. Обратитесь к команде сервиса.")
            try:
                job_id = jobs.submit(path, getattr(request.state, "user", {}).get("id"), remaining_time(), capacity)
            except QueueFull:
                raise HTTPException(status_code=429, detail="Очередь анализа заполнена. Попробуйте позже.", headers={"Retry-After": "5"})
            except JobTimeout:
                raise HTTPException(status_code=408, detail="Передача и обработка файла превысили допустимое время")
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
        row = jobs.get(job_id)
        if not row:
            raise HTTPException(status_code=404, detail="Job not found")
        user = getattr(request.state, "user", None)
        if user and row["owner"] != user["id"]:
            raise HTTPException(status_code=404, detail="Job not found")
        return jobs.response(row)

    return app
