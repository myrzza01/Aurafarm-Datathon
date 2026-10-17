import logging
import os
import secrets
import tempfile
from pathlib import Path
from typing import Annotated, Callable
from uuid import uuid4

from fastapi import BackgroundTasks, Cookie, Depends, FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles

from backend.analyzer import AnalyzerNotConfigured, analyze_video
from backend.schemas import AnalysisResult, JobCreated, JobResponse

logger = logging.getLogger(__name__)
key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def create_app(
    analyzer: Callable[[Path], AnalysisResult] = analyze_video,
    *,
    api_key: str | None = None,
    upload_dir: Path | None = None,
    max_upload_bytes: int = 100 * 1024 * 1024,
) -> FastAPI:
    expected_key = api_key or os.environ.get("API_KEY")
    if not expected_key:
        raise RuntimeError("Set API_KEY before starting the server")
    storage = upload_dir or Path("data/uploads")
    storage.mkdir(parents=True, exist_ok=True)
    jobs: dict[str, JobResponse] = {}
    app = FastAPI(title="ForteBank Video Review API", version="0.1.0")
    frontend = Path(__file__).resolve().parent.parent / "frontend"
    ui_token = secrets.token_urlsafe(32)
    app.mount("/assets", StaticFiles(directory=frontend / "assets"), name="assets")

    def require_key(key: Annotated[str | None, Depends(key_header)]):
        if key is None or not secrets.compare_digest(key.encode(), expected_key.encode()):
            raise HTTPException(status_code=401, detail="Invalid API key")

    def require_ui_session(
        request: Request, ui_session: Annotated[str | None, Cookie()] = None,
    ):
        if ui_session is None or not secrets.compare_digest(ui_session.encode(), ui_token.encode()):
            raise HTTPException(status_code=401, detail="Откройте главную страницу заново")
        if request.method == "POST" and request.headers.get("X-UI-Request") != "1":
            raise HTTPException(status_code=403, detail="Invalid UI request")

    @app.get("/", include_in_schema=False)
    def home(request: Request):
        response = FileResponse(frontend / "index.html", headers={"Cache-Control": "no-store"})
        response.set_cookie(
            "ui_session", ui_token, httponly=True, samesite="strict",
            secure=request.url.scheme == "https", max_age=8 * 60 * 60,
        )
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
    def analyze(file: UploadFile, background_tasks: BackgroundTasks):
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
    def get_job(job_id: str):
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        return jobs[job_id]

    return app
