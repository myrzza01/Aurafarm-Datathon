"""Start the analysis budget before FastAPI receives/parses multipart uploads."""
import asyncio
import time

from starlette.formparsers import MultiPartException
from starlette.responses import JSONResponse


class AnalysisDeadlineMiddleware:
    def __init__(self, app, timeout: float):
        self.app, self.timeout = app, timeout

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST" or scope["path"] not in {
            "/api/v1/analyze", "/app/api/v1/analyze",
        }:
            return await self.app(scope, receive, send)
        state = scope.setdefault("state", {})
        state["analysis_deadline"] = time.monotonic() + self.timeout
        state["upload_timed_out"] = False
        response_started = False

        async def bounded_receive():
            remaining = state["analysis_deadline"] - time.monotonic()
            try:
                if remaining <= 0:
                    raise asyncio.TimeoutError()
                return await asyncio.wait_for(receive(), remaining)
            except asyncio.TimeoutError:
                state["upload_timed_out"] = True
                # Starlette closes all partially spooled files on MultiPartException.
                # Its ordinary 400 is translated to 408 below.
                raise MultiPartException("Передача и обработка файла превысили допустимое время") from None

        async def bounded_send(message):
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            if state["upload_timed_out"] and message["type"] == "http.response.start":
                message = {**message, "status": 408}
            await send(message)

        try:
            # Leave at most five seconds for bounded validator shutdown/HTTP error
            # delivery. Configuration caps analysis at 290s, below the 300s case limit.
            await asyncio.wait_for(self.app(scope, bounded_receive, bounded_send), self.timeout + 5)
        except asyncio.TimeoutError:
            if not response_started:
                response = JSONResponse(status_code=408, content={
                    "detail": "Передача и обработка файла превысили допустимое время",
                })
                await response(scope, receive, send)
            else:
                raise
