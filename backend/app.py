"""FastAPI application factory.

Mounts the static frontend at ``/``, includes the API routers under ``/api``,
and reattaches to an already-running queue runner on startup. Same-origin
(no CORS), no auth — bind to 127.0.0.1 (see scripts/run.sh).
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from core import config
from core import queue as q
from core.queue import QueueLocked

from . import process
from .api import datasets, health, hardware, models, queue as queue_api, stream, weights

STATIC_DIR = config.PROJECT_ROOT / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.shutdown_event = asyncio.Event()
    config.ensure_dirs()
    # Reattach: if a runner is alive we leave it alone; otherwise clean up any
    # stale "running" state left by a crash / hard kill.
    if not process.is_running():
        process._reconcile(q.load())
    yield
    # Signal all open SSE streams to exit so uvicorn can shut down cleanly.
    app.state.shutdown_event.set()


def _json_error(code: int, message: str) -> JSONResponse:
    return JSONResponse(status_code=code, content={"detail": message})


def create_app() -> FastAPI:
    app = FastAPI(title="YOLO Training Studio", lifespan=lifespan)

    # --- exception handlers (core raises plain exceptions) ---
    @app.exception_handler(ValueError)
    async def _value_error(request: Request, exc: ValueError):
        return _json_error(400, str(exc))

    @app.exception_handler(FileNotFoundError)
    async def _not_found(request: Request, exc: FileNotFoundError):
        return _json_error(404, f"Not found: {exc}")

    @app.exception_handler(KeyError)
    async def _key_error(request: Request, exc: KeyError):
        return _json_error(404, f"Not found: {exc}")

    @app.exception_handler(QueueLocked)
    async def _queue_locked(request: Request, exc: QueueLocked):
        return _json_error(409, str(exc))

    @app.exception_handler(process.ProcessError)
    async def _process_error(request: Request, exc: process.ProcessError):
        return _json_error(exc.code, str(exc))

    # --- API routers ---
    for r in (health.router, hardware.router, datasets.router, weights.router,
              queue_api.router, models.router, stream.router):
        app.include_router(r, prefix="/api")

    # --- static frontend (must be mounted last; "/" catches the rest) ---
    if STATIC_DIR.is_dir():
        app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

    return app


app = create_app()
