"""Queue endpoints: build/edit/remove/reorder/clear tasks + start/stop/status."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from core import engine, models
from core import queue as q
from .. import process

router = APIRouter()


class TaskRequest(BaseModel):
    family: str
    size: str
    init: str
    dataset: str
    epochs: int
    batch: int = -1
    imgsz: int = 640
    device: str = "auto"


class ReorderRequest(BaseModel):
    order: list[str]


class ClearRequest(BaseModel):
    scope: str = "all"


# --- task CRUD ---------------------------------------------------------------
@router.get("/queue")
def get_queue() -> dict:
    return q.load()


@router.post("/queue/tasks")
def add_task(req: TaskRequest) -> dict:
    task = engine.build_task(
        family=req.family, size=req.size, init=req.init, dataset=req.dataset,
        epochs=req.epochs, batch=req.batch, imgsz=req.imgsz, device=req.device,
    )
    q.add(task)
    return task


@router.put("/queue/tasks/{task_id}")
def edit_task(task_id: str, req: TaskRequest) -> dict:
    existing = q.get(task_id)
    if existing is None:
        raise KeyError(task_id)
    rebuilt = engine.build_task(
        family=req.family, size=req.size, init=req.init, dataset=req.dataset,
        epochs=req.epochs, batch=req.batch, imgsz=req.imgsz, device=req.device,
    )
    fields = {k: rebuilt[k] for k in (
        "family", "size", "model", "init", "model_source", "pretrained",
        "dataset", "dataset_name", "epochs", "batch", "imgsz", "device",
        "name", "project", "patience", "workers",
    )}
    return q.update(task_id, fields)


@router.delete("/queue/tasks/{task_id}")
def remove_task(task_id: str) -> dict:
    q.remove(task_id)
    return {"removed": True, "id": task_id}


@router.post("/queue/reorder")
def reorder(req: ReorderRequest) -> dict:
    return q.reorder(req.order)


@router.post("/queue/clear")
def clear(req: ClearRequest) -> dict:
    if req.scope == "all" and process.is_running():
        process.stop()  # "clear all" implies canceling the in-flight run too
    else:
        process.status()  # reconcile a stale "running" flag before mutating
    return q.clear(req.scope)


# --- run control -------------------------------------------------------------
@router.post("/queue/start")
def start() -> JSONResponse:
    result = process.start()
    return JSONResponse(status_code=202, content=result)


@router.post("/queue/stop")
def stop() -> dict:
    return process.stop()


@router.get("/queue/status")
def status() -> dict:
    return process.status()


@router.get("/queue/tasks/{task_id}/metrics")
def task_metrics(task_id: str) -> dict:
    task = q.get(task_id)
    if task is None:
        raise KeyError(task_id)
    run_dir = Path(task.get("run_dir") or "") if task.get("run_dir") else \
        Path(task["project"]) / task["name"]
    series = models.metric_series(run_dir) if run_dir.exists() else {"epoch": []}
    return {
        "task_id": task_id,
        "name": task.get("name"),
        "status": task.get("status"),
        "epochs": task.get("epochs"),
        "series": series,
    }
