"""Training-queue model + JSON persistence (training_queue.json).

Writes are atomic (temp file + ``os.replace``). While the queue is *running*
only appending pending tasks and stopping are allowed; editing / removing /
reordering existing tasks raises :class:`QueueLocked` (mapped to HTTP 409).
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime

from .config import QUEUE_FILE

QUEUE_VERSION = 2


class QueueLocked(Exception):
    """Raised when a mutating op is attempted on a running queue."""


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _empty() -> dict:
    return {
        "version": QUEUE_VERSION,
        "created": _now(),
        "updated": _now(),
        "status": "idle",          # idle | running | completed
        "runner_pid": None,
        "total_tasks": 0,
        "completed_tasks": 0,
        "failed_tasks": 0,
        "tasks": [],
    }


def _normalize(data: dict) -> dict:
    """Bring a loaded dict up to the v2 shape (best-effort for legacy v1)."""
    if not isinstance(data, dict):
        return _empty()
    base = _empty()
    base.update({k: data[k] for k in data if k in base})
    base["tasks"] = data.get("tasks", []) or []
    base["version"] = QUEUE_VERSION
    # Legacy v1 used status "pending" for the whole queue; map to idle/completed.
    if base["status"] not in ("idle", "running", "completed"):
        base["status"] = "completed" if data.get("finished") else "idle"
    return base


def load() -> dict:
    if not QUEUE_FILE.exists():
        return _empty()
    try:
        with open(QUEUE_FILE) as f:
            return _normalize(json.load(f))
    except Exception:
        return _empty()


def _recompute(data: dict) -> None:
    tasks = data.get("tasks", [])
    data["total_tasks"] = len(tasks)
    data["completed_tasks"] = sum(1 for t in tasks if t.get("status") == "completed")
    data["failed_tasks"] = sum(1 for t in tasks if t.get("status") == "failed")


def save(data: dict) -> dict:
    _recompute(data)
    data["updated"] = _now()
    QUEUE_FILE.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(QUEUE_FILE.parent), prefix=".queue_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, QUEUE_FILE)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return data


# --- queries -----------------------------------------------------------------
def get(task_id: str, data: dict | None = None) -> dict | None:
    data = data or load()
    for t in data["tasks"]:
        if t.get("id") == task_id:
            return t
    return None


def _index(data: dict, task_id: str) -> int:
    for i, t in enumerate(data["tasks"]):
        if t.get("id") == task_id:
            return i
    return -1


# --- mutations ---------------------------------------------------------------
def add(task: dict) -> dict:
    """Append a task. Allowed even while running."""
    data = load()
    data["tasks"].append(task)
    save(data)
    return task


def update(task_id: str, fields: dict) -> dict:
    data = load()
    if data["status"] == "running":
        raise QueueLocked("Cannot edit tasks while the queue is running")
    idx = _index(data, task_id)
    if idx < 0:
        raise KeyError(task_id)
    if data["tasks"][idx].get("status") != "pending":
        raise QueueLocked("Only pending tasks can be edited")
    data["tasks"][idx].update(fields)
    save(data)
    return data["tasks"][idx]


def remove(task_id: str) -> None:
    data = load()
    if data["status"] == "running":
        raise QueueLocked("Cannot remove tasks while the queue is running")
    idx = _index(data, task_id)
    if idx < 0:
        raise KeyError(task_id)
    if data["tasks"][idx].get("status") == "running":
        raise QueueLocked("Cannot remove a running task")
    data["tasks"].pop(idx)
    save(data)


def reorder(order: list[str]) -> dict:
    data = load()
    if data["status"] == "running":
        raise QueueLocked("Cannot reorder tasks while the queue is running")
    by_id = {t["id"]: t for t in data["tasks"]}
    if set(order) != set(by_id):
        raise ValueError("Reorder list must contain exactly the existing task ids")
    data["tasks"] = [by_id[i] for i in order]
    save(data)
    return data


def clear(scope: str = "all") -> dict:
    data = load()
    if scope == "all":
        if data["status"] == "running":
            raise QueueLocked("Cannot clear all tasks while the queue is running")
        data["tasks"] = []
        data["status"] = "idle"
    elif scope == "completed":
        data["tasks"] = [t for t in data["tasks"]
                         if t.get("status") not in ("completed", "failed", "canceled")]
    elif scope == "pending":
        if data["status"] == "running":
            raise QueueLocked("Cannot clear pending tasks while the queue is running")
        data["tasks"] = [t for t in data["tasks"] if t.get("status") != "pending"]
    else:
        raise ValueError(f"Unknown scope: {scope!r}")
    save(data)
    return data


def has_pending(data: dict | None = None) -> bool:
    data = data or load()
    return any(t.get("status") == "pending" for t in data["tasks"])
