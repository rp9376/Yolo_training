"""Standalone queue runner: ``python -m core.queue_runner``.

Runs pending tasks sequentially, persisting state after every transition.
Writes a human log to ``runs/queue_progress.log`` and structured one-line JSON
events to ``runs/queue_events.log``. Honours SIGINT/SIGTERM by marking the
running task ``canceled`` and exiting cleanly.
"""

from __future__ import annotations

import json
import os
import signal
import time
import traceback
from datetime import datetime

from . import engine
from . import queue as q
from .config import EVENTS_LOG, PROGRESS_LOG, RUNS_DIR

_stop = False
COOLDOWN_SECONDS = 30


def _handle_stop(signum, frame):
    global _stop
    _stop = True
    raise KeyboardInterrupt()


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(message: str, task_id: str | None = None) -> None:
    line = f"[{_ts()}] {message}"
    try:
        with open(PROGRESS_LOG, "a") as f:
            f.write(line + "\n")
    except OSError:
        pass
    print(line, flush=True)
    event({"type": "log", "message": message, "task_id": task_id})


def event(ev: dict) -> None:
    ev = {"ts": time.time(), **ev}
    try:
        with open(EVENTS_LOG, "a") as f:
            f.write(json.dumps(ev) + "\n")
    except OSError:
        pass


def _set_status(data: dict, idx: int, status: str, **fields) -> None:
    data["tasks"][idx]["status"] = status
    for k, v in fields.items():
        data["tasks"][idx][k] = v
    q.save(data)
    event({"type": "status", "task_id": data["tasks"][idx].get("id"),
           "name": data["tasks"][idx].get("name"), "status": status})


def _interruptible_sleep(seconds: int) -> None:
    for _ in range(seconds):
        if _stop:
            return
        time.sleep(1)


def _next_pending_index(data: dict) -> int:
    for i, t in enumerate(data["tasks"]):
        if t.get("status") == "pending":
            return i
    return -1


def main() -> int:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    data = q.load()
    data["status"] = "running"
    data["runner_pid"] = os.getpid()
    q.save(data)
    event({"type": "status", "status": "running", "scope": "queue"})
    log("Queue runner started.")

    first = True
    try:
        while not _stop:
            data = q.load()  # reload to pick up appended tasks / external edits
            idx = _next_pending_index(data)
            if idx < 0:
                break

            task = data["tasks"][idx]
            task_id = task.get("id")

            if not first:
                log(f"Waiting {COOLDOWN_SECONDS}s for GPU memory to free...", task_id)
                _interruptible_sleep(COOLDOWN_SECONDS)
                if _stop:
                    break
            first = False

            log(f"Starting task {task['name']} "
                f"({task['model']}, {task['epochs']} epochs, imgsz {task['imgsz']})",
                task_id)
            start = datetime.now()
            _set_status(data, idx, "running",
                        started_at=start.isoformat(timespec="seconds"))

            try:
                def cb(ev, _tid=task_id):
                    ev = dict(ev)
                    ev["task_id"] = _tid
                    event(ev)

                result = engine.train_one(task, on_event=cb)
                duration = str(datetime.now() - start)

                # Reload (the task list may have grown) and locate this task.
                data = q.load()
                idx = next((i for i, t in enumerate(data["tasks"])
                            if t.get("id") == task_id), idx)
                _set_status(
                    data, idx, "completed",
                    finished_at=datetime.now().isoformat(timespec="seconds"),
                    duration=duration,
                    best_epoch=result["best_epoch"],
                    best_fitness=result["best_fitness"],
                    run_dir=result["run_dir"],
                )
                log(f"Completed {task['name']} in {duration} "
                    f"(best epoch {result['best_epoch']}, "
                    f"fitness {result['best_fitness']:.4f})", task_id)
                event({"type": "done", "task_id": task_id, "name": task["name"],
                       "status": "completed", **result})

            except KeyboardInterrupt:
                data = q.load()
                idx = next((i for i, t in enumerate(data["tasks"])
                            if t.get("id") == task_id), idx)
                _set_status(data, idx, "canceled",
                            finished_at=datetime.now().isoformat(timespec="seconds"),
                            error="Canceled by user")
                log(f"Canceled {task['name']}.", task_id)
                event({"type": "done", "task_id": task_id, "name": task["name"],
                       "status": "canceled"})
                break

            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                data = q.load()
                idx = next((i for i, t in enumerate(data["tasks"])
                            if t.get("id") == task_id), idx)
                _set_status(data, idx, "failed",
                            finished_at=datetime.now().isoformat(timespec="seconds"),
                            error=err)
                log(f"FAILED {task['name']}: {err}", task_id)
                log(traceback.format_exc(), task_id)
                event({"type": "done", "task_id": task_id, "name": task["name"],
                       "status": "failed", "error": err})
                # Continue to the next task (matches legacy behaviour).
    finally:
        data = q.load()
        data["status"] = "completed"
        data["runner_pid"] = None
        data["finished"] = _ts()
        q.save(data)
        event({"type": "status", "status": "completed", "scope": "queue"})
        log("Queue runner finished.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        # Stop requested before/between tasks.
        try:
            data = q.load()
            data["status"] = "completed"
            data["runner_pid"] = None
            q.save(data)
            event({"type": "status", "status": "completed", "scope": "queue"})
        except Exception:
            pass
        raise SystemExit(0)
