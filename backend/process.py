"""Queue-runner subprocess manager.

Launches ``python -m core.queue_runner`` in its own session (process group) so
training survives a backend restart; the backend reattaches via the PID file.
The runner command is overridable with ``YOLO_STUDIO_RUNNER_CMD`` (JSON list)
to support testing.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import psutil

from core import queue as q
from core.config import EVENTS_LOG, OUTPUT_LOG, PID_FILE, RUNS_DIR

# Directory containing the `core` / `backend` packages. This is the runner's
# working dir so `python -m core.queue_runner` resolves regardless of where the
# data root (YOLO_STUDIO_ROOT) points.
_CODE_ROOT = Path(__file__).resolve().parent.parent


def _emit(message: str) -> None:
    """Append a log event so it streams to the Monitor console via SSE.

    ``stop()`` runs in the web-server process, not the runner, so it has no
    other path to the live console. Mirrors ``core.queue_runner.event``'s
    one-line-JSON format (the SSE stream tails the same file).
    """
    try:
        with open(EVENTS_LOG, "a") as f:
            f.write(json.dumps({"ts": time.time(), "type": "log",
                                "message": message}) + "\n")
    except OSError:
        pass


class ProcessError(Exception):
    """Raised on illegal process-manager transitions (carries an HTTP code)."""

    def __init__(self, message: str, code: int = 409):
        super().__init__(message)
        self.code = code


def _read_pid() -> int | None:
    try:
        return int(PID_FILE.read_text().strip())
    except (OSError, ValueError):
        return None


def _alive(pid: int | None) -> bool:
    if not pid:
        return False
    # If the runner is our own child and has already exited, reap it now so it
    # doesn't linger as a zombie (a zombie still answers os.kill(pid, 0)).
    try:
        reaped, _ = os.waitpid(pid, os.WNOHANG)
        if reaped == pid:
            return False
    except ChildProcessError:
        pass  # not our child (e.g. after a backend restart) — fall through
    except OSError:
        pass

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists but owned by another user
    except OSError:
        return False

    # A not-yet-reaped zombie answers os.kill but is effectively dead.
    try:
        with open(f"/proc/{pid}/stat") as f:
            state = f.read().split(") ", 1)[1].split(" ", 1)[0]
        if state == "Z":
            return False
    except (OSError, IndexError):
        pass
    return True


def _clear_pid() -> None:
    try:
        PID_FILE.unlink()
    except OSError:
        pass


def is_running() -> bool:
    return _tree_alive(_read_pid())


def _runner_cmd() -> list[str]:
    override = os.environ.get("YOLO_STUDIO_RUNNER_CMD")
    if override:
        return json.loads(override)
    return [sys.executable, "-m", "core.queue_runner"]


def _reconcile(data: dict) -> dict:
    """Fix stale state when the runner died without a clean shutdown."""
    changed = False
    if data.get("status") == "running":
        data["status"] = "completed"
        changed = True
    for t in data.get("tasks", []):
        if t.get("status") == "running":
            t["status"] = "canceled"
            t["error"] = t.get("error") or "Runner stopped unexpectedly"
            changed = True
    if changed:
        q.save(data)
    return data


def start() -> dict:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    pid = _read_pid()
    if _alive(pid):
        raise ProcessError(f"Queue runner already running (pid {pid})", 409)
    _clear_pid()
    if not q.has_pending():
        raise ProcessError("No pending tasks to run", 400)

    logf = open(OUTPUT_LOG, "a")
    try:
        proc = subprocess.Popen(
            _runner_cmd(),
            cwd=str(_CODE_ROOT),
            stdout=logf,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=os.environ.copy(),
        )
    finally:
        logf.close()
    PID_FILE.write_text(str(proc.pid))
    return {"started": True, "pid": proc.pid}


def _capture_groups(pid: int) -> set[int]:
    """Process-group ids of the runner and every descendant alive right now.

    This must be captured *before* we signal anything. The runner starts in its
    own session (pgid == pid) and ultralytics runs the torch DDP launcher as a
    plain child (same group), but torch's elastic launcher spawns each GPU
    worker with start_new_session=True — so every worker lives in its *own*
    session/group, escaping ``killpg(runner_pid)``. Worse, once we kill the
    runner/launcher the workers are reparented to init and the ppid chain that
    psutil walks is severed, so we can no longer discover them. We therefore
    enumerate the whole tree while it is still connected and remember each
    distinct group id, so we can kill every group individually afterwards.
    """
    groups: set[int] = set()
    if not pid:
        return groups
    try:
        groups.add(os.getpgid(pid))
    except (ProcessLookupError, PermissionError, OSError):
        groups.add(pid)  # runner is its own group leader by construction
    try:
        root = psutil.Process(pid)
        procs = [root, *root.children(recursive=True)]
    except psutil.NoSuchProcess:
        return groups
    for p in procs:
        try:
            groups.add(os.getpgid(p.pid))
        except (ProcessLookupError, PermissionError, psutil.NoSuchProcess, OSError):
            continue
    return groups


def _signal_groups(groups: set[int], sig: int) -> None:
    """Send ``sig`` to every captured process group (each worker is a leader)."""
    for pgid in groups:
        try:
            os.killpg(pgid, sig)
        except (ProcessLookupError, PermissionError, OSError):
            pass


def _groups_alive(groups: set[int]) -> bool:
    """True if any *live* process is still a member of one of the captured groups.

    Catches the escaped DDP workers (each its own group) and any dataloader
    workers they spawned (which stay in their worker's group), even after the
    runner and launcher have already exited. Zombies are ignored: a killed
    process lingers in the table (and still answers ``getpgid``) until its
    parent reaps it, but it is no longer doing any work.
    """
    if not groups:
        return False
    for p in psutil.process_iter(["status"]):
        try:
            if p.info["status"] == psutil.STATUS_ZOMBIE:
                continue
            if os.getpgid(p.pid) in groups:
                return True
        except (ProcessLookupError, psutil.NoSuchProcess, PermissionError):
            continue
    return False


def _tree_alive(pid: int) -> bool:
    """True if the runner or anything left in its process tree is still alive.

    During normal operation the runner stays alive for the whole run (it blocks
    in ultralytics' ``subprocess.run`` waiting on the DDP launcher), so the
    cheap ``_alive`` check answers immediately. The group scan is a fallback for
    the window where the runner has exited but workers it spawned are still
    running — including DDP workers that escaped into their own sessions.
    """
    if _alive(pid):
        return True
    return _groups_alive(_capture_groups(pid))


def stop(grace: float = 6.0) -> dict:
    pid = _read_pid()
    if not _tree_alive(pid):
        _emit("⏹ Stop requested, but no training process is running.")
        _clear_pid()
        _reconcile(q.load())
        return {"stopped": False, "reason": "not running"}

    # Snapshot every process group in the tree *now*, while the runner ->
    # launcher -> worker chain is still intact. Killing the runner first would
    # orphan the DDP workers (they run in their own sessions) and make them
    # undiscoverable, so we must record their groups before signalling anyone.
    groups = _capture_groups(pid)
    _emit(f"⏹ Stop requested — terminating training (pid {pid}, "
          f"{len(groups)} process group(s)). Sending SIGTERM…")

    _signal_groups(groups, signal.SIGTERM)
    deadline = time.time() + grace
    while time.time() < deadline:
        _alive(pid)  # reap the runner if it has exited (it is our child)
        if not _groups_alive(groups):
            break
        # New dataloader workers can appear inside a group mid-shutdown; keep
        # the snapshot current for anything still reachable via the runner.
        groups |= _capture_groups(pid)
        time.sleep(0.25)
    if _groups_alive(groups):
        _emit("⚠ Training did not exit on SIGTERM; forcing SIGKILL.")
        _signal_groups(groups, signal.SIGKILL)
        time.sleep(0.5)
    _alive(pid)  # final reap so the dead runner doesn't linger as a zombie
    _clear_pid()
    _reconcile(q.load())
    still = _groups_alive(groups)
    if still:
        _emit("❌ Stop failed: some training processes are still alive.")
    else:
        _emit("✅ Training stopped.")
    return {"stopped": not still}


def status() -> dict:
    pid = _read_pid()
    running = _tree_alive(pid)
    data = q.load()
    if not running:
        _clear_pid()
        data = _reconcile(data)

    running_task_id = None
    if running:
        for t in data.get("tasks", []):
            if t.get("status") == "running":
                running_task_id = t.get("id")
                break

    return {
        "running": running,
        "pid": pid if running else None,
        "running_task_id": running_task_id,
        "queue_status": data.get("status"),
        "counts": {
            "total": data.get("total_tasks", 0),
            "completed": data.get("completed_tasks", 0),
            "failed": data.get("failed_tasks", 0),
        },
    }
