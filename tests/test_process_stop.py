"""Regression tests for stopping training, including the DDP topology.

The hard case: torch's elastic launcher spawns each GPU worker with
``start_new_session=True``, so workers live in their *own* sessions/process
groups rather than the runner's. Killing the runner first orphans them and
severs the ppid chain, so a naive ``killpg(runner_pid)`` + post-mortem psutil
walk leaks the workers (and the GPU memory). ``process.stop`` must capture the
groups up front and kill each one individually.
"""

from __future__ import annotations

import json
import os
import signal
import sys
import time

import pytest

from backend import process
from core import queue as q


def _wait_for(path, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.exists(path):
            return True
        time.sleep(0.05)
    return False


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    try:
        state = open(f"/proc/{pid}/stat").read().split(") ", 1)[1].split(" ", 1)[0]
        return state != "Z"
    except (OSError, IndexError):
        return True


def _seed_pending():
    q.save({"tasks": [{"id": "x", "name": "t", "status": "pending"}],
            "status": "idle", "total_tasks": 1, "completed_tasks": 0,
            "failed_tasks": 0})


def test_stop_kills_escaped_session_worker(root, monkeypatch):
    """A worker in its own session (like a DDP worker) must still be killed."""
    worker_pid_file = root / "runs" / "worker.pid"
    # Runner spawns a grandchild with start_new_session=True (escaping the
    # runner's group) that ignores SIGTERM, mimicking a stuck GPU worker.
    runner_code = (
        "import os, signal, subprocess, sys, time\n"
        "worker = (\n"
        "  \"import os, signal, time\\n\"\n"
        "  \"signal.signal(signal.SIGTERM, signal.SIG_IGN)\\n\"\n"
        f"  \"open(r'{worker_pid_file}','w').write(str(os.getpid()))\\n\"\n"
        "  \"time.sleep(120)\\n\")\n"
        "subprocess.Popen([sys.executable,'-c',worker], start_new_session=True)\n"
        "time.sleep(120)\n"
    )
    monkeypatch.setenv("YOLO_STUDIO_RUNNER_CMD",
                       json.dumps([sys.executable, "-c", runner_code]))
    _seed_pending()

    res = process.start()
    runner_pid = res["pid"]
    try:
        assert _wait_for(worker_pid_file), "worker never started"
        worker_pid = int(worker_pid_file.read_text())

        # The worker really did escape the runner's process group.
        assert os.getpgid(worker_pid) == worker_pid
        assert os.getpgid(worker_pid) != os.getpgid(runner_pid)
        assert process.is_running()

        process.stop(grace=2.0)

        deadline = time.time() + 5
        while time.time() < deadline and _alive(worker_pid):
            time.sleep(0.1)
        assert not _alive(worker_pid), "escaped-session worker was not killed"
        assert not _alive(runner_pid)
    finally:
        for pid in (runner_pid,):
            try:
                os.killpg(pid, signal.SIGKILL)
            except OSError:
                pass


def test_stop_when_not_running(root):
    _seed_pending()
    out = process.stop()
    assert out["stopped"] is False
