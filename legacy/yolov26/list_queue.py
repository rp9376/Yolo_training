#!/usr/bin/env python3
"""
YOLOv26 Training Queue Viewer

Displays the current training queue in a readable table format.

Usage:
  python list_queue.py
"""

# Re-exec with the project venv if needed (allows `python3 script.py` without activating).
import sys as _sys, os as _os
_venv_py = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '.venv', 'bin', 'python3'))
if _os.path.exists(_venv_py) and _os.path.abspath(_sys.executable) != _venv_py:
    _os.execv(_venv_py, [_venv_py] + _sys.argv)
del _venv_py, _sys, _os

import json
import sys
from pathlib import Path

QUEUE_FILE = Path(__file__).parent / "training_queue.json"

# ANSI color codes
RESET  = "\033[0m"
BOLD   = "\033[1m"
YELLOW = "\033[33m"
CYAN   = "\033[36m"
GREEN  = "\033[32m"
RED    = "\033[31m"
DIM    = "\033[2m"

STATUS_COLORS = {
    "pending":    YELLOW,
    "running":    CYAN,
    "completed":  GREEN,
    "failed":     RED,
    "skipped":    DIM,
}


def colorize_status(status: str) -> str:
    color = STATUS_COLORS.get(status.lower(), "")
    return f"{color}{status}{RESET}"


def short_dataset(path: str) -> str:
    """Return just the dataset folder name, trimmed for display."""
    parts = Path(path).parts
    # data.yaml is the last part; go up one to get the folder name
    folder = parts[-2] if len(parts) >= 2 else parts[-1]
    return folder[:20]


def print_queue(data: dict) -> None:
    tasks = data.get("tasks", [])
    total     = data.get("total_tasks", len(tasks))
    completed = data.get("completed_tasks", 0)
    failed    = data.get("failed_tasks", 0)
    created   = data.get("created", "unknown")
    status    = data.get("status", "unknown")

    width = 80
    print()
    print(BOLD + "=" * width + RESET)
    print(BOLD + "        YOLOv26 Training Queue".center(width) + RESET)
    print(BOLD + "=" * width + RESET)
    print()
    print(f"  Queue file : {QUEUE_FILE}")
    print(f"  Created    : {created}")
    print(f"  Status     : {colorize_status(status)}")
    print(f"  Progress   : {GREEN}{completed}{RESET}/{total} completed"
          f"  |  {RED}{failed}{RESET} failed"
          f"  |  {YELLOW}{total - completed - failed}{RESET} pending")
    print()

    if not tasks:
        print("  No tasks in queue.\n")
        return

    # Column widths
    col = {"idx": 3, "name": 24, "model": 8, "dataset": 22, "ep": 5,
           "batch": 5, "imgsz": 5, "pretrained": 10, "status": 10}

    header = (
        f"  {'#':<{col['idx']}}  "
        f"{'Name':<{col['name']}}  "
        f"{'Model':<{col['model']}}  "
        f"{'Dataset':<{col['dataset']}}  "
        f"{'Ep':>{col['ep']}}  "
        f"{'Batch':>{col['batch']}}  "
        f"{'Img':>{col['imgsz']}}  "
        f"{'Pretrained':<{col['pretrained']}}  "
        f"Status"
    )

    print(BOLD + header + RESET)
    print("  " + "-" * (width - 2))

    for i, task in enumerate(tasks, start=1):
        dataset_short = short_dataset(task.get("dataset", ""))
        task_status   = task.get("status", "unknown")
        pretrained    = "yes" if task.get("pretrained") else "no"

        row = (
            f"  {i:<{col['idx']}}  "
            f"{task.get('name', ''):<{col['name']}}  "
            f"{task.get('model', ''):<{col['model']}}  "
            f"{dataset_short:<{col['dataset']}}  "
            f"{task.get('epochs', '')!s:>{col['ep']}}  "
            f"{task.get('batch', '')!s:>{col['batch']}}  "
            f"{task.get('imgsz', '')!s:>{col['imgsz']}}  "
            f"{pretrained:<{col['pretrained']}}  "
            f"{colorize_status(task_status)}"
        )
        print(row)

    print()


def main() -> None:
    if not QUEUE_FILE.exists():
        print(f"{RED}Error:{RESET} Queue file not found: {QUEUE_FILE}", file=sys.stderr)
        print("Run setup_queue.py first to create the queue.", file=sys.stderr)
        sys.exit(1)

    try:
        with open(QUEUE_FILE) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"{RED}Error:{RESET} Could not parse queue file: {e}", file=sys.stderr)
        sys.exit(1)

    print_queue(data)


if __name__ == "__main__":
    main()
