#!/usr/bin/env python3
"""
YOLOv8 Training Cleanup Script

Cleans up queue files, logs, and optionally trained model outputs.

Usage:
  python cleanup.py              # Interactive mode
  python cleanup.py --all        # Clean everything including runs/
  python cleanup.py --logs-only  # Clean only logs and queue file
"""

# Re-exec with the project venv if needed (allows `python3 script.py` without activating).
import sys as _sys, os as _os
_venv_py = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '.venv', 'bin', 'python3'))
if _os.path.exists(_venv_py) and _os.path.abspath(_sys.executable) != _venv_py:
    _os.execv(_venv_py, [_venv_py] + _sys.argv)
del _venv_py, _sys, _os

import sys
import shutil
from pathlib import Path
from datetime import datetime

SCRIPT_DIR   = Path(__file__).parent
QUEUE_FILE   = SCRIPT_DIR / "training_queue.json"
PROGRESS_LOG = SCRIPT_DIR / "queue_progress.log"
OUTPUT_LOG   = SCRIPT_DIR / "queue_output.log"
RUNS_DIR     = SCRIPT_DIR / "runs"


def print_header():
    print("=" * 70)
    print("        YOLOv8 Training Cleanup Script")
    print("=" * 70)
    print()


def get_size_mb(path):
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size / (1024 * 1024)
    return sum(f.stat().st_size for f in path.rglob('*') if f.is_file()) / (1024 * 1024)


def fmt_size(mb):
    return f"{mb:.1f} MB" if mb < 1024 else f"{mb / 1024:.1f} GB"


def delete_file(path, label):
    if not path.exists():
        print(f"  ⏭️  {label}: Not found")
        return False
    size = get_size_mb(path)
    try:
        path.unlink()
        print(f"  ✅ {label}: Deleted ({fmt_size(size)})")
        return True
    except Exception as e:
        print(f"  ❌ {label}: Failed - {e}")
        return False


def delete_directory(path, label):
    if not path.exists():
        print(f"  ⏭️  {label}: Not found")
        return False
    size = get_size_mb(path)
    try:
        shutil.rmtree(path)
        print(f"  ✅ {label}: Deleted ({fmt_size(size)})")
        return True
    except Exception as e:
        print(f"  ❌ {label}: Failed - {e}")
        return False


def show_status():
    print("Current Status:")
    print("-" * 70)
    items = [
        (QUEUE_FILE,   "Queue file (training_queue.json)"),
        (PROGRESS_LOG, "Progress log (queue_progress.log)"),
        (OUTPUT_LOG,   "Output log (queue_output.log)"),
        (RUNS_DIR,     "Training outputs (runs/)"),
    ]
    total = 0
    for path, label in items:
        if path.exists():
            size = get_size_mb(path)
            total += size
            icon = "📄" if path.is_file() else "📁"
            print(f"  {icon} {label}: {fmt_size(size)}")
        else:
            print(f"  ⚪ {label}: Not found")
    print("-" * 70)
    print(f"  Total: {fmt_size(total)}\n")


def clean_logs():
    print("\n🧹 Cleaning logs and queue...")
    print("-" * 70)
    delete_file(QUEUE_FILE,   "Queue file")
    delete_file(PROGRESS_LOG, "Progress log")
    delete_file(OUTPUT_LOG,   "Output log")
    print("-" * 70)


def clean_runs():
    print("\n🧹 Cleaning training outputs...")
    print("-" * 70)
    delete_directory(RUNS_DIR, "runs/")
    print("-" * 70)


def confirm(prompt):
    while True:
        r = input(f"{prompt} [y/N]: ").strip().lower()
        if r in ("", "n", "no"):
            return False
        if r in ("y", "yes"):
            return True
        print("Please enter Y or N.")


def main():
    print_header()

    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg == "--all":
            clean_logs()
            clean_runs()
            print("\n✅ Cleanup complete!")
        elif arg == "--logs-only":
            clean_logs()
            print("\n✅ Cleanup complete!")
        elif arg in ("--help", "-h"):
            print(__doc__)
        else:
            print(f"Unknown argument: {arg}")
            sys.exit(1)
        return

    # Interactive mode
    show_status()
    print("What would you like to clean?")
    print("-" * 70)
    print("  1. Logs and queue only (keep trained models)")
    print("  2. Everything (logs, queue, and trained models)")
    print("  3. Cancel\n")

    while True:
        choice = input("Select [1-3]: ").strip()
        if choice == "1":
            if confirm("\n⚠️  Clean logs and queue?"):
                clean_logs()
                print("\n✅ Cleanup complete!")
            else:
                print("❌ Cancelled.")
            break
        elif choice == "2":
            print("\n⚠️  WARNING: This will delete all trained models!")
            if confirm("Are you sure?"):
                clean_logs()
                clean_runs()
                print("\n✅ Cleanup complete!")
            else:
                print("❌ Cancelled.")
            break
        elif choice == "3":
            print("❌ Cancelled.")
            break
        else:
            print("Please enter 1, 2, or 3.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise
