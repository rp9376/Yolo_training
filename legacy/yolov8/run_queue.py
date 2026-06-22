#!/usr/bin/env python3
"""
YOLOv8 Training Queue Runner - Execute Training Queue

Executes the training tasks defined in training_queue.json sequentially.
Failed tasks are logged; the queue continues to the next task.

Usage:
  python run_queue.py                                          # Foreground
  nohup ../../.venv/bin/python3 run_queue.py > queue_output.log 2>&1 &  # Background
"""

# Re-exec with the project venv if needed (allows `python3 script.py` without activating).
import sys as _sys, os as _os
_venv_py = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '.venv', 'bin', 'python3'))
if _os.path.exists(_venv_py) and _os.path.abspath(_sys.executable) != _venv_py:
    _os.execv(_venv_py, [_venv_py] + _sys.argv)
del _venv_py, _sys, _os

import os
import sys
import json
import time
import traceback
import csv
from pathlib import Path
from datetime import datetime

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics not installed. Run: pip install ultralytics")
    sys.exit(1)

SCRIPT_DIR = Path(__file__).parent
QUEUE_FILE = SCRIPT_DIR / "training_queue.json"
LOG_FILE   = SCRIPT_DIR / "queue_progress.log"


# =============================================================================
# LOGGING
# =============================================================================

def log(message, also_print=True):
    timestamp   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] {message}"
    with open(LOG_FILE, 'a') as f:
        f.write(log_message + "\n")
    if also_print:
        print(log_message)


def log_sep():
    log("=" * 60)


# =============================================================================
# QUEUE MANAGEMENT
# =============================================================================

def load_queue():
    if not QUEUE_FILE.exists():
        return None
    with open(QUEUE_FILE, 'r') as f:
        return json.load(f)


def save_queue(queue_data):
    with open(QUEUE_FILE, 'w') as f:
        json.dump(queue_data, f, indent=2)


def get_pending_tasks(queue_data):
    return [t for t in queue_data["tasks"] if t["status"] == "pending"]


def update_task_status(queue_data, task_index, status, error=None):
    queue_data["tasks"][task_index]["status"] = status
    if error:
        queue_data["tasks"][task_index]["error"] = str(error)
    if status == "completed":
        queue_data["completed_tasks"] = queue_data.get("completed_tasks", 0) + 1
    elif status == "failed":
        queue_data["failed_tasks"] = queue_data.get("failed_tasks", 0) + 1
    save_queue(queue_data)


# =============================================================================
# TRAINING
# =============================================================================

def train_task(task):
    model_yaml = f"{task['model']}.yaml"
    log(f"Initializing {task['model']} from scratch...")

    model = YOLO(model_yaml)
    results = model.train(
        data=task["dataset"],
        epochs=task["epochs"],
        batch=task["batch"],
        imgsz=task["imgsz"],
        device=task["device"],
        project=task["project"],
        name=task["name"],
        patience=task["patience"],
        workers=task["workers"],
        exist_ok=True,
        pretrained=False,
        verbose=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=15.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0005,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.2,
    )

    # Extract best epoch
    best_fitness = 0.0
    best_epoch   = 0
    try:
        save_dir   = Path(task["project"]) / task["name"]
        results_csv = save_dir / "results.csv"
        if results_csv.exists():
            with open(results_csv, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        map50    = float(row.get('metrics/mAP50(B)', 0))
                        map50_95 = float(row.get('metrics/mAP50-95(B)', 0))
                        fitness  = 0.1 * map50 + 0.9 * map50_95
                        epoch    = int(row.get('epoch', 0))
                        if fitness > best_fitness:
                            best_fitness = fitness
                            best_epoch   = epoch
                    except (ValueError, TypeError):
                        continue
    except Exception as e:
        log(f"⚠️  Could not extract best epoch: {e}")

    log_sep()
    log(f"📊 TRAINING SUMMARY")
    log(f"   Best Epoch:   {best_epoch}")
    log(f"   Best Fitness: {best_fitness:.4f}")
    log_sep()

    return results, best_epoch


# =============================================================================
# MAIN
# =============================================================================

def print_summary(queue_data):
    log_sep()
    log("QUEUE EXECUTION COMPLETE")
    log_sep()
    log(f"Total tasks: {queue_data['total_tasks']}")
    log(f"Completed:   {queue_data.get('completed_tasks', 0)}")
    log(f"Failed:      {queue_data.get('failed_tasks', 0)}")
    log_sep()
    log("Task Details:")
    for i, task in enumerate(queue_data["tasks"], 1):
        icon = "✅" if task["status"] == "completed" else "❌" if task["status"] == "failed" else "⏸️"
        log(f"  {icon} Task #{i} ({task['name']}): {task['status']}")
        if task.get("error"):
            log(f"      Error: {task['error']}")
        if task.get("best_epoch"):
            log(f"      Best epoch: {task['best_epoch']}")
    log_sep()
    log("Output Locations:")
    for task in queue_data["tasks"]:
        if task["status"] == "completed":
            log(f"  📁 {Path(task['project']) / task['name']}")
    log_sep()


def main():
    log_sep()
    log("YOLOv8 Training Queue Runner Starting")
    log_sep()

    queue_data = load_queue()
    if queue_data is None:
        log(f"❌ ERROR: No queue file found at {QUEUE_FILE}")
        log("   Run 'python setup_queue.py' first.")
        sys.exit(1)

    if queue_data["status"] == "completed":
        log("⚠️  Queue already completed. Create a new queue to run more tasks.")
        sys.exit(0)

    queue_data["status"]  = "running"
    queue_data["started"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    save_queue(queue_data)

    pending = get_pending_tasks(queue_data)
    total   = queue_data["total_tasks"]
    log(f"Found {len(pending)} pending task(s) out of {total} total")
    log_sep()

    for i, task in enumerate(queue_data["tasks"]):
        task_num = i + 1
        if task["status"] != "pending":
            log(f"Skipping Task #{task_num} ({task['name']}): {task['status']}")
            continue

        log_sep()
        log(f"STARTING Task #{task_num}/{total}: {task['name']}")
        log_sep()
        log(f"  Model:      {task['model']}")
        log(f"  Epochs:     {task['epochs']}")
        log(f"  Batch:      {'Auto' if task['batch'] == -1 else task['batch']}")
        log(f"  Image Size: {task['imgsz']}px")
        log(f"  Dataset:    {task['dataset']}")
        log_sep()

        update_task_status(queue_data, i, "running")

        try:
            start_time           = datetime.now()
            results, best_epoch  = train_task(task)
            duration             = datetime.now() - start_time

            update_task_status(queue_data, i, "completed")
            queue_data["tasks"][i]["duration"]   = str(duration)
            queue_data["tasks"][i]["best_epoch"] = best_epoch
            save_queue(queue_data)

            log_sep()
            log(f"✅ Task #{task_num} COMPLETED in {duration}")
            log(f"   Best model at epoch {best_epoch}")
            log_sep()

            if task_num < total:
                log("Waiting 30s for GPU memory to free...")
                time.sleep(30)

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            update_task_status(queue_data, i, "failed", error_msg)
            log_sep()
            log(f"❌ Task #{task_num} FAILED: {error_msg}")
            log(traceback.format_exc(), also_print=False)
            log_sep()
            log("Continuing to next task...")
            if task_num < total:
                log("Waiting 30s for GPU memory to free...")
                time.sleep(30)

    queue_data["status"]   = "completed"
    queue_data["finished"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    save_queue(queue_data)
    print_summary(queue_data)
    log("Queue execution finished.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("\n⚠️  Queue interrupted by user.")
        sys.exit(1)
    except Exception as e:
        log(f"❌ Fatal error: {e}")
        log(traceback.format_exc())
        sys.exit(1)
