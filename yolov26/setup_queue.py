#!/usr/bin/env python3
"""
YOLOv26 Training Queue Setup - Interactive Queue Configuration

Sets up a queue of YOLOv26 training tasks saved to training_queue.json.
Run this first, then execute with: python run_queue.py
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
import torch
from pathlib import Path
from datetime import datetime

SCRIPT_DIR   = Path(__file__).parent
DATASETS_DIR = SCRIPT_DIR.parent / "datasets"
WEIGHTS_DIR  = SCRIPT_DIR.parent / "weights"
QUEUE_FILE   = SCRIPT_DIR / "training_queue.json"

# =============================================================================
# CONFIGURATION PRESETS
# =============================================================================

MODEL_SIZES = {
    "1": ("yolo26n", "Nano     - Fastest, lowest accuracy"),
    "2": ("yolo26s", "Small    - Fast, good accuracy"),
    "3": ("yolo26m", "Medium   - Balanced speed/accuracy"),
    "4": ("yolo26l", "Large    - Slower, high accuracy"),
    "5": ("yolo26x", "XLarge   - Slowest, highest accuracy"),
}

EPOCH_PRESETS = {
    "1": (5,    "Quick test (5 epochs)"),
    "2": (50,   "Short training (50 epochs)"),
    "3": (100,  "Standard training (100 epochs)"),
    "4": (200,  "Extended training (200 epochs)"),
    "5": (300,  "Production training (300 epochs)"),
    "6": (500,  "Maximum training (500 epochs)"),
    "c": (None, "Custom - Enter your own value"),
}

BATCH_PRESETS = {
    "1": (-1,  "Auto (let YOLO decide based on GPU memory)"),
    "2": (16,  "Small (16) - Safe for limited VRAM"),
    "3": (32,  "Medium (32) - Good balance"),
    "4": (64,  "Large (64) - Fast if you have VRAM"),
    "5": (128, "XLarge (128) - Maximum speed, needs lots of VRAM"),
}

IMAGE_SIZE_PRESETS = {
    "1": (640,  "Standard (640px) - Default, good balance"),
    "2": (1280, "Large (1280px) - Better for small objects"),
}

DEFAULTS = {"patience": 50, "workers": 8}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def clear_screen():
    os.system('clear' if os.name != 'nt' else 'cls')


def print_header():
    clear_screen()
    print("=" * 70)
    print("        YOLOv26 Training Queue Setup - Multi-GPU")
    print("=" * 70)
    print()


def get_gpu_info():
    if not torch.cuda.is_available():
        return None, 0
    gpu_count = torch.cuda.device_count()
    gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
    return gpu_names, gpu_count


def print_gpu_status():
    gpu_names, gpu_count = get_gpu_info()
    if gpu_count == 0:
        print("⚠️  WARNING: No GPUs detected! Training will be very slow on CPU.\n")
        return 0
    print(f"🖥️  Detected {gpu_count} GPU(s):")
    for i, name in enumerate(gpu_names):
        vram = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        print(f"   GPU {i}: {name} ({vram:.1f} GB VRAM)")
    print()
    return gpu_count


def find_datasets():
    datasets = []
    if not DATASETS_DIR.exists():
        DATASETS_DIR.mkdir(parents=True)
        return datasets
    for item in sorted(DATASETS_DIR.iterdir()):
        if item.is_dir() and (item / "data.yaml").exists():
            datasets.append((item.name, item / "data.yaml"))
    return datasets


def select_dataset():
    datasets = find_datasets()
    if not datasets:
        print("❌ No datasets found!")
        print(f"\nPlease add a dataset to: {DATASETS_DIR}")
        return None
    print("📁 Available Datasets:")
    print("-" * 40)
    for i, (name, _) in enumerate(datasets, 1):
        print(f"  {i}. {name}")
    print()
    while True:
        choice = input(f"Select dataset [1-{len(datasets)}]: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(datasets):
            return datasets[int(choice) - 1][1]
        print("Invalid selection. Try again.")


def select_option(prompt, options, default_key=None):
    print(f"\n{prompt}")
    print("-" * 40)
    for key, (value, description) in options.items():
        marker = " ← DEFAULT" if key == default_key else ""
        print(f"  {key}. {description}{marker}")
    print()
    while True:
        choice = input(f"Select [{'/'.join(options.keys())}] (Enter for default): ").strip().lower()
        if choice == "" and default_key:
            return options[default_key][0]
        if choice in options:
            value = options[choice][0]
            if value is None:
                while True:
                    custom = input("Enter number of epochs: ").strip()
                    if custom.isdigit() and int(custom) > 0:
                        return int(custom)
                    print("Please enter a positive integer.")
            return value
        print("Invalid selection. Try again.")


def select_init_mode(model_name):
    pt_file = WEIGHTS_DIR / f"{model_name}.pt"
    has_pt  = pt_file.exists()

    print(f"\n🔧 Model Initialization:")
    print("-" * 40)
    if has_pt:
        print(f"  1. Pretrained weights  ({pt_file.name}) ← RECOMMENDED")
        print(f"  2. From scratch        ({model_name}.yaml)")
        default = "1"
    else:
        print(f"  1. Pretrained weights  ({pt_file.name}) - NOT FOUND")
        print(f"  2. From scratch        ({model_name}.yaml) ← ONLY OPTION")
        default = "2"
    print()

    while True:
        choice = input("Select [1-2] (Enter for default): ").strip()
        if choice == "":
            choice = default
        if choice == "1":
            if not has_pt:
                print(f"⚠️  Weight file not found: {pt_file}")
                continue
            return str(pt_file), True
        if choice == "2":
            return f"{model_name}.yaml", False
        print("Please enter 1 or 2.")


def get_device_string(gpu_count):
    if gpu_count == 0:
        return "cpu"
    elif gpu_count == 1:
        return "0"
    return ",".join(str(i) for i in range(gpu_count))


def configure_task(task_number, gpu_count, cached_dataset=None):
    print_header()
    print(f"📋 Configuring Task #{task_number}")
    print("=" * 70)
    print()

    if cached_dataset:
        print(f"📁 Last dataset: {cached_dataset}")
        use_same = input("Use same dataset? [Y/n]: ").strip().lower()
        dataset = cached_dataset if use_same in ("", "y", "yes") else select_dataset()
    else:
        dataset = select_dataset()

    if dataset is None:
        return None, None

    print(f"\n✅ Dataset: {dataset}")

    model        = select_option("🔧 Select Model Size:",         MODEL_SIZES,       "1")
    model_source, pretrained = select_init_mode(model)
    epochs       = select_option("⏱️  Select Training Duration:", EPOCH_PRESETS,     "5")
    batch        = select_option("📦 Select Batch Size:",         BATCH_PRESETS,     "4")
    imgsz        = select_option("📐 Select Image Size:",         IMAGE_SIZE_PRESETS,"1")

    if batch != -1 and gpu_count > 1 and batch % gpu_count != 0:
        adjusted = max(gpu_count, (batch // gpu_count) * gpu_count)
        print(f"\n⚠️  Adjusting batch {batch} → {adjusted} (multiple of {gpu_count} GPUs)")
        batch = adjusted

    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = model.replace("yolo26", "")
    run_name    = f"{model_short}_e{epochs}_{timestamp}"

    task = {
        "dataset":      str(dataset),
        "model":        model,
        "model_source": model_source,
        "pretrained":   pretrained,
        "epochs":       epochs,
        "batch":        batch,
        "imgsz":        imgsz,
        "device":       get_device_string(gpu_count),
        "project":      str(SCRIPT_DIR / "runs" / "detect"),
        "name":         run_name,
        "patience":     DEFAULTS["patience"],
        "workers":      DEFAULTS["workers"],
        "status":       "pending",
    }
    return task, dataset


def display_queue_summary(tasks):
    print("\n" + "=" * 70)
    print("                    QUEUE SUMMARY")
    print("=" * 70)
    for i, task in enumerate(tasks, 1):
        batch_str  = "Auto" if task["batch"] == -1 else str(task["batch"])
        init_label = "pretrained" if task["pretrained"] else "from scratch"
        print(f"\n  Task #{i}:")
        print(f"    Model:      {task['model']} ({init_label})")
        print(f"    Epochs:     {task['epochs']}")
        print(f"    Batch:      {batch_str}")
        print(f"    Image Size: {task['imgsz']}px")
        print(f"    Run Name:   {task['name']}")
    print("\n" + "=" * 70)


def save_queue(tasks):
    queue_data = {
        "created":         datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status":          "pending",
        "total_tasks":     len(tasks),
        "completed_tasks": 0,
        "failed_tasks":    0,
        "tasks":           tasks,
    }
    with open(QUEUE_FILE, 'w') as f:
        json.dump(queue_data, f, indent=2)
    return QUEUE_FILE


# =============================================================================
# MAIN
# =============================================================================

def main():
    try:
        print_header()
        gpu_count = print_gpu_status()
        print("Set up a queue of YOLOv26 training tasks.")
        print(f"Queue will be saved to: {QUEUE_FILE}")
        print("Execute later with: python run_queue.py\n")

        tasks = []
        cached_dataset = None
        task_number = 1

        while True:
            task, cached_dataset = configure_task(task_number, gpu_count, cached_dataset)
            if task is None:
                if not tasks:
                    print("\n❌ Queue setup cancelled - no tasks configured.")
                    sys.exit(1)
                break

            tasks.append(task)
            display_queue_summary(tasks)

            add_more = input("Add another task? [y/N]: ").strip().lower()
            if add_more not in ("y", "yes"):
                break
            task_number += 1

        print_header()
        display_queue_summary(tasks)

        queue_file = save_queue(tasks)
        print(f"\n✅ Queue saved to: {queue_file}")
        print(f"📋 {len(tasks)} task(s) ready.\n")
        print("=" * 70)
        print("                    NEXT STEPS")
        print("=" * 70)
        print("""
  Foreground:
    python run_queue.py

  Background (close terminal safely):
    nohup ../../.venv/bin/python3 run_queue.py > queue_output.log 2>&1 &

  Monitor:
    tail -f queue_progress.log
    tail -f queue_output.log
""")
        print("=" * 70)

    except KeyboardInterrupt:
        print("\n\n⚠️  Setup interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()
