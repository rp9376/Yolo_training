#!/usr/bin/env python3
"""
YOLOv26 Training Script - Interactive Multi-GPU Training

This script provides an interactive menu for training YOLOv26 models.
- Always uses ALL available GPUs automatically
- Supports training from scratch (YAML config) or from pretrained weights
- Only learns classes from your dataset
"""

# Re-exec with the project venv if needed (allows `python3 script.py` without activating).
import sys as _sys, os as _os
_venv_py = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '.venv', 'bin', 'python3'))
if _os.path.exists(_venv_py) and _os.path.abspath(_sys.executable) != _venv_py:
    _os.execv(_venv_py, [_venv_py] + _sys.argv)
del _venv_py, _sys, _os

import os
import sys
import torch
from pathlib import Path
from datetime import datetime

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics not installed. Run: pip install ultralytics")
    sys.exit(1)

SCRIPT_DIR   = Path(__file__).parent
DATASETS_DIR = SCRIPT_DIR.parent / "datasets"
WEIGHTS_DIR  = SCRIPT_DIR.parent / "weights"

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
    "1": (50,  "Quick test (50 epochs)"),
    "2": (100, "Standard training (100 epochs)"),
    "3": (200, "Extended training (200 epochs)"),
    "4": (300, "Production training (300 epochs)"),
    "5": (500, "Maximum training (500 epochs)"),
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

DEFAULTS = {
    "model":      "yolo26x",
    "epochs":     300,
    "batch":      -1,
    "imgsz":      640,
    "patience":   50,
    "workers":    8,
    "pretrained": True,   # Use pretrained weights by default
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def clear_screen():
    os.system('clear' if os.name != 'nt' else 'cls')


def print_header():
    clear_screen()
    print("=" * 70)
    print("        YOLOv26 Training - Multi-GPU")
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


def find_available_weights():
    """Find available pretrained weight files in weights/."""
    if not WEIGHTS_DIR.exists():
        return []
    return [f for f in sorted(WEIGHTS_DIR.glob("yolo26*.pt"))]


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
        choice = input(f"Select [1-{len(options)}] (Enter for default): ").strip()
        if choice == "" and default_key:
            return options[default_key][0]
        if choice in options:
            return options[choice][0]
        print("Invalid selection. Try again.")


def select_init_mode(model_name):
    """Choose between pretrained weights or from-scratch YAML."""
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
        choice = input(f"Select [1-2] (Enter for default): ").strip()
        if choice == "":
            choice = default
        if choice == "1":
            if not has_pt:
                print(f"⚠️  Weight file not found: {pt_file}")
                print(f"   Place {model_name}.pt in: {WEIGHTS_DIR}")
                continue
            return str(pt_file), True
        if choice == "2":
            return f"{model_name}.yaml", False
        print("Please enter 1 or 2.")


def confirm_settings(settings):
    print("\n" + "=" * 70)
    print("                    TRAINING CONFIGURATION")
    print("=" * 70)
    init_str = f"Pretrained ({Path(settings['model_source']).name})" if settings['pretrained'] else "From scratch (YAML)"
    print(f"""
  Dataset:        {settings['dataset']}
  Model:          {settings['model']}
  Initialization: {init_str}
  Epochs:         {settings['epochs']}
  Batch Size:     {'Auto' if settings['batch'] == -1 else settings['batch']}
  Image Size:     {settings['imgsz']}px
  GPUs:           {settings['device']}
  Output:         {settings['project']}/{settings['name']}
""")
    print("=" * 70)
    while True:
        confirm = input("\nStart training? [Y/n]: ").strip().lower()
        if confirm in ("", "y", "yes"):
            return True
        if confirm in ("n", "no"):
            return False
        print("Please enter Y or N.")


def get_training_config():
    print_header()
    gpu_count = print_gpu_status()

    dataset = select_dataset()
    if dataset is None:
        return None
    print(f"\n✅ Selected: {dataset}")

    print("\n" + "-" * 40)
    print("🚀 QUICK START: Use production defaults?")
    print("   (XLarge model, 300 epochs, auto batch, pretrained)")
    print("-" * 40)
    quick = input("Use defaults? [Y/n]: ").strip().lower()

    if quick in ("", "y", "yes"):
        model      = DEFAULTS["model"]
        epochs     = DEFAULTS["epochs"]
        batch      = DEFAULTS["batch"]
        imgsz      = DEFAULTS["imgsz"]
        pt_file    = WEIGHTS_DIR / f"{model}.pt"
        if pt_file.exists():
            model_source = str(pt_file)
            pretrained   = True
        else:
            model_source = f"{model}.yaml"
            pretrained   = False
    else:
        model        = select_option("🔧 Select Model Size:",         MODEL_SIZES,       "5")
        epochs       = select_option("⏱️  Select Training Duration:", EPOCH_PRESETS,     "4")
        batch        = select_option("📦 Select Batch Size:",         BATCH_PRESETS,     "1")
        imgsz        = select_option("📐 Select Image Size:",         IMAGE_SIZE_PRESETS,"1")
        model_source, pretrained = select_init_mode(model)

    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = model.replace("yolo26", "")
    run_name    = f"{model_short}_e{epochs}_{timestamp}"

    if gpu_count == 0:
        device = "cpu"
    elif gpu_count == 1:
        device = "0"
    else:
        device = ",".join(str(i) for i in range(gpu_count))

    return {
        "dataset":      str(dataset),
        "model":        model,
        "model_source": model_source,
        "pretrained":   pretrained,
        "epochs":       epochs,
        "batch":        batch,
        "imgsz":        imgsz,
        "device":       device,
        "project":      str(SCRIPT_DIR / "runs" / "detect"),
        "name":         run_name,
        "patience":     DEFAULTS["patience"],
        "workers":      DEFAULTS["workers"],
    }


# =============================================================================
# TRAINING
# =============================================================================

def train_model(settings):
    print("\n" + "=" * 70)
    print("                    STARTING TRAINING")
    print("=" * 70 + "\n")

    init_label = "pretrained weights" if settings['pretrained'] else "scratch (YAML)"
    print(f"📝 Initializing {settings['model']} from {init_label}...")
    print(f"   Source: {settings['model_source']}\n")

    model = YOLO(settings['model_source'])
    results = model.train(
        data=settings["dataset"],
        epochs=settings["epochs"],
        batch=settings["batch"],
        imgsz=settings["imgsz"],
        device=settings["device"],
        project=settings["project"],
        name=settings["name"],
        patience=settings["patience"],
        workers=settings["workers"],
        exist_ok=True,
        pretrained=settings["pretrained"],
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
    return results


def print_results(settings):
    output_dir  = Path(settings["project"]) / settings["name"]
    weights_dir = output_dir / "weights"
    print("\n" + "=" * 70)
    print("                    TRAINING COMPLETE!")
    print("=" * 70)
    print(f"""
  ✅ Training finished successfully!

  📁 Output Directory: {output_dir}
  🏆 Best Model:  {weights_dir}/best.pt
  📊 Last Model:  {weights_dir}/last.pt

  🔍 Inference:
     yolo detect predict model={weights_dir}/best.pt source=your_image.jpg

  📋 Validate:
     yolo detect val model={weights_dir}/best.pt data={settings['dataset']}
""")
    print("=" * 70)


# =============================================================================
# MAIN
# =============================================================================

def main():
    try:
        settings = get_training_config()
        if settings is None:
            print("\n❌ Training cancelled - no dataset available.")
            sys.exit(1)
        if not confirm_settings(settings):
            print("\n❌ Training cancelled by user.")
            sys.exit(0)
        train_model(settings)
        print_results(settings)
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()
