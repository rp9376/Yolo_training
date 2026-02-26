#!/usr/bin/env python3
"""
YOLOv8 Training Script - Interactive Multi-GPU Training from Scratch

This script provides an interactive menu for training YOLOv8 models.
- Always uses ALL available GPUs automatically
- Trains from scratch (no pretrained classification weights)
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

SCRIPT_DIR = Path(__file__).parent
DATASETS_DIR = SCRIPT_DIR.parent / "datasets"

# =============================================================================
# CONFIGURATION PRESETS
# =============================================================================

MODEL_SIZES = {
    "1": ("yolov8n", "Nano     - Fastest, lowest accuracy (~3.2M params)"),
    "2": ("yolov8s", "Small    - Fast, good accuracy (~11.2M params)"),
    "3": ("yolov8m", "Medium   - Balanced speed/accuracy (~25.9M params)"),
    "4": ("yolov8l", "Large    - Slower, high accuracy (~43.7M params)"),
    "5": ("yolov8x", "XLarge   - Slowest, highest accuracy (~68.2M params)"),
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
    "model":    "yolov8x",
    "epochs":   300,
    "batch":    -1,
    "imgsz":    640,
    "patience": 50,
    "workers":  8,
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def clear_screen():
    os.system('clear' if os.name != 'nt' else 'cls')


def print_header():
    clear_screen()
    print("=" * 70)
    print("        YOLOv8 Training from Scratch - Multi-GPU")
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
    """Find available datasets in the shared datasets directory."""
    datasets = []
    if not DATASETS_DIR.exists():
        DATASETS_DIR.mkdir(parents=True)
        return datasets
    for item in sorted(DATASETS_DIR.iterdir()):
        if item.is_dir():
            yaml_file = item / "data.yaml"
            if yaml_file.exists():
                datasets.append((item.name, yaml_file))
    return datasets


def select_dataset():
    datasets = find_datasets()
    if not datasets:
        print("❌ No datasets found!")
        print(f"\nPlease add a dataset to: {DATASETS_DIR}")
        print("Expected structure:")
        print("  datasets/")
        print("  └── your_dataset/")
        print("      ├── data.yaml")
        print("      ├── train/images/ & train/labels/")
        print("      └── valid/images/ & valid/labels/")
        return None

    print("📁 Available Datasets:")
    print("-" * 40)
    for i, (name, path) in enumerate(datasets, 1):
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


def confirm_settings(settings):
    print("\n" + "=" * 70)
    print("                    TRAINING CONFIGURATION")
    print("=" * 70)
    print(f"""
  Dataset:      {settings['dataset']}
  Model:        {settings['model']} (from scratch)
  Epochs:       {settings['epochs']}
  Batch Size:   {'Auto' if settings['batch'] == -1 else settings['batch']}
  Image Size:   {settings['imgsz']}px
  GPUs:         {settings['device']}
  Output:       {settings['project']}/{settings['name']}
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
    print("   (XLarge model, 300 epochs, auto batch)")
    print("-" * 40)
    quick = input("Use defaults? [Y/n]: ").strip().lower()

    if quick in ("", "y", "yes"):
        model  = DEFAULTS["model"]
        epochs = DEFAULTS["epochs"]
        batch  = DEFAULTS["batch"]
        imgsz  = DEFAULTS["imgsz"]
    else:
        model  = select_option("🔧 Select Model Size:",         MODEL_SIZES,      "5")
        epochs = select_option("⏱️  Select Training Duration:", EPOCH_PRESETS,    "4")
        batch  = select_option("📦 Select Batch Size:",         BATCH_PRESETS,    "1")
        imgsz  = select_option("📐 Select Image Size:",         IMAGE_SIZE_PRESETS,"1")

    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = model.replace("yolov8", "")
    run_name    = f"{model_short}_e{epochs}_{timestamp}"

    if gpu_count == 0:
        device = "cpu"
    elif gpu_count == 1:
        device = "0"
    else:
        device = ",".join(str(i) for i in range(gpu_count))

    return {
        "dataset": str(dataset),
        "model":   model,
        "epochs":  epochs,
        "batch":   batch,
        "imgsz":   imgsz,
        "device":  device,
        "project": str(SCRIPT_DIR / "runs" / "detect"),
        "name":    run_name,
        "patience": DEFAULTS["patience"],
        "workers":  DEFAULTS["workers"],
    }


# =============================================================================
# TRAINING
# =============================================================================

def train_model(settings):
    print("\n" + "=" * 70)
    print("                    STARTING TRAINING")
    print("=" * 70 + "\n")

    model_yaml = f"{settings['model']}.yaml"
    print(f"📝 Initializing {settings['model']} from scratch...")
    print(f"   (Model will only learn classes from your dataset)\n")

    model = YOLO(model_yaml)
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
