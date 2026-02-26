#!/usr/bin/env python3
"""
YOLOv26 Model Validation Script

Validates a trained YOLOv26 model and displays:
- Model parameters and architecture details
- Class labels
- Performance metrics (mAP, precision, recall, F1)
- Per-class breakdown

Usage:
  python validate.py
"""

# Re-exec with the project venv if needed (allows `python3 script.py` without activating).
import sys as _sys, os as _os
_venv_py = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '.venv', 'bin', 'python3'))
if _os.path.exists(_venv_py) and _os.path.abspath(_sys.executable) != _venv_py:
    _os.execv(_venv_py, [_venv_py] + _sys.argv)
del _venv_py, _sys, _os

import sys
import torch
import yaml
from pathlib import Path
from datetime import datetime

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics not installed.")
    sys.exit(1)

SCRIPT_DIR   = Path(__file__).parent
DATASETS_DIR = SCRIPT_DIR.parent / "datasets"
RUNS_DIR     = SCRIPT_DIR / "runs" / "detect"


def section(title):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def load_data_config(yaml_path):
    try:
        with open(yaml_path) as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading data config: {e}")
        return None


def display_model_info(model, data_config):
    section("MODEL ARCHITECTURE & PARAMETERS")
    model.info(detailed=True, verbose=True)

    total     = sum(p.numel() for p in model.model.parameters())
    trainable = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    print(f"\n📊 Parameters:")
    print(f"  Total:      {total:,}")
    print(f"  Trainable:  {trainable:,}")
    print(f"  Size (MB):  {total * 4 / 1024 / 1024:.2f}")
    print(f"  Device:     {next(model.model.parameters()).device}")

    if data_config:
        section("CLASS LABELS")
        classes     = data_config.get('names', [])
        num_classes = data_config.get('nc', len(classes))
        print(f"\n📋 Total Classes: {num_classes}")
        print(f"\n{'Index':<8} {'Class Name'}")
        print("-" * 40)
        for idx, name in enumerate(classes):
            print(f"  {idx:<6} {name}")


def run_validation(model, data_yaml_path, split='val'):
    section(f"VALIDATION RESULTS - {split.upper()} SET")
    print(f"\n🔍 Running validation on {split} dataset...\n")

    try:
        results = model.val(
            data=data_yaml_path,
            split=split,
            batch=16,
            imgsz=640,
            device=0 if torch.cuda.is_available() else 'cpu',
            plots=True,
            save_json=True,
            verbose=True,
        )

        section("PERFORMANCE METRICS")
        print(f"\n📈 Overall:")
        print(f"  mAP50:      {results.box.map50:.4f}")
        print(f"  mAP50-95:   {results.box.map:.4f}")
        print(f"  Precision:  {results.box.mp:.4f}")
        print(f"  Recall:     {results.box.mr:.4f}")
        prec = results.box.mp
        rec  = results.box.mr
        f1   = 2 * prec * rec / (prec + rec + 1e-16)
        print(f"  F1-Score:   {f1:.4f}")

        if hasattr(results.box, 'maps') and results.box.maps is not None:
            print(f"\n📊 Per-Class mAP50-95:")
            print("-" * 60)
            p_list = results.box.p if hasattr(results.box, 'p') else [0] * len(results.box.maps)
            r_list = results.box.r if hasattr(results.box, 'r') else [0] * len(results.box.maps)
            print(f"{'Class':<20} {'mAP50-95':<12} {'Precision':<12} {'Recall'}")
            print("-" * 60)
            for i, (m, p, r) in enumerate(zip(results.box.maps, p_list, r_list)):
                name = results.names[i] if i < len(results.names) else f"Class {i}"
                print(f"{name:<20} {m:.4f}       {p:.4f}       {r:.4f}")

        if hasattr(results, 'speed'):
            print(f"\n⚡ Speed (ms per image):")
            for k, v in results.speed.items():
                print(f"  {k.capitalize():<14} {v:.2f} ms")
            total_ms = sum(results.speed.values())
            print(f"  {'FPS':<14} {1000 / total_ms:.1f}")

        return results
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        return None


def find_models():
    if not RUNS_DIR.exists():
        return []

    models = []
    for subdir in sorted(RUNS_DIR.iterdir()):
        if not subdir.is_dir():
            continue
        weights_dir = subdir / "weights"
        for pt_name in ("best.pt", "last.pt"):
            pt = weights_dir / pt_name
            if pt.exists():
                mtime    = pt.stat().st_mtime
                size_mb  = pt.stat().st_size / (1024 * 1024)
                model_letter = "?"
                epochs   = "?"
                args_file = subdir / "args.yaml"
                if args_file.exists():
                    try:
                        with open(args_file) as f:
                            args = yaml.safe_load(f)
                        m_str = args.get('model', '')
                        for s in ['n', 's', 'm', 'l', 'x']:
                            if f'yolo26{s}' in m_str:
                                model_letter = s.upper()
                                break
                        epochs = args.get('epochs', '?')
                    except Exception:
                        pass
                models.append({
                    'path': pt, 'name': subdir.name, 'type': pt_name.replace('.pt', ''),
                    'mtime': mtime, 'size_mb': size_mb,
                    'model_letter': model_letter, 'epochs': epochs,
                })
                break

    models.sort(key=lambda x: x['mtime'], reverse=True)
    return models


def find_datasets():
    datasets = []
    if DATASETS_DIR.exists():
        for item in sorted(DATASETS_DIR.iterdir()):
            if item.is_dir() and (item / "data.yaml").exists():
                datasets.append((item.name, item / "data.yaml"))
    return datasets


def select_item(label, items, display_fn):
    print(f"\n{label}")
    print("-" * 80)
    for i, item in enumerate(items, 1):
        print(f"  {i}. {display_fn(item)}")
    while True:
        choice = input(f"\nSelect [1-{len(items)}] or 'q' to quit: ").strip().lower()
        if choice == 'q':
            return None
        if choice.isdigit() and 1 <= int(choice) <= len(items):
            return items[int(choice) - 1]
        print("Invalid selection.")


def main():
    print("\n" + "=" * 80)
    print("  YOLOv26 MODEL VALIDATION & ANALYSIS TOOL")
    print("=" * 80)

    if torch.cuda.is_available():
        print(f"\n✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("\n⚠️  No GPU - using CPU (slower)")

    models = find_models()
    if not models:
        print(f"\n❌ No trained models found in: {RUNS_DIR}")
        return

    fmt_model = lambda m: (
        f"[{m['model_letter']}] {m['name']}  "
        f"{m['type']}  {m['size_mb']:.1f}MB  "
        f"epochs={m['epochs']}  "
        f"{datetime.fromtimestamp(m['mtime']).strftime('%Y-%m-%d %H:%M')}"
    )
    selected = select_item("📦 Available Trained Models:", models, fmt_model)
    if selected is None:
        return
    weights_path = selected['path']
    print(f"\n✅ Model: {weights_path}")

    datasets = find_datasets()
    if not datasets:
        print(f"\n❌ No datasets found in {DATASETS_DIR}")
        return

    ds = select_item("📁 Select Dataset for Validation:", datasets, lambda d: d[0])
    if ds is None:
        return
    data_yaml = ds[1]
    print(f"✅ Dataset: {data_yaml}")

    data_config = load_data_config(data_yaml)

    print("\n📦 Loading model...")
    try:
        model = YOLO(str(weights_path))
        print("✅ Model loaded.")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    display_model_info(model, data_config)

    if input("\n🔍 Run validation on val set? [Y/n]: ").strip().lower() in ("", "y", "yes"):
        val_results = run_validation(model, str(data_yaml), split='val')
        if val_results and input("\n🔍 Also run on TEST set? [y/N]: ").strip().lower() in ("y", "yes"):
            run_validation(model, str(data_yaml), split='test')

    section("VALIDATION COMPLETE")
    print(f"\n✅ Done! Results saved to: {weights_path.parent.parent / 'val'}\n")


if __name__ == "__main__":
    main()
