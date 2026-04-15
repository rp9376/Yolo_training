#!/usr/bin/env python3
"""
Extract and Organize Trained YOLOv26 Models

Scans runs/ for completed training runs and copies best weights with
descriptive names based on model size, dataset, image size, and fitness.

Usage:
  python extract_models.py
  python extract_models.py --output-dir my_models
  python extract_models.py --runs-dir runs/detect
"""

# Re-exec with the project venv if needed (allows `python3 script.py` without activating).
import sys as _sys, os as _os
_venv_py = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '.venv', 'bin', 'python3'))
if _os.path.exists(_venv_py) and _os.path.abspath(_sys.executable) != _venv_py:
    _os.execv(_venv_py, [_venv_py] + _sys.argv)
del _venv_py, _sys, _os

import argparse
import csv
import shutil
import yaml
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent


def clean_dataset_name(data_path):
    try:
        name = Path(data_path).parent.name
        for suffix in [
            '.v1i.yolov8', '.v2i.yolov8', '.v3i.yolov8', '.v4i.yolov8',
            '.v5i.yolov8', '.v6i.yolov8', '.v7i.yolov8', '.yolov8',
            'roboflow-fast-model-augmented3x', ' images',
        ]:
            name = name.replace(suffix, '')
        name = name.replace(' ', '_').replace('-', '_').replace('.', '_').lower()
        while '__' in name:
            name = name.replace('__', '_')
        return name.strip('_') or 'unknown'
    except Exception:
        return 'unknown'


def get_model_size(model_str):
    model_str = str(model_str).lower()
    for s in ['n', 's', 'm', 'l', 'x']:
        if f'yolo26{s}' in model_str:
            return s
    # fallback: check generic yolo sizes
    for s in ['n', 's', 'm', 'l', 'x']:
        if model_str.endswith(s + '.pt') or model_str.endswith(s + '.yaml'):
            return s
    return 'unknown'


def extract_best_epoch(results_csv):
    best_fitness = 0.0
    best_epoch   = 0
    total_epochs = 0
    try:
        with open(results_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    map50    = float(row.get('metrics/mAP50(B)', 0))
                    map50_95 = float(row.get('metrics/mAP50-95(B)', 0))
                    fitness  = 0.1 * map50 + 0.9 * map50_95
                    epoch    = int(row.get('epoch', 0))
                    total_epochs = max(total_epochs, epoch)
                    if fitness > best_fitness:
                        best_fitness = fitness
                        best_epoch   = epoch
                except (ValueError, TypeError):
                    continue
    except Exception:
        pass
    return best_epoch, best_fitness, total_epochs


def extract_metadata(run_dir):
    args_file   = run_dir / "args.yaml"
    results_csv = run_dir / "results.csv"
    best_pt     = run_dir / "weights" / "best.pt"

    if not (args_file.exists() and results_csv.exists() and best_pt.exists()):
        return None

    try:
        with open(args_file) as f:
            args = yaml.safe_load(f)

        best_epoch, best_fitness, total_epochs = extract_best_epoch(results_csv)

        return {
            'run_name':     run_dir.name,
            'run_dir':      run_dir,
            'model_size':   get_model_size(args.get('model', '')),
            'imgsz':        args.get('imgsz', 640),
            'dataset_name': clean_dataset_name(args.get('data', '')),
            'batch':        args.get('batch', 'auto'),
            'epochs_cfg':   args.get('epochs', 0),
            'total_epochs': total_epochs,
            'best_epoch':   best_epoch,
            'best_fitness': best_fitness,
            'file_size_mb': best_pt.stat().st_size / (1024 * 1024),
            'best_pt_path': best_pt,
            'data_path':    args.get('data', ''),
        }
    except Exception as e:
        print(f"⚠️  Error reading {run_dir.name}: {e}")
        return None


def make_model_name(meta):
    parts = [meta['model_size']]
    if meta['imgsz'] != 640:
        parts.append(str(meta['imgsz']))
    dataset = meta['dataset_name'][:20]
    parts.append(dataset)
    fitness_pct = int(meta['best_fitness'] * 100)
    if fitness_pct > 0:
        parts.append(f"f{fitness_pct}")
    return "yolo26_" + "_".join(parts) + ".pt"


def scan_runs(runs_dir):
    runs_path = Path(runs_dir)
    if not runs_path.exists():
        print(f"❌ Runs directory not found: {runs_path}")
        return []

    run_dirs = [
        d for d in runs_path.glob("*")
        if d.is_dir() and (d / "weights" / "best.pt").exists()
    ]
    print(f"Found {len(run_dirs)} completed run(s)")
    return [m for d in run_dirs if (m := extract_metadata(d))]


def organize_models(models, output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"\n📁 Output: {out}")
    print("=" * 70)

    by_dataset = {}
    for m in models:
        by_dataset.setdefault(m['dataset_name'], []).append(m)

    copied = 0
    for dataset, group in by_dataset.items():
        print(f"\n📊 Dataset: {dataset}")
        print("-" * 70)
        for meta in sorted(group, key=lambda x: x['best_fitness'], reverse=True):
            name = make_model_name(meta)
            dest = out / name
            try:
                shutil.copy2(meta['best_pt_path'], dest)
                copied += 1
                print(f"✅ {name}")
                print(f"   Source:  {meta['run_name']}")
                print(f"   Model:   yolo26{meta['model_size']} @ {meta['imgsz']}px")
                print(f"   Epochs:  {meta['best_epoch']}/{meta['total_epochs']}")
                print(f"   Fitness: {meta['best_fitness']:.4f}")
                print(f"   Size:    {meta['file_size_mb']:.1f} MB")
            except Exception as e:
                print(f"❌ Failed to copy {name}: {e}")

    print("\n" + "=" * 70)
    print(f"✅ Copied {copied} model(s) to {out}")
    return copied


def main():
    parser = argparse.ArgumentParser(description='Extract organized YOLOv26 models')
    parser.add_argument('--output-dir', default=str(SCRIPT_DIR / 'extracted_models'),
                        help='Output directory (default: extracted_models/)')
    parser.add_argument('--runs-dir',   default=str(SCRIPT_DIR / 'runs' / 'detect'),
                        help='Runs directory to scan (default: runs/detect)')
    args = parser.parse_args()

    print("=" * 70)
    print("YOLOv26 Model Extractor")
    print("=" * 70)

    models = scan_runs(args.runs_dir)
    if not models:
        print(f"\n⚠️  No completed runs found in: {args.runs_dir}")
        return

    print(f"✅ Found {len(models)} model(s) with valid metadata")
    organize_models(models, args.output_dir)


if __name__ == "__main__":
    main()
