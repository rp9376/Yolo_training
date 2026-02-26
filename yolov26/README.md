# YOLOv26 Training

Train YOLOv26 object detection models using all available GPUs.
Supports training from pretrained weights or from scratch via YAML config.

## Quick Start

```bash
# From the project root
source .venv/bin/activate
cd yolov26

# Single training session (interactive)
python train.py

# Or set up a queue and run unattended
python setup_queue.py
nohup ../../.venv/bin/python3 run_queue.py > queue_output.log 2>&1 &
```

## Scripts

| Script | Description |
|--------|-------------|
| `train.py` | Interactive training — select dataset, model size, init mode, epochs, batch |
| `setup_queue.py` | Configure multiple training tasks saved to `training_queue.json` |
| `run_queue.py` | Execute all pending queue tasks sequentially |
| `cleanup.py` | Remove queue file, logs, optionally `runs/` |
| `validate.py` | Validate a trained model with full metrics |
| `extract_models.py` | Copy best weights with descriptive filenames to `extracted_models/` |

## Model Sizes

| Name | Speed | Accuracy |
|------|-------|----------|
| yolo26n | Fastest | Lower |
| yolo26s | Fast | Good |
| yolo26m | Medium | Better |
| yolo26l | Slower | High |
| yolo26x | Slowest | Highest |

## Initialization Modes

Unlike YOLOv8 (always from scratch), YOLOv26 supports two init modes:

| Mode | Description | When to use |
|------|-------------|-------------|
| **Pretrained weights** | Load `weights/yolo26{size}.pt` | Recommended — faster convergence |
| **From scratch** | Load `yolo26{size}.yaml` | No pretrained weights available |

Place pretrained weight files in `../weights/` (e.g., `yolo26n.pt`, `yolo26x.pt`).

## Training Options

- **Model**: n / s / m / l / x
- **Init**: pretrained `.pt` or from-scratch `.yaml`
- **Epochs**: 50 / 100 / 200 / 300 / 500 or custom
- **Batch**: Auto (recommended) or 16 / 32 / 64 / 128
- **Image size**: 640px (default) or 1280px

## Training Queue

### Setup

```bash
python setup_queue.py
```

Each task lets you choose:
- Dataset (auto-discovered from `../datasets/`)
- Model size
- Initialization mode (pretrained or from scratch)
- Epochs, batch size, image size

### Execute

```bash
# Foreground
python run_queue.py

# Background — safe to close terminal
nohup ../../.venv/bin/python3 run_queue.py > queue_output.log 2>&1 &
```

### Monitor

```bash
tail -f queue_progress.log
tail -f queue_output.log
cat training_queue.json
```

### Cleanup

```bash
python cleanup.py               # Interactive
python cleanup.py --logs-only   # Logs and queue only
python cleanup.py --all         # Everything including runs/
```

## Outputs

Training outputs saved under `runs/detect/<run_name>/`:

```
runs/detect/n_e300_20260220_143022/
├── weights/
│   ├── best.pt
│   └── last.pt
├── results.csv
├── results.png
├── confusion_matrix.png
└── args.yaml
```

## Inference

```bash
yolo detect predict model=runs/detect/<run_name>/weights/best.pt source=image.jpg
yolo export model=runs/detect/<run_name>/weights/best.pt format=onnx
```

## GPU Configuration

Scripts automatically use ALL available GPUs:
- Single GPU → `device=0`
- Multi-GPU → `device=0,1,2,3`
- No GPU → CPU fallback

## Troubleshooting

**Weight file not found** — Download or place `yolo26{size}.pt` in `../weights/`. Use "from scratch" mode if not available.

**Out of memory** — Use smaller batch or model size; reduce image size to 640.

**Poor results** — Train for 300+ epochs; use larger model (l, x); add more data.
