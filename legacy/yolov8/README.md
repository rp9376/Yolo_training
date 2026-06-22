# YOLOv8 Training

Train YOLOv8 object detection models from scratch using all available GPUs.
Models only learn classes from your dataset — no pretrained COCO classes.

## Quick Start

```bash
# From the project root
source .venv/bin/activate
cd yolov8

# Single training session (interactive)
python train.py

# Or set up a queue and run unattended
python setup_queue.py
nohup ../../.venv/bin/python3 run_queue.py > queue_output.log 2>&1 &
```

## Scripts

| Script | Description |
|--------|-------------|
| `train.py` | Interactive training — select dataset, model size, epochs, batch |
| `setup_queue.py` | Configure multiple training tasks saved to `training_queue.json` |
| `run_queue.py` | Execute all pending queue tasks sequentially |
| `cleanup.py` | Remove queue file, logs, optionally `runs/` |
| `validate.py` | Validate a trained model with full metrics |
| `extract_models.py` | Copy best weights with descriptive filenames to `extracted_models/` |

## Model Sizes

| Name | Parameters | Speed | Accuracy |
|------|-----------|-------|----------|
| YOLOv8n | ~3.2M | Fastest | Lower |
| YOLOv8s | ~11.2M | Fast | Good |
| YOLOv8m | ~25.9M | Medium | Better |
| YOLOv8l | ~43.7M | Slower | High |
| YOLOv8x | ~68.2M | Slowest | Highest |

## Training Options

- **Model**: n / s / m / l / x
- **Epochs**: 50 / 100 / 200 / 300 / 500 or custom
- **Batch**: Auto (recommended) or 16 / 32 / 64 / 128
- **Image size**: 640px (default) or 1280px

All models train **from scratch** using `yolov8{size}.yaml` (YAML config, not pretrained `.pt`).
This ensures clean class labels — only what is in your dataset.

## Training Queue

### Setup

```bash
python setup_queue.py
```

- Configure dataset, model, epochs, batch, image size for each task
- Reuse the same dataset across tasks if needed
- All tasks saved to `training_queue.json`

### Execute

```bash
# Foreground
python run_queue.py

# Background — safe to close terminal
nohup ../../.venv/bin/python3 run_queue.py > queue_output.log 2>&1 &
```

### Monitor

```bash
tail -f queue_progress.log   # High-level task progress
tail -f queue_output.log     # Full training output
cat training_queue.json      # Raw queue state
```

### Cleanup

```bash
python cleanup.py               # Interactive
python cleanup.py --logs-only   # Remove logs and queue, keep models
python cleanup.py --all         # Remove everything including runs/
```

## Outputs

Training outputs are saved under `runs/detect/<run_name>/`:

```
runs/detect/x_e300_20260220_143022/
├── weights/
│   ├── best.pt     ← Best model (by validation fitness)
│   └── last.pt     ← Final epoch model
├── results.csv
├── results.png
├── confusion_matrix.png
└── args.yaml
```

## Inference

```bash
# Predict on image
yolo detect predict model=runs/detect/<run_name>/weights/best.pt source=image.jpg

# Predict on video
yolo detect predict model=runs/detect/<run_name>/weights/best.pt source=video.mp4

# Export to ONNX
yolo export model=runs/detect/<run_name>/weights/best.pt format=onnx

# Export to TensorRT
yolo export model=runs/detect/<run_name>/weights/best.pt format=engine
```

## GPU Configuration

The scripts automatically use ALL available GPUs:
- Single GPU → `device=0`
- Multi-GPU → `device=0,1,2,3` (all GPUs)
- No GPU → CPU fallback (very slow)

## Troubleshooting

**Out of memory** — Use smaller batch or model size, or reduce image size to 640.

**Poor results** — Train for 300+ epochs; use model `l` or `x`; add more data; check label quality.

**"No datasets found"** — Ensure `datasets/your_dataset/data.yaml` exists one level above `yolov8/`.
