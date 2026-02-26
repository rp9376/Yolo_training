# YOLO Model Training

Multi-GPU training framework supporting **YOLOv8** and **YOLOv26** object detection models.
All models train on custom datasets only — no COCO pretrained class interference.

## Project Structure

```
.
├── datasets/               ← Shared datasets (used by both YOLO versions)
│   └── your_dataset/
│       ├── data.yaml
│       ├── train/images/ & train/labels/
│       ├── valid/images/ & valid/labels/
│       └── test/images/  & test/labels/  (optional)
│
├── weights/                ← Base pretrained weight files (.pt)
│   └── yolo26n.pt
│
├── yolov8/                 ← YOLOv8 training system
│   ├── train.py            # Interactive single-run training
│   ├── setup_queue.py      # Build a training queue
│   ├── run_queue.py        # Execute the queue
│   ├── cleanup.py          # Clean logs, queue, and optionally runs/
│   ├── validate.py         # Validate a trained model
│   ├── extract_models.py   # Copy best weights with descriptive names
│   └── runs/               # Training outputs (auto-created, gitignored)
│
├── yolov26/                ← YOLOv26 training system (same structure)
│   ├── train.py
│   ├── setup_queue.py
│   ├── run_queue.py
│   ├── cleanup.py
│   ├── validate.py
│   ├── extract_models.py
│   └── runs/
│
└── .venv/                  ← Python virtual environment
```

## Setup

```bash
# Install dependencies (once)
python3 -m venv .venv
source .venv/bin/activate
pip install ultralytics torch torchvision
```

## Quick Start

### YOLOv8

```bash
source .venv/bin/activate
cd yolov8
python train.py          # Interactive single run
```

### YOLOv26

```bash
source .venv/bin/activate
cd yolov26
python train.py          # Interactive single run
```

## Training Queue (Unattended / Multiple Experiments)

Run multiple experiments back-to-back without manual intervention:

```bash
source .venv/bin/activate
cd yolov8          # or yolov26

# Step 1: Configure the queue interactively
python setup_queue.py

# Step 2: Run in background (terminal can be closed)
nohup ../../.venv/bin/python3 run_queue.py > queue_output.log 2>&1 &

# Monitor progress
tail -f queue_progress.log

# After training: clean up for next run
python cleanup.py
```

## Adding Datasets

Export from Roboflow in **YOLOv8** format and place under `datasets/`:

```
datasets/my_dataset/
├── data.yaml
├── train/images/  & train/labels/
├── valid/images/  & valid/labels/
└── test/images/   & test/labels/   (optional)
```

Both `yolov8/` and `yolov26/` automatically discover all datasets in `datasets/`.

## Tools

| Script | Purpose |
|--------|---------|
| `train.py` | Interactive single training session |
| `setup_queue.py` | Build a queue of multiple experiments |
| `run_queue.py` | Execute the queue (unattended) |
| `cleanup.py` | Remove logs/queue/runs |
| `validate.py` | Evaluate a trained model with full metrics |
| `extract_models.py` | Copy best weights with named outputs |

See `yolov8/README.md` and `yolov26/README.md` for version-specific details.
