"""The training engine: build a task dict, then run the verbatim YOLO.train call.

``ultralytics``/``torch`` are imported lazily inside :func:`train_one` only, so
importing this module never pulls in the heavy stack.
"""

from __future__ import annotations

import csv
import os
import time
import uuid
from datetime import datetime
from pathlib import Path

from . import naming, weights
from .config import (AUGMENTATION, BATCH_PRESETS, DATASETS_DIR, DEFAULTS,
                     FAMILIES, IMGSZ_PRESETS, SIZES, family_prefix,
                     runs_project_dir)


def _resolve_dataset(dataset: str) -> tuple[str, str]:
    """Return ``(abs_data_yaml, dataset_name)`` from a name or a data.yaml path."""
    p = Path(dataset)
    if p.name == "data.yaml" and p.exists():
        return str(p.resolve()), p.parent.name
    cand = DATASETS_DIR / dataset / "data.yaml"
    if cand.exists():
        return str(cand.resolve()), dataset
    raise ValueError(f"Dataset not found: {dataset!r}")


def build_task(family: str, size: str, init: str, dataset: str,
               epochs: int, batch: int, imgsz: int,
               device: str = "auto") -> dict:
    """Validate inputs and return a complete pending task dict (schema §6)."""
    if family not in FAMILIES:
        raise ValueError(f"Unknown family: {family!r}")
    if size not in SIZES:
        raise ValueError(f"Unknown size: {size!r}")
    if init not in ("pretrained", "scratch"):
        raise ValueError("init must be 'pretrained' or 'scratch'")

    try:
        epochs = int(epochs)
    except (TypeError, ValueError):
        raise ValueError("epochs must be an integer")
    if epochs <= 0:
        raise ValueError("epochs must be positive")

    try:
        batch = int(batch)
    except (TypeError, ValueError):
        raise ValueError("batch must be an integer")
    if batch != -1 and batch <= 0:
        raise ValueError("batch must be -1 (auto) or a positive integer")

    try:
        imgsz = int(imgsz)
    except (TypeError, ValueError):
        raise ValueError("imgsz must be an integer")
    if imgsz <= 0:
        raise ValueError("imgsz must be positive")

    if device not in ("auto", "cpu", "all") and not all(p.isdigit() for p in str(device).split(",")):
        raise ValueError(f"Invalid device: {device!r}")

    data_yaml, dataset_name = _resolve_dataset(dataset)
    model_source, pretrained = weights.resolve(family, size, init)
    model = f"{family_prefix(family)}{size}"
    name = naming.run_name(size, epochs)
    project = str(runs_project_dir(family))

    return {
        "id": str(uuid.uuid4()),
        "family": family,
        "size": size,
        "model": model,
        "init": init,
        "model_source": model_source,
        "pretrained": pretrained,
        "dataset": data_yaml,
        "dataset_name": dataset_name,
        "epochs": epochs,
        "batch": batch,
        "imgsz": imgsz,
        "device": device,
        "patience": DEFAULTS["patience"],
        "workers": DEFAULTS["workers"],
        "name": name,
        "project": project,
        "status": "pending",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "started_at": None,
        "finished_at": None,
        "duration": None,
        "best_epoch": None,
        "best_fitness": None,
        "error": None,
        "run_dir": None,
    }


def resolve_device(device: str) -> str:
    """Resolve ``"auto"`` / ``"all"`` to the concrete device string for YOLO.

    * ``"auto"`` → first GPU (``"0"``), or ``"cpu"``.  Single-GPU avoids DDP.
    * ``"all"``  → all GPUs (``"0,1,2,..."``), triggering DDP.
    * Any other value is returned unchanged (``"cpu"``, ``"0"``, ``"0,1"`` …).
    """
    if device not in ("auto", "all"):
        return device
    try:
        import torch
        n = torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        n = 0
    if n <= 0:
        return "cpu"
    if device == "all" and n > 1:
        return ",".join(str(i) for i in range(n))
    return "0"


def _matching_ptxjit_lib() -> str | None:
    """Path to the ``libnvidia-ptxjitcompiler`` matching the active ``libcuda``,
    *only* when the installed soname symlink points at a different (stale)
    driver version.

    After an NVIDIA driver upgrade the ``libnvidia-ptxjitcompiler.so.1`` symlink
    is sometimes left pointing at the old driver build while ``libcuda.so.1`` is
    updated. NCCL JIT-compiles its device kernels through that library, so the
    version mismatch SIGSEGVs every DDP rank during process-group init (ordinary
    single-GPU training is unaffected because it runs precompiled SASS). We
    detect the mismatch and return the correct library to ``LD_PRELOAD``; the
    version is read from ``libcuda`` so this keeps working across driver bumps.
    Returns ``None`` when there is no mismatch or the files can't be located.
    """
    for libdir in ("/usr/lib/x86_64-linux-gnu", "/usr/lib64", "/usr/lib"):
        libcuda = os.path.join(libdir, "libcuda.so.1")
        if not os.path.exists(libcuda):
            continue
        ver = os.path.basename(os.path.realpath(libcuda)).split("libcuda.so.")[-1]
        if not ver[:1].isdigit():
            return None
        matching = os.path.join(libdir, f"libnvidia-ptxjitcompiler.so.{ver}")
        soname = os.path.join(libdir, "libnvidia-ptxjitcompiler.so.1")
        if not os.path.exists(matching):
            return None
        if not os.path.exists(soname) or os.path.realpath(soname) != os.path.realpath(matching):
            return matching
        return None  # soname already matches the driver; nothing to do
    return None


def _configure_ddp_env(on_event=None) -> None:
    """Prepare the environment so NCCL/DDP workers start cleanly on this host.

    Workers are launched by ``torch.distributed.run`` as child processes that
    inherit ``os.environ``, so setting these here propagates to every rank.

    * ``LD_PRELOAD`` the ptxjitcompiler matching the active driver — see
      :func:`_matching_ptxjit_lib` (fixes a SIGSEGV in every rank at init).
    * ``NCCL_CUMEM_HOST_ENABLE=0`` — NCCL's cuMem host allocations fail when the
      container runtime disables NUMA, which otherwise crashes the SHM transport.
    * ``NCCL_IB_DISABLE=1`` / ``OMP_NUM_THREADS=1`` — single-node hygiene.
    """
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("NCCL_CUMEM_HOST_ENABLE", "0")

    preload = _matching_ptxjit_lib()
    if preload:
        existing = os.environ.get("LD_PRELOAD", "")
        if preload not in existing.split(":"):
            os.environ["LD_PRELOAD"] = preload + (f":{existing}" if existing else "")
        if on_event is not None:
            on_event({"type": "log",
                      "message": f"Multi-GPU: preloading {os.path.basename(preload)} to match "
                                 f"the active driver (PTX JIT mismatch workaround)."})


def _best_epoch_from_csv(results_csv: Path) -> tuple[int, float]:
    best_fitness = 0.0
    best_epoch = 0
    try:
        with open(results_csv, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    map50 = float(row.get("metrics/mAP50(B)", 0) or 0)
                    map5095 = float(row.get("metrics/mAP50-95(B)", 0) or 0)
                    epoch = int(float(row.get("epoch", 0) or 0))
                    fitness = 0.1 * map50 + 0.9 * map5095
                    if fitness > best_fitness:
                        best_fitness = fitness
                        best_epoch = epoch
                except (ValueError, TypeError):
                    continue
    except Exception:
        pass
    return best_epoch, best_fitness


def _fake_train(task: dict, on_event=None) -> dict:
    """Fabricate a completed run without torch/ultralytics.

    Enabled by the ``YOLO_STUDIO_FAKE_TRAIN`` env var. This is a *test seam*:
    it lets the full pipeline (API -> process manager -> queue_runner) be
    exercised in well under a second with no GPU and no heavy imports. It is
    never enabled in normal operation.
    """
    run_dir = Path(task["project"]) / task["name"]
    (run_dir / "weights").mkdir(parents=True, exist_ok=True)
    (run_dir / "weights" / "best.pt").write_bytes(b"PK\x03\x04 fake-best")
    (run_dir / "weights" / "last.pt").write_bytes(b"PK\x03\x04 fake-last")
    (run_dir / "args.yaml").write_text(
        f"task: detect\nmode: train\nmodel: {task['model_source']}\n"
        f"data: {task['dataset']}\nepochs: {task['epochs']}\n"
        f"imgsz: {task['imgsz']}\nbatch: {task['batch']}\n"
        f"project: {task['project']}\nname: {task['name']}\n"
        f"pretrained: {task['pretrained']}\n"
    )
    rows = [(1, 0.20, 0.10), (2, 0.60, 0.40), (3, 0.50, 0.30)]
    header = ("epoch,time,train/box_loss,train/cls_loss,train/dfl_loss,"
              "metrics/precision(B),metrics/recall(B),metrics/mAP50(B),"
              "metrics/mAP50-95(B),val/box_loss,val/cls_loss,val/dfl_loss\n")
    lines = [header]
    for ep, m50, m5095 in rows:
        lines.append(f"{ep},{ep*1.0},1,1,1,0.7,0.6,{m50},{m5095},1,1,1\n")
        if on_event is not None:
            on_event({"type": "epoch", "epoch": ep, "total": len(rows),
                      "metrics": {"metrics/mAP50(B)": m50, "metrics/mAP50-95(B)": m5095}})
        time.sleep(0.05)
    (run_dir / "results.csv").write_text("".join(lines))
    try:
        from PIL import Image
        Image.new("RGB", (16, 16), (20, 30, 40)).save(run_dir / "results.png")
    except Exception:
        (run_dir / "results.png").write_bytes(b"\x89PNG\r\n")
    best_epoch, best_fitness = _best_epoch_from_csv(run_dir / "results.csv")
    return {"run_dir": str(run_dir), "best_epoch": best_epoch,
            "best_fitness": round(best_fitness, 4), "device": "cpu"}


def train_one(task: dict, on_event=None) -> dict:
    """Run the verbatim YOLO.train call for ``task`` and summarise the result.

    ``on_event`` (optional) is called with small dicts: ``{"type": "epoch", ...}``
    during training (best-effort; reliable on single-process CPU/1-GPU runs).
    """
    if os.environ.get("YOLO_STUDIO_FAKE_TRAIN"):
        return _fake_train(task, on_event)

    from ultralytics import YOLO  # lazy import (heavy)

    device = resolve_device(task.get("device", "auto"))
    is_ddp = "," in device

    # DDP workers are spawned by torch.distributed.run and inherit the
    # environment; configure NCCL and the driver-library workaround so they
    # start cleanly on this host (see _configure_ddp_env).
    if is_ddp:
        _configure_ddp_env(on_event)

    model_source = task["model_source"]
    project = task["project"]
    name = task["name"]
    Path(project).mkdir(parents=True, exist_ok=True)

    model = YOLO(model_source)

    if on_event is not None:
        def _epoch_cb(trainer):
            try:
                epoch = int(getattr(trainer, "epoch", 0)) + 1
                total = int(getattr(trainer, "epochs", task["epochs"]))
                metrics = {}
                raw = getattr(trainer, "metrics", None) or {}
                for k in ("metrics/mAP50(B)", "metrics/mAP50-95(B)",
                          "metrics/precision(B)", "metrics/recall(B)"):
                    if k in raw:
                        metrics[k] = float(raw[k])
                on_event({"type": "epoch", "epoch": epoch, "total": total,
                          "metrics": metrics})
            except Exception:
                pass
        try:
            model.add_callback("on_fit_epoch_end", _epoch_cb)
        except Exception:
            pass

    # With DDP, each GPU gets its own dataloader workers. Cap per-GPU workers
    # so total worker count (n_gpus × workers) doesn't exhaust /dev/shm.
    gpu_count = len(device.split(",")) if is_ddp else 1
    workers = max(1, task["workers"] // gpu_count)

    # AutoBatch (batch=-1) is unsupported under multi-GPU/DDP — ultralytics
    # raises before training. Substitute a valid batch that's a multiple of the
    # GPU count (its own suggested default of 8 images per GPU) so "All GPUs"
    # doesn't hard-fail when the batch was left on Auto.
    batch = task["batch"]
    if is_ddp and batch is not None and batch < 1:
        batch = gpu_count * 8
        if on_event is not None:
            on_event({"type": "log",
                      "message": f"Auto batch is unsupported with multiple GPUs; "
                                 f"using batch={batch} ({gpu_count} GPUs × 8)."})

    model.train(
        data=task["dataset"],
        epochs=task["epochs"],
        batch=batch,
        imgsz=task["imgsz"],
        device=device,
        project=project,
        name=name,
        patience=task["patience"],
        workers=workers,
        exist_ok=True,
        pretrained=task["pretrained"],
        verbose=True,
        **AUGMENTATION,
    )

    run_dir = Path(project) / name
    best_epoch, best_fitness = _best_epoch_from_csv(run_dir / "results.csv")
    return {
        "run_dir": str(run_dir),
        "best_epoch": best_epoch,
        "best_fitness": round(best_fitness, 4),
        "device": device,
    }
