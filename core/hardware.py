"""CPU/memory/disk/GPU snapshot.

GPU info comes from pynvml when available, falling back to parsing
``nvidia-smi``; if neither works we return an empty GPU list. This module
**never raises** — it degrades gracefully — and never imports torch. It is
cheap enough to call about once per second.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time

import psutil

from .config import RUNS_DIR

_MIB = 1024 * 1024


# --- CPU / memory / disk -----------------------------------------------------
def _cpu() -> dict:
    try:
        per_core = psutil.cpu_percent(interval=None, percpu=True) or []
        if per_core:
            overall = round(sum(per_core) / len(per_core), 1)
        else:
            overall = psutil.cpu_percent(interval=None)
        freq = None
        try:
            freq = psutil.cpu_freq()
        except Exception:
            freq = None
        try:
            load = list(psutil.getloadavg())
        except (AttributeError, OSError):
            load = []
        return {
            "percent": overall,
            "per_core": [round(x, 1) for x in per_core],
            "cores": psutil.cpu_count(logical=True) or 0,
            "freq_mhz": round(freq.current) if freq else 0,
            "load_avg": [round(x, 2) for x in load],
        }
    except Exception:
        return {"percent": 0.0, "per_core": [], "cores": 0, "freq_mhz": 0, "load_avg": []}


def _memory() -> dict:
    try:
        m = psutil.virtual_memory()
        return {"total": m.total, "used": m.used, "available": m.available,
                "percent": m.percent}
    except Exception:
        return {"total": 0, "used": 0, "available": 0, "percent": 0.0}


def _swap() -> dict:
    try:
        s = psutil.swap_memory()
        return {"total": s.total, "used": s.used, "percent": s.percent}
    except Exception:
        return {"total": 0, "used": 0, "percent": 0.0}


def _disk() -> dict:
    path = RUNS_DIR if RUNS_DIR.exists() else RUNS_DIR.parent
    try:
        d = psutil.disk_usage(str(path))
        return {"path": str(path), "total": d.total, "used": d.used,
                "free": d.free, "percent": d.percent}
    except Exception:
        return {"path": str(path), "total": 0, "used": 0, "free": 0, "percent": 0.0}


# --- GPUs --------------------------------------------------------------------
def _gpus_pynvml() -> list[dict]:
    import pynvml  # raises ImportError if absent

    pynvml.nvmlInit()
    try:
        out = []
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(h)
            if isinstance(name, bytes):
                name = name.decode(errors="replace")
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
            except Exception:
                util = 0
            try:
                temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
            except Exception:
                temp = 0
            try:
                power = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
            except Exception:
                power = 0.0
            try:
                power_limit = pynvml.nvmlDeviceGetEnforcedPowerLimit(h) / 1000.0
            except Exception:
                try:
                    power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(h) / 1000.0
                except Exception:
                    power_limit = 0.0
            out.append({
                "index": i, "name": str(name),
                "mem_total": int(mem.total), "mem_used": int(mem.used),
                "mem_free": int(mem.free), "util": int(util), "temp": int(temp),
                "power": round(power, 1), "power_limit": round(power_limit, 1),
            })
        return out
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def _gpus_smi() -> list[dict]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        raise RuntimeError("nvidia-smi not found")
    fields = ("index,name,memory.total,memory.used,memory.free,"
              "utilization.gpu,temperature.gpu,power.draw,power.limit")
    proc = subprocess.run(
        [exe, f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=6,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "nvidia-smi failed")

    def _num(v, cast=float):
        v = v.strip()
        if v in ("", "[N/A]", "[Not Supported]", "N/A"):
            return 0
        try:
            return cast(v)
        except ValueError:
            return 0

    gpus = []
    for line in proc.stdout.strip().splitlines():
        if not line.strip():
            continue
        p = [x.strip() for x in line.split(",")]
        if len(p) < 9:
            continue
        gpus.append({
            "index": int(_num(p[0], int)),
            "name": p[1],
            "mem_total": int(_num(p[2], float)) * _MIB,
            "mem_used": int(_num(p[3], float)) * _MIB,
            "mem_free": int(_num(p[4], float)) * _MIB,
            "util": int(_num(p[5], float)),
            "temp": int(_num(p[6], float)),
            "power": round(_num(p[7], float), 1),
            "power_limit": round(_num(p[8], float), 1),
        })
    return gpus


def _gpus() -> tuple[list[dict], str]:
    try:
        return _gpus_pynvml(), "pynvml"
    except Exception:
        pass
    try:
        return _gpus_smi(), "nvidia-smi"
    except Exception:
        pass
    return [], "none"


def snapshot() -> dict:
    """Return the full hardware snapshot. Never raises."""
    gpus, backend = _gpus()
    return {
        "timestamp": time.time(),
        "cpu": _cpu(),
        "memory": _memory(),
        "swap": _swap(),
        "disk": _disk(),
        "gpus": gpus,
        "gpu_backend": backend,
    }


def gpu_count() -> int:
    gpus, _ = _gpus()
    return len(gpus)


if __name__ == "__main__":
    print(json.dumps(snapshot(), indent=2))
