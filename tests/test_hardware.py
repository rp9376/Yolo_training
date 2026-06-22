from core import hardware


def test_snapshot_shape():
    s = hardware.snapshot()
    for key in ("timestamp", "cpu", "memory", "swap", "disk", "gpus", "gpu_backend"):
        assert key in s
    assert isinstance(s["gpus"], list)
    for k in ("percent", "per_core", "cores", "freq_mhz", "load_avg"):
        assert k in s["cpu"]
    for k in ("total", "used", "available", "percent"):
        assert k in s["memory"]
    for k in ("path", "total", "used", "free", "percent"):
        assert k in s["disk"]


def test_gpus_pynvml(monkeypatch):
    fake = [{"index": 0, "name": "FakeGPU", "mem_total": 100, "mem_used": 10,
             "mem_free": 90, "util": 50, "temp": 40, "power": 1.0, "power_limit": 2.0}]
    monkeypatch.setattr(hardware, "_gpus_pynvml", lambda: fake)
    s = hardware.snapshot()
    assert s["gpu_backend"] == "pynvml"
    assert s["gpus"] == fake


def test_gpus_smi_fallback(monkeypatch):
    def boom():
        raise RuntimeError("no pynvml")
    fake = [{"index": 0, "name": "SmiGPU", "mem_total": 1, "mem_used": 0,
             "mem_free": 1, "util": 0, "temp": 0, "power": 0.0, "power_limit": 0.0}]
    monkeypatch.setattr(hardware, "_gpus_pynvml", boom)
    monkeypatch.setattr(hardware, "_gpus_smi", lambda: fake)
    s = hardware.snapshot()
    assert s["gpu_backend"] == "nvidia-smi"
    assert s["gpus"] == fake


def test_gpus_none(monkeypatch):
    def boom():
        raise RuntimeError("nope")
    monkeypatch.setattr(hardware, "_gpus_pynvml", boom)
    monkeypatch.setattr(hardware, "_gpus_smi", boom)
    s = hardware.snapshot()
    assert s["gpu_backend"] == "none"
    assert s["gpus"] == []
