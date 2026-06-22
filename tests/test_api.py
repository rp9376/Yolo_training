import io
import os
import time
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import make_synthetic_dataset as synth
from backend.app import app
from core import config


@pytest.fixture
def client(root):
    with TestClient(app) as c:
        yield c


def _zip_bytes(src_dir: Path) -> io.BytesIO:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for f in Path(src_dir).rglob("*"):
            if f.is_file():
                zf.write(f, f.relative_to(src_dir).as_posix())
    buf.seek(0)
    return buf


# --- health / hardware / weights --------------------------------------------
def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    data = r.json()
    assert data["ok"] is True
    assert "python" in data
    assert "gpu_backend" in data


def test_hardware(client):
    r = client.get("/api/hardware")
    assert r.status_code == 200
    data = r.json()
    for key in ("cpu", "memory", "disk", "gpus", "gpu_backend"):
        assert key in data


def test_weights(client):
    r = client.get("/api/weights")
    assert r.status_code == 200
    data = r.json()
    assert "yolov8" in data and "yolov26" in data
    assert "n" in data["yolov26"]


# --- datasets ----------------------------------------------------------------
def test_dataset_upload_list_delete(client, tmp_path):
    src = synth.make_dataset(tmp_path / "src")
    buf = _zip_bytes(src)
    r = client.post("/api/datasets/upload",
                    files={"file": ("ds.zip", buf, "application/zip")},
                    data={"name": "apiup"})
    assert r.status_code == 200, r.text
    info = r.json()
    assert info["valid"]

    listing = client.get("/api/datasets").json()
    assert any(d["name"] == info["name"] for d in listing)

    r = client.delete(f"/api/datasets/{info['name']}")
    assert r.status_code == 200 and r.json()["deleted"]


def test_dataset_register(client, tmp_path):
    src = synth.make_dataset(tmp_path / "ext")
    r = client.post("/api/datasets/register", json={"path": str(src), "name": "apireg"})
    assert r.status_code == 200, r.text
    assert r.json()["source"] == "registered"


def test_dataset_register_bad_path(client, tmp_path):
    r = client.post("/api/datasets/register", json={"path": str(tmp_path / "nope")})
    assert r.status_code == 400


# --- queue CRUD --------------------------------------------------------------
def test_queue_build_reorder_remove_clear(client):
    synth.make_dataset(config.DATASETS_DIR / "synth")
    r1 = client.post("/api/queue/tasks", json={
        "family": "yolov26", "size": "n", "init": "scratch",
        "dataset": "synth", "epochs": 5, "batch": -1, "imgsz": 640})
    assert r1.status_code == 200, r1.text
    t1 = r1.json()
    assert t1["model"] == "yolo26n"
    assert t1["name"].startswith("n_e5_")

    r2 = client.post("/api/queue/tasks", json={
        "family": "yolov8", "size": "s", "init": "scratch",
        "dataset": "synth", "epochs": 50})
    t2 = r2.json()

    q = client.get("/api/queue").json()
    assert q["total_tasks"] == 2

    r = client.post("/api/queue/reorder", json={"order": [t2["id"], t1["id"]]})
    assert r.status_code == 200
    assert [t["id"] for t in r.json()["tasks"]] == [t2["id"], t1["id"]]

    r = client.delete(f"/api/queue/tasks/{t1['id']}")
    assert r.status_code == 200

    r = client.post("/api/queue/clear", json={"scope": "all"})
    assert r.status_code == 200
    assert client.get("/api/queue").json()["total_tasks"] == 0


def test_queue_build_invalid_dataset(client):
    r = client.post("/api/queue/tasks", json={
        "family": "yolov26", "size": "n", "init": "scratch",
        "dataset": "ghost", "epochs": 5})
    assert r.status_code == 400


def test_queue_start_no_pending(client):
    r = client.post("/api/queue/start")
    assert r.status_code == 400


# --- full start -> running -> completed lifecycle (fake-train subprocess) ----
def test_queue_lifecycle(client, monkeypatch):
    monkeypatch.setenv("YOLO_STUDIO_FAKE_TRAIN", "1")
    synth.make_dataset(config.DATASETS_DIR / "synth")
    task = client.post("/api/queue/tasks", json={
        "family": "yolov26", "size": "n", "init": "scratch",
        "dataset": "synth", "epochs": 3, "imgsz": 320, "batch": 2}).json()

    r = client.post("/api/queue/start")
    assert r.status_code == 202, r.text
    assert r.json()["started"] and r.json()["pid"]

    # Poll status until the runner finishes (fake train is sub-second).
    deadline = time.time() + 30
    final = None
    try:
        while time.time() < deadline:
            st = client.get("/api/queue/status").json()
            if not st["running"] and st["queue_status"] == "completed":
                final = st
                break
            time.sleep(0.3)
    finally:
        client.post("/api/queue/stop")

    assert final is not None, "runner did not complete in time"
    assert final["counts"]["completed"] == 1

    # The task is completed and shows up as a model.
    q = client.get("/api/queue").json()
    assert q["tasks"][0]["status"] == "completed"
    assert q["tasks"][0]["best_epoch"] == 2

    detail = client.get(f"/api/queue/tasks/{task['id']}/metrics").json()
    assert len(detail["series"]["epoch"]) == 3

    models = client.get("/api/models").json()
    assert any(m["run_name"] == task["name"] for m in models)


def test_queue_clear_all_stops_running(client, monkeypatch):
    import json as _json
    import sys as _sys

    synth.make_dataset(config.DATASETS_DIR / "synth")
    client.post("/api/queue/tasks", json={
        "family": "yolov26", "size": "n", "init": "scratch",
        "dataset": "synth", "epochs": 5})

    # Stand in for the runner with a long-lived subprocess so "clear all" has
    # something real to kill, independent of the actual training pipeline.
    monkeypatch.setenv("YOLO_STUDIO_RUNNER_CMD",
                        _json.dumps([_sys.executable, "-c", "import time; time.sleep(60)"]))
    r = client.post("/api/queue/start")
    assert r.status_code == 202, r.text
    pid = r.json()["pid"]

    deadline = time.time() + 5
    while time.time() < deadline and not client.get("/api/queue/status").json()["running"]:
        time.sleep(0.1)
    assert client.get("/api/queue/status").json()["running"]

    r = client.post("/api/queue/clear", json={"scope": "all"})
    assert r.status_code == 200, r.text
    assert client.get("/api/queue").json()["total_tasks"] == 0

    deadline = time.time() + 5
    while time.time() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        pytest.fail("runner process was not killed by clear-all")


# --- models against a fabricated run -----------------------------------------
def test_models_api(client, tmp_path):
    ds = synth.make_dataset(config.DATASETS_DIR / "synth")
    run = synth.make_fake_run(config.RUNS_DIR, name="n_e3_api",
                              data_yaml=str(ds / "data.yaml"))
    listing = client.get("/api/models").json()
    assert any(m["run_name"] == "n_e3_api" for m in listing)

    detail = client.get("/api/models/n_e3_api").json()
    assert detail["best_epoch"] == 2
    assert "results.png" in detail["artifacts"]

    art = client.get("/api/models/n_e3_api/artifact/results.png")
    assert art.status_code == 200
    assert art.headers["content-type"].startswith("image/")

    dl = client.get("/api/models/n_e3_api/download?which=best")
    assert dl.status_code == 200
    assert "attachment" in dl.headers.get("content-disposition", "")


def test_artifact_missing_returns_404(client, tmp_path):
    ds = synth.make_dataset(config.DATASETS_DIR / "synth")
    synth.make_fake_run(config.RUNS_DIR, name="n_e3_api2", data_yaml=str(ds / "data.yaml"))
    r = client.get("/api/models/n_e3_api2/artifact/nope.png")
    assert r.status_code == 404


def test_model_not_found(client):
    r = client.get("/api/models/does_not_exist")
    assert r.status_code == 404


# --- SSE ---------------------------------------------------------------------
def test_sse_yields_event(root):
    # Drive the async generator directly with a request that disconnects
    # immediately, so we get the initial events without the heartbeat sleep
    # and without TestClient's streaming-teardown hang.
    import asyncio

    from backend.api import stream

    class FakeRequest:
        async def is_disconnected(self):
            return True

    async def collect():
        import asyncio as _aio
        ev = _aio.Event()
        out = []
        async for chunk in stream._generate(FakeRequest(), ev):
            out.append(chunk)
        return out

    chunks = asyncio.run(collect())
    joined = "".join(chunks)
    assert any(c.startswith("event:") for c in chunks)
    assert "event: status" in joined
