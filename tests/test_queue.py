import pytest

from core import queue as q


def _task(tid, status="pending"):
    return {"id": tid, "status": status, "name": tid}


def test_add_and_get(root):
    q.add(_task("t1"))
    data = q.load()
    assert data["total_tasks"] == 1
    assert q.get("t1")["id"] == "t1"
    assert q.get("missing") is None


def test_counts_recomputed(root):
    q.add(_task("a", "completed"))
    q.add(_task("b", "failed"))
    q.add(_task("c", "pending"))
    d = q.load()
    assert d["total_tasks"] == 3
    assert d["completed_tasks"] == 1
    assert d["failed_tasks"] == 1


def test_update_remove_reorder(root):
    q.add(_task("a"))
    q.add(_task("b"))
    q.update("a", {"epochs": 5})
    assert q.get("a")["epochs"] == 5
    q.reorder(["b", "a"])
    assert [t["id"] for t in q.load()["tasks"]] == ["b", "a"]
    q.remove("a")
    assert q.get("a") is None


def test_reorder_validates_ids(root):
    q.add(_task("a"))
    with pytest.raises(ValueError):
        q.reorder(["a", "ghost"])


def test_running_lock(root):
    q.add(_task("a"))
    d = q.load()
    d["status"] = "running"
    q.save(d)

    with pytest.raises(q.QueueLocked):
        q.update("a", {"epochs": 5})
    with pytest.raises(q.QueueLocked):
        q.remove("a")
    with pytest.raises(q.QueueLocked):
        q.reorder(["a"])
    # Appending pending tasks is allowed while running.
    q.add(_task("b"))
    assert q.get("b") is not None


def test_clear_scopes(root):
    q.add(_task("a", "completed"))
    q.add(_task("b", "pending"))
    q.clear("completed")
    assert q.get("a") is None and q.get("b") is not None
    q.clear("all")
    assert q.load()["total_tasks"] == 0


def test_atomic_roundtrip(root):
    q.add(_task("a", "completed"))
    d1 = q.load()
    q.save(d1)
    d2 = q.load()
    assert d1["tasks"] == d2["tasks"]
    assert q.QUEUE_VERSION == d2["version"]
