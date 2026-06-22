from core import naming


def test_run_name_format():
    assert naming.run_name("x", 300, "20260616_120000") == "x_e300_20260616_120000"


def test_run_name_generates_timestamp():
    name = naming.run_name("n", 5)
    assert name.startswith("n_e5_")
    assert len(name.split("_")) == 4  # n, e5, date, time


def test_export_name_basic():
    meta = {"family": "yolov26", "size": "x", "imgsz": 640,
            "dataset_name": "military", "best_fitness": 0.42}
    n = naming.export_name(meta)
    assert n.startswith("yolo26_x_military")
    assert "f42" in n
    assert n.endswith(".pt")
    assert "640" not in n  # default imgsz omitted


def test_export_name_imgsz_and_last():
    meta = {"family": "yolov8", "size": "m", "imgsz": 1280,
            "dataset_name": "foo", "best_fitness": 0.0}
    # imgsz != 640 included, no fitness segment (0), "_last" suffix appended.
    assert naming.export_name(meta, which="last") == "yolov8_m_1280_foo_last.pt"


def test_clean_dataset_name():
    # The ".v7i.yolov8" suffix is stripped entirely.
    assert naming.clean_dataset_name(
        "/x/Military Vehicle Detection.v7i.yolov8/data.yaml"
    ) == "military_vehicle_detection"
