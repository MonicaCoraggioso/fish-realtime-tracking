"""Checks on the trained YOLOv8 weights.

Skipped automatically when `ultralytics` is not installed or the weights are not
present (e.g. on GitHub Actions). To run them locally, download
`yolo8m_live_tracking.pt` from the GitHub Release into `1_yolo_detector/` and run:

    pip install ultralytics
    pytest tests/test_yolo_model.py -v
"""
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "1_yolo_detector" / "yolo8m_live_tracking.pt"
TEST_SET = ROOT / "1_yolo_detector" / "dataset" / "test"

ultralytics = pytest.importorskip("ultralytics")
pytestmark = pytest.mark.skipif(not WEIGHTS.exists(), reason="trained weights not downloaded")


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union


def label_box(path, size=128):
    _, xc, yc, w, h = map(float, path.read_text().split())
    return [(xc - w / 2) * size, (yc - h / 2) * size, (xc + w / 2) * size, (yc + h / 2) * size]


@pytest.fixture(scope="module")
def model():
    return ultralytics.YOLO(str(WEIGHTS))


def test_model_has_single_fish_class(model):
    assert list(model.names.values()) == ["fish"]


def test_fish_is_found_on_every_test_image(model):
    for img in sorted((TEST_SET / "images").glob("*.jpg")):
        result = model(str(img), imgsz=128, conf=0.25, verbose=False)[0]  # Ultralytics default threshold
        assert len(result.boxes) >= 1, f"no fish detected in {img.name}"


def test_best_box_overlaps_ground_truth(model):
    ious = []
    for img in sorted((TEST_SET / "images").glob("*.jpg")):
        truth = label_box(TEST_SET / "labels" / f"{img.stem}.txt")
        boxes = model(str(img), imgsz=128, conf=0.01, verbose=False)[0].boxes.xyxy.cpu().numpy()
        ious.append(max((iou(b, truth) for b in boxes), default=0.0))
    assert np.mean(ious) > 0.5
