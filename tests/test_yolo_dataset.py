"""Integrity checks on the bundled YOLO training dataset (yolo/dataset/)."""
import re
from pathlib import Path

import pytest
import yaml
from PIL import Image

DATASET = Path(__file__).resolve().parents[1] / "yolo" / "dataset"
SPLITS = {"train": 106, "valid": 11, "test": 5}


def stems(split, sub, ext):
    return {p.stem for p in (DATASET / split / sub).glob(f"*{ext}")}


def source_frame(stem):
    """Roboflow names augmented copies 'image_<N>_jpg.rf.<hash>': N identifies the source frame."""
    return int(re.match(r"image_(\d+)_jpg", stem).group(1))


def test_data_yaml_declares_single_fish_class():
    cfg = yaml.safe_load((DATASET / "data.yaml").read_text())
    assert cfg["nc"] == 1 and cfg["names"] == ["fish"]
    for key in ("train", "val", "test"):
        assert (DATASET / cfg[key]).is_dir()


@pytest.mark.parametrize("split, n", SPLITS.items())
def test_every_image_has_a_label(split, n):
    images, labels = stems(split, "images", ".jpg"), stems(split, "labels", ".txt")
    assert images == labels
    assert len(images) == n


@pytest.mark.parametrize("split", SPLITS)
def test_labels_are_one_valid_yolo_box(split):
    for label in (DATASET / split / "labels").glob("*.txt"):
        rows = [r.split() for r in label.read_text().strip().splitlines()]
        assert len(rows) == 1, f"{label.name}: expected exactly one fish"
        cls, *box = rows[0]
        assert cls == "0"
        xc, yc, w, h = map(float, box)
        assert 0 <= xc <= 1 and 0 <= yc <= 1 and 0 < w <= 1 and 0 < h <= 1


@pytest.mark.parametrize("split", SPLITS)
def test_images_match_training_resolution(split):
    for img in (DATASET / split / "images").glob("*.jpg"):
        assert Image.open(img).size == (128, 128)


def test_no_leakage_between_splits():
    """Augmented copies of a frame must stay in one split, or validation scores are inflated."""
    frames = {s: {source_frame(x) for x in stems(s, "images", ".jpg")} for s in SPLITS}
    assert not frames["train"] & frames["valid"]
    assert not frames["train"] & frames["test"]
    assert not frames["valid"] & frames["test"]
