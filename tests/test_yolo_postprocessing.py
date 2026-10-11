"""Tests for the decision rules applied to YOLO detections in the real-time tracker."""
import numpy as np
import pytest

from conftest import load_module

yp = load_module("acquisition/utilities/yolo_postprocessing.py", "acq_yolo_postprocessing")


def test_box_centers_are_rescaled_to_camera_pixels():
    # YOLO sees a 2x-downscaled frame: a box centred at (10, 20) is at (20, 40) on camera
    np.testing.assert_allclose(yp.box_centers([[5, 15, 15, 25]]), [[20.0, 40.0]])


def test_single_detection_is_used():
    assert yp.select_detection([[5, 15, 15, 25]], [0.9], prev_pos=(0, 0)) == (20.0, 40.0, 0.9)


def test_dust_is_rejected_by_nearest_neighbour_rule():
    # the fish was at (100, 100); a more confident dust particle sits far away
    boxes = [[0, 0, 4, 4],        # dust, centre (4, 4) on camera
             [48, 48, 52, 52]]    # fish, centre (100, 100) on camera
    x, y, conf = yp.select_detection(boxes, [0.95, 0.40], prev_pos=(98, 101))
    assert (x, y, conf) == (100.0, 100.0, 0.40)


@pytest.mark.parametrize("boxes", [np.empty((0, 4)), [[np.nan] * 4]])
def test_no_valid_detection_keeps_previous_position(boxes):
    assert yp.select_detection(boxes, [0.5] * len(boxes), prev_pos=(12, 34)) == (12.0, 34.0, 0.0)


def test_confidence_weight_shape():
    assert yp.confidence_weight(0.0) == 0.0
    assert yp.confidence_weight(yp.TH_CONF) == pytest.approx(1 - np.exp(-1))
    assert yp.confidence_weight(0.9) > 0.999
    w = [yp.confidence_weight(c) for c in np.linspace(0, 1, 50)]
    assert np.all(np.diff(w) > 0)                     # monotonic


def test_fusion_trusts_confident_yolo_and_falls_back_to_kinematics():
    yolo, kine = (100.0, 100.0), (110.0, 90.0)
    assert yp.fuse_positions(yolo, kine, conf=0.0) == kine
    np.testing.assert_allclose(yp.fuse_positions(yolo, kine, conf=0.99), yolo, atol=1e-3)
    x, _ = yp.fuse_positions(yolo, kine, conf=0.05)
    assert 100 < x < 110                              # in between for an uncertain detection


def test_kinematic_prediction_steps_along_heading_and_stays_in_image():
    x, y = yp.kinematic_prediction((300, 300), (1, 0), v_max_fish=50, dt=1 / 20,
                                   mm_per_pix=0.05, img_size=600)
    assert (x, y) == (400.0, 300.0)                   # step = 2 * 50 * 0.05 / 0.05 px
    x, _ = yp.kinematic_prediction((590, 300), (1, 0), 50, 1 / 20, 0.05, 600)
    assert x == 600.0                                 # clipped at the image border


def test_fish_lost_threshold():
    assert yp.is_fish_lost(0.01)
    assert not yp.is_fish_lost(0.5)
