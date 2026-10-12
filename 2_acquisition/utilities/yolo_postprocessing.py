"""
YOLO detection post-processing for the real-time tracker.

Pure-NumPy functions implementing the decision rules applied to YOLOv8 outputs
in the tracking thread of `freely_swimming.ipynb` (function `run_tracking`),
described in Chapter 2.3 of the PhD thesis (M. Coraggioso, Sorbonne Université,
2025, https://theses.hal.science/tel-05569582):

1. YOLO runs on frames downscaled by 2 (128x128 input), so box coordinates are
   scaled back to camera pixels.
2. When several objects are detected (dust, partial occlusion), the detection
   closest to the fish position at t-1 is kept (nearest-neighbour rule).
3. When nothing valid is detected (no box, or only NaN boxes), the previous position is kept with
   confidence 0.
4. Near the arena edges, the YOLO position is fused with a kinematic prediction
   (previous position + step along the heading), weighted by a smooth function
   of the YOLO confidence.
5. The fish is declared lost when the confidence falls below a threshold.

Keeping these rules in plain functions makes them unit-testable without a
camera, a GPU or the trained weights (see `tests/test_yolo_postprocessing.py`).

Author: Monica Coraggioso
"""
from typing import Sequence, Tuple

import numpy as np

DOWNSCALE = 2          # camera frame -> YOLO input (128x128)
TH_CONF = 0.1          # confidence scale of the fusion weight
LOST_CONF = 0.03       # below this confidence the fish is considered lost


def box_centers(xyxy: np.ndarray, scale: float = DOWNSCALE) -> np.ndarray:
    """Centres of YOLO boxes ``[x1, y1, x2, y2]``, rescaled to camera pixels.

    Returns an array of shape (n_boxes, 2).
    """
    xyxy = np.atleast_2d(np.asarray(xyxy, dtype=float))
    cx = (xyxy[:, 0] + xyxy[:, 2]) / 2 * scale
    cy = (xyxy[:, 1] + xyxy[:, 3]) / 2 * scale
    return np.column_stack([cx, cy])


def select_detection(xyxy: np.ndarray, confs: Sequence[float],
                     prev_pos: Sequence[float],
                     scale: float = DOWNSCALE) -> Tuple[float, float, float]:
    """Pick the fish among YOLO detections.

    Parameters
    ----------
    xyxy : array, shape (n, 4)
        YOLO boxes in YOLO-input pixels (may be empty).
    confs : sequence of float, length n
        YOLO confidence of each box.
    prev_pos : (x, y)
        Fish position at t-1, in camera pixels.

    Returns
    -------
    x, y, conf : position in camera pixels and confidence of the chosen box.
        If there is no valid detection, returns ``prev_pos`` with conf 0.
    """
    xyxy = np.asarray(xyxy, dtype=float).reshape(-1, 4)
    if len(xyxy) == 0:
        return float(prev_pos[0]), float(prev_pos[1]), 0.0

    centers = box_centers(xyxy, scale)
    dist = np.hypot(centers[:, 0] - prev_pos[0], centers[:, 1] - prev_pos[1])
    idx = int(np.nanargmin(dist)) if not np.isnan(dist).all() else 0

    if np.isnan(xyxy[idx]).any():
        return float(prev_pos[0]), float(prev_pos[1]), 0.0
    return float(centers[idx, 0]), float(centers[idx, 1]), float(confs[idx])


def confidence_weight(conf: float, th_conf: float = TH_CONF) -> float:
    """Weight given to the YOLO estimate: ``1 - exp(-conf / th_conf)``.

    0 for conf = 0, rises quickly and saturates at 1, so a moderately
    confident detection already dominates the kinematic prediction.
    (Same function as ``weighed_conf`` in the experiment notebook.)
    """
    return float(1.0 - np.exp(-conf / th_conf))


def kinematic_prediction(prev_pos: Sequence[float], head_dir: Sequence[float],
                         v_max_fish: float, dt: float, mm_per_pix: float,
                         img_size: float) -> Tuple[float, float]:
    """Predicted position: one step of length ``2 * v_max * dt`` along the heading.

    Distances in camera pixels; the prediction is clipped to the image.
    """
    step = (v_max_fish * dt / mm_per_pix) * 2
    x = min(max(prev_pos[0] + step * head_dir[0], 0), img_size)
    y = min(max(prev_pos[1] + step * head_dir[1], 0), img_size)
    return float(x), float(y)


def fuse_positions(yolo_xy: Sequence[float], kine_xy: Sequence[float],
                   conf: float, th_conf: float = TH_CONF) -> Tuple[float, float]:
    """Confidence-weighted fusion of the YOLO and kinematic estimates."""
    w = confidence_weight(conf, th_conf)
    x = w * (yolo_xy[0] - kine_xy[0]) + kine_xy[0]
    y = w * (yolo_xy[1] - kine_xy[1]) + kine_xy[1]
    return float(x), float(y)


def is_fish_lost(conf: float, threshold: float = LOST_CONF) -> bool:
    """True when the YOLO confidence is too low to trust the detection."""
    return conf < threshold
