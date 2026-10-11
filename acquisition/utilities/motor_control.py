"""
Motor-following law of the tracking stage, as described in the PhD thesis.

Reference implementation of Section 2.3.3 ("Motor piloting") of
M. Coraggioso, *Multiscale locomotor dynamics in Danionella cerebrum*,
PhD thesis, Sorbonne Université (2025), https://theses.hal.science/tel-05569582

The motor is pulled toward the fish by a virtual spring in series with a
dashpot. At every motor update (period ``dt_motor``):

    step:   dx' = (dx / r) * V * dt_motor,   dy' = (dy / r) * V * dt_motor      (Eq. 2.2)
    speed:  V(t+1) = alpha * r + (1 - beta) * V(t)                             (Eq. 2.3)

where (dx, dy) is the fish offset from the centre of the camera ROI and r its
norm. Eq. 2.3 is written in the thesis as V(t+1) = -alpha*r + (1-beta)*V(t) with
r measured from the fish to the target; here r is the (positive) distance from
the target to the fish, so V is the speed toward the fish.

    gain:     alpha = V_MAX / R_MAX       (R_MAX: radius of the ROI)
    damping:  beta  = 0.8 * sqrt(alpha * dt_motor)   (calibrated with a projected moving dot)

V_MAX ranges from 100 mm/s for young, slow larvae to 300 mm/s for older,
faster fish. The motor also never leaves software limits registered at the
arena edges (crossing them would trigger a GRBL ALARM).

The live experiment code (`run_motor` / `speed_osc` in `freely_swimming.ipynb`)
is not modified; this module documents the control law and makes it testable.

Author: Monica Coraggioso
"""
from typing import Tuple

import numpy as np


def gain(v_max: float, r_max: float) -> float:
    """Response gain alpha = V_MAX / R_MAX (units 1/s if V in mm/s and R in mm)."""
    return v_max / r_max


def damping(alpha: float, dt_motor: float) -> float:
    """Damping beta = 0.8 * sqrt(alpha * dt_motor), keeping the virtual spring underdamped."""
    return 0.8 * np.sqrt(alpha * dt_motor)


def next_speed(r: float, v_prev: float, alpha: float, beta: float) -> float:
    """Eq. 2.3: speed toward the fish at the next motor update."""
    return alpha * r + (1 - beta) * v_prev


def motor_step(dx: float, dy: float, v: float, dt_motor: float) -> Tuple[float, float]:
    """Eq. 2.2: relative motor displacement along the direction of the fish."""
    r = np.hypot(dx, dy)
    if r == 0:
        return 0.0, 0.0
    return float(dx / r * v * dt_motor), float(dy / r * v * dt_motor)


def clip_to_limits(x: float, y: float, x_lim: Tuple[float, float],
                   y_lim: Tuple[float, float]) -> Tuple[float, float]:
    """Keep the motor target inside the software limits registered at the arena edges."""
    return float(np.clip(x, *x_lim)), float(np.clip(y, *y_lim))
