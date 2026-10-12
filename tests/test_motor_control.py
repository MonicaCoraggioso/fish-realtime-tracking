"""Tests for the motor-following law described in the thesis (Section 2.3.3)."""
import numpy as np
import pytest

from conftest import load_module

mc = load_module("2_acquisition/utilities/motor_control.py", "acq_motor_control")

DT_MOTOR = 1 / 18          # motor thread runs at ~18 Hz (thesis, Fig. 2.3)
# (V_MAX [mm/s], ROI radius [mm]) for young and old fish (ROI diameter 20 mm at 2 wpf, 35 mm at 5 wpf)
REGIMES = {"young": (100, 10.0), "old": (300, 17.5)}


def simulate(fish_xy, v_max, r_max, n_steps=80):
    """Closed loop: fish trajectory (n_steps, 2) -> motor positions and fish-to-centre distances."""
    alpha = mc.gain(v_max, r_max)
    beta = mc.damping(alpha, DT_MOTOR)
    motor, v, motors, dists = np.zeros(2), 0.0, [], []
    for fish in fish_xy[:n_steps]:
        dx, dy = fish - motor
        r = np.hypot(dx, dy)
        v = mc.next_speed(r, v, alpha, beta)
        motor = motor + mc.motor_step(dx, dy, v, DT_MOTOR)
        motors.append(motor.copy())
        dists.append(np.hypot(*(fish - motor)))
    return np.array(motors), np.array(dists)


def test_gain_and_damping_formulas():
    alpha = mc.gain(100, 10)
    assert alpha == 10
    assert mc.damping(alpha, DT_MOTOR) == pytest.approx(0.8 * np.sqrt(10 / 18))


def test_speed_grows_with_distance_and_relaxes_when_centred():
    alpha, beta = 10.0, 0.6
    assert mc.next_speed(5, 0, alpha, beta) > mc.next_speed(1, 0, alpha, beta)
    v = 50.0
    for _ in range(20):
        v = mc.next_speed(0, v, alpha, beta)   # fish centred: speed decays geometrically
    assert v == pytest.approx(50 * 0.4 ** 20)


def test_step_points_toward_the_fish():
    dx, dy = mc.motor_step(3, 4, v=10, dt_motor=0.1)
    assert (dx, dy) == pytest.approx((0.6, 0.8))          # unit direction (0.6, 0.8), length 1 mm
    assert mc.motor_step(0, 0, v=10, dt_motor=0.1) == (0.0, 0.0)


@pytest.mark.parametrize("regime", REGIMES)
def test_recentres_a_still_fish_smoothly(regime):
    v_max, r_max = REGIMES[regime]
    fish = np.tile([8.0, 0.0], (80, 1))                   # fish stopped 8 mm off-centre
    motors, dists = simulate(fish, v_max, r_max)
    assert dists[-1] < 0.01                               # converges onto the fish
    assert motors[:, 0].max() < 8.0 * 1.25                # underdamped: small, bounded overshoot


@pytest.mark.parametrize("regime", REGIMES)
def test_keeps_a_swimming_fish_inside_the_roi(regime):
    v_max, r_max = REGIMES[regime]
    t = np.arange(80) * DT_MOTOR
    speed = v_max / 4                                     # fish cruising at a quarter of V_MAX
    fish = np.column_stack([speed * t, np.zeros_like(t)])
    _, dists = simulate(fish, v_max, r_max)
    assert dists.max() < r_max                            # never leaves the field of view


def test_motor_stays_within_software_limits():
    assert mc.clip_to_limits(-5, 500, (0, 400), (0, 400)) == (0.0, 400.0)
    assert mc.clip_to_limits(120, 80, (0, 400), (0, 400)) == (120.0, 80.0)
