"""Tests for hardware-independent helpers of the acquisition code."""
import numpy as np
import pytest

from conftest import load_module

_fit = load_module("2_acquisition/utilities/my_fit.py", "acq_my_fit")
_sanity = load_module("2_acquisition/utilities/sanity_test.py", "acq_sanity_test")
fit_line, build_line, build_line_inv = _fit.fit_line, _fit.build_line, _fit.build_line_inv
calculate_distance, check_motor_movement, send_sms = (
    _sanity.calculate_distance, _sanity.check_motor_movement, _sanity.send_sms)


def test_fit_line_recovers_slope_and_intercept():
    assert fit_line(0, 1, 2, 5) == (2.0, 1.0)


def test_build_line_and_inverse_agree():
    m, b = 2.0, 1.0
    x, y = build_line(m, b, 0, 10)
    np.testing.assert_allclose(y, m * x + b)
    x_inv, y_inv = build_line_inv(m, b, 1, 21)
    np.testing.assert_allclose(m * x_inv + b, y_inv)


def test_calculate_distance():
    assert calculate_distance((0, 0), (3, 4)) == 5.0


@pytest.mark.parametrize("end, moved", [((20, 0), True), ((5, 0), False)])
def test_check_motor_movement(end, moved):
    positions = [(0, 0)] * 10 + [end]
    assert check_motor_movement(positions, last_points=11) is moved


def test_send_sms_is_skipped_without_credentials(monkeypatch, capsys):
    for var in ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_FROM_NUMBER"):
        monkeypatch.delenv(var, raising=False)
    send_sms("+000", "test")              # must not raise or try to import twilio
    assert "skipped" in capsys.readouterr().out
