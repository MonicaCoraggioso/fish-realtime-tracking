"""Tests for the pose/kinematics cleaning used to build the analysis datasets."""
import numpy as np
import pandas as pd
import pytest

import data_processing_functions as dp


def straight_fish(n_frames=20, n_points=21, length=5.0, noise=0.0, seed=0):
    """Synthetic fish: a straight midline along +x, starting at the origin."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0, length, n_points)
    pose = np.zeros((n_frames, n_points, 2))
    pose[:, :, 0] = x
    pose[:, 1:, :] += noise * rng.standard_normal((n_frames, n_points - 1, 2))
    return pose


# --- metadata parsing --------------------------------------------------------

@pytest.mark.parametrize("path", [
    "data/fs_10dpf_05Dec24/fish_2/postprocessed_runs/run1_postprocessing.h5",
    r"D:\data\fs_10dpf_05Dec24\fish_2\postprocessed_runs\run1_postprocessing.h5",
])
def test_extract_info_from_path(path):
    assert dp.extract_info_from_path(path) == ("10_05Dec24_2_1", "10", "05Dec24", "2", "1")


def test_extract_info_from_path_no_match():
    assert dp.extract_info_from_path("not/an/experiment.h5") == (None,) * 5


# --- outlier filters ---------------------------------------------------------

def test_iqr_test_flags_outlier():
    values = np.array([1.0, 1.1, 0.9, 1.05, 0.95, 10.0])
    mask = dp.iqr_test(values)
    assert mask.tolist() == [True, True, True, True, True, False]


@pytest.mark.parametrize("method", ["iqr", "z_score", "mod_z_score"])
def test_filter_L_keeps_typical_lengths(method):
    L = np.r_[np.full(50, 5.0) + np.linspace(-0.05, 0.05, 50), 50.0]
    mask = dp.filter_L(L, which=method)
    assert mask[:-1].all()
    assert not mask[-1]


def test_filter_L_unknown_method():
    with pytest.raises(ValueError):
        dp.filter_L(np.ones(5), which="nope")


# --- geometry ----------------------------------------------------------------

def test_cmpt_thetas_known_angles():
    xy = np.array([[[0, 0], [1, 0], [1, 1], [0, 1]]], dtype=float)
    np.testing.assert_allclose(dp.cmpt_thetas(xy), [[0.0, np.pi / 2, np.pi]])


def test_cmpt_thetas_df_columns():
    pose = straight_fish(n_frames=3, n_points=5)
    df = dp.cmpt_thetas_df(np.arange(3), pose)
    assert list(df.columns) == ["t", "theta_1", "theta_2", "theta_3", "theta_4"]
    np.testing.assert_allclose(df.filter(like="theta").to_numpy(), 0.0)


def test_invert_ij_swaps_only_misordered_frames():
    pose = straight_fish(n_frames=2, n_points=5)
    pose[1, [1, 2]] = pose[1, [2, 1]]          # introduce a tracking swap in frame 1
    fixed, n_swapped = dp.invert_ij(pose.copy(), 1, 2)
    assert n_swapped == 1
    np.testing.assert_allclose(fixed, straight_fish(n_frames=2, n_points=5))


def test_spline_filter_accepts_uniform_midline():
    pose = straight_fish(n_frames=5)
    regularized, valid = dp.spline_interpolation_filter(pose)
    assert valid.all()
    np.testing.assert_allclose(regularized, pose, atol=1e-6)


# --- kinematics --------------------------------------------------------------

def test_kine_preprocessing_drops_nans_and_unwraps():
    phi_deg = np.array([170.0, 179.0, -179.0, -170.0, np.nan])
    kine = pd.DataFrame({"t": range(5), "X": 0.0, "Y": 0.0, "phi": phi_deg})
    out = dp.kine_preprocessing(kine)
    assert len(out) == 4
    assert np.all(np.abs(np.diff(out["phi"])) < np.pi)   # no artificial jump at ±180°


# --- full pose pipeline ------------------------------------------------------

def test_pose_preprocessing_end_to_end():
    pose = straight_fish(n_frames=30, n_points=21)
    pose[3] = np.nan                                 # a lost frame
    t = np.arange(30) / 250.0
    L = np.full(30, 5.0)
    t_out, L_out, pose_out = dp.pose_preprocessing(t, L, pose, dropfirst=True)
    assert len(t_out) == len(L_out) == len(pose_out) == 29
    assert pose_out.shape[1:] == (20, 2)
    assert not np.isnan(pose_out).any()
