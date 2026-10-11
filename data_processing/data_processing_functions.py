import os, re, glob
import numpy as np
import pandas as pd
from scipy.interpolate import splprep, splev


def extract_info_from_path(filepath):
    """
    Extract experiment metadata from HDF5 file path.

    Parameters
    ----------
    filepath : str
        Path to postprocessed HDF5 file

    Returns
    -------
    tuple
        (ID, dpf, date, fish, run) or (None, None, None, None, None) if no match
        ID format: 'dpf_date_fish_run' (e.g., '10_05Dec24_2_1')

    Examples
    --------
    >>> path = 'fs_10dpf_05Dec24/fish_2/postprocessed_runs/run1_postprocessing.h5'
    >>> extract_info_from_path(path)
    ('10_05Dec24_2_1', '10', '05Dec24', '2', '1')
    """
    # Pattern: fs_{dpf}dpf_{date}/fish_{fish}/postprocessed_runs/run{run}_postprocessing.h5
    pattern = r'fs_(\d+)dpf_(\d{2}[A-Za-z]{3}\d{2})[\\/]+fish_(\d+)[\\/]+postprocessed_runs[\\/]+run(\d+)_postprocessing\.h5'
    match = re.search(pattern, filepath)

    if not match:
        return None, None, None, None, None

    dpf, date, fish, run = match.groups()
    ID = f'{dpf}_{date}_{fish}_{run}'

    return ID, dpf, date, fish, run


def get_info_exps(data_path, verbose=False):
    """
    Scan directory for postprocessed experiments and build metadata table.

    Parameters
    ----------
    data_path : str
        Root directory containing experiment folders
    verbose : bool, default=True
        Print summary statistics

    Returns
    -------
    pd.DataFrame
        Columns: ['ID', 'age', 'date', 'fish', 'run']
        Sorted by age, date, fish, run
    """
    # Find all postprocessed files
    pattern = os.path.join(data_path, '**', 'run*_postprocessing.h5')
    postprocessing_paths = glob.glob(pattern, recursive=True)

    # Extract metadata from each path
    records = []
    for path in postprocessing_paths:
        ID, dpf, date, fish, run = extract_info_from_path(path)
        if ID:
            records.append({
                'ID': ID,
                'age': int(dpf),
                'date': date,
                'fish': int(fish),
                'run': int(run)
            })

    # Build and sort dataframe
    info_exps = pd.DataFrame(records)
    info_exps = info_exps.sort_values(['age', 'date', 'fish', 'run']).reset_index(drop=True)

    # Print summary
    if verbose:
        print('Experiments Summary:')
        print(f"  Ages (dpf): {info_exps['age'].nunique()}")
        print(f"  Fish: {info_exps.groupby('date')['fish'].nunique().sum()}")
        print(f"  Runs: {len(info_exps)}")
        print()
        print(info_exps)

    return info_exps


def get_datapath_file(info, datapath):
    """Return full path to the postprocessed HDF5 file for an experiment.

    Parameters:
        info: sequence-like with elements [dpf, day, fish, run]
        datapath: base data directory

    Returns:
        str: full path to the run postprocessing HDF5 file
    """
    file_path = os.path.join(
        datapath,
        f'fs_{info[0]}dpf_{info[1]}',
        f'fish_{info[2]}',
        'postprocessed_runs',
        f"run{info[3]}_postprocessing.h5"
    )
    return file_path


def get_kine(file):
    """
    Extract kinematic traces from an open HDF5 `file`.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - t: time stamps [s]
        - X, Y: positions [mm]
        - phi: orientation [deg]
    """
    t = file['post-processing/tvals'][:]  # time stamps [s]
    X = file['post-processing/X'][:]  # mm
    Y = file['post-processing/Y'][:]  # mm
    phi = file['post-processing/orient'][:]  # deg

    return pd.DataFrame({'t': t, 'X': X, 'Y': Y, 'phi': phi})


def kine_preprocessing(kine):
    """
    Clean kinematic data: remove NaNs and unwrap orientation.

    Parameters
    ----------
    kine : pd.DataFrame
        Kinematic data with 'phi' column (degrees)

    Returns
    -------
    pd.DataFrame
        Cleaned data with phi in continuous radians
    """
    # Remove NaNs
    kine = kine.dropna()

    # Convert degrees to radians and unwrap if needed
    phi = np.radians(kine['phi'].values)
    if np.any(np.abs(np.diff(phi)) > np.pi):
        phi = np.unwrap(phi % (2 * np.pi), discont=np.pi)

    kine.loc[:, 'phi'] = phi

    return kine


def get_pose(file):
    """
    Return spline length and spline coordinates from an open HDF5 file.

    Parameters
    ----------
    file : h5py.File-like
        Open HDF5 file containing expected datasets.

    Returns
    -------
    tuple
        (L, t, pose_xy)
        - L : float — spline length in mm from `fish_id/spline_length_mm`
        - t : ndarray — time stamps [s] from `post-processing/tvals`
        - pose_xy : ndarray — spline coordinates [mm] from `post-processing/spline_mm_zero`
    """

    L = file['fish_id/spline_length_mm'][0]  # mm
    t = file['post-processing/tvals'][:]
    pose_xy = file['post-processing/spline_mm_zero'][:]

    return t, L, pose_xy


def iqr_test(values, rm_extreme_only=False):
    """
    Identify outliers using Interquartile Range (IQR) method.

    Parameters
    ----------
    values : array-like
        Numerical dataset
    rm_extreme_only : bool, default=False
        If True, use factor=3 (extreme outliers only)
        If False, use factor=1.5 (standard threshold)

    Returns
    -------
    np.ndarray
        Boolean mask: True = inlier, False = outlier
    """
    factor = 3 if rm_extreme_only else 1.5

    Q1 = np.percentile(values, 25)
    Q3 = np.percentile(values, 75)
    IQR = Q3 - Q1

    lb = Q1 - factor * IQR
    ub = Q3 + factor * IQR

    return (values >= lb) & (values <= ub)


def filter_L(L, which='iqr'):
    """
    Filter spline length outliers using statistical tests.

    Parameters
    ----------
    L : array-like
        Spline lengths
    which : str, default='iqr'
        Method: 'z_score', 'mod_z_score', or 'iqr'

    Returns
    -------
    np.ndarray
        Boolean mask of valid values
    """
    if which == 'z_score':
        z = (L - np.mean(L)) / np.std(L)
        return np.abs(z) < 3

    if which == 'mod_z_score':
        median_L = np.median(L)
        MAD = max(np.median(np.abs(L - median_L)), 1e-8)
        mod_z = 0.6745 * (L - median_L) / MAD
        return mod_z < 3.5

    if which == 'iqr':
        return iqr_test(L)

    raise ValueError(f"Unknown method: {which}")


def cmpt_thetas(xy):
    """
    Compute angles between consecutive body points.

    Parameters
    ----------
    xy : array-like
        Shape (n_frames, n_points, 2) - body point coordinates

    Returns
    -------
    np.ndarray
        Shape (n_frames, n_points-1) - angles in radians
    """
    dxdy = xy[:, 1:, :] - xy[:, :-1, :]
    return np.arctan2(dxdy[:, :, 1], dxdy[:, :, 0])


def invert_ij(pose, i, j):
    """
    Swap points i and j if x_i > x_j (correct tracking errors).

    Parameters
    ----------
    pose : array-like
        Shape (n_frames, n_points, 2) - body point coordinates
    i, j : int
        Point indices to check and potentially swap

    Returns
    -------
    tuple
        (pose, n_swapped) - corrected pose and number of swaps
    """
    swapped = 0

    for idx in range(len(pose)):
        if pose[idx, i, 0] > pose[idx, j, 0]:
            pose[idx, [i, j], :] = pose[idx, [j, i], :]
            swapped += 1

    return pose, swapped


def spline_interpolation_filter(pose, tolerance=0.01):
    """
    Regularize fish pose with spline interpolation and filter by segment uniformity.

    For each frame:
    - Fits B-spline to body points (anchored at first point)
    - Resamples to equally spaced points
    - Filters frames where segment lengths are non-uniform (std > tolerance)

    Parameters
    ----------
    pose : array-like
        Shape (n_frames, n_points, 2) - body point coordinates
    tolerance : float, default=0.01
        Max std of segment lengths (higher = more permissive)

    Returns
    -------
    tuple
        (pose_interpolated, valid_mask)
        - pose_interpolated: regularized poses (same shape as input)
        - valid_mask: boolean array indicating successful fits
    """
    n_frames, n_points, _ = pose.shape
    pose_interpolated = np.zeros_like(pose)

    # Spline parameters
    s = 0.0  # No smoothing (interpolation only)
    u_eval = np.linspace(0, 1, n_points)

    valid_mask = np.zeros(n_frames, dtype=bool)

    for i in range(n_frames):
        x, y = pose[i, :, 0], pose[i, :, 1]

        # Weight first point heavily to anchor at origin
        w = np.ones(n_points)
        w[0] = 1e10

        try:
            # Fit and evaluate spline
            tck, _ = splprep([x, y], w=w, s=s)
            x_new, y_new = splev(u_eval, tck)
            pose_interpolated[i] = np.column_stack([x_new, y_new])

            # Check segment length uniformity
            deltas = np.diff(pose_interpolated[i], axis=0)
            distances = np.linalg.norm(deltas, axis=1)

            valid_mask[i] = np.std(distances) < tolerance

        except Exception:
            # Spline fit failed - keep invalid
            pose_interpolated[i] = pose[i]
            valid_mask[i] = False

    return pose_interpolated, valid_mask


def pose_preprocessing(t, L, pose, dropfirst = False, verbose = False):
    """
    Clean and regularize fish pose data through multiple quality filters.

    Applies sequential filters:
    1. Remove NaN values
    2. Remove outliers in spline length L (IQR method)
    3. Optionally drop first point and recenter
    4. Remove frames with abnormal head angles (>20°)
    5. Correct swapped points 10-11
    6. Apply spline interpolation to regularize shape

    Parameters
    ----------
    t : np.ndarray
        Timestamps
    L : np.ndarray
        Spline lengths
    pose : np.ndarray
        Shape (n_frames, n_points, 2) - fish body point coordinates
    dropfirst : bool, default=False
        If True, remove first point and recenter
    verbose : bool, default=False
        Print filtering statistics

    Returns
    -------
    tuple
        (t, L, pose) - filtered arrays
    """
    threshold_theta_head = np.pi / 9  # 20 degrees
    discarded = []
    N = len(pose)

    # 1 - Remove NaN frames
    idxs = ~np.any(np.isnan(pose), axis=(1, 2))
    discarded.append(np.sum(~idxs))
    t, L, pose = t[idxs], L[idxs], pose[idxs]

    # 2 - Filter spline length outliers (IQR)
    idxs = filter_L(L, which='iqr')
    discarded.append(np.sum(~idxs))
    t, L, pose = t[idxs], L[idxs], pose[idxs]

    # 3 - Drop first point and recenter
    if dropfirst:
        pose = pose[:, 1:, :] - pose[:, 1:2, :]

    # 4 - Remove abnormal head angles
    thetas_head = cmpt_thetas(pose[:, :5, :])
    idxs = np.all(np.abs(thetas_head) < threshold_theta_head, axis=1)
    discarded.append(np.sum(~idxs))
    t, L, pose = t[idxs], L[idxs], pose[idxs]

    # 5 - Correct swapped points 10-11
    pose, swapped = invert_ij(pose, 9, 10)

    # 6 - Spline interpolation to regularize shape
    pose, idxs = spline_interpolation_filter(pose)
    discarded.append(np.sum(~idxs))
    t, L, pose = t[idxs], L[idxs], pose[idxs]

    if verbose:
        print('** Pose Preprocessing Summary')
        print(f'Total frames: {N}')
        print(f'  - NaNs: {discarded[0]} removed, {N - discarded[0]} left')
        print(f'  - Length outliers: {discarded[1]} removed, {N - sum(discarded[:2])} left')
        print(f'  - Abnormal heads: {discarded[2]} removed, {N - sum(discarded[:3])} left')
        print(f'  - Failed spline fits: {discarded[3]} removed, {N - sum(discarded[:4])} left')
        print(f'Kept: {N - sum(discarded[:4])}/{N} ({100 * (N - sum(discarded[:4])) / N:.2f}%)')

    return t, L, pose


def cmpt_thetas_df(t, xy):
    """
    Compute angles between consecutive body points.

    Parameters
    ----------
    xy : array-like
        Shape (n_frames, n_points, 2) - body point coordinates
    t : array-like, optional
        Timestamps for each frame (required if as_dataframe=True)
    as_dataframe : bool, default=False
        If True, return DataFrame with columns ['t', 'theta_1', 'theta_2', ...]

    Returns
    -------
    np.ndarray or pd.DataFrame
        If as_dataframe=False: array of shape (n_frames, n_points-1)
        If as_dataframe=True: DataFrame with timestamps and angle columns
    """
    # Compute angles between consecutive points
    dxdy = xy[:, 1:, :] - xy[:, :-1, :]
    thetas = np.arctan2(dxdy[:, :, 1], dxdy[:, :, 0])

    # Build DataFrame
    df = pd.DataFrame({'t': t})
    theta_cols = {f'theta_{i}': thetas[:, i - 1] for i in range(1, thetas.shape[1] + 1)}

    return pd.concat([df, pd.DataFrame(theta_cols)], axis=1)


def print_h5(datapath, **dataframes):
    """Export experimental data"""

    if not os.path.exists(datapath):
        os.makedirs(datapath)

    for name, df in dataframes.items():
        df.to_hdf(os.path.join(datapath, f'{name}.h5'), key='df', mode='w', complevel=9, complib='blosc')
    return

