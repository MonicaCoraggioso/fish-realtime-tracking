# Data processing: building the analysis-ready datasets

This script takes the per-run output of [`../3_pose_estimation/`](../3_pose_estimation/), i.e. raw tracking data from longitudinal fish behavioral experiments, applying quality filters and consolidating results into unified datasets.

## Overview

The script scans a directory of raw HDF5 files, preprocesses kinematic and pose data for each experiment, and saves consolidated datasets for downstream analysis.

**Key processing steps:**
1. **Kinematic data** (X, Y, phi): Remove NaNs, unwrap orientation angles
2. **Pose data** (body shape): Filter outliers, regularize with spline interpolation, compute segment angles
3. **Consolidation**: Combine all experiments into single datasets with unique IDs


## Usage

```bash
pip install pandas h5py numpy scipy
python data_processing.py --input path/to/data_input --output path/to/data_output
```

The processed datasets used in the paper (`INFO.h5`, `KINE.h5`, `POSE.h5`, about 3 GB) are on [Zenodo](https://doi.org/10.5281/zenodo.17650136) (`Data_processing.zip`). You only need to run this script to rebuild them from post-processed runs.

## Input Data Structure

The script expects raw HDF5 files organized as follows:
```
data_input/
└── fs_{dpf}dpf_{date}/
        └── fish_{fish}/
            └── postprocessed_runs/
                └── run{run}_postprocessing.h5
```

**Example:**
```
data_input/fs_10dpf_05Dec24/fish_2/postprocessed_runs/run1_postprocessing.h5
```

## Output Data Structure

Results are saved to:
```
data_output/
└── INFO.h5
└── KINE.h5
└── POSE.h5
```

**Output datasets:**

1. **INFO** (metadata table)
   - Columns: `ID`, `age`, `date`, `fish`, `run`, `duration`, etc.
   - One row per experiment

2. **KINE** (kinematic data)
   - Columns: `ID`, `t`, `X`, `Y`, `phi`
   - Cleaned and preprocessed trajectories
   - `phi` unwrapped to continuous radians

3. **POSE** (body shape data)
   - Columns: `ID`, `t`, `L`, `theta_1`, `theta_2`, ..., `theta_19`
   - `L`: spline length (body size proxy)
   - `theta_i`: angle between body segments i and i+1

---

## Data Processing Details

### Kinematic Preprocessing
- **NaN removal**: Discard frames with missing position/orientation
- **Angle unwrapping**: Convert orientation from wrapped (±π) to continuous radians
  - Handles cases where fish completes multiple rotations
  - Prevents artificial jumps at ±π boundary

### Pose Preprocessing
Applies sequential filters:
1. **NaN removal**: Discard frames with missing body points
2. **Length outliers**: Remove frames with abnormal body size (IQR test)
3. **Head angle filter**: Discard frames with unrealistic head curvature (>20°)
4. **Point swap correction**: Fix tracking errors where points 10-11 are inverted
5. **Spline regularization**: Fit B-spline to body points for equal spacing
   - Only keep frames with uniform segment lengths (std < 0.01)

## Author

Vito Dichio (ENS–PSL), as part of the collaboration behind the paper.
