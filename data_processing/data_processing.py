#!/usr/bin/env python3
"""
Longitudinal Fish Experiment Data Processing Pipeline

This script processes raw kinematic and pose data from multiple fish experiments,
applies quality filters, and consolidates results into combined datasets.

Input:
    - Raw HDF5 files in data_input/ following the structure:
      fs_{dpf}dpf_{date}/fish_{fish}/postprocessed_runs/run{run}_postprocessing.h5

Output:
    - Consolidated HDF5 file in data_output/ containing:
      * INFO: experiment metadata (age, date, fish, run, etc.)
      * KINE: kinematic data (ID, t, X, Y, phi)
      * POSE: body shape data (ID, t, L, theta_1, theta_2, ...)

Processing steps:
    1. Scan input directory for all postprocessed experiments
    2. For each experiment:
       - Load raw kinematic data (X, Y, phi)
       - Clean and unwrap orientation angles
       - Load pose data (body point coordinates)
       - Filter outliers and regularize with spline interpolation
       - Compute body segment angles
    3. Concatenate all experiments into unified datasets
    4. Save to HDF5 format

Usage:
    python data_processing.py --input data_input --output data_output

Author: Vito Dichio (vito.dichio@ens.fr)
Date: 2025-01-29
"""

import argparse
import os
import sys
import pandas as pd
import h5py

CODE_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--input", default=os.path.join(CODE_DIR, "data_input"),
                        help="Folder with fs_{dpf}dpf_{date}/fish_{n}/postprocessed_runs/*.h5")
    parser.add_argument("--output", default=os.path.join(CODE_DIR, "data_output"),
                        help="Folder where INFO.h5, KINE.h5 and POSE.h5 are written")
    return parser.parse_args()


def main():
    """Main data processing pipeline."""

    # ================================
    # Setup paths
    # ================================

    args = parse_args()
    sys.path.insert(0, CODE_DIR)
    input_path = args.input
    output_path = args.output
    os.makedirs(output_path, exist_ok=True)

    # Import processing utilities
    import data_processing_functions as utils

    print("=" * 60)
    print("LONGITUDINAL DATA PROCESSING PIPELINE")
    print("=" * 60)

    # ================================
    # Load experiment metadata
    # ================================

    print("\n1. Scanning experiments...")
    info_exps = utils.get_info_exps(input_path, verbose=True)

    if info_exps.empty:
        print("❌ No experiments found. Exiting.")
        return

    # ================================
    # Process each experiment
    # ================================

    print("\n2. Processing individual experiments...")

    INFO = info_exps.copy(deep=True)
    KINE = []
    POSE = []

    for idx, ID in enumerate(info_exps['ID'], 1):
        print(f"\n[{idx}/{len(info_exps)}] Processing: {ID}")

        try:
            # Extract experiment metadata
            info = ID.split('_')

            # Load raw data
            filepath = utils.get_datapath_file(info, input_path)
            with h5py.File(filepath, "r") as file:

                # Process kinematics (X, Y, phi)
                kine = utils.get_kine(file)
                kine = utils.kine_preprocessing(kine)
                kine.insert(0, 'ID', ID)

                # Process pose (body shape)
                t, L, pose_xy = utils.get_pose(file)
                t, L, pose_xy = utils.pose_preprocessing(t, L, pose_xy, dropfirst=True, verbose=False)
                pose = utils.cmpt_thetas_df(t, pose_xy)
                pose.insert(0, 'ID', ID)
                pose.insert(2, 'L', L)

            # Accumulate
            KINE.append(kine)
            POSE.append(pose)

            print(f"  ✓ KINE: {len(kine)} frames, POSE: {len(pose)} frames")

        except Exception as e:
            print(f"  ❌ Failed: {e}")
            continue

    # ================================
    # Consolidate and save
    # ================================

    print("\n3. Consolidating datasets...")

    KINE = pd.concat(KINE, ignore_index=True)
    POSE = pd.concat(POSE, ignore_index=True)

    print(f"  Total KINE frames: {len(KINE):,}")
    print(f"  Total POSE frames: {len(POSE):,}")

    print("\n4. Saving to HDF5...")
    utils.print_h5(output_path, INFO=INFO, KINE=KINE, POSE=POSE)

    print("\n" + "=" * 60)
    print("✅ PROCESSING COMPLETE")
    print(f"   Experiments: {len(info_exps)}")
    print(f"   Output: {output_path}")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()