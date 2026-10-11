"""
Image preprocessing functions for fish tracking.

This module provides functions for:
- Image downsampling and Gaussian filtering
- Background subtraction
- Image binarization (thresholding)
- Noise removal and contour filtering
- Edge artifact removal from motor backlash

Core routines are implemented in core_preprocessing (self-contained in this package).
residual_edge_removal is specific to postprocessing (motor backlash artifact).

Author: Monica Coraggioso
"""
from .core_preprocessing import (
    preprocess_im,
    background_sub,
    bw_image,
    noise_removal,
    area_contours,
)

import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.signal import find_peaks, peak_widths


def residual_edge_removal(image, plot=None):
    """Remove edge residue artifacts caused by motor backlash.

    As a result of backlash, information about the position of the motor acquires
    noise over the course of the experiment and thus the mask becomes misaligned
    to the camera ROI.

    The idea is to remove any lines due to the edge by integrating the intensities
    along the x-axis and y-axis. If the integral has a pronounced peak it means
    that the line is present and the corresponding portion is set to zero in the
    binarized image.

    Parameters
    ----------
    image : 2D np.array, dtype=uint8
        Final binarized image, already preprocessed and multiplied by 'mask'
    plot : bool, optional
        True if you want to plot the peak detection process

    Returns
    -------
    image : 2D np.array, dtype=uint8
        Binarized image with 0 at residual edge lines

    Notes
    -----
    This function is specific to postprocessing and not needed for real-time
    tracking in code/.
    """
    kernel = np.ones((3, 3), np.uint8)

    row_intensities = np.sum(image, axis=1)
    p_r, _ = find_peaks(row_intensities, prominence=1.5, height=100)
    if len(p_r) != 0:
        max_p_r = np.argmax(row_intensities[p_r])
        if row_intensities[p_r[max_p_r]] > 100:
            results_r = peak_widths(row_intensities, p_r, rel_height=0.7)
            aa = int(results_r[2][0]), int(results_r[1][0])
            bb = int(results_r[3][0]), int(results_r[1][0])
            image[aa[0]:bb[0]] = 0
            image = cv2.erode(image, kernel)

    col_intensities = np.sum(image, axis=0)
    p_c, _ = find_peaks(col_intensities, prominence=1.5, height=100)
    if len(p_c) != 0:
        max_p_c = np.argmax(col_intensities[p_c])
        if col_intensities[p_c[max_p_c]] > 100:
            results_c = peak_widths(col_intensities, p_c, rel_height=0.7)
            cc = int(results_c[2][0]), int(results_c[1][0])
            dd = int(results_c[3][0]), int(results_c[1][0])
            image[:, cc[0]:dd[0]] = 0
            image = cv2.erode(image, kernel)

    if plot is not None:
        fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
        ax = axes.ravel()
        ax[0].plot(row_intensities, label='row_intensities')
        if len(p_r) != 0:
            ax[0].plot([p_r[max_p_r]], row_intensities[p_r[max_p_r]], 'ro', label='peak')
            ax[0].plot(aa[0], aa[1], 'go', label='left_bases')
            ax[0].plot(bb[0], bb[1], 'yo', label='right_bases')
        ax[0].legend()
        ax[1].plot(col_intensities, label='col_intensities')
        if len(p_c) != 0:
            ax[1].plot([p_c[max_p_c]], col_intensities[p_c[max_p_c]], 'ro', label='peak')
            ax[1].plot(cc[0], cc[1], 'go', label='left_bases')
            ax[1].plot(dd[0], dd[1], 'yo', label='right_bases')
        ax[1].legend()
    return image


__all__ = [
    'preprocess_im',
    'background_sub',
    'bw_image',
    'noise_removal',
    'area_contours',
    'residual_edge_removal',
]
