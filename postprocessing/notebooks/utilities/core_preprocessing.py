"""
Core image preprocessing functions for fish tracking post-processing.

This module provides the fundamental preprocessing routines used by the
post-processing pipeline: downsampling, filtering, background subtraction,
binarization, and contour-based noise removal. It is self-contained within
the postprocessing package and does not depend on other parts of the repository.

Author: Monica Coraggioso
"""
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.transform import downscale_local_mean
import cv2


def preprocess_im(frame, gaussian_sigma, downsampling):
    """Downsample and apply Gaussian filter.

    Parameters
    ----------
    frame : np.ndarray, shape (H, W), dtype uint8
        Raw grayscale frame.
    gaussian_sigma : float
        Standard deviation for Gaussian kernel (e.g. 1.0).
    downsampling : int
        Factor by which to downsample by local averaging (e.g. 2).

    Returns
    -------
    frame : np.ndarray, dtype float32
        Preprocessed frame.
    """
    frame = downscale_local_mean(frame, (downsampling, downsampling))
    frame = gaussian_filter(frame, gaussian_sigma)
    return frame.astype(np.float32)


def background_sub(frame, background):
    """Background subtraction.

    Parameters
    ----------
    frame : np.ndarray, dtype float32
        Frame already processed with preprocess_im.
    background : np.ndarray, dtype float32
        Background image already processed with preprocess_im.

    Returns
    -------
    frame : np.ndarray, dtype uint8
        Subtracted frame, non-negative.
    """
    frame = background - frame
    frame[frame < 0] = 0
    return frame.astype(np.uint8)


def bw_image(im_raw, gaussian_sigma, downsampling, background, th_img):
    """Full binarization pipeline: preprocess, background subtract, threshold.

    Parameters
    ----------
    im_raw : np.ndarray
        Raw input frame.
    gaussian_sigma : float
        Gaussian blur sigma for preprocess_im.
    downsampling : int
        Downsampling factor for preprocess_im.
    background : np.ndarray, dtype float32
        Preprocessed background image.
    th_img : int
        Binarization threshold (0–255).

    Returns
    -------
    img_bw : np.ndarray, dtype uint8
        Binary image (preprocess + background sub + threshold).
    img_bg : np.ndarray, dtype uint8
        Grayscale after preprocess + background sub.
    img_ds : np.ndarray, dtype float32
        Preprocessed (downsampled + filtered) only.
    """
    img_ds = preprocess_im(im_raw, gaussian_sigma, downsampling)
    img_bg = background_sub(img_ds, background)
    _, img_bw = cv2.threshold(img_bg, th_img, 255, cv2.THRESH_BINARY)
    return img_bw, img_bg, img_ds


def noise_removal(img_bw, small_th, big_th=None):
    """Remove contours by area and fill small holes with morphological closing.

    Parameters
    ----------
    img_bw : np.ndarray, dtype uint8
        Binary image.
    small_th : float
        Contours with area < small_th are filled with zeros.
    big_th : float, optional
        If given, contours with area > big_th are also filled (e.g. edge artifacts).

    Returns
    -------
    img_bw2 : np.ndarray, dtype uint8
        Cleaned binary image (3x3 morphological closing applied).
    """
    cnts = cv2.findContours(img_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        area = cv2.contourArea(c)
        if area < small_th:
            cv2.fillPoly(img_bw, [c], color=0)
        if big_th is not None and area > big_th:
            cv2.fillPoly(img_bw, [c], color=0)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img_bw2 = cv2.morphologyEx(img_bw, cv2.MORPH_CLOSE, kernel, iterations=2)
    return img_bw2


def area_contours(img):
    """Compute contour areas for a binary image.

    Parameters
    ----------
    img : np.ndarray, dtype uint8
        Binary image.

    Returns
    -------
    np.ndarray
        Array of contour areas.
    """
    cnts = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    areas = [cv2.contourArea(c) for c in cnts]
    return np.array(areas, dtype=float)
