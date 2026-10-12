"""
Image preprocessing: downsampling, Gaussian filtering, and binarization.

This module provides functions for preprocessing video frames before tracking,
including downsampling, Gaussian filtering, background subtraction, and
binarization. These operations are essential for robust fish tracking.

Author: Monica Coraggioso
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.transform import downscale_local_mean
import cv2
from scipy.signal import find_peaks, peak_widths
from typing import Tuple, Optional


def preprocess_im(frame: np.ndarray, gaussian_sigma: float,
                  downsampling: int) -> np.ndarray:
    """
    Preprocess image: downsampling and Gaussian filtering.
    
    Applies downsampling by local averaging and Gaussian smoothing to reduce
    noise and computational load.
    
    Parameters
    ----------
    frame : np.ndarray
        Input image (2D array, dtype=uint8).
    gaussian_sigma : float
        Standard deviation for Gaussian kernel in pixels. Typical value: 1.
    downsampling : int
        Integer factor by which to down-sample image by local averaging.
        Typical value: 2.
    
    Returns
    -------
    np.ndarray
        Preprocessed image (2D array, dtype=float32).
    
    Examples
    --------
    >>> processed = preprocess_im(raw_frame, gaussian_sigma=1, downsampling=2)
    """
    # Downsample by local averaging
    frame = downscale_local_mean(frame, downsampling)
    
    # Apply Gaussian filter for noise reduction
    frame = gaussian_filter(frame, gaussian_sigma)
    
    # Convert to float32
    frame = frame.astype(np.float32)
    
    return frame


def background_sub(frame: np.ndarray, background: np.ndarray,
                   convert: Optional[type] = None) -> np.ndarray:
    """
    Background subtraction: subtract background from frame.
    
    Computes frame - background, clamping negative values to zero. This
    highlights objects that are darker than the background.
    
    Parameters
    ----------
    frame : np.ndarray
        Current frame (2D array, dtype=float32), already preprocessed.
    background : np.ndarray
        Background image (2D array, dtype=float32), already preprocessed.
    convert : type, optional
        If specified, convert frame to this type before processing.
    
    Returns
    -------
    np.ndarray
        Background-subtracted image (2D array, dtype=uint8).
    
    Examples
    --------
    >>> subtracted = background_sub(frame, bg, convert=None)
    """
    if convert is not None:
        frame = frame.astype(np.float32)
    
    # Subtract background (background - frame)
    frame = (background - frame)
    
    # Clamp negative values to zero
    frame[frame < 0] = 0
    
    # Convert to uint8
    frame = frame.astype(np.uint8)
    
    return frame


def background_sub2(frame: np.ndarray, background: np.ndarray,
                    convert: Optional[type] = None) -> np.ndarray:
    """
    Background subtraction: subtract background from frame (alternative).
    
    Computes frame - background without clamping negative values. This
    highlights objects that are brighter than the background.
    
    Parameters
    ----------
    frame : np.ndarray
        Current frame (2D array, dtype=float32), already preprocessed.
    background : np.ndarray
        Background image (2D array, dtype=float32), already preprocessed.
    convert : type, optional
        If specified, convert frame to this type before processing.
    
    Returns
    -------
    np.ndarray
        Background-subtracted image (2D array, dtype=uint8).
    
    Notes
    -----
    Unlike `background_sub()`, this version does not clamp negative values,
    allowing detection of both brighter and darker objects.
    """
    if convert is not None:
        frame = frame.astype(np.float32)
    
    # Subtract background (frame - background)
    frame = (frame - background)
    
    # Note: negative values are not clamped here
    frame = frame.astype(np.uint8)
    
    return frame


def background_sub_uint8(frame: np.ndarray, background: np.ndarray) -> np.ndarray:
    """
    Background subtraction for uint8 images.
    
    Simplified version that works directly with uint8 arrays.
    
    Parameters
    ----------
    frame : np.ndarray
        Current frame (2D array, dtype=float32), already preprocessed.
    background : np.ndarray
        Background image (2D array, dtype=float32), already preprocessed.
    
    Returns
    -------
    np.ndarray
        Background-subtracted image (2D array, dtype=uint8).
    """
    # Subtract background and clamp negative values
    frame = (background - frame)
    frame[frame < 0] = 0
    
    return frame


def bw_image(im_raw: np.ndarray, gaussian_sigma: float, downsampling: int,
             background: np.ndarray, th_img: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Complete image preprocessing pipeline: binarization.
    
    Applies full preprocessing pipeline: downsampling, Gaussian filtering,
    background subtraction, and thresholding to create a binary image.
    
    Parameters
    ----------
    im_raw : np.ndarray
        Raw input image (2D array, dtype=uint8).
    gaussian_sigma : float
        Standard deviation for Gaussian kernel. Typical value: 1.
    downsampling : int
        Downsampling factor. Typical value: 2.
    background : np.ndarray
        Preprocessed background image (2D array, dtype=float32).
    th_img : int
        Threshold value for binarization (0-255). Must be chosen per experiment.
    
    Returns
    -------
    img_bw : np.ndarray
        Binary image after full preprocessing pipeline (dtype=uint8).
    img_bg : np.ndarray
        Image after preprocessing and background subtraction (dtype=uint8).
    img_ds : np.ndarray
        Image after preprocessing only (dtype=float32).
    
    Examples
    --------
    >>> img_bw, img_bg, img_ds = bw_image(raw_img, 1, 2, bg, 50)
    """
    # Preprocess: downsampling and Gaussian filtering
    img_ds = preprocess_im(im_raw, gaussian_sigma, downsampling)
    
    # Background subtraction
    img_bg = background_sub(img_ds, background)
    
    # Binarization
    img_bw = cv2.threshold(img_bg, th_img, 1, cv2.THRESH_BINARY)[1]
    
    return img_bw, img_bg, img_ds


def bw_image_simple(im_raw: np.ndarray, downsampling: int,
                    background: np.ndarray, th_img: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simplified image preprocessing: binarization without Gaussian filter.
    
    Similar to `bw_image()` but skips Gaussian filtering for faster processing.
    Useful when noise levels are low.
    
    Parameters
    ----------
    im_raw : np.ndarray
        Raw input image (2D array, dtype=uint8).
    downsampling : int
        Downsampling factor. Typical value: 2.
    background : np.ndarray
        Preprocessed background image (2D array, dtype=float32).
    th_img : int
        Threshold value for binarization (0-255). Must be chosen per experiment.
    
    Returns
    -------
    img_bw : np.ndarray
        Binary image after preprocessing (dtype=uint8).
    img_bg : np.ndarray
        Image after preprocessing and background subtraction (dtype=uint8).
    im_ds : np.ndarray
        Image after downsampling only (dtype=float32).
    
    Notes
    -----
    Unlike `bw_image()`, this function does not apply Gaussian filtering,
    which can be faster but may be more sensitive to noise.
    """
    # Downsampling only (no Gaussian filter)
    im_ds = downscale_local_mean(im_raw, downsampling)
    im_ds = im_ds.astype(np.float32)
    
    # Background subtraction
    img_bg = background_sub(im_ds, background)
    
    # Binarization
    img_bw = cv2.threshold(img_bg, th_img, 1, cv2.THRESH_BINARY)[1]
    
    return img_bw, img_bg, im_ds


def noise_removal(img_bw: np.ndarray, small_th: float,
                  big_th: Optional[float] = None) -> np.ndarray:
    """
    Remove noise objects and fill small holes in contours.
    
    Filters out contours that are too small (noise) or too large (artifacts).
    Also applies morphological closing to fill small holes in the fish contour.
    
    Parameters
    ----------
    img_bw : np.ndarray
        Binary image after preprocessing and binarization (dtype=uint8).
    small_th : float
        Minimum contour area. All contours with area < small_th are removed.
    big_th : float, optional
        Maximum contour area. If specified, all contours with area > big_th
        are removed. Use with caution when fish is attached to edge.
    
    Returns
    -------
    np.ndarray
        Cleaned binary image (dtype=uint8).
    
    Notes
    -----
    Morphological closing is applied with a 3x3 pixel kernel and 2 iterations
    to fill small holes in the image.
    
    Examples
    --------
    >>> cleaned = noise_removal(img_bw, small_th=10, big_th=1000)
    """
    # Find all contours
    cnts = cv2.findContours(img_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    
    # Remove contours that are too small or too large
    for c in cnts:
        area = cv2.contourArea(c)
        if area < small_th:
            cv2.fillPoly(img_bw, [c], color=0)
        if big_th is not None:
            if area > big_th:
                cv2.fillPoly(img_bw, [c], color=0)
    
    # Morphological closing to fill small holes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img_bw2 = cv2.morphologyEx(img_bw, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    return img_bw2


def area_contours(img: np.ndarray) -> np.ndarray:
    """
    Calculate areas of all contours in a binary image.
    
    Parameters
    ----------
    img : np.ndarray
        Binary image (2D array, dtype=uint8) after preprocessing and
        binarization.
    
    Returns
    -------
    np.ndarray
        Array of contour areas in pixels.
    
    Examples
    --------
    >>> areas = area_contours(img_bw)
    >>> print(f"Found {len(areas)} objects")
    """
    # Find all contours
    cnts = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    
    # Calculate area for each contour
    area = []
    for c in cnts:
        area.append(cv2.contourArea(c))
    
    return np.array(area)
