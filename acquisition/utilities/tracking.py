"""
Online tracking: position of the fish + tail points.

This module provides functions for tracking fish position, orientation, and tail
segmentation in real-time video streams. It uses computer vision techniques
including centroid calculation, ellipse fitting, and contour analysis.

Author: Monica Coraggioso
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import cv2
from typing import Tuple, List, Optional, Union
from scipy.stats import skew
from scipy.signal import find_peaks
from utilities.preprocessing import area_contours


def centroid(img: np.ndarray) -> Tuple[List[float], List[float]]:
    """
    Calculate centroid coordinates for all objects in a binary image.
    
    Uses image moments to compute the center of mass for each detected contour.
    This is a "lazy centroid" approach that finds all objects without filtering.
    
    Parameters
    ----------
    img : np.ndarray
        Binary image (2D array, dtype=uint8 or float32) with objects as white
        pixels on black background.
    
    Returns
    -------
    cX : List[float]
        X-coordinates of centroids in pixels.
    cY : List[float]
        Y-coordinates of centroids in pixels.
    """
    cX = []
    cY = []
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    for c in contours:
        # Calculate moments for each contour
        M = cv2.moments(c)
        
        # Calculate x,y coordinate of center
        if M["m00"] != 0:
            cX.append(M["m10"] / M["m00"])
            cY.append(M["m01"] / M["m00"])
        else:
            cX.append(0)
            cY.append(0)
    
    return cX, cY


def centroid2(img: np.ndarray) -> Tuple[List[float], List[float], List]:
    """
    Calculate centroid coordinates and return contours.
    
    Similar to `centroid()` but also returns the detected contours for further
    processing.
    
    Parameters
    ----------
    img : np.ndarray
        Binary image (2D array, dtype=uint8 or float32) with objects as white
        pixels on black background.
    
    Returns
    -------
    cX : List[float]
        X-coordinates of centroids in pixels.
    cY : List[float]
        Y-coordinates of centroids in pixels.
    contours : List
        Detected contours from OpenCV.
    """
    cX = []
    cY = []
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    for c in contours:
        # Calculate moments for each contour
        M = cv2.moments(c)
        
        # Calculate x,y coordinate of center
        if M["m00"] != 0:
            cX.append(M["m10"] / M["m00"])
            cY.append(M["m01"] / M["m00"])
        else:
            cX.append(0)
            cY.append(0)
    
    return cX, cY, contours


def find_nearest(array: Union[np.ndarray, List], value: float) -> int:
    """
    Find the index of the array element closest to a given value.
    
    Parameters
    ----------
    array : np.ndarray or List
        Array to search in.
    value : float
        Target value to find the nearest element to.
    
    Returns
    -------
    int
        Index of the element closest to the target value.
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def fish_position(img_bw: np.ndarray, area0: float, first_loop: bool,
                  areas: List[float], xfish: List[float], 
                  yfish: List[float]) -> Tuple[float, float, bool]:
    """
    Track fish position using area and position constraints.
    
    When multiple objects are detected, this function uses area and position
    constraints to select the most likely fish object. It maintains tracking
    continuity by preferring objects closest to the previous fish position.
    
    Parameters
    ----------
    img_bw : np.ndarray
        Binary image after preprocessing and thresholding.
    area0 : float
        Expected fish area (used for filtering).
    first_loop : bool
        Whether this is the first iteration (affects fallback behavior).
    areas : List[float]
        List of detected object areas (maintained across frames).
    xfish : List[float]
        List of previous fish x-positions (maintained across frames).
    yfish : List[float]
        List of previous fish y-positions (maintained across frames).
    
    Returns
    -------
    xfish_current : float
        Current fish x-position in pixels.
    yfish_current : float
        Current fish y-position in pixels.
    fish_lost : bool
        True if fish could not be detected.
    
    See Also
    --------
    centroid : Calculate centroids of all objects.
    """
    fish_lost = False
    
    # Get areas and centroids of all detected objects
    a = area_contours(img_bw)
    xc, yc = centroid(img_bw)
    
    if len(a) == 1:
        # Single object detected - assume it's the fish
        areas.append(a[0])
        xfish.append(xc[0])
        yfish.append(yc[0])
    
    elif len(a) == 0:
        # No objects detected - fish lost
        print('ERROR: fish lost')
        fish_lost = True
        if first_loop:
            print('ERROR: fish lost at first loop, re-start!')
            xfish.append(150)
            yfish.append(150)
        else:
            # Use last known position
            xfish.append(xfish[-1])
            yfish.append(yfish[-1])
    
    else:
        # Multiple objects detected - use constraints to select fish
        # Find object with area closest to mean fish area
        idx_a = find_nearest(a, np.mean(areas))
        
        if first_loop:
            idx_px = idx_a
            idx_py = idx_a
        else:
            # Find object closest to previous fish position
            idx_px = find_nearest(xc, xfish[-1])
            idx_py = find_nearest(yc, yfish[-1])
        
        # Position constraint is more stable than area constraint
        if idx_px == idx_py:
            idx_p = idx_px
            xtmp, ytmp = xc[idx_p], yc[idx_p]
        else:
            print('WARNING: the control is idx_px!=idx_py')
            # Fallback to area-based selection
            xtmp, ytmp = xc[idx_a], yc[idx_a]
        
        xfish.append(xtmp)
        yfish.append(ytmp)
    
    return xfish[-1], yfish[-1], fish_lost


class structtype():
    """Simple structure class for storing ellipse properties."""
    pass


def get_ellipse(Img: np.ndarray) -> structtype:
    """
    Calculate fish position and orientation using image moments.
    
    Computes the equivalent ellipse of a binary object using image moments.
    The ellipse provides position (centroid), orientation (theta), and shape
    parameters (major/minor axes). The orientation is corrected to point from
    tail to head using skew analysis.
    
    Parameters
    ----------
    Img : np.ndarray
        Binary image (2D array, dtype=float32) with object as white pixels
        on black background.
    
    Returns
    -------
    E : structtype
        Structure containing ellipse properties:
        - x, y : centroid position in pixels
        - theta : orientation angle in radians [-pi, pi]
        - w, l : minor and major axis lengths
        - x1, y1, x2, y2 : focal point coordinates (x1,y1 is head)
        Returns np.nan for all fields if object is lost.
    
    References
    ----------
    Inspired by: http://www.raphael.candelier.fr/
    
    See Also
    --------
    img_draw_ellipse : Visualize the computed ellipse.
    """
    E = structtype()
    
    # Object's pixels coordinates
    i, j = np.nonzero(Img)
    
    # Get the image moments
    M = cv2.moments(Img)
    E.m00 = M["m00"]
    E.m10 = M["m10"]
    E.m01 = M["m01"]
    E.m11 = M["m11"]
    E.m02 = M["m02"]
    E.m20 = M["m20"]
    
    # Calculate ellipse properties
    if E.m00 != 0:
        # Barycenter (centroid)
        E.x = E.m10 / E.m00
        E.y = E.m01 / E.m00
        
        # Central moments (intermediary step)
        a = E.m20 / E.m00 - E.x**2
        b = 2 * (E.m11 / E.m00 - E.x * E.y)
        c = E.m02 / E.m00 - E.y**2
        
        # Orientation (radians [-pi, pi])
        E.theta = (1/2 * math.atan(b/(a-c)) + (a < c) * np.pi/2)
        
        # Minor and major axis
        E.w = np.sqrt(8 * (a + c - np.sqrt(b**2 + (a-c)**2))) / 2
        E.l = np.sqrt(8 * (a + c + np.sqrt(b**2 + (a-c)**2))) / 2
        
        # Ellipse focal points
        d = np.sqrt(E.l**2 - E.w**2)
        E.x1 = E.x + d * math.cos(E.theta)
        E.y1 = E.y + d * math.sin(E.theta)
        E.x2 = E.x - d * math.cos(E.theta)
        E.y2 = E.y - d * math.sin(E.theta)
        
        # Correct orientation to point from tail to head using skew
        theta = E.theta + np.pi/2
        pos_trasl = np.array([i - np.mean(i), j - np.mean(j)])
        rot = np.array([[np.cos(theta), -np.sin(theta)], 
                       [np.sin(theta), np.cos(theta)]])
        tmp = np.dot(rot, pos_trasl)
        
        if skew(tmp[0]) > 0:
            # Fix direction: flip 180 degrees
            E.theta = np.mod(E.theta + np.pi, 2 * np.pi)
            tmp = [E.x1, E.y1]
            # Swap F1 and F2 (head and tail)
            E.x1 = E.x2  # head
            E.y1 = E.y2
            E.x2 = tmp[0]
            E.y2 = tmp[1]
    else:
        # Object lost - return NaN values
        E.x = np.nan
        E.y = np.nan
        E.x2 = np.nan
        E.y2 = np.nan
        E.theta = np.nan
        return E
    
    return E


def img_draw_ellipse(img: np.ndarray, E: structtype, u: float, v: float, 
                     w: float, l: float, t_rot: float) -> None:
    """
    Display the ellipse associated with the image.
    
    Visualizes the computed ellipse overlaid on the image, showing the
    centroid, ellipse outline, and head position.
    
    Parameters
    ----------
    img : np.ndarray
        Binary image (2D array, dtype=float32) with object as white pixels
        on black background.
    E : structtype
        Ellipse structure from `get_ellipse()`.
    u : float
        X-position of the center.
    v : float
        Y-position of the center.
    w : float
        Minor axis radius.
    l : float
        Major axis radius.
    t_rot : float
        Rotation angle in radians.
    
    See Also
    --------
    get_ellipse : Calculate ellipse properties.
    """
    b = w
    a = l
    if w > l:
        # If w is the ellipse major axis, it stays along x-axis
        a = w
        b = l
    
    # Generate ellipse points
    t = np.linspace(0, 2*np.pi, 100)
    Ell = np.array([a * np.cos(t), b * np.sin(t)])
    
    # Rotation matrix
    R_rot = np.array([[np.cos(t_rot), -np.sin(t_rot)],
                     [np.sin(t_rot), np.cos(t_rot)]])
    
    # Apply rotation
    Ell_rot = np.zeros((2, Ell.shape[1]))
    for i in range(Ell.shape[1]):
        Ell_rot[:, i] = np.dot(R_rot, Ell[:, i])
    
    # Plot visualization
    plt.plot(u, v, 'ro')  # centroid
    plt.plot(u + Ell_rot[0, :], v + Ell_rot[1, :], 'darkorange')  # rotated ellipse
    plt.plot(E.x1, E.y1, 'bo')  # head
    plt.imshow(img, origin='lower')
    plt.show()


def tracking_tail(x: float, y: float, orientation: float, img_bw: np.ndarray,
                  n_segments: int, window_radius: float, tail_length: float,
                  segment_length: float) -> np.ndarray:
    """
    Track fish tail by segmenting it into discrete points.
    
    Starting from the fish position and orientation, segments the tail into
    n_segments points by iteratively finding the next segment along the tail.
    
    Parameters
    ----------
    x : float
        Fish x-position in pixels.
    y : float
        Fish y-position in pixels.
    orientation : float
        Fish orientation angle in radians.
    img_bw : np.ndarray
        Binary image with fish as white pixels.
    n_segments : int
        Number of segments in the tail curve.
    window_radius : float
        Radius of ROI for finding next segment point.
    tail_length : float
        Total tail length (evaluated per frame).
    segment_length : float
        Length of each segment (tail_length / n_segments).
    
    Returns
    -------
    tail_points : np.ndarray
        Array of shape (n_segments+1, 2) containing [x, y] coordinates of
        tail points. Contains np.nan if tail tracking fails.
    
    See Also
    --------
    fish_position : Get fish position.
    get_ellipse : Get fish orientation.
    """
    tail_points = np.full((n_segments + 1, 2), np.nan)
    
    # Tail tracking only if object is detected
    if not np.isnan(x) and not np.isnan(orientation):
        initial_x, initial_y = x, y
        theta_tail = orientation
        
        # Initial segment direction
        initial_dx = segment_length * np.cos(theta_tail)
        initial_dy = segment_length * np.sin(theta_tail)
        
        tail_points[0] = [initial_x, initial_y]
        
        # Segment the tail iteratively
        for s in range(1, n_segments + 1):
            # Note: next_segment function would need to be implemented
            # For now, this is a placeholder structure
            # initial_x, initial_y, initial_dx, initial_dy = next_segment(
            #     img_bw, initial_x, initial_y, initial_dx, initial_dy,
            #     window_radius, segment_length)
            
            # Placeholder: would need actual next_segment implementation
            break
        
    return tail_points


def track_fish(img_bw: np.ndarray, xc: List[float], yc: List[float], idx: int,
               n_segments: int, window_radius: float, tail_length: float,
               segment_length: float) -> Tuple[float, float, float, np.ndarray]:
    """
    Complete fish tracking: position, orientation, and tail.
    
    Combines position tracking, orientation estimation, and tail segmentation
    when the fish object index is known.
    
    Parameters
    ----------
    img_bw : np.ndarray
        Final preprocessed binary image (multiplied by mask).
    xc : List[float]
        X-coordinates of all object centroids.
    yc : List[float]
        Y-coordinates of all object centroids.
    idx : int
        Index of the object that is the fish.
    n_segments : int
        Number of tail segments.
    window_radius : float
        Radius for tail segmentation.
    tail_length : float
        Total tail length.
    segment_length : float
        Length of each tail segment.
    
    Returns
    -------
    xfish_t : float
        Fish x-position in pixels.
    yfish_t : float
        Fish y-position in pixels.
    theta : float
        Fish orientation angle in radians.
    tail_points : np.ndarray
        Tail point coordinates.
    
    See Also
    --------
    tracking_tail : Segment the tail.
    """
    # Get fish position
    xfish_t, yfish_t = xc[idx], yc[idx]
    
    # Get fish orientation
    E = get_ellipse(img_bw)
    theta = E.theta
    
    # Track tail
    tail_points = tracking_tail(xfish_t, yfish_t, theta, img_bw, n_segments,
                                window_radius, tail_length, segment_length)
    
    return xfish_t, yfish_t, theta, tail_points
