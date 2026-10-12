"""
    Coordinate transformations and orientation calculations for fish body splines.
    
    This module provides functions for:
    - Transforming spline coordinates between different reference frames
    - Calculating center of mass (COM) and orientation from spline data
    - Rotating and translating spline points for registration alignment
    
    Author: Monica Coraggioso
"""
import numpy as np
import matplotlib.pyplot as plt
import math
import cv2
from numba import njit
from scipy.stats import skew
from scipy.signal import find_peaks
from utilities.preprocessing import area_contours

def create_nan_array_like(arr, len_rec):
    """
    Creates a numpy array filled with NaN values with the same shape as the input array.
    """
    if isinstance(arr, list):
        shape = (len(arr),) + np.shape(arr[0])
    else:
        shape = np.shape(arr)
    tmp=np.full(shape, np.nan)
    return tmp[:len_rec]

def rotate_2darray(points, angle, start_crop, end_crop):
    """
    Rotates a set of 2D points around their centroid by a specified angle.

    Parameters:
    points (np.ndarray): The input array of points with shape (N, 2), where N is the number of points.
    angle (float): The angle by which to rotate the points, in degrees.

    Returns:
    np.ndarray: The rotated points.
    """
    points = np.asarray(points)

    # Convert angle from degrees to radians
    angle_rad = np.deg2rad(angle)
    
    # Calculate the centroid of the points
    centroid = ( start_crop[1]+(end_crop[1]-start_crop[1])/2, start_crop[0]+(end_crop[0]-start_crop[0])/2 )
    
    # Translate points to the origin
    translated_points = points - centroid
    
    # Define the rotation matrix
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)
    rotation_matrix = np.array([[cos_angle, -sin_angle],
                                [sin_angle,  cos_angle]])
    
    # Apply the rotation matrix to the points
    rotated_points = np.dot(translated_points, rotation_matrix.T)
    
    # Translate the points back to their original location
    rotated_points += centroid
    
    return rotated_points

def trasform_spline_data(spline_avg, orient_sleap, reg_angle, reg_transl, discarded_frame_set, start_crop, end_crop):
    len_rec = len(spline_avg)
    spline_sy0 = []
    com_sy0 = []
    
    for frame in range(len_rec):
        if frame not in discarded_frame_set:
            # Rotate spline_avg as you did for the image cropped
            rotated_points = rotate_2darray(spline_avg[frame], -np.rad2deg(orient_sleap[frame]), start_crop[frame], end_crop[frame])
            # Apply rotation1: reg_angle
            #rotated_points_reg = rotate_2darray(rotated_points, -reg_angle[frame], start_crop[frame], end_crop[frame])
            rotated_points_reg = rotate_2darray(rotated_points, -reg_angle[frame], start_crop[frame], end_crop[frame])
            # Apply translation1: reg_transl
            T_corr = np.array([-reg_transl[frame][1], -reg_transl[frame][0]])  # Fixed the syntax error here
            translated_points = rotated_points_reg + T_corr
            # Rotate back
            spline_reg = rotate_2darray(translated_points, np.rad2deg(orient_sleap[frame]), start_crop[frame], end_crop[frame])

            spline_sy0.append(spline_reg)
            com_sy0.append(spline_reg[5])  # Assuming 5 is the index for the center of mass, adjust if necessary
        else:
            spline_sy0.append(np.full_like(spline_avg[frame], np.nan))
            com_sy0.append(np.array([np.nan, np.nan]))

    return spline_sy0, com_sy0




def transform_points_after_reg(points, head_reg, reg_angle_list, reg_transl_list, discarded_frame_set):
    """
    Apply the registration transformations (rotation + translation) to the same set of points across all frames.

    Parameters
    ----------
    points : np.ndarray of shape (N, 2)
        Points to transform (e.g., [(20, 15), (10, 15)]).
    head_reg : list of np.ndarray
        Image stack used for registration (same as in registration_process).
    ref_img : np.ndarray
        Reference image used for registration.
    reg_angle_list : list of float
        Rotation angles per frame (in degrees).
    reg_transl_list : list of [y, x]
        Translation vectors per frame.
    discarded_frame_set : set or list
        Set of frame indices to skip.

    Returns
    -------
    transformed_all : list of np.ndarray
        Transformed point array per frame. NaNs if frame was discarded.
    """
    len_rec = len(head_reg)
    transformed_all = []

    for frame in range(len_rec):
        if frame in discarded_frame_set:
            transformed_all.append(np.full_like(points, np.nan))
            continue

        im = head_reg[frame]
        angle = reg_angle_list[frame]
        shift = reg_transl_list[frame]  # [y, x]

        # --- Rotate around center ---
        h, w = im.shape[:2]
        center = np.array([w / 2, h / 2])

        # Move to origin, apply rotation
        points_centered = points - center
        theta = -np.deg2rad(angle)
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ])
        rotated = points_centered @ R.T
        rotated_points = rotated + center

        # --- Translate ---
        translated_points = rotated_points - np.array([shift[1], shift[0]])  # [x, y] order

        transformed_all.append(translated_points)

    return transformed_all


def undo_crop_and_rotation(transformed_points_after_reg, orient_sleap, start_crop, end_crop):
    """
    Transforms points from the cropped & rotated space back to original image coordinates.

    Parameters
    ----------
    transformed_points_after_reg : list of (N, 2) arrays
        Points in the rotated, cropped frame.
    orient_sleap : array of float
        Rotation angles per frame in radians (clockwise).
    start_crop : list of (2,) arrays or tuples
        Starting coordinates [y, x] of the crop per frame.
    end_crop : list of (2,) arrays or tuples
        Ending coordinates [y, x] of the crop per frame.

    Returns
    -------
    points_in_original_frame : list of (N, 2) arrays
        Transformed points in the original image coordinate space.
    """
    points_in_original_frame = []

    for frame, pts in enumerate(transformed_points_after_reg):
        if np.isnan(pts).all():
            points_in_original_frame.append(np.full_like(pts, np.nan))
            continue

        # --- Rotation angle (undo clockwise => rotate counterclockwise) ---
        theta = orient_sleap[frame]  # in radians, counter-clockwise

        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ])

        # --- Rotation center: center of cropped image in (x, y) ---
        crop_size = np.array(end_crop[frame]) - np.array(start_crop[frame])  # [y, x]
        center_cropped = (crop_size / 2)[::-1]  # convert to (x, y)

        # --- Rotate back around crop center ---
        pts_centered = pts - center_cropped
        rotated_back = pts_centered @ R.T + center_cropped

        # --- Shift back to full image coordinates ---
        shift = np.array([start_crop[frame][1], start_crop[frame][0]])  # [x, y]
        original_pts = rotated_back + shift

        points_in_original_frame.append(original_pts)

    return points_in_original_frame


def calculate_orientation(orient_sleap, reg_angle):
    """
    Plots the orientation data.

    Parameters:
    orient_sleap (np.ndarray): The orientation data from sleap.
    reg_angle (np.ndarray): The registered angle data.

    Returns:
    None
    """
    # Angle periodicity
    phi = np.array(orient_sleap)  # rad
    a = np.array(phi)
    a[~np.isnan(a)] = np.unwrap(a[~np.isnan(a)])
    phin = np.rad2deg(a)  # deg

    phi_c = np.deg2rad(reg_angle)  # rad
    a = np.array(phi_c)
    a[~np.isnan(a)] = np.unwrap(a[~np.isnan(a)])
    phin_c = np.rad2deg(a)
    #print('Nan in phin?', str(np.isnan(phin).any()))
    #print('Nan in phin_c?', str(np.isnan(phin_c).any()))

    orient = +phin - phin_c  # deg

    fig, ax = plt.subplots(3, 1, dpi=100, sharex=True)
    ax[0].plot(phin)
    ax[0].set_ylabel('deg')
    ax[0].set_title('orient_sleap: phin')
    ax[1].plot(phin_c)
    ax[1].set_title('correction: phin_c')
    ax[1].set_ylabel('deg')
    ax[2].plot(orient)
    ax[2].set_title('orientation: phin+phin_c')
    ax[2].set_ylabel('deg')
    ax[2].set_xlabel('frame')
    plt.show()

    return orient

def convert_pix2mm(r_array_pix, xROIcenter, yROIcenter, mm_per_pix, tvals):
    # Convert list of lists into a numpy array
    array_pix = np.array([arr for sublist in r_array_pix for arr in sublist])
    print(f'Shape array in pix={array_pix.shape}, shape time={tvals.shape}')

    # Check if the input array has the shape (270191, 21, 2)
    if array_pix.ndim == 3 and array_pix.shape[1] == 21 and array_pix.shape[2] == 2:
        fish_x_off = (array_pix[..., 0] - xROIcenter) * mm_per_pix
        fish_y_off = (array_pix[..., 1] - yROIcenter) * mm_per_pix
        
        # Stack the x and y offsets to form the final position array
        array_mm = np.stack((fish_x_off, fish_y_off), axis=-1)
    else:
        # Handle the case where the input array has the shape (270191, 2)
        fish_x_off = (array_pix[:, 0] - xROIcenter) * mm_per_pix
        fish_y_off = (array_pix[:, 1] - yROIcenter) * mm_per_pix
        
        # Stack the x and y offsets to form the final position array
        array_mm = np.column_stack((fish_x_off, fish_y_off))

    print(f'Shape array in mm={array_mm.shape}')
    return array_pix, array_mm