"""
    Image registration and alignment functions for fish tracking.

    This module provides functions for:
    - Cropping images around fish head region
    - Extracting orientation from spline data
    - Preparing images for registration (rotation, cropping)
    - Defining reference images for registration alignment

    Author: Monica Coraggioso
"""
import numpy as np
import matplotlib.pyplot as plt
import math
import cv2
from numba import njit
from scipy.stats import skew
from scipy.signal import find_peaks
from scipy.ndimage import rotate
from time import sleep
from tqdm.notebook import tqdm

from .exceptions import RegistrationError
from .logging_config import get_logger

logger = get_logger(__name__)

def cropping_img(c, dcrop, image):
    if int(c[1])-dcrop<0:
        mm=max(0,int(c[1])-dcrop)
        MM=mm+2*dcrop
    if int(c[1])+dcrop>image.shape[0]:
        MM=min(image.shape[0],int(c[1])+dcrop)
        mm=MM-2*dcrop
    else:
        mm=int(c[1])-dcrop
        MM=int(c[1])+dcrop

    if int(c[0])-dcrop<0:
        nn=max(0,int(c[0])-dcrop)
        NN=nn+2*dcrop
    if int(c[0])+dcrop>image.shape[0]:
        NN=min(image.shape[0],int(c[0])+dcrop)
        nn=NN-2*dcrop
    else:
        nn=int(c[0])-dcrop
        NN=int(c[0])+dcrop
    start_crop=([mm,nn])
    end_crop=([MM,NN])
    return image[mm:MM,nn:NN], start_crop, end_crop

def calculate_angle(spline_avg_frame):
    return np.arctan2(spline_avg_frame[0][1] - spline_avg_frame[6][1],
                      spline_avg_frame[0][0] - spline_avg_frame[6][0])
    
def extract_orientation(spline_points):
    orient_sleap = []
    for frame in range(len(spline_points)):
        if len(spline_points[frame]) > 0:
            orient_sleap.append(calculate_angle(spline_points[frame]))
        else:
            orient_sleap.append(np.nan)
    return orient_sleap

def crop_head_regions(len_rec, myVideo, t1_loc, head_loc, discarded_frame_set, dcrop=15):
    """Crop head regions from video frames.

    Extracts square regions around the midpoint between t1 and head positions
    for each frame.

    Parameters
    ----------
    len_rec : int
        Number of frames to process
    myVideo : np.ndarray
        Video array (height, width, frames)
    t1_loc : list
        T1 keypoint locations for each frame
    head_loc : list
        Head keypoint locations for each frame
    discarded_frame_set : set
        Frame indices to skip
    dcrop : int, default=15
        Crop radius in pixels

    Returns
    -------
    cropped_images : list of np.ndarray
        Cropped head regions (or zeros for discarded/invalid frames)
    start_crop : list of np.ndarray
        Top-left coordinates of crop regions
    end_crop : list of np.ndarray
        Bottom-right coordinates of crop regions
    """
    cropped_images = []
    start_crop = []
    end_crop = []

    progress_bar = tqdm(total=len_rec, desc='Cropping head regions', unit='frame')

    for frame in range(len_rec):
        if frame not in discarded_frame_set:
            image = myVideo[:, :, frame]
            # Midpoint between t1 and head keypoints
            center = (t1_loc[frame] + head_loc[frame]) / 2

            img_crop, start, end = cropping_img(center, dcrop, image)

            # Check if crop is valid (square)
            if img_crop.shape[0] == img_crop.shape[1]:
                cropped_images.append(img_crop)
            else:
                # Invalid crop dimensions
                cropped_images.append(np.zeros((dcrop * 2, dcrop * 2)))

            start_crop.append(start)
            end_crop.append(end)
        else:
            # Discarded frame - use empty image
            cropped_images.append(np.zeros((dcrop * 2, dcrop * 2)))
            start_crop.append(np.array([np.nan, np.nan]))
            end_crop.append(np.array([np.nan, np.nan]))

        progress_bar.update(1)

    progress_bar.close()
    return cropped_images, start_crop, end_crop


def rotate_head_crops(cropped_images, spline_avg, discarded_frame_set):
    """Rotate cropped head images by their orientation.

    Calculates orientation from spline data and rotates each cropped
    head region to align with a standard orientation.

    Parameters
    ----------
    cropped_images : list of np.ndarray
        Cropped head regions from crop_head_regions()
    spline_avg : list of np.ndarray
        Spline data for each frame
    discarded_frame_set : set
        Frame indices to skip

    Returns
    -------
    rotated_images : list of np.ndarray
        Rotated head regions
    orientations : list of float
        Calculated orientations in radians (or np.nan for discarded frames)
    """
    rotated_images = []
    orientations = []

    progress_bar = tqdm(total=len(cropped_images), desc='Rotating head crops', unit='frame')

    for frame, img_crop in enumerate(cropped_images):
        if frame not in discarded_frame_set and not np.all(img_crop == 0):
            # Calculate orientation from spline
            orientation = calculate_angle(spline_avg[frame])
            orientations.append(orientation)

            # Rotate by orientation (convert to degrees)
            rotated = rotate(img_crop, np.rad2deg(orientation), reshape=False)
            rotated_images.append(rotated)
        else:
            # Discarded or invalid frame
            rotated_images.append(img_crop)  # Keep as-is (zeros)
            orientations.append(np.nan)

        progress_bar.update(1)

    progress_bar.close()
    return rotated_images, orientations


def preparation4registration(len_rec, spline_avg, discarded_frame_set, myVideo, t1_loc, head_loc, dcrop=15):
    """Prepare head regions for image registration.

    This function orchestrates the preparation pipeline:
    1. Crop head regions around midpoint between t1 and head keypoints
    2. Calculate orientations from spline data
    3. Rotate cropped regions to align orientation

    Parameters
    ----------
    len_rec : int
        Number of frames to process
    spline_avg : list of np.ndarray
        Spline data for each frame (for orientation calculation)
    discarded_frame_set : set
        Frame indices to skip (invalid detections)
    myVideo : np.ndarray
        Video array with shape (height, width, frames)
    t1_loc : list of np.ndarray
        T1 keypoint locations [(x, y), ...] for each frame
    head_loc : list of np.ndarray
        Head keypoint locations [(x, y), ...] for each frame
    dcrop : int, default=15
        Crop radius in pixels (final image will be 2*dcrop square)

    Returns
    -------
    head_reg : list of np.ndarray
        Rotated and cropped head regions ready for registration
    start_crop : list of np.ndarray
        Top-left coordinates of crop regions in original images
    end_crop : list of np.ndarray
        Bottom-right coordinates of crop regions in original images
    orient_sleap : list of float
        Calculated orientations in radians for each frame
    """
    # Step 1: Crop head regions
    cropped_images, start_crop, end_crop = crop_head_regions(
        len_rec, myVideo, t1_loc, head_loc, discarded_frame_set, dcrop
    )

    # Step 2: Rotate crops by orientation
    head_reg, orient_sleap = rotate_head_crops(
        cropped_images, spline_avg, discarded_frame_set
    )

    return head_reg, start_crop, end_crop, orient_sleap

def def_reg_image(len_rec, head_reg, last_x = None, max_y = None):
    img_aux = []
    tmp = np.random.randint(0, len_rec, 100)
    for i in tmp:
        img_aux.append(head_reg[i])
    ref_img = np.array(img_aux).mean(axis=0)
    
    if last_x is None :
        # --- Sum along axes ---
        sum_y = np.sum(ref_img, axis=1)  # sum over x-axis
        sum_x = np.sum(ref_img, axis=0)  # sum over y-axis

        # --- Max Y ---
        max_y = np.argmax(sum_y)

        # --- Last X where sum > 200 ---
        above_thresh_x = np.where(sum_x > 200)[0]
        if len(above_thresh_x) > 0:
            last_x = above_thresh_x[-1]
        else:
            last_x = None

    # --- Plot result ---
    plt.figure(figsize=(6, 6))
    plt.imshow(ref_img, cmap='gray', origin='lower')

    plt.axhline(max_y, color='red', linestyle='--', label=f'Fixed Y = {max_y}')
    if last_x is not None:
        plt.axvline(last_x, color='blue', linestyle='--', label=f'Fixed  = {last_x}')

    plt.title('ref_img with max Y and thresholded X')
    plt.legend()
    plt.tight_layout()
    plt.show()

    return ref_img, last_x, max_y

def registration_process(head_reg, ref_img, discarded_frame_set):
    len_rec = len(head_reg)  # Assuming head_reg contains the registration data

    progress_bar = tqdm(total=len_rec, desc='registration', unit='frame')

    reg_transl = []
    video_reg = []
    reg_angle = []

    for frame in range(len_rec): 
        if frame not in discarded_frame_set:
            im = head_reg[frame].copy()
            try:
                [shift,err,zero] = phase_cross_correlation(ref_img, im, upsample_factor=100, normalization=None)
                #print('reg:',frame, err)
                if err > 0.2:#0.2:         
                    angle = reg_angle_single_frame(frame, head_reg, reg_angle)
                    im = rotate(im, -angle, reshape=False)
                    [shift,err0,zero] = phase_cross_correlation(ref_img, im, upsample_factor=50, normalization=None)
                else:
                    angle = 0
                trasl = np.float32([[1, 0, shift[1]], [0, 1, shift[0]]])
                img_transf = cv2.warpAffine(im, trasl, (im.shape[1], im.shape[0])) 

                video_reg.append(img_transf)
                reg_transl.append(shift)
                reg_angle.append(angle) 

            except (ValueError, RuntimeError) as e:
                logger.warning(f"Registration failed at frame {frame}: {str(e)}")
                video_reg.append(np.zeros((30,30)))
                reg_transl.append(np.array([0,0]))
                reg_angle.append(0)
        else:
            video_reg.append(np.zeros((30,30)))
            reg_transl.append(np.array([0,0])) 
            reg_angle.append(0)

        progress_bar.update(1)
    progress_bar.close()

    if len_rec == len(reg_transl) == len(video_reg) == len(reg_angle):
        print('ok')
    else:
        print(f'ERROR:{len(reg_transl)}, {len(video_reg)}, {len(reg_angle)}')
    
    return video_reg, reg_transl, reg_angle


from scipy.optimize import curve_fit

def quadratic_function(x, a, b, c):
    return a * x**2 + b * x + c

def parabola(x, a, b, c):
    return a * x**2 + b * x + c

def linear(x, m, b):
    return m * x + b

def sigmoid(x, a, b, c, d):
    return a / (1 + np.exp(-c * (x - b))) + d

import numpy as np
from tqdm import tqdm
from scipy.ndimage import rotate
from skimage.registration import phase_cross_correlation
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import math

def quadratic_function(x, a, b, c):
    """Quadratic function for curve fitting: y = ax² + bx + c."""
    return a * x**2 + b * x + c


def _coarse_angle_search(reference, im, angle_range=(-95, 95), n_angles=10):
    """Perform coarse angular search over a wide range.

    Phase 1 of angle detection: test a small number of angles across a wide range
    to find the approximate rotation angle.

    Parameters
    ----------
    reference : np.ndarray
        Reference image (typically mirrored version of im)
    im : np.ndarray
        Image to register
    angle_range : tuple of (float, float), default=(-95, 95)
        Range of angles to search (min_angle, max_angle) in degrees
    n_angles : int, default=10
        Number of angles to test in the range

    Returns
    -------
    angles : np.ndarray
        Array of tested angles
    errors : list of float
        Registration errors for each angle
    """
    angles = np.linspace(angle_range[0], angle_range[1], n_angles)
    errors = []

    for angle in angles:
        rotated_ref = rotate(reference, angle, reshape=False)
        shift, err, zero = phase_cross_correlation(
            rotated_ref, im, upsample_factor=1, normalization=None
        )
        errors.append(err)

    return angles, errors


def _refine_angle_with_interpolation(angles, errors):
    """Refine angle estimate using gradient-based interpolation.

    Uses the gradient of errors to find where it crosses zero (local minimum).

    Parameters
    ----------
    angles : np.ndarray
        Array of angles tested
    errors : list of float
        Registration errors for each angle

    Returns
    -------
    refined_angle : float
        Refined angle estimate, or None if minimum is at boundary
    id_min : int
        Index of minimum error in original arrays

    Notes
    -----
    Returns None for refined_angle if the minimum is at the boundary,
    indicating the search range may need adjustment.
    """
    id_min = np.argmin(errors)

    # Check if minimum is at boundary
    if id_min == 0 or id_min == len(angles) - 1:
        return None, id_min

    # Interpolate gradient to find zero crossing (minimum)
    f = interp1d(angles, np.gradient(errors), kind='linear')
    x_interp = np.linspace(angles[id_min - 1], angles[id_min + 1], 100)
    y_interp = f(x_interp)

    # Find angle where gradient is closest to zero
    refined_angle = x_interp[np.argmin(np.abs(y_interp))]

    return round(refined_angle, 2), id_min


def _fine_angle_search(reference, im, center_angle, n_angles=8):
    """Perform fine angular search around a center angle.

    Phase 2 of angle detection: test more angles in a narrow range around
    the coarse estimate for precise angle determination.

    Parameters
    ----------
    reference : np.ndarray
        Reference image (typically mirrored version of im)
    im : np.ndarray
        Image to register
    center_angle : float
        Center angle for fine search (degrees)
    n_angles : int, default=8
        Number of angles to test (range will be ±n_angles/2 around center)

    Returns
    -------
    angles : list of float
        Array of tested angles
    errors : list of float
        Registration errors for each angle (with higher upsampling)
    """
    angles = []
    errors = []

    for i in range(n_angles):
        angle = center_angle + (i - n_angles / 2)
        rotated_ref = rotate(reference, angle, reshape=False)
        shift, err, zero = phase_cross_correlation(
            rotated_ref, im, upsample_factor=50, normalization=None
        )
        angles.append(angle)
        errors.append(err)

    return angles, errors


def _fit_quadratic_minimum(angles, errors):
    """Fit quadratic to errors and find precise minimum angle.

    Parameters
    ----------
    angles : list of float
        Angles tested in fine search
    errors : list of float
        Registration errors for each angle

    Returns
    -------
    min_angle : float
        Angle at the quadratic minimum

    Raises
    ------
    RuntimeError
        If curve fitting fails
    """
    # Fit quadratic to the errors
    params, covariance = curve_fit(quadratic_function, angles, errors)
    a, b, c = params

    # Evaluate fitted quadratic on fine grid
    x_fit = np.linspace(angles[0], angles[-1], 50)
    y_fit = quadratic_function(x_fit, a, b, c)

    # Find minimum
    id_min = np.argmin(y_fit)
    min_angle = x_fit[id_min]

    return round(min_angle, 2)


def reg_angle_single_frame(frame, head_reg, corr_angle):
    """Find registration angle for a single frame using two-phase search.

    This function determines the rotation angle needed to align a head image
    with its mirror image (for symmetry-based registration). It uses a coarse-to-fine
    search strategy:
    1. Coarse search over ±95° to find approximate angle
    2. Fine search around coarse estimate for precise angle

    Parameters
    ----------
    frame : int
        Frame number to process
    head_reg : list of np.ndarray
        List of head region images
    corr_angle : list of float
        List of angles from previous frames (used as fallback)

    Returns
    -------
    angle : float
        Registration angle in degrees, divided by 4 (due to mirror symmetry).
        Returns np.nan if registration fails.

    Notes
    -----
    The angle is divided by 4 at the end because:
    1. Mirror reference creates 2x angle sensitivity (divide by 2)
    2. Additional factor of 2 for final correction (divide by 2 again)
    """
    try:
        im = head_reg[frame].copy()
        reference = im[::-1, :]  # Mirror image as reference

        # Phase 1: Coarse angular search over wide range
        coarse_angles, coarse_errors = _coarse_angle_search(reference, im)

        # Refine coarse estimate using interpolation
        refined_angle, id_min = _refine_angle_with_interpolation(
            coarse_angles, coarse_errors
        )

        # Determine fine search parameters based on coarse results
        if refined_angle is None:
            # Minimum at boundary - use previous angle as fallback
            center_angle = corr_angle[frame - 1] if frame > 0 else 0
            n_fine_angles = 80  # Wider search needed
        elif frame > 0 and math.isclose(
            90, np.abs(refined_angle - corr_angle[frame - 1]), abs_tol=10
        ):
            # Large change from previous frame - may be error
            center_angle = refined_angle
            n_fine_angles = 80  # Wider search for verification
        else:
            # Normal case - refined angle looks good
            center_angle = refined_angle
            n_fine_angles = 8  # Narrow fine search

        # Phase 2: Fine angular search around coarse estimate
        fine_angles, fine_errors = _fine_angle_search(
            reference, im, center_angle, n_angles=n_fine_angles
        )

        # Fit quadratic and find precise minimum
        angle_final = _fit_quadratic_minimum(fine_angles, fine_errors)

        # Divide by 4 due to mirror symmetry properties
        return angle_final / 4

    except (ValueError, RuntimeError, IndexError) as e:
        # Return nan if any step fails (curve fitting, interpolation, etc.)
        logger.warning(f"Angle detection failed at frame {frame}: {str(e)}")
        return np.nan