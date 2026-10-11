"""
YOLO-based tracking: image cropping utilities for object detection.

This module provides functions for cropping image regions around YOLO-detected
objects or predicted positions. These crops are used for focused tracking
and processing of the fish.

Author: Monica Coraggioso
"""

import numpy as np
from PIL import Image
from typing import Tuple, Union, List


def crop_around_rectangle(img: np.ndarray, rect_coords: List[float],
                          scale_factor: int = 2, crop_size: int = 120) -> np.ndarray:
    """
    Crop image around a YOLO-detected bounding box.
    
    YOLO detects objects as rectangles (bounding boxes). This function crops
    a fixed-size region centered on the detected rectangle, with optional
    scaling to account for image downsampling.
    
    Parameters
    ----------
    img : np.ndarray
        Input image (2D or 3D array).
    rect_coords : List[float]
        Rectangle coordinates [x1, y1, x2, y2] from YOLO detection.
    scale_factor : int, default=2
        Scaling factor to apply to coordinates (accounts for downsampling).
        Default is 2 for 2x downsampled images.
    crop_size : int, default=120
        Size of the square crop region in pixels.
    
    Returns
    -------
    np.ndarray
        Cropped image as numpy array.
    
    Examples
    --------
    >>> rect = [50, 50, 150, 150]  # [x1, y1, x2, y2]
    >>> cropped = crop_around_rectangle(img, rect, scale_factor=2, crop_size=120)
    """
    image = Image.fromarray(img)
    
    # Extract coordinates from YOLO detection with scaling
    x1, y1 = rect_coords[0] * scale_factor, rect_coords[1] * scale_factor
    x2, y2 = rect_coords[2] * scale_factor, rect_coords[3] * scale_factor
    
    # Calculate center of the rectangle
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    
    # Calculate cropping box to create a square area centered on the rectangle
    crop_x1 = max(0, int(center_x - crop_size / 2))
    crop_y1 = max(0, int(center_y - crop_size / 2))
    crop_x2 = crop_x1 + crop_size
    crop_y2 = crop_y1 + crop_size
    
    # Ensure cropping box is within image boundaries
    image_width, image_height = image.size
    if crop_x2 > image_width:
        crop_x1 = image_width - crop_size
        crop_x2 = image_width
    if crop_y2 > image_height:
        crop_y1 = image_height - crop_size
        crop_y2 = image_height
    
    # Crop the image
    cropped_image = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
    return np.array(cropped_image)


def crop_around_point(image: np.ndarray, center_coords: Union[np.ndarray, List[float]],
                      crop_size: int = 100) -> np.ndarray:
    """
    Crop image around a predicted fish position.
    
    Crops a fixed-size region centered on the fish position estimated at time
    t-1 (e.g., using a Kalman filter). This is useful for focused tracking
    when the approximate position is known.
    
    Parameters
    ----------
    image : np.ndarray
        Input image (2D array).
    center_coords : np.ndarray or List[float]
        Center coordinates [x, y] of the crop region.
    crop_size : int, default=100
        Size of the square crop region in pixels.
    
    Returns
    -------
    np.ndarray
        Cropped image as numpy array.
    
    Raises
    ------
    ValueError
        If center_coords does not contain exactly two values [x, y].
    
    Examples
    --------
    >>> center = [300, 300]  # [x, y]
    >>> cropped = crop_around_point(img, center, crop_size=100)
    """
    center_coords = np.asarray(center_coords)
    if center_coords.shape != (2,):
        raise ValueError("center_coords should contain exactly two values: [x, y].")
    
    # Extract center coordinates
    center_x, center_y = center_coords
    
    # Calculate cropping box to create a square area centered on the point
    crop_x1 = max(0, int(center_x - crop_size / 2))
    crop_y1 = max(0, int(center_y - crop_size / 2))
    crop_x2 = crop_x1 + crop_size
    crop_y2 = crop_y1 + crop_size
    
    # Ensure cropping box is within image boundaries
    image_height, image_width = image.shape[0], image.shape[1]
    if crop_x2 > image_width:
        crop_x1 = image_width - crop_size
        crop_x2 = image_width
    if crop_y2 > image_height:
        crop_y1 = image_height - crop_size
        crop_y2 = image_height
    
    # Crop the image using the calculated box
    cropped_image = image[crop_y1:crop_y2, crop_x1:crop_x2]
    return np.array(cropped_image)
