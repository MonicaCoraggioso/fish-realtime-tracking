"""
Point Grey camera initialization and capture functions.

This module provides functions for initializing, capturing frames from, and
cleaning up Point Grey (FLIR) cameras using the Spinnaker SDK.

Author: Monica Coraggioso
"""

import numpy as np
from typing import Tuple


def _initialize(camera) -> None:
    """
    Initialize the Point Grey camera and begin acquisition.
    
    Parameters
    ----------
    camera : Camera object
        Point Grey camera object from rotpy.camera.
    
    Notes
    -----
    This function initializes the camera and starts image acquisition.
    Must be called before capturing frames.
    """
    camera.init_cam()
    camera.begin_acquisition()


def _cleanup(camera) -> None:
    """
    Clean up camera resources and end acquisition.
    
    Parameters
    ----------
    camera : Camera object
        Point Grey camera object from rotpy.camera.
    
    Notes
    -----
    This function stops acquisition, deinitializes the camera, and releases
    resources. Should be called when done using the camera.
    """
    camera.end_acquisition()
    camera.deinit_cam()
    camera.release()


def capture(camera) -> Tuple[np.ndarray, float]:
    """
    Capture a single frame from the Point Grey camera.
    
    Parameters
    ----------
    camera : Camera object
        Point Grey camera object from rotpy.camera (must be initialized).
    
    Returns
    -------
    img : np.ndarray
        Captured image as 2D numpy array (dtype=uint8). Image is rotated
        90 degrees clockwise to align with Shapeoko axis.
    timestamp : float
        Frame timestamp in seconds (converted from nanoseconds).
    
    Notes
    -----
    The image is rotated 90 degrees clockwise (k=1) to align with the
    motor coordinate system. To rotate 180 degrees, use k=2.
    """
    image = camera.get_next_image()
    
    # Get timestamp in seconds (converted from nanoseconds)
    timestamp = image.get_frame_timestamp() / 1e9
    
    # Extract image data
    img = np.frombuffer(image.get_image_data(), 
                       dtype=image.get_pix_fmt_int_type())
    img = np.reshape(img, (image.get_width(), image.get_height()), order='F')
    
    # Release image buffer
    image.release()
    
    # Rotate 90 degrees clockwise to align with Shapeoko axis
    img = np.rot90(img, k=1)
    
    return img, timestamp
