"""
Collection of functions for saving and loading data.

This module provides utilities for saving experimental data, including numpy
arrays, images, and HDF5 files. Functions handle file I/O operations for
persistent storage of tracking data and video frames.

Author: Monica Coraggioso
"""

import numpy as np
import os
from typing import List, Any, Union


def saveList(myList: List[Any], filename: str) -> None:
    """
    Save a Python list to a NumPy .npy file.
    
    Parameters
    ----------
    myList : List[Any]
        Python list to save.
    filename : str
        Output filename. Should include the '.npy' extension.
    
    Examples
    --------
    >>> saveList([1, 2, 3], 'data.npy')
    Saved successfully!
    """
    np.save(filename, myList)
    print("Saved successfully!")


def loadList(filename: str) -> List[Any]:
    """
    Load a Python list from a NumPy .npy file.
    
    Parameters
    ----------
    filename : str
        Input filename. Should include the '.npy' extension.
    
    Returns
    -------
    List[Any]
        Loaded Python list.
    
    Examples
    --------
    >>> data = loadList('data.npy')
    """
    tempNumpyArray = np.load(filename)
    return tempNumpyArray.tolist()


def cropImage(image: np.ndarray, center_x: float, center_y: float,
              crop_size: int) -> np.ndarray:
    """
    Crop an image around a center point.
    
    Extracts a square region of specified size centered on the given coordinates.
    Useful for saving video frames focused on the fish while maintaining
    coordinate tracking.
    
    Parameters
    ----------
    image : np.ndarray
        Input grayscale image (2D array).
    center_x : float
        X-coordinate of the center point (centroid).
    center_y : float
        Y-coordinate of the center point (centroid).
    crop_size : int
        Size of the square crop region. Should be odd for equal pixels in
        each direction from center.
    
    Returns
    -------
    np.ndarray
        Cropped image region.
    
    Notes
    -----
    Important: Keep track of the centroid and crop_size so coordinates can be
    reconstructed later. The crop is performed as:
    [center - crop_size//2 : center + crop_size//2 + 1]
    
    Examples
    --------
    >>> cropped = cropImage(img, 300, 300, 500)
    """
    start_x = int(center_x - crop_size // 2)
    end_x = int(center_x + crop_size // 2 + 1)
    start_y = int(center_y - crop_size // 2)
    end_y = int(center_y + crop_size // 2 + 1)
    
    return image[start_x:end_x, start_y:end_y]
