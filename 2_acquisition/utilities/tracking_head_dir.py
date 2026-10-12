"""
    Online tracking: position of the fish + tail points
    
    Author: Monica Coraggioso
"""
import numpy as np

def angle_between(v1, v2):
    # Calculate the angle in degrees between two normalized vectors
    dot_product = np.clip(np.dot(v1, v2), -1.0, 1.0)  # Clip to avoid numerical issues
    angle_rad = np.arccos(dot_product)  # Arc cosine gives angle in radians
    angle_deg = np.degrees(angle_rad)  # Convert to degrees
    return angle_deg

def check_direction_difference(directions, threshold_degrees=100):
    if len(directions) < 2:
        raise ValueError("Array must contain at least two direction vectors.")
    
    # Check for NaNs in the last two direction vectors
    v1 = directions[-1]
    v2 = directions[-2]
    
    if np.isnan(v1).any() or np.isnan(v2).any():
        return False  # or you could raise an exception

    # Calculate the angle between the two vectors
    angle_diff = angle_between(v1, v2)
    
    # Check if the angle difference is less than the threshold
    is_smaller = angle_diff < threshold_degrees
    
    return is_smaller