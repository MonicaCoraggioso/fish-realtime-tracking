"""
    Analysis functions for fish body spline data.
    
    This module provides functions for:
    - Spline transformation and alignment (rotation, translation)
    - Principal Component Analysis (PCA) of spline angles
    - Spline length calculations
    - Moving RMS calculations for signal processing
    
    Author: Monica Coraggioso
"""
import numpy as np
import matplotlib.pyplot as plt
import math
import cv2

def moving_rms(data, window_size):
    squared_data = data**2  # Square the data
    squared_sum = np.cumsum(squared_data)  # Cumulative sum of squared values
    squared_sum[window_size:] = squared_sum[window_size:] - squared_sum[:-window_size]
    rms = np.sqrt(squared_sum / window_size)
    return rms

def spline_transformed(len_run, spline_mm, orient):
    spline_mm_zero=[] #aligned in zero
    y_L=[]
    for frame in range(len_run): 
        spline_traslated=spline_mm[frame]-spline_mm[frame][0]
        angle_deg=-(orient[frame]) % 360
        cos_angle, sin_angle = np.cos(np.deg2rad(angle_deg)), np.sin(np.deg2rad(angle_deg))
        rotation_matrix = np.array([[cos_angle, -sin_angle],[sin_angle,  cos_angle]])
        spline_rotated = np.dot(spline_traslated, rotation_matrix.T)
        
        spline_mm_zero.append(spline_rotated) #mm
        
        if not np.isnan(spline_mm[frame]).any():
            y_L.append(spline_mm_zero[frame][-1, 1]) 
        else:
            y_L.append(np.nan)
    y_L=np.array(y_L)
    return spline_mm_zero, y_L

def reconstruct_segments(angles, segments):
    x = np.cumsum(segments * np.cos(np.radians(angles)))
    y = np.cumsum(segments * np.sin(np.radians(angles)))
    return x, y

def extract_and_reconstruct(coordinates,segment_length):
    # Extract Relative Angles
    vectors = coordinates[1:] - coordinates[:-1]
    angles_rad = np.arctan2(vectors[:, 1], vectors[:, 0])  # Compute angles in radians
    angles_deg = np.degrees(angles_rad)  # Convert angles to degrees
    # Reconstruct the Original Set of Points
    reconstructed_coordinates = [coordinates[0]]
    for i in range(len(angles_deg)):
        normalized_segment=segment_length/tail_length
        delta_x = normalized_segment * np.cos(np.radians(angles_deg[i]))
        delta_y = normalized_segment * np.sin(np.radians(angles_deg[i]))
        print('--',reconstructed_coordinates[-1])
        new_point = reconstructed_coordinates[-1] + np.array([delta_x, delta_y])
        print(i,new_point )
        reconstructed_coordinates.append(new_point)
    print(reconstructed_coordinates)
    reconstructed_coordinates = np.array(reconstructed_coordinates)
    return angles_deg, reconstructed_coordinates

def calculate_spline_length(points):
    # Calculate the differences between consecutive points
    diffs = np.diff(points, axis=0)
    
    # Calculate the Euclidean distance for each segment
    segment_lengths = np.linalg.norm(diffs, axis=1)
    
    # Sum the segment lengths to get the total length
    total_length = np.sum(segment_lengths)
    
    return total_length

def spline_length_mm(len_run, spline_mm_zero):
    spline_length_mm=[]
    for frame in range(len_run):
        spline_length_mm.append(calculate_spline_length(spline_mm_zero[frame]))
    return spline_length_mm

def PCA_spline(len_run, spline_mm_zero):
    len_run = len(spline_mm_zero)
    indices_pca = []
    dangles_spline_xpca = []
    dangles_spline_xpca_nans = []

    for frame in range(len_run):
        spline_length_mm = calculate_spline_length(spline_mm_zero[frame])
        n_coordinates = spline_mm_zero[frame] / spline_length_mm

        vectors = n_coordinates[1:] - n_coordinates[:-1]
        angles_rad = np.arctan2(vectors[:, 1], vectors[:, 0])  # Compute angles in radians

        if not np.isnan(spline_mm_zero[frame]).any():
            indices_pca.append(frame)
            dangles_spline_xpca.append(np.degrees(angles_rad))
            dangles_spline_xpca_nans.append(np.degrees(angles_rad))
        else:
            dangles_spline_xpca_nans.append(np.full_like(np.degrees(angles_rad), np.nan))

    dangles_spline_points = np.vstack(dangles_spline_xpca)  # shape (n_samples, n_features)
    dangles_spline_points_nans = np.vstack(dangles_spline_xpca_nans)
    
    print('shape dangles_spline_points', dangles_spline_points.shape)
    print('shape dangles_spline_points_nans', dangles_spline_points_nans.shape)
    print('shape indices_pca', len(indices_pca))
    
    return dangles_spline_points, dangles_spline_points_nans, indices_pca