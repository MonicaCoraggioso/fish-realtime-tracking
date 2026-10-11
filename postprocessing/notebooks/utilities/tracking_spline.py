"""
    Online tracking: position of the fish + tail points

    Author: Monica Coraggioso
"""
import numpy as np
import matplotlib.pyplot as plt
import math
import cv2
from numba import njit
from scipy.stats import skew
from scipy.signal import find_peaks
from time import sleep
from tqdm.notebook import tqdm

from .preprocessing import area_contours
from .core_tracking import (
    next_segment,
    next_segment_head,
    tracking_tail,
    tracking_head_extrapolate,
)


def spline_tracking_new(myVideo, points_sy0, mean_fish_L, n_segments, window_radius, frame_fish_L=None):
    spline_points = []
    len_rec = myVideo.shape[2]
    progress_bar = tqdm(total=len_rec, desc='tracking spline', unit='frame')

    for frame in range(len_rec):
        # Use head and next point to determine direction
        head = points_sy0[frame][0]
        next_point = points_sy0[frame][1]
        
        if np.any(np.isnan(head)) or np.any(np.isnan(next_point)):
            spline_points.append(np.full((n_segments+1, 2), np.nan))
            progress_bar.update(1)
            continue

        # Compute fish length and direction
        if frame_fish_L is not None and np.isnan(frame_fish_L[frame]):
            fish_length = frame_fish_L[frame]
        else:
            fish_length = mean_fish_L
        segment_length = fish_length / n_segments
        direction = np.arctan2(next_point[1] - head[1], next_point[0] - head[0])

        # Head coordinates
        START_X, START_Y = head[0], head[1]

        # Track points backward along the body (tailward)
        points = tracking_tail(
            START_X, START_Y, direction,
            myVideo[:, :, frame], n_segments,
            window_radius, fish_length, segment_length
        )

        spline_points.append(points)
        progress_bar.update(1)

    progress_bar.close()
    return spline_points




def spline_tracking(myVideo, profile_L, head_L, mean_fish_L, fish_length, n_segments, window_radius, tail_direction, t4_loc , head_loc):
    spline_points_rev=[]
    spline_points=[] 
    spline_avg=[]
    len_rec=myVideo[:,:,:].shape[2]
    progress_bar = tqdm(total=len_rec, desc='tracking spline', unit='frame')
    for frame in range(myVideo[:,:,:].shape[2]):
        if ~np.isnan(profile_L[frame][0]) and ~np.isnan( head_L[frame][0]):
            fish_length=profile_L[frame][0] #pix
        else:
            fish_length=mean_fish_L #pix
        segment_length=fish_length/n_segments       
    
        direction=tail_direction[frame][0] #head_direction[frame][0]+np.pi
    
        #START_X, START_Y = t4_loc[frame][0][0],t4_loc[frame][1][0]
        #spline_points_rev.append(tracking_head_extrapolate(START_X, START_Y, direction, myVideo[:,:,frame], n_segments, window_radius, fish_length, segment_length))
    
        START_X, START_Y = head_loc[frame][0][0],head_loc[frame][1][0]
        spline_points.append(tracking_head_extrapolate(START_X, START_Y, direction+np.pi, myVideo[:,:,frame], n_segments, window_radius, fish_length, segment_length))
    
        #a=spline_points_rev[frame]
        #a = a[::-1]
        #weights = np.concatenate([
        #    a[:11],
        #    spline_points[frame][-10:],])
        #spline_avg.append(weights)
        progress_bar.update(1)
    progress_bar.close()  
    return spline_points#spline_avg



############other functions for discarding frames based on scores and alignment checks
import numpy as np
from tqdm import tqdm

def is_between(a, b, c, tol=10):
    a, b, c = np.array(a), np.array(b), np.array(c)
    return np.abs(np.linalg.norm(a - b) + np.linalg.norm(b - c) - np.linalg.norm(a - c)) < tol


def discared_frames(len_rec,instance_scores, th_score,spline_avg):
    discarded_frame=[]
    ok_frame=[]
    for frame in range(len_rec):
        if instance_scores[0][frame]>th_score: 
            if np.isnan(spline_avg[frame]).any():  
                discarded_frame.append(frame) #frame discarded because nans in pose tracking
            else:
                ok_frame.append(frame)
        else:
            discarded_frame.append(frame) #frame discarded with Sleap instance_score metric
    if len(discarded_frame)+len(ok_frame)==len_rec:
        print(f'Discarded {len(discarded_frame)} frames over {len_rec} = {round((len(discarded_frame)/len_rec)*100,0)}%')
    else:
        print('ERROR: something is wrong')
    discarded_frame_set = set(discarded_frame)
    return ok_frame, discarded_frame, discarded_frame_set


def discared_frames_based_on_scores(len_rec,instance_scores, point_scores, head_loc, t1_loc, t2_loc, t3_loc, t4_loc,th_score = 2, th_score_points = 0.5):
    discarded_frame=[]
    ok_frame=[]
    for frame in range(len_rec):
        
        head = np.array(head_loc[frame], dtype=float).flatten()
        t1 = np.array(t1_loc[frame], dtype=float).flatten()
        t2 = np.array(t2_loc[frame], dtype=float).flatten()
        t3 = np.array(t3_loc[frame], dtype=float).flatten()
        t4 = np.array(t4_loc[frame], dtype=float).flatten()
            

        # Alignment checks
        check1 = is_between(head, t1, t2)
        check2 = is_between(t1, t2, t3)
        check3 = is_between(t2, t3, t4)
        is_aligned = all([check1, check2, check3])

        if instance_scores[0][frame]>th_score and (point_scores[0][:,frame] > th_score_points).all() and is_aligned:  
            ok_frame.append(frame)
        else:
            discarded_frame.append(frame) #frame discarded with Sleap instance_score metric
    if len(discarded_frame)+len(ok_frame)==len_rec:
        print(f'Discarded {len(discarded_frame)} frames over {len_rec} = {round((len(discarded_frame)/len_rec)*100,0)}%')
    else:
        print('ERROR: something is wrong')
    discarded_frame_set = set(discarded_frame)
    return ok_frame, discarded_frame, discarded_frame_set