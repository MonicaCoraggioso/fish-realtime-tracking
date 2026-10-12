"""
    Collection of function useful for reading data
    
    Author: Monica Coraggioso
"""
import numpy as np
from PIL import Image
import re
import os
import cv2
import h5py

def extract_run_movie(filename):
    """
    Extract run and movie numbers from a filename like 'converted_run2_movie4.mp4'.
    Returns a tuple (run_number, movie_number).
    """
    match = re.search(r"run(\d+)_movie(\d+)", filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return (9999, 9999)  # fallback for unmatched filenames (puts them at the end)


def sorted_alphanumeric(data):
    '''os.listdir() returns a list containing the names of the entries in the directory given by 'data'. The list is in arbitrary order, so you have to write a natural sorting (meaning 1, 2, 10 instead of 1, 10, 2) yourself.
    
    ---You can now use this function to sort a list:
        lst = sorted_alphanumeric(os.listdir(path_movie))
        print(lst)
    '''
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    
    return sorted(data, key=alphanum_key)

def readingMovie(path_movie):
    """read images in 'path_movie'
    """
    img_movie = []
    frame=-1
    lst = sorted_alphanumeric(os.listdir(path_movie))

    for file in lst:
        frame=frame+1
        im =  Image.open(os.path.join(path_movie,file))    
        img_movie.append(np.array(im))

    if len(img_movie)==len(os.listdir(path_movie)):
        print('Movie uploaded: ', len(os.listdir(path_movie)), 'images')
    else:
        print('ERROR uploading movie')
        
    return img_movie

def read_mp4_video(path_file, frame2d):
    cap = cv2.VideoCapture(path_file) # Open the video file
    if not cap.isOpened(): # Check if the video file opened successfully
        print("Error: Could not open video file.")
        exit()
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) 
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if frame2d: # Create a variable to store frames
        myVideo = np.zeros((height, width, total_frames), dtype=np.uint8)
    else:
        myVideo = np.zeros((height, width, total_frames, 3), dtype=np.uint8)
    
    for frame in range(total_frames): # Read and store frames
        ret, frame_data = cap.read()
        if not ret:
            print(f"Error reading frame {frame}.")
            break
        if frame2d:
            gray_frame = cv2.cvtColor(frame_data, cv2.COLOR_BGR2GRAY)
            myVideo[:, :, frame] = gray_frame
        else:
            myVideo[:, :, frame, :] = frame_data
    cap.release()# Release the video capture object
    
    # Display some video information
    #print(f"Video Resolution: {width}x{height}")
    #print(f"Frames per Second (FPS): {fps}")
    #print(f"Total Frames: {total_frames}")
    # Now, 'myVideo' contains all the frames of the video
    # You can access individual frames using myVideo[:, :, frame]
    # For example, to access the first frame: first_frame = myVideo[:, :, 0]
    return myVideo, width, height, fps, total_frames

def extract_id_from_filename(filename,mp4):
    if mp4:
        match = re.search(r'movie_(\d+).mp4', filename)
    else:
        match = re.search(r'movie_(\d+).npy', filename)
    if match:
        return int(match.group(1))
    else:
        return None

import os
import numpy as np

def process_batch(file_list, path_movie):
    '''Manage Batching: Implement a way to load and process only 'batch_size' files at a time.
    Memory Management: This script processes each batch independently and resets myVideo after each batch, which should help manage memory usage. If your videos are large, consider checking and possibly clearing more resources or using more disk-based processing.
    %myVideo, frame_per_video, n_selected_video = process_batch(batch_files, path_movie)
    '''
    myVideo = None
    frame_per_video = []
    n_selected_video = 0
    
    for file in file_list:
        full_path = os.path.join(path_movie, file)
        movie_id = int(file.split('_')[1].split('.')[0])
        
        if '.mp4' in file:
            # Assuming read_mp4_video function returns video array and metadata
            video_data = read_mp4_video(full_path, True)
            video, width, height, fps, total_frames = video_data
            
            if myVideo is None:
                myVideo = video
            else:
                myVideo = np.concatenate((myVideo, video), axis=2)
            
            frame_per_video.append(total_frames)
        else:
            video = np.load(full_path, allow_pickle=True)
            
            if myVideo is None:
                myVideo = video
            else:
                myVideo = np.concatenate((myVideo, video), axis=2)
            
            frame_per_video.append(video.shape[2])
        
        n_selected_video += 1
    
    return myVideo, frame_per_video, n_selected_video