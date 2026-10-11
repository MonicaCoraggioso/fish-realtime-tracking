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

def sorted_alphanumeric(data):
    '''os.listdir() returns a list containing the names of the entries in the directory given by 'data'. The list is in arbitrary order, so you have to write a natural sorting (meaning 1, 2, 10 instead of 1, 10, 2) yourself.
    
    ---You can now use this function to sort a list:
        lst = sorted_alphanumeric(os.listdir(path_movie))
        print(lst)
    '''
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    
    return sorted(data, key=alphanum_key)

def load_mp4_as_numpy(video_path, as_gray=True):
    cap = cv2.VideoCapture(video_path)

    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if as_gray:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # shape: (H, W)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   # shape: (H, W, 3)
        frames.append(frame)
    cap.release()

    if not frames:
        raise ValueError(f"No frames loaded from {video_path}")

    video_array = np.stack(frames, axis=-1) if as_gray else np.stack(frames, axis=-1)  # shape: (H, W, T) or (H, W, 3, T)
    return video_array

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


def h5tree_view(file):
    """Display HDF5 file structure in tree-like format.
   
    Parameters
    ----------
        :file: h5py.File , opened HDF5 file.
    """
    import h5py
    assert type(file) == h5py._hl.files.File
   
    def view_h5attributes(obj, depth=0):
        atts = obj.attrs
        deep = "│   "*depth
        ks = atts.keys()
        for i,k in enumerate(ks):
            if i == len(ks)-1:
                d = deep + "└──"
            else:
                d = deep + "├──"
            try:
                print(d, f'🏷️{k} = `{atts[k].decode("utf-8")}`')
            except (UnicodeDecodeError, AttributeError):
                print(d, f'🏷️{k} = `{atts[k]}`')
           
    def view_h5object(obj, depth=0):
        name = obj.name.split("/")[-1]
        deep = "│   "*depth + "├──"
        if type(obj) == h5py._hl.group.Group:
            print(deep,f"📁{name}")
            view_h5attributes(obj, depth=depth+1)
            for k in list(obj):
                view_h5object(obj[k], depth=depth+1)
        else:
            print(deep,f"🔢{name}", f"⚙️{obj.shape}{obj.dtype}")
            view_h5attributes(obj, depth=depth+1)

    print(".", file.filename)
    for k in list(file):
        view_h5object(file[k], depth=0)