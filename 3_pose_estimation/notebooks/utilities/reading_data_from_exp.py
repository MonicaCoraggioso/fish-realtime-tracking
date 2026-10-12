"""
    Functions for reading experimental data files.
    
    This module provides utilities for:
    - Reading HDF5 files from experimental recordings
    - Displaying HDF5 file structure
    - Extracting timestamps and metadata from experimental data
    
    Author: Monica Coraggioso
"""
import numpy as np
from PIL import Image
import re
import os
import cv2
import h5py

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

def extract_time_from_date(byte_string):
    decoded_string = byte_string.decode('utf-8')
    time_slicing = decoded_string[11:19]
    return time_slicing
