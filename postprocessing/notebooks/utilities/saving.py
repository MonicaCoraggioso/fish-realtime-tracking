"""
    Utility functions for saving and loading data files.
    
    This module provides functions for:
    - Saving Python lists as NumPy arrays (.npy format)
    - Loading saved NumPy arrays back to Python lists
    - Creating directories for data organization
    
    Author: Monica Coraggioso
"""
import numpy as np
import os

def saveList(myList,filename):
    # the filename should mention the extension 'npy'
    np.save(filename,myList)
    print("Saved successfully!")


def loadList(filename):
    # the filename should mention the extension 'npy'
    tempNumpyArray=np.load(filename)
    return tempNumpyArray.tolist()

import os
import errno

def create_folder(path):
    try:
        os.makedirs(path)
        print(f"Folder created at: {path}")
    except OSError as e:
        if e.errno == errno.EEXIST:
            print(f"Error: Folder already exists at: {path}")
        else:
            raise  # Re-raise the exception if a different error occurred