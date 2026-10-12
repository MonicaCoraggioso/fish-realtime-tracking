"""
    Line-fitting helpers used for the pixel-to-mm and camera-to-motor calibration.
    
    Author: Monica Coraggioso
"""
import numpy as np

def fit_line(x1, y1, x2, y2):
    m = (y2 - y1) / (x2 - x1)
    b = y1 - (m * x1)
    # The equation of the line is now "y = mx + b"
    return round(m,4),round(b,4)

def build_line(m,b,x_start,x_end):
    _aux=np.linspace(x_start,x_end,100)
    _fit=m*_aux+b
    return _aux,_fit

def build_line_inv(m,b,y_start,y_end):
    x_values = []
    y_values = []
    for y in range(int(y_start), int(y_end) + 1):
        x = (y - b) / m
        x_values.append(x)
        y_values.append(y)
    return np.array(x_values), np.array(y_values)