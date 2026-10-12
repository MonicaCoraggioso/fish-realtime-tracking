"""
    Some useful functions to performe smoothing and moving average of the signal
    
    Author: Monica Coraggioso
"""
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

class Fish_live:
   
    @classmethod
    def sma(cls, data, window):
        """
        calculates Simple Moving Average

        """
        if len(data) < window:
            return None
        return sum(data[-window:]) / float(window)
 
    @classmethod
    def ema(cls, data, window):
        """returns a 'window'-size period Exponential Moving Average for the time series 'data'

        """
        if len(data) < 2 * window:
            current_ema=np.nan
            return current_ema
            #raise ValueError("data is too short")
        c = 2.0 / (window + 1)
        current_ema = cls.sma(data[-window*2:-window], window)
        for value in data[-window:]:
            current_ema = (c * value) + ((1 - c) * current_ema)
        return current_ema

    @classmethod
    def position(cls, data_x, data_y, window):
        if len(data_x)<window:
            return np.full(window, 150), np.full(window, 150) 
        else:
            return gaussian_filter(data_x[-window:], sigma=1), gaussian_filter(data_y[-window:], sigma=1) 
   
    @classmethod
    def speed(cls, data_x, data_y, data_t):
        if len(data_t) > 1:
            x_speed = np.diff(data_x)/np.diff(data_t)
            y_speed = np.diff(data_y)/np.diff(data_t)
            return np.mean(x_speed), np.mean(y_speed)
        else:
            return np.nan, np.nan

    @classmethod
    def forecast_position(cls, vX, vY, dt):
        X_forecast=dt*vX
        Y_forecast=dt*vY
        return X_forecast, Y_forecast



