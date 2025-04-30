import pandas as pd
import numpy as np
from scipy.signal import savgol_filter

def get_slope(price: pd.Series, window: int = 21, polyorder: int = 2) -> pd.Series:
    """
    Calculate smoothed slope (first derivative) using Savitzky-Golay filter.
    """
    slope = pd.Series(index=price.index, dtype=float)
    if len(price) >= window:
        filtered = savgol_filter(price.values, window_length=window, polyorder=polyorder, deriv=1)
        slope.iloc[window-1:] = filtered[window-1:]
    return slope.fillna(0)

def get_acceleration(price: pd.Series, window: int = 21, polyorder: int = 2) -> pd.Series:
    """
    Calculate smoothed acceleration (second derivative) using Savitzky-Golay filter.
    """
    accel = pd.Series(index=price.index, dtype=float)
    if len(price) >= window:
        filtered = savgol_filter(price.values, window_length=window, polyorder=polyorder, deriv=2)
        accel.iloc[window-1:] = filtered[window-1:]
    return accel.fillna(0)
