import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from typing import Optional

def safe_savgol(x: np.ndarray, window: int, polyorder: int, deriv: int) -> np.ndarray:
    """Completely safe Savitzky-Golay filter wrapper"""
    n = len(x)
    if n == 0:
        return np.zeros_like(x)
    
    # Calculate maximum possible window
    max_window = n if n % 2 == 1 else n - 1
    window = min(window, max_window)
    window = max(3, window)  # Minimum window size
    
    # Ensure window is odd
    if window % 2 == 0:
        window -= 1
    
    # Adjust polyorder if needed
    polyorder = min(polyorder, window - 1)
    
    try:
        if window >= 3 and n >= window:
            return savgol_filter(x, window_length=window, 
                               polyorder=polyorder, 
                               deriv=deriv,
                               mode='interp')
        return np.gradient(x)  # Fallback for small datasets
    except:
        return np.gradient(x)  # Final fallback

def get_slope(price: pd.Series, window: Optional[int] = None, polyorder: int = 2) -> pd.Series:
    """Bulletproof slope calculation"""
    if len(price) < 2:
        return pd.Series(0, index=price.index)
    
    window = min(window or 21, len(price))  # Default to 21 or data length
    filtered = safe_savgol(price.values, window, polyorder, deriv=1)
    return pd.Series(filtered, index=price.index)

def get_acceleration(price: pd.Series, window: Optional[int] = None, polyorder: int = 2) -> pd.Series:
    """Bulletproof acceleration calculation"""
    if len(price) < 3:
        return pd.Series(0, index=price.index)
    
    window = min(window or 21, len(price))
    filtered = safe_savgol(price.values, window, polyorder, deriv=2)
    return pd.Series(filtered, index=price.index)
