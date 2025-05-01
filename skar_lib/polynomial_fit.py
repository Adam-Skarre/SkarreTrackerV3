import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from typing import Optional

def get_slope(price: pd.Series, window: Optional[int] = None, polyorder: int = 2) -> pd.Series:
    """
    Completely robust slope calculation that handles all edge cases.
    """
    if len(price) < 2:
        return pd.Series(0, index=price.index)
    
    # Auto-calculate safe window size
    max_window = len(price)
    if window is None:
        window = min(21, max_window)
    else:
        window = min(window, max_window)
    
    # Ensure valid window parameters
    window = max(3, window)  # Minimum window size
    if window % 2 == 0:  # Must be odd
        window -= 1
    
    # Ensure valid polyorder
    polyorder = min(polyorder, window - 1)
    
    try:
        # Fallback to gradient for very small windows
        if window < 3 or len(price) < window:
            return pd.Series(np.gradient(price.values), index=price.index)
        
        return pd.Series(
            savgol_filter(price.values, 
                         window_length=window,
                         polyorder=polyorder,
                         deriv=1,
                         mode='interp'),
            index=price.index
        )
    except Exception as e:
        print(f"Warning: Using gradient fallback due to: {str(e)}")
        return pd.Series(np.gradient(price.values), index=price.index)

def get_acceleration(price: pd.Series, window: Optional[int] = None, polyorder: int = 2) -> pd.Series:
    """Robust acceleration calculation using slope of slope"""
    return get_slope(get_slope(price, window, polyorder), window, polyorder)
