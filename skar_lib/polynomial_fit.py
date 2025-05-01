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
    max_window = len(price) - 1 if len(price) % 2 == 0 else len(price)
    window = min(window or 21, max_window) if max_window >= 3 else 3
    window = max(3, window)  # Minimum window size
    window = window if window % 2 == 1 else window - 1  # Ensure odd
    
    # Adjust polyorder if needed
    polyorder = min(polyorder, window - 1)
    
    try:
        # Use simpler calculation for very small windows
        if window < 3:
            return pd.Series(np.gradient(price.values), index=price.index)
        
        return pd.Series(
            savgol_filter(price.values, window_length=window, 
                         polyorder=polyorder, deriv=1, mode='interp'),
            index=price.index
        )
    except Exception:
        # Fallback to numpy gradient if anything goes wrong
        return pd.Series(np.gradient(price.values), index=price.index)

def get_acceleration(price: pd.Series, window: Optional[int] = None, polyorder: int = 2) -> pd.Series:
    """Robust acceleration calculation using slope of slope"""
    return get_slope(get_slope(price, window, polyorder), window, polyorder)
