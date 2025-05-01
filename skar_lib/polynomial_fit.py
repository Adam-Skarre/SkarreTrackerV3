import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from typing import Optional

def get_slope(price: pd.Series, window: Optional[int] = None, polyorder: int = 2) -> pd.Series:
    """
    Calculate smoothed slope (first derivative) with automatic window adjustment.
    Guaranteed to never raise the window_length error.
    """
    if len(price) < 2:  # Need at least 2 points for a slope
        return pd.Series(0, index=price.index)
    
    # Calculate safe window size
    max_possible_window = len(price) - 1 if len(price) % 2 == 0 else len(price)
    default_window = min(21, max_possible_window)
    window = min(window, max_possible_window) if window else default_window
    
    # Ensure window meets Savitzky-Golay requirements
    window = max(3, window)  # Minimum window size
    if window % 2 == 0:  # Must be odd
        window -= 1
    
    # Adjust polyorder if needed
    polyorder = min(polyorder, window - 1)
    
    try:
        filtered = savgol_filter(
            price.values,
            window_length=window,
            polyorder=polyorder,
            deriv=1,
            mode='interp'  # This is the default that was causing issues
        )
        return pd.Series(filtered, index=price.index)
    except Exception as e:
        print(f"Slope calculation warning: {str(e)}")
        return pd.Series(0, index=price.index)

def get_acceleration(price: pd.Series, window: Optional[int] = None, polyorder: int = 2) -> pd.Series:
    """Calculate second derivative with the same safety checks."""
    return get_slope(get_slope(price, window, polyorder), window, polyorder)
