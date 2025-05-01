import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from typing import Optional

def get_slope(price: pd.Series, 
              window: Optional[int] = None, 
              polyorder: int = 2) -> pd.Series:
    """
    Calculate smoothed slope (first derivative) using Savitzky-Golay filter.
    Automatically adjusts window size for small datasets.
    
    Args:
        price: Price series
        window: Optional window size. If None, auto-calculates based on data length
        polyorder: Polynomial order (default: 2)
    
    Returns:
        pd.Series with slope values
    """
    # Auto-calculate window size if not specified
    if window is None:
        window = min(21, len(price))
    
    # Ensure valid window size
    window = max(3, min(window, len(price)))
    if window % 2 == 0:  # Savitzky-Golay requires odd window size
        window -= 1
    
    # Handle edge cases
    if len(price) < 3:
        return pd.Series(0, index=price.index)
    
    try:
        filtered = savgol_filter(price.values, 
                               window_length=window,
                               polyorder=min(polyorder, window-1),
                               deriv=1)
        return pd.Series(filtered, index=price.index)
    except Exception as e:
        print(f"Error in get_slope: {e}")
        return pd.Series(0, index=price.index)

def get_acceleration(price: pd.Series, 
                    window: Optional[int] = None,
                    polyorder: int = 2) -> pd.Series:
    """
    Calculate smoothed acceleration (second derivative) with auto-window adjustment.
    """
    return get_slope(get_slope(price, window, polyorder), window, polyorder)
