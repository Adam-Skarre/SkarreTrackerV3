import pandas as pd
import numpy as np

def generate_signals(slope: pd.Series, accel: pd.Series, entry: float, exit_: float, use_acceleration=True) -> pd.Series:
    """
    Generate trade signals based on slope and optional acceleration.
    Signal:
        1 = Buy
       -1 = Sell
        0 = Hold
    """
    signal = pd.Series(0, index=slope.index)

    if use_acceleration:
        signal[(slope > entry) & (accel > 0)] = 1
        signal[(slope < exit_) & (accel < 0)] = -1
    else:
        signal[slope > entry] = 1
        signal[slope < exit_] = -1

    return signal
