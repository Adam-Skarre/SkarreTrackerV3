import pandas as pd
import numpy as np
from typing import Callable, Tuple
from skar_lib.backtester import backtest

def run_sensitivity(
    price: pd.Series,
    signal_fn_factory: Callable[[float, int], Callable[[pd.Series], pd.Series]],
    entry_range: Tuple[float, float, float],
    hold_range: Tuple[int, int, int]
) -> pd.DataFrame:
    """
    Run grid search over combinations of entry_threshold and min_hold.
    For each combo, create a signal function and run a backtest.

    Args:
        price: price series
        signal_fn_factory: function that returns a signal function given (entry_threshold, min_hold)
        entry_range: tuple of (start, stop, step) for entry threshold
        hold_range: tuple of (start, stop, step) for min_hold period

    Returns:
        DataFrame with index = entry_threshold, columns = min_hold, values = Sharpe Ratio
    """

    entry_values = np.arange(*entry_range)
    hold_values = np.arange(*hold_range)

    results = pd.DataFrame(index=entry_values, columns=hold_values, dtype=float)

    for entry in entry_values:
        for hold in hold_values:
            signal_fn = signal_fn_factory(entry, hold)
            try:
                signal = signal_fn(price)
                result = backtest(price, signal)
                sharpe = result["metrics"].get("sharpe", np.nan)
            except Exception:
                sharpe = np.nan
            results.loc[entry, hold] = sharpe

    results.index.name = "Entry Threshold"
    results.columns.name = "Min Hold Period"
    return results
