import pandas as pd
from typing import Callable, List, Dict
from skar_lib.backtester import backtest

def run_walkforward(
    price: pd.Series,
    signal_fn: Callable[[pd.Series], pd.Series],
    train_window: int,
    test_window: int,
    step: int
) -> pd.DataFrame:
    """
    Run walk-forward backtest across time splits.
    For each fold, use `signal_fn()` on test set and run backtest.
    """
    results: List[Dict] = []
    n = len(price)
    idx = price.index

    for start in range(0, n - train_window - test_window + 1, step):
        t0, t1 = start, start + train_window
        u0, u1 = t1, t1 + test_window

        test_price = price.iloc[u0:u1]
        signal = signal_fn(test_price)
        metrics = backtest(test_price, signal)["metrics"]

        results.append({
            "train_start": idx[t0].strftime("%Y-%m-%d"),
            "train_end": idx[t1 - 1].strftime("%Y-%m-%d"),
            "test_start": idx[u0].strftime("%Y-%m-%d"),
            "test_end": idx[u1 - 1].strftime("%Y-%m-%d"),
            **metrics
        })

    return pd.DataFrame(results)
