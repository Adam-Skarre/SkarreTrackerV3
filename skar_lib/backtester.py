import pandas as pd
import numpy as np

def backtest(price: pd.Series, signal: pd.Series) -> dict:
    """
    Simple backtest engine:
      - Buys when signal == 1
      - Sells when signal == -1
      - Holds otherwise
    Returns a dict with 'metrics' and 'trade_log' (both pandas objects).
    """

    # 1. Align signal to next bar (we execute on the next period)
    position = signal.shift(1).fillna(0)

    # 2. Compute returns
    returns = price.pct_change().fillna(0)
    strat_returns = returns * position

    # 3. Build equity curve
    equity = (1 + strat_returns).cumprod()

    # 4. Build the trade_log DataFrame by passing the Series directly.
    #    This guarantees each column shares the same Date index.
    trades = pd.DataFrame({
        "Price": price,
        "Signal": signal,
        "Position": position,
        "Return": strat_returns,
        "Equity": equity
    })

    # 5. Compute summary metrics
    total_return = equity.iloc[-1] - 1
    sharpe = (strat_returns.mean() / strat_returns.std(ddof=1)) * np.sqrt(252) \
             if strat_returns.std(ddof=1) != 0 else np.nan
    max_drawdown = (equity / equity.cummax() - 1).min()

    metrics = {
        "total_return": round(total_return, 6),
        "sharpe":       round(sharpe, 6),
        "max_drawdown": round(max_drawdown, 6)
    }

    return {"metrics": metrics, "trade_log": trades}
