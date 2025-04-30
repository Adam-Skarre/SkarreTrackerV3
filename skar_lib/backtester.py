import pandas as pd

def backtest(price: pd.Series, signal: pd.Series) -> dict:
    """
    Simple backtest engine:
    - Buys when signal == 1
    - Sells when signal == -1
    - Holds otherwise
    Returns metrics and trade log.
    """
    position = signal.shift(1).fillna(0)
    daily_returns = price.pct_change().fillna(0)
    strategy_returns = daily_returns * position

    equity = (1 + strategy_returns).cumprod()

    trades = pd.DataFrame({
        "Date": price.index,
        "Price": price.values,
        "Signal": signal.values,
        "Position": position.values,
        "Return": strategy_returns.values,
        "Equity": equity.values
    })

    # Summary metrics
    total_return = equity.iloc[-1] - 1
    sharpe = strategy_returns.mean() / strategy_returns.std() * (252 ** 0.5) if strategy_returns.std() != 0 else 0
    max_drawdown = (equity / equity.cummax() - 1).min()

    metrics = {
        "total_return": round(total_return, 4),
        "sharpe": round(sharpe, 4),
        "max_drawdown": round(max_drawdown, 4)
    }

    return {"metrics": metrics, "trade_log": trades}
