import pandas as pd
import numpy as np

def backtest(price: pd.Series, signal: pd.Series,
             initial_cash=100000,
             commission=1.0,
             slippage_pct=0.001,
             delay_days=1) -> dict:
    """
    Enhanced backtest engine with slippage, commission, delayed execution, and risk metrics.
    """

    # 1. Execution delay
    signal = signal.shift(delay_days).fillna(0)
    position = signal.replace({1: 1, -1: -1, 0: 0}).ffill().fillna(0)

    # 2. Effective trading price with slippage
    effective_price = price * (1 + slippage_pct)

    # 3. Returns calculation
    returns = price.pct_change().fillna(0)
    strat_returns = returns * position.shift(1)

    # 4. Trade costs
    trades = position.diff().fillna(0).abs()
    trade_costs = trades * commission / initial_cash
    net_returns = strat_returns - trade_costs

    # 5. Equity curve
    equity = (1 + net_returns).cumprod() * initial_cash

    # Align all series before constructing the trade log
    common_index = price.index
    for series in [signal, position, strat_returns, net_returns, equity]:
        common_index = common_index.intersection(series.index)

    price = price.loc[common_index]
    signal = signal.loc[common_index]
    position = position.loc[common_index]
    strat_returns = strat_returns.loc[common_index]
    net_returns = net_returns.loc[common_index]
    equity = equity.loc[common_index]

    # 6. Trade log
    trade_log = pd.DataFrame({
        "Price": price,
        "Signal": signal,
        "Position": position,
        "Return": strat_returns,
        "Net Return": net_returns,
        "Equity": equity
    })

    # 7. Metrics
    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    daily_ret = equity.pct_change().dropna()
    sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252) if daily_ret.std() != 0 else np.nan
    sortino = (daily_ret.mean() / daily_ret[daily_ret < 0].std()) * np.sqrt(252) if len(daily_ret[daily_ret < 0]) > 0 else np.nan
    max_drawdown = (equity / equity.cummax() - 1).min()
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (252 / len(equity)) - 1
    turnover = trades.sum() / len(price)

    metrics = {
        "total_return": round(total_return, 6),
        "sharpe":       round(sharpe, 6),
        "sortino":      round(sortino, 6),
        "max_drawdown": round(max_drawdown, 6),
        "CAGR":         round(cagr, 6),
        "turnover":     round(turnover, 6)
    }

    return {"metrics": metrics, "trade_log": trade_log}
