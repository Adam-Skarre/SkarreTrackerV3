import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.graph_objects as go
from datetime import datetime

from skar_lib.data_loader import get_data
from skar_lib.backtester import backtest
from skar_lib.signal_logic import generate_signals
from skar_lib.polynomial_fit import get_slope, get_acceleration
from skar_lib.walkforward import run_walkforward
from skar_lib.sensitivity import run_sensitivity

st.set_page_config(page_title="Skarre Tracker Dashboard V3", layout="wide")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select View", ["About", "Live Signal", "Backtest V1", "Walk-Forward", "Sensitivity Analysis"])

# About Page
if page == "About":
    st.title(" Skarre Tracker Dashboard — V3")
    st.markdown(
        """
        This V3 dashboard features:
        - Signal generation using slope + acceleration
        - V1 backtest logic with entry/exit tuning
        - V3 walk-forward validation with fold-by-fold metrics
        """
    )

# Live Signal Viewer
elif page == "Live Signal":
    st.header(" Live Signal Viewer")
    ticker = st.sidebar.text_input("Ticker", value="SPY")
    start = st.sidebar.date_input("Start Date", datetime(2022, 1, 1))
    end = st.sidebar.date_input("End Date", datetime.today())

    df = get_data(ticker, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    price = df["Price"]
    slope = get_slope(price)
    accel = get_acceleration(price)

    entry = st.sidebar.number_input("Entry Threshold", value=0.5)
    exit_ = st.sidebar.number_input("Exit Threshold", value=-0.5)
    use_acc = st.sidebar.checkbox("Use Acceleration", True)

    signals = generate_signals(slope, accel, entry, exit_, use_acc)

    # Plot price and signals
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=price.index, y=price, mode='lines', name='Price'
    ))
    buys = signals[signals == 1]
    sells = signals[signals == -1]
    if not buys.empty:
        fig.add_trace(go.Scatter(
            x=buys.index, y=price.loc[buys.index], mode='markers', name='Buy',
            marker=dict(symbol='triangle-up', color='green', size=10)
        ))
    if not sells.empty:
        fig.add_trace(go.Scatter(
            x=sells.index, y=price.loc[sells.index], mode='markers', name='Sell',
            marker=dict(symbol='triangle-down', color='red', size=10)
        ))
    fig.update_layout(xaxis_title='Date', yaxis_title='Price & Signals')
    st.plotly_chart(fig, use_container_width=True)

# V1 Backtest
elif page == "Backtest V1":
    st.header("Backtest V1: Original Skarre Signal")
    ticker = st.sidebar.text_input("Ticker", value="SPY")
    start = st.sidebar.date_input("Start Date", datetime(2022, 1, 1))
    end = st.sidebar.date_input("End Date", datetime.today())

    df = get_data(ticker, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    price = df["Price"]
    slope = get_slope(price)
    accel = get_acceleration(price)

    entry = st.sidebar.number_input("Entry Threshold", value=0.5)
    exit_ = st.sidebar.number_input("Exit Threshold", value=-0.5)
    use_acc = st.sidebar.checkbox("Use Acceleration", True)

    signals = generate_signals(slope, accel, entry, exit_, use_acc)

    # DEBUG: check lengths before backtest
    st.write(f"▶️ Price length: {len(price)}    Signals length: {len(signals)}")

    result = backtest(price, signals)

    # Plot equity curve with SPY benchmark
    st.subheader("Equity Curve (Strategy vs SPY Buy & Hold)")
    strategy_equity = result["trade_log"]["Equity"]
    spy_benchmark = (1 + price.pct_change().fillna(0)).cumprod() * 100000

    equity_df = pd.DataFrame({
        "Strategy": strategy_equity,
        "SPY (Buy & Hold)": spy_benchmark
    })
    st.line_chart(equity_df)

    # Show performance metrics
    st.subheader("Performance Metrics")
    for key, val in result["metrics"].items():
        st.metric(label=key, value=f"{val:.2%}" if 'return' in key.lower() or 'drawdown' in key.lower() or 'cagr' in key.lower() else f"{val:.2f}")

# Walk-Forward Validation
elif page == "Walk-Forward":
    st.header("Walk-Forward Validation (V3)")
    ticker = st.sidebar.text_input("Ticker", value="SPY")
    entry = st.sidebar.number_input("Entry Threshold", value=0.5)
    exit_ = st.sidebar.number_input("Exit Threshold", value=-0.5)
    train_window = st.sidebar.number_input("Train Window (days)", min_value=30, value=252)
    test_window = st.sidebar.number_input("Test Window (days)", min_value=30, value=63)
    step = st.sidebar.number_input("Step Size (days)", min_value=1, value=63)

    df = get_data(ticker, "2000-01-01", datetime.today().strftime("%Y-%m-%d"))
    price = df["Price"]

    def signal_fn(p):
        s = get_slope(p)
        a = get_acceleration(p)
        return generate_signals(s, a, entry, exit_, True)

    df_folds = run_walkforward(price, signal_fn, train_window, test_window, step)
    st.subheader("Fold-by-Fold Results")
    st.dataframe(df_folds)
    
    if not df_folds.empty:
        st.line_chart(df_folds[["return", "sharpe", "max_drawdown"]])
 elif page == "Sensitivity Analysis":
    st.header("📊 Sensitivity Analysis: Sharpe vs Entry & Min Hold")

    ticker = st.sidebar.text_input("Ticker", value="SPY")
    start = st.sidebar.date_input("Start Date", datetime(2018, 1, 1))
    end = st.sidebar.date_input("End Date", datetime.today())

    df = get_data(ticker, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    price = df["Price"]

    # Define parameter sweep ranges
    entry_min = st.sidebar.number_input("Entry Min", value=0.1)
    entry_max = st.sidebar.number_input("Entry Max", value=1.0)
    entry_step = st.sidebar.number_input("Entry Step", value=0.1)

    hold_min = st.sidebar.number_input("Min Hold Min", value=1)
    hold_max = st.sidebar.number_input("Min Hold Max", value=10)
    hold_step = st.sidebar.number_input("Min Hold Step", value=1)

    use_acc = st.sidebar.checkbox("Use Acceleration", value=True)

    def signal_fn_factory(entry, hold):
        def inner(price_window):
            slope = get_slope(price_window)
            accel = get_acceleration(price_window)
            return generate_signals(slope, accel, entry, -entry, use_acc=use_acc, min_hold=hold)
        return inner

    st.write("Running grid search...")
    grid_df = run_sensitivity(
        price,
        signal_fn_factory,
        entry_range=(entry_min, entry_max + 0.001, entry_step),
        hold_range=(hold_min, hold_max + 1, hold_step)
    )

    st.subheader("📈 Sharpe Ratio Heatmap")
    st.dataframe(grid_df)

    import plotly.express as px
    fig = px.imshow(grid_df.values,
                    labels=dict(x="Min Hold", y="Entry Threshold", color="Sharpe"),
                    x=grid_df.columns.astype(str),
                    y=grid_df.index.astype(str),
                    color_continuous_scale="Viridis",
                    aspect="auto")
    st.plotly_chart(fig)
