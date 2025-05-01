import streamlit as st
st.set_page_config(page_title="Skarre Tracker Dashboard V3", layout="wide")

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Debug loader path
import skar_lib.data_loader as dl
st.write(f"Loading data_loader from: {dl.__file__}")

# Core functionality imports
from skar_lib.data_loader import get_data
from skar_lib.backtester import backtest
from skar_lib.signal_logic import generate_signals
from skar_lib.polynomial_fit import get_slope, get_acceleration
from skar_lib.walkforward import run_walkforward
from skar_lib.sensitivity import run_sensitivity

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select View", [
    "About",
    "Live Signal",
    "Backtest V1",
    "Walk-Forward",
    "Sensitivity Analysis"
])

# --- ABOUT PAGE ---
if page == "About":
    st.title("Skarre Tracker Dashboard — V3")
    st.markdown(
        """
        **Features:**
        - Signal generation via slope & acceleration
        - Costed backtest with entry/exit tuning
        - Walk-forward validation with fold-by-fold metrics
        - Sensitivity heatmap for threshold & hold periods
        """
    )

# --- LIVE SIGNAL VIEWER ---
elif page == "Live Signal":
    st.header("Live Signal Viewer")
    
    # Input controls in sidebar
    with st.sidebar:
        st.subheader("Parameters")
        ticker = st.text_input("Ticker Symbol", value="SPY", 
                             help="Enter a valid stock ticker (e.g., SPY, AAPL)")
        
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=datetime(2022, 1, 1),
                                     max_value=datetime.today())
        with col2:
            end_date = st.date_input("End Date", value=datetime.today(),
                                   min_value=start_date,
                                   max_value=datetime.today())
        
        st.markdown("---")
        st.subheader("Signal Settings")
        entry = st.number_input("Entry Threshold", value=0.5, min_value=0.0, step=0.1,
                              help="Slope value required to generate buy signal")
        exit_ = st.number_input("Exit Threshold", value=-0.5, max_value=0.0, step=0.1,
                              help="Slope value required to generate sell signal")
        use_acc = st.checkbox("Use Acceleration", value=True,
                            help="Include acceleration in signal generation")
        window_size = st.slider("Smoothing Window", min_value=5, max_value=61, 
                              value=21, step=2,
                              help="Larger values create smoother signals")

    try:
        # Data loading with validation
        with st.spinner(f"Loading {ticker} data..."):
            df = get_data(ticker, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
            
            if df.empty:
                st.error("No data returned - check ticker symbol and date range")
                st.stop()
                
            if "Price" not in df.columns:
                st.error("Data format error - missing 'Price' column")
                st.stop()
                
            price = df["Price"].dropna()
            if len(price) < 10:
                st.warning(f"Limited data ({len(price)} points) - results may be unreliable")
                if len(price) < 3:
                    st.error("Insufficient data points (need at least 3)")
                    st.stop()

        # Dynamic window adjustment
        safe_window = min(window_size, len(price))
        safe_window = max(3, safe_window)  # Minimum window size
        if safe_window % 2 == 0:  # Must be odd
            safe_window -= 1

        # Indicator calculation
        with st.spinner("Calculating indicators..."):
            slope = get_slope(price, window=safe_window)
            accel = get_acceleration(price, window=safe_window)
            signals = generate_signals(slope, accel, entry, exit_, use_acc)

        # Visualization
        fig = go.Figure()
        
        # Price line
        fig.add_trace(go.Scatter(
            x=price.index,
            y=price,
            mode='lines',
            name='Price',
            line=dict(color='#1f77b4', width=2)
        ))
        
        # Buy signals
        buy_signals = signals[signals == 1]
        if not buy_signals.empty:
            fig.add_trace(go.Scatter(
                x=buy_signals.index,
                y=price.loc[buy_signals.index],
                mode='markers',
                name='Buy',
                marker=dict(
                    symbol='triangle-up',
                    size=12,
                    color='green',
                    line=dict(width=1, color='DarkSlateGrey')
                )
            ))
        
        # Sell signals
        sell_signals = signals[signals == -1]
        if not sell_signals.empty:
            fig.add_trace(go.Scatter(
                x=sell_signals.index,
                y=price.loc[sell_signals.index],
                mode='markers',
                name='Sell',
                marker=dict(
                    symbol='triangle-down',
                    size=12,
                    color='red',
                    line=dict(width=1, color='DarkSlateGrey')
                )
            ))
        
        fig.update_layout(
            title=f"{ticker} Price with Trading Signals",
            xaxis_title='Date',
            yaxis_title='Price',
            hovermode='x unified',
            showlegend=True,
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Indicators plot
        with st.expander("Show Technical Indicators"):
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=slope.index,
                y=slope,
                name='Slope',
                line=dict(color='purple')
            ))
            if use_acc:
                fig2.add_trace(go.Scatter(
                    x=accel.index,
                    y=accel,
                    name='Acceleration',
                    line=dict(color='orange')
                ))
            fig2.update_layout(
                title="Slope and Acceleration Indicators",
                xaxis_title='Date',
                yaxis_title='Value'
            )
            st.plotly_chart(fig2, use_container_width=True)
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.stop()

    # Plot price and signals
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=price.index, y=price, mode='lines', name='Price'))
    buys = signals[signals == 1]
    sells = signals[signals == -1]
    if not buys.empty:
        fig.add_trace(go.Scatter(
            x=buys.index, y=price.loc[buys.index], mode='markers', name='Buy',
            marker=dict(symbol='triangle-up', size=10)
        ))
    if not sells.empty:
        fig.add_trace(go.Scatter(
            x=sells.index, y=price.loc[sells.index], mode='markers', name='Sell',
            marker=dict(symbol='triangle-down', size=10)
        ))
    fig.update_layout(xaxis_title='Date', yaxis_title='Price & Signals')
    st.plotly_chart(fig, use_container_width=True)

# --- V1 BACKTEST ---
elif page == "Backtest V1":
    st.header("Backtest V1: Original Skarre Signal")
    ticker = st.sidebar.text_input("Ticker", value="SPY")
    start_date = st.sidebar.date_input("Start Date", datetime(2022, 1, 1))
    end_date = st.sidebar.date_input("End Date", datetime.today())

    df = get_data(ticker, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
    price = df["Price"]
    slope = get_slope(price)
    accel = get_acceleration(price)

    entry = st.sidebar.number_input("Entry Threshold", value=0.5)
    exit_ = st.sidebar.number_input("Exit Threshold", value=-0.5)
    use_acc = st.sidebar.checkbox("Use Acceleration", value=True)

    signals = generate_signals(slope, accel, entry, exit_, use_acc)

    st.write(f"▶️ Price length: {len(price)}    Signals length: {len(signals)}")
    if price.empty or signals.empty:
        st.warning("❗ Price or signal data is empty.")
    else:
        result = backtest(price, signals)

        st.subheader("Equity Curve (Strategy vs SPY Buy & Hold)")
        strat_eq = result["trade_log"]["Equity"]
        spy_eq = (1 + price.pct_change().fillna(0)).cumprod() * 100000
        eq_df = pd.DataFrame({"Strategy": strat_eq, "SPY (Buy & Hold)": spy_eq})
        st.line_chart(eq_df)

        st.subheader("Performance Metrics")
        for key, val in result["metrics"].items():
            label = key.replace('_', ' ').title()
            if any(x in key.lower() for x in ['return','drawdown','cagr']):
                st.metric(label=label, value=f"{val:.2%}")
            else:
                st.metric(label=label, value=f"{val:.2f}")

# --- WALK-FORWARD VALIDATION ---
elif page == "Walk-Forward":
    st.header("Walk-Forward Validation (V3)")
    ticker = st.sidebar.text_input("Ticker", value="SPY")
    entry = st.sidebar.number_input("Entry Threshold", value=0.5)
    exit_ = st.sidebar.number_input("Exit Threshold", value=-0.5)
    train_win = st.sidebar.number_input("Train Window (days)", min_value=30, value=252)
    test_win = st.sidebar.number_input("Test Window (days)", min_value=30, value=63)
    step = st.sidebar.number_input("Step Size (days)", min_value=1, value=63)

    df = get_data(ticker, "2000-01-01", datetime.today().strftime("%Y-%m-%d"))
    if df.empty or "Price" not in df:
        st.warning("❗ Data not loaded properly.")
    else:
        price = df["Price"]
        def signal_fn(p):
            s = get_slope(p)
            a = get_acceleration(p)
            return generate_signals(s, a, entry, exit_, use_acc=True)

        folds = run_walkforward(price, signal_fn, train_win, test_win, step)
        st.subheader("Fold-by-Fold Results")
        st.dataframe(folds)
        if not folds.empty:
            st.line_chart(folds[["return","sharpe","max_drawdown"]])

# --- SENSITIVITY ANALYSIS ---
elif page == "Sensitivity Analysis":
    st.header("📊 Sensitivity Analysis: Sharpe vs Entry & Min Hold")
    ticker = st.sidebar.text_input("Ticker", value="SPY")
    start_date = st.sidebar.date_input("Start Date", datetime(2018, 1, 1))
    end_date = st.sidebar.date_input("End Date", datetime.today())

    df = get_data(ticker, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
    price = df.get("Price", pd.Series())
    if price.empty:
        st.warning("❗ No price data.")
    else:
        entry_min = st.sidebar.number_input("Entry Min", value=0.1)
        entry_max = st.sidebar.number_input("Entry Max", value=1.0)
        entry_step = st.sidebar.number_input("Entry Step", value=0.1)
        hold_min = st.sidebar.number_input("Min Hold Min", value=1)
        hold_max = st.sidebar.number_input("Min Hold Max", value=10)
        hold_step = st.sidebar.number_input("Min Hold Step", value=1)
        use_acc = st.sidebar.checkbox("Use Acceleration", value=True)

        st.write("Running sensitivity grid search...")
        grid = run_sensitivity(
            price,
            lambda p, e, h: generate_signals(
                get_slope(p), get_acceleration(p), e, -e, use_acc, min_hold=h
            ),
            entry_range=(entry_min, entry_max + entry_step/2, entry_step),
            hold_range=(hold_min, hold_max + 1, hold_step)
        )
        st.subheader("Sharpe Ratio Heatmap")
        st.dataframe(grid)

        fig = px.imshow(
            grid.values,
            labels=dict(x="Hold Period", y="Entry Threshold", color="Sharpe"),
            x=list(grid.columns),
            y=list(grid.index),
            aspect="auto"
        )
        fig.update_layout(yaxis_autorange='reversed')
        st.plotly_chart(fig, use_container_width=True)
