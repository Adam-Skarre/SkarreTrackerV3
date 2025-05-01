# app.py
import streamlit as st
import pandas as pd
from skar_lib import data_loader, signal_logic, backtester

st.title("📈 Skarre Tracker V3")

# --- Sidebar for inputs ---
ticker = st.sidebar.selectbox("Select Ticker", ["SPY", "QQQ", "DIA"])
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2024-12-31"))

# --- Load data ---
st.write(f"Loading data for {ticker}...")
df = data_loader.load_price_data(ticker, start_date, end_date)
st.line_chart(df["Close"])

# --- Generate signals ---
st.write("Generating signals...")
signals = signal_logic.generate_signals(df)

# --- Run backtest ---
st.write("Running backtest...")
results = backtester.run_backtest(df, signals)

# --- Show results ---
st.subheader("📊 Equity Curve")
st.line_chart(results["equity_curve"])

st.subheader("📌 Performance Metrics")
for k, v in results["metrics"].items():
    st.metric(k, f"{v:.2f}")
