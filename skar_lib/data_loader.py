import yfinance as yf
import pandas as pd

def get_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    # Download historical data using yfinance
    df = yf.download(ticker, start=start, end=end)

    # 🔍 Debug: Print the actual columns returned
    print(f"[DEBUG] Downloaded columns for {ticker}: {list(df.columns)}")

    # Fallback logic: use 'Adj Close' if available, otherwise 'Close'
    if "Adj Close" in df.columns:
        df = df[["Adj Close"]].rename(columns={"Adj Close": "Price"})
    elif "Close" in df.columns:
        df = df[["Close"]].rename(columns={"Close": "Price"})
    else:
        raise ValueError(f"No 'Adj Close' or 'Close' column found in {ticker} data.")

    return df
