import yfinance as yf
import pandas as pd

def get_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end)

    print(f"[DEBUG] Downloaded columns for {ticker}: {list(df.columns)}")  # Confirm it's updated

    if "Adj Close" in df.columns:
        df = df[["Adj Close"]].rename(columns={"Adj Close": "Price"})
    elif "Close" in df.columns:
        df = df[["Close"]].rename(columns={"Close": "Price"})
    else:
        raise ValueError(f"No 'Adj Close' or 'Close' column found in {ticker} data.")

    return df
