import yfinance as yf
import pandas as pd

def get_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Download historical price data using yfinance.
    Returns a DataFrame with a 'Price' column (adjusted close).
    """
    df = yf.download(ticker, start=start, end=end, progress=False)
    if "Adj Close" in df.columns:
    df = df[["Adj Close"]].rename(columns={"Adj Close": "Price"})
elif "Close" in df.columns:
    df = df[["Close"]].rename(columns={"Close": "Price"})
else:
    raise ValueError("No 'Adj Close' or 'Close' column found in downloaded data.")
    df = df.dropna()
    return df
