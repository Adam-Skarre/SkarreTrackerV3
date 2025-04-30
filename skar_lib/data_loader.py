import yfinance as yf
import pandas as pd

def get_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Download historical price data using yfinance.
    Returns a DataFrame with a 'Price' column (adjusted close).
    """
    df = yf.download(ticker, start=start, end=end, progress=False)
    df = df[["Adj Close"]].rename(columns={"Adj Close": "Price"})
    df = df.dropna()
    return df
