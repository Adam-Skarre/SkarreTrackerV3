import yfinance as yf
import pandas as pd

def get_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Downloads historical price data for the given ticker using yfinance.
    Returns a DataFrame with a single column: 'Price'.

    If 'Adj Close' is available, it will be used.
    Otherwise, it will fall back to 'Close'.
    """
    df = yf.download(ticker, start=start, end=end)

    # Print to debug what yfinance actually returned
    print(f"Downloaded columns for {ticker}: {list(df.columns)}")

    if "Adj Close" in df.columns:
        df = df[["Adj Close"]].rename(columns={"Adj Close": "Price"})
    elif "Close" in df.columns:
        df = df[["Close"]].rename(columns={"Close": "Price"})
    else:
        raise ValueError("No 'Adj Close' or 'Close' column found in downloaded data.")

    return df
