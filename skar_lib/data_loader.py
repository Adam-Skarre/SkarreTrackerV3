import yfinance as yf
import pandas as pd

print("LOADED CORRECT data_loader.py")

def get_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    # Fetch data with adjusted prices included
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)

    if df.empty:
        print(f"[WARNING] No data returned for ticker {ticker} between {start} and {end}.")
        return pd.DataFrame()

    print(f"[DEBUG] Available columns for {ticker}: {list(df.columns)}")

    # Prioritize 'Adj Close' if available; fallback to 'Close'
    if 'Adj Close' in df.columns:
        df = df[['Adj Close']].rename(columns={'Adj Close': 'Price'})
    elif 'Close' in df.columns:
        df = df[['Close']].rename(columns={'Close': 'Price'})
    else:
        raise ValueError(f"No 'Adj Close' or 'Close' column found in data for ticker {ticker}.")

    return df
