import yfinance as yf
import pandas as pd

print("LOADED CORRECT data_loader.py")

def get_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, progress=False)

    if df.empty:
        raise ValueError(f"No data returned for ticker {ticker} between {start} and {end}.")

    print(f"[DEBUG] Downloaded columns for {ticker}: {list(df.columns)}")

    if "Adj Close" in df.columns:
        df = df[["Adj Close"]].rename(columns={"Adj Close": "Price"})
    elif "Close" in df.columns:
        df = df[["Close"]].rename(columns={"Close": "Price"})
    else:
        raise ValueError(f"No 'Adj Close' or 'Close' column in {ticker} data.")

    return df
