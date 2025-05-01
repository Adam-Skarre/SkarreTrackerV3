import yfinance as yf
import pandas as pd

print("LOADED CORRECT data_loader.py")

def get_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    # Download full dataset
    df = yf.download(ticker, start=start, end=end, progress=False)

    if df.empty:
        raise ValueError(f"No data returned for ticker {ticker} between {start} and {end}.")

    print(f"[DEBUG] Downloaded columns for {ticker}: {list(df.columns)}")

    # Check for adjusted or close price
    if "Adj Close" in df.columns:
        df['Price'] = df['Adj Close']
    elif "Close" in df.columns:
        df['Price'] = df['Close']
    else:
        raise ValueError(f"No 'Adj Close' or 'Close' column in {ticker} data.")

    # OPTIONAL: Keep the full DataFrame + just add the 'Price' column
    # You can create a reduced DataFrame if needed later:
    # reduced_df = df[['Price', 'Adj Open', 'HL_PCT', 'PCT_change', 'Adj Volume']]

    return df
