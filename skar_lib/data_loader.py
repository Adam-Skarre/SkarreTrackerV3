import yfinance as yf
import pandas as pd

print("LOADED CORRECT data_loader.py")

def get_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, progress=False)
    
    if df.empty:
        print(f"[WARNING] No data returned for ticker {ticker} between {start} and {end}.")
        return pd.DataFrame()
    
    price_column = "Adj Close" if "Adj Close" in df.columns else "Close"
    
    # Only keep the price column and rename it
    df = df[[price_column]].rename(columns={price_column: "Price"})
    
    return df
