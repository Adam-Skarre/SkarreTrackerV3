# skar_lib/data_loader.py
import pandas as pd
import yfinance as yf
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_data(ticker, start_date, end_date, data_source='yfinance'):
    """
    Universal data loader with robust error handling
    Supports multiple data sources and column name variations
    """
    df = pd.DataFrame()
    
    try:
        if data_source == 'yfinance':
            df = _fetch_yfinance_data(ticker, start_date, end_date)
        elif data_source == 'csv':
            df = _fetch_csv_data(ticker, start_date, end_date)
        else:
            raise ValueError(f"Unsupported data source: {data_source}")

        # Standardize column names
        df = _standardize_columns(df)
        
        # Validate dataframe
        if df.empty:
            raise ValueError(f"No data found for {ticker} between {start_date} and {end_date}")
            
        logger.info(f"Successfully loaded data for {ticker}")
        return df

    except Exception as e:
        logger.error(f"Data loading failed: {str(e)}")
        raise

def _fetch_yfinance_data(ticker, start_date, end_date):
    """Fetch data from Yahoo Finance with enhanced error handling"""
    try:
        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=True  # Automatically adjusts OHLC for corporate actions
        )
        
        # Handle different yfinance version formats
        data.columns = data.columns.str.lower().str.replace(' ', '_')
        return data

    except Exception as e:
        raise ConnectionError(f"Yahoo Finance API error: {str(e)}")

def _standardize_columns(df):
    """Normalize column names across different data sources"""
    column_mapping = {
        'adj_close': 'price',
        'close': 'price',
        'adjusted_close': 'price',
        'adj_close': 'price',
        'px_last': 'price'  # For Bloomberg-style data
    }
    
    # Find first matching column
    for source_col in column_mapping.keys():
        if source_col in df.columns:
            df = df[[source_col]].rename(columns={source_col: 'price'})
            df = df[['price']]  # Ensure single-column output
            return df
            
    raise KeyError("No valid price column found. Available columns: " + str(df.columns.tolist()))

# Example CSV handler for local files
def _fetch_csv_data(ticker, start_date, end_date):
    """Load data from local CSV files"""
    try:
        df = pd.read_csv(f'data/{ticker}.csv', parse_dates=['Date'])
        df = df.set_index('Date')
        return df
    except FileNotFoundError:
        raise ValueError(f"CSV file not found for ticker: {ticker}")

# Test the function
if __name__ == "__main__":
    # Test parameters
    test_ticker = 'SPY'
    test_start = '2020-01-01'
    test_end = '2023-01-01'
    
    try:
        test_df = get_data(test_ticker, test_start, test_end)
        print(test_df.head())
    except Exception as e:
        print(f"Test failed: {str(e)}")
