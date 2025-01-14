import yfinance as yf
import pandas as pd
import os

def download_stock_data(ticker, start_date, end_date):
    """
    Download historical stock price data from Yahoo Finance.

    Parameters:
    ticker (str): Stock ticker symbol (e.g., 'AAPL').
    start_date (str): Start date for the data in 'YYYY-MM-DD' format.
    end_date (str): End date for the data in 'YYYY-MM-DD' format.

    Returns:
    pd.DataFrame: DataFrame containing the historical stock data.
    """
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

def handle_missing_data(stock_data):
    """
    Handle missing data in the stock DataFrame.

    Parameters:
    stock_data (pd.DataFrame): DataFrame containing stock price data.

    Returns:
    pd.DataFrame: DataFrame with missing data handled.
    """
    stock_data.ffill(inplace=True)  # Updated to use ffill() instead of fillna(method='ffill')
    return stock_data

def save_data_to_csv(stock_data, filename):
    """
    Save the stock data to a CSV file.

    Parameters:
    stock_data (pd.DataFrame): DataFrame containing stock price data.
    filename (str): Filename for the CSV file.
    """
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        stock_data.to_csv(filename, index=True)
        print(f"Data saved to {filename}")  # Added logging to confirm file save
    except Exception as e:
        print(f"Error saving data to {filename}: {e}")  # Added error handling

if __name__ == "__main__":
    # Define the stock ticker and date range
    ticker = 'AAPL'
    start_date = '2018-01-01'
    end_date = '2023-01-01'

    # Download the stock data
    stock_data = download_stock_data(ticker, start_date, end_date)

    # Handle missing data
    stock_data = handle_missing_data(stock_data)

    # Save the data to a CSV file using an absolute path
    save_data_to_csv(stock_data, os.path.join(os.getcwd(), 'data', 'raw_data.csv'))
