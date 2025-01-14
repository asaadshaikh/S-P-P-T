# Data Documentation for Stock Price Prediction Project

This README file contains important information regarding the datasets used in the Stock Price Prediction project.

## Data Sources

The primary source of historical stock price data is the `yfinance` library, which allows for easy access to financial data from Yahoo Finance. The data includes daily stock prices, trading volume, and other relevant financial metrics.

## Dataset Description

The dataset consists of historical stock prices for the selected stock ticker (e.g., AAPL, GOOG) over the past specified number of years (e.g., 5, 10). The key features in the dataset include:

- **Date**: The date of the stock price record.
- **Open**: The price at which the stock opened on that day.
- **High**: The highest price of the stock during the day.
- **Low**: The lowest price of the stock during the day.
- **Close**: The price at which the stock closed on that day.
- **Volume**: The number of shares traded during the day.
- **Adjusted Close**: The closing price adjusted for dividends and stock splits.

## Handling Missing Data

In the data acquisition phase, missing values are handled using forward fill or interpolation methods to ensure continuity in the time series analysis.

## Technical Indicators

The project also calculates several technical indicators to aid in analysis and prediction, including:

- **Moving Average**: A commonly used indicator to smooth out price data.
- **Relative Strength Index (RSI)**: A momentum oscillator that measures the speed and change of price movements.
- **Moving Average Convergence Divergence (MACD)**: A trend-following momentum indicator that shows the relationship between two moving averages of a security’s price.

## Usage

The data will be preprocessed and used to train a Long Short-Term Memory (LSTM) neural network model for stock price prediction. The results will be evaluated and visualized in the accompanying Jupyter notebook located in the `notebooks` directory.