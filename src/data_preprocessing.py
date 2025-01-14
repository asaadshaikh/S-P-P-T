import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(data):
    # Calculate Simple Moving Average (SMA)
    data['SMA'] = data['Close'].rolling(window=20).mean()

    # Calculate Relative Strength Index (RSI)
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))

    # Calculate Bollinger Bands
    data['Bollinger_Middle'] = data['Close'].rolling(window=20).mean()
    data['Bollinger_Upper'] = data['Bollinger_Middle'] + 2 * data['Close'].rolling(window=20).std()
    data['Bollinger_Lower'] = data['Bollinger_Middle'] - 2 * data['Close'].rolling(window=20).std()

    # Drop NaN values
    data.dropna(inplace=True)

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['Close']])  # Only use 'Close' for LSTM

    # Prepare the dataset
    X = []
    y = []
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i, :])
        y.append(scaled_data[i, 0])  # Predict the 'Close' price

    X, y = np.array(X), np.array(y)

    return X, y, scaler
