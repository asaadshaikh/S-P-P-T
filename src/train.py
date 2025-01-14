import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os

# Ensure the models directory exists
os.makedirs('d:/Projects/FirstProgram/stock-price-prediction/models', exist_ok=True)

# Load preprocessed data
data = pd.read_csv('data/processed_data.csv')

# Normalize the 'Close' prices
scaler = MinMaxScaler(feature_range=(0, 1))
data['Close'] = scaler.fit_transform(data[['Close']])

# Prepare data for LSTM
def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

# Function to create the LSTM model
def create_model(lstm_units=100, dropout_rate=0.2):
    model = Sequential()
    model.add(LSTM(lstm_units, return_sequences=True, input_shape=(60, 1)))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(lstm_units, return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(lstm_units, return_sequences=False))
    model.add(Dropout(dropout_rate))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Split data into features and target
time_step = 60
X, y = create_dataset(data[['Close']].values, time_step)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = create_model()

# Define callbacks
early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('d:/Projects/FirstProgram/stock-price-prediction/models/stock_price_model.h5', save_best_only=True)

# Fit the model
model.fit(X_train, y_train, epochs=50, batch_size=64, verbose=1, callbacks=[early_stopping, model_checkpoint])

# Make predictions
predictions = model.predict(X_test)

# Reshape predictions for inverse transform
predictions = predictions.reshape(-1, 1)

# Inverse transform predictions and y_test
predictions = scaler.inverse_transform(predictions)
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# Evaluate the model
rmse = np.sqrt(mean_squared_error(y_test, predictions))
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"R²: {r2}")

# Plot actual vs predicted
plt.figure(figsize=(10, 6))
plt.plot(y_test, color='blue', label='Actual Stock Price')
plt.plot(predictions, color='red', label='Predicted Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.savefig('d:/Projects/FirstProgram/stock-price-prediction/plots/actual_vs_predicted.png')
plt.show()