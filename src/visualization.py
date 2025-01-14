import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import numpy as np

# Load the trained model
model = load_model('models/stock_price_model.keras')

# Load preprocessed data
data = pd.read_csv('data/processed_data.csv')

# Prepare data for LSTM
def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

# Split data into features and target
time_step = 60
X, y = create_dataset(data[['Close']].values, time_step)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Make predictions
predictions = model.predict(X)

# Plot actual vs predicted
plt.figure(figsize=(10, 6))
plt.plot(y, color='blue', label='Actual Stock Price')
plt.plot(predictions, color='red', label='Predicted Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.savefig('plots/actual_vs_predicted.png')
plt.show()
