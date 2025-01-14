import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import pandas as pd
from data_preprocessing import preprocess_data
import os

# Create plots directory if it doesn't exist
plots_dir = 'd:/Projects/FirstProgram/stock-price-prediction/plots'
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)
    print(f"Created directory: {plots_dir}")
else:
    print(f"Directory already exists: {plots_dir}")

# Load the model
model = load_model('models/stock_price_model.keras')

# Print the model's input shape
print(f"Model input shape: {model.input_shape}")

# Load and preprocess the data
raw_data = pd.read_csv('data/processed_data.csv')
X, y, scaler = preprocess_data(raw_data)

# Print the shape of the input data
print(f"Shape of input data X: {X.shape}")

# Make predictions
predicted_stock_price = model.predict(X)
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

# Plot the results
plt.figure(figsize=(14, 5))
plt.plot(scaler.inverse_transform(y.reshape(-1, 1)), color='blue', label='Actual Stock Price')
plt.plot(predicted_stock_price, color='red', label='Predicted Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()

# Save the plot
plot_path = os.path.join(plots_dir, 'stock_price_prediction.png')
plt.savefig(plot_path)
print(f"Plot saved successfully at: {plot_path}")
plt.show()
