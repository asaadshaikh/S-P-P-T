
from flask import Flask, render_template, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = load_model('d:/Projects/FirstProgram/stock-price-prediction/models/stock_price_model.keras')

# Load preprocessed data
data = pd.read_csv('d:/Projects/FirstProgram/stock-price-prediction/data/processed_data.csv')

# Prepare data for LSTM
def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the request
    input_data = request.json['data']
    input_data = np.array(input_data).reshape(1, -1, 1)

    # Make prediction
    prediction = model.predict(input_data)
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
