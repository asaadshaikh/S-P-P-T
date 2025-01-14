# Stock Price Prediction

This project predicts stock prices using an LSTM model. It includes data preprocessing, model training, evaluation, and a Flask web application for deployment.

## Project Structure

```
stock-price-prediction/
├── app/
│   ├── app.py
│   └── templates/
│       └── index.html
├── data/
│   ├── raw_data.csv
│   └── processed_data.csv
├── models/
│   └── stock_price_model.keras
├── plots/
│   └── actual_vs_predicted.png
├── src/
│   ├── data_acquisition.py
│   ├── data_preprocessing.py
│   ├── train.py
│   └── visualization.py
└── README.md
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/stock-price-prediction.git
   cd stock-price-prediction
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Data Preprocessing
Run the data preprocessing script:
```bash
python src/data_preprocessing.py
```

### Model Training
Train the LSTM model:
```bash
python src/train.py
```

### Visualization
Generate plots of actual vs predicted stock prices:
```bash
python src/visualization.py
```

### Deployment
Run the Flask web application:
```bash
python app/app.py
```

Open your browser and navigate to `http://localhost:5000` to use the web interface.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
