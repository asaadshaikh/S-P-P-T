from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def create_lstm_model(input_shape, num_layers=2, num_neurons=50, dropout_rate=0.2):
    model = Sequential()
    
    for _ in range(num_layers):
        model.add(LSTM(num_neurons, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(dropout_rate))
    
    model.add(LSTM(num_neurons))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))  # Output layer for regression
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model