def plot_stock_prices(data, title='Stock Prices'):
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(14, 7))
    plt.plot(data['Date'], data['Close'], label='Close Price', color='blue')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid()
    plt.show()

def plot_technical_indicators(data, indicators, title='Technical Indicators'):
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(14, 7))
    for indicator in indicators:
        plt.plot(data['Date'], data[indicator], label=indicator)
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid()
    plt.show()

def calculate_rmse(actual, predicted):
    from sklearn.metrics import mean_squared_error
    import numpy as np
    
    return np.sqrt(mean_squared_error(actual, predicted))
