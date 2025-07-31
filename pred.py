import numpy as np
import pandas as pd
import yfinance as yf
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error

available_companies = {
    'TCS': 'TCS.NS',
    'Infosys': 'INFY.NS',
    'Reliance': 'RELIANCE.NS',
    'HDFC Bank': 'HDFCBANK.NS',
    'ICICI Bank': 'ICICIBANK.NS',
    'Wipro': 'WIPRO.NS',
    'HCL Tech': 'HCLTECH.NS',
    'SBI': 'SBIN.NS',
    'Bharti Airtel': 'BHARTIARTL.NS',
    'L&T': 'LT.NS',
    'Bajaj Finance': 'BAJFINANCE.NS',
    'Axis Bank': 'AXISBANK.NS',
    'Maruti Suzuki': 'MARUTI.NS',
    'Asian Paints': 'ASIANPAINT.NS',
    'Titan': 'TITAN.NS',
    'Ultratech Cement': 'ULTRACEMCO.NS',
    'Nestle India': 'NESTLEIND.NS',
    'Tech Mahindra': 'TECHM.NS',
    'Power Grid': 'POWERGRID.NS',
    'Coal India': 'COALINDIA.NS',
    'ONGC': 'ONGC.NS',
    'JSW Steel': 'JSWSTEEL.NS',
    'NTPC': 'NTPC.NS',
    'Hindalco': 'HINDALCO.NS',
    'Sun Pharma': 'SUNPHARMA.NS'
}

def create_dataset(data, time_step=60):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

def predict_stock(company_input):
    ticker = available_companies[company_input]
    
    df = yf.download(ticker, start='2015-01-01', end='2024-12-31')
    close_prices = df['Close'].values.reshape(-1, 1)
    dates = df.index

    scaler = joblib.load(f"models/{company_input}/scaler.pkl")
    data_scaled = scaler.transform(close_prices)

    X, y = create_dataset(data_scaled)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    lstm_model = load_model(f"models/{company_input}/lstm_model.h5")
    gru_model = load_model(f"models/{company_input}/gru_model.h5")

    lstm_pred = lstm_model.predict(X_test)
    gru_pred = gru_model.predict(X_test)

    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
    lstm_pred_actual = scaler.inverse_transform(lstm_pred)
    gru_pred_actual = scaler.inverse_transform(gru_pred)

    lstm_rmse = np.sqrt(mean_squared_error(y_test_actual, lstm_pred_actual))
    gru_rmse = np.sqrt(mean_squared_error(y_test_actual, gru_pred_actual))

    lstm_mape = np.mean(np.abs((y_test_actual - lstm_pred_actual) / y_test_actual)) * 100
    gru_mape = np.mean(np.abs((y_test_actual - gru_pred_actual) / y_test_actual)) * 100

    lstm_accuracy = 100 - lstm_mape
    gru_accuracy = 100 - gru_mape


    plt.figure(figsize=(12, 6))
    plt.plot(dates, close_prices, label='Actual Close Price', color='blue')
    plt.title(f"{company_input} Stock Price Trend")
    plt.xlabel('Date')
    plt.ylabel('Price (INR)')
    plt.legend()
    plt.grid()
    plt.show()

    last_60_days = data_scaled[-60:]
    last_60_days = last_60_days.reshape(1, 60, 1)

    lstm_next = lstm_model.predict(last_60_days)
    gru_next = gru_model.predict(last_60_days)

    lstm_next_price = scaler.inverse_transform(lstm_next)[0][0]
    gru_next_price = scaler.inverse_transform(gru_next)[0][0]

    print(f"\nModel Evaluation:")
    print(f"LSTM RMSE: {lstm_rmse:.4f}")
    print(f"LSTM Accuracy: {lstm_accuracy:.2f}%")
    print(f"GRU RMSE: {gru_rmse:.4f}")
    print(f"GRU Accuracy: {gru_accuracy:.2f}%")

    print(f"\nPredicted Next Day Close Price:")
    print(f"LSTM Model: ₹{lstm_next_price:.2f}")
    print(f"GRU Model: ₹{gru_next_price:.2f}")

if __name__ == "__main__":
    print("\nAvailable Companies:")
    for name in available_companies.keys():
        print("-", name)
    
    company_name = input("\nEnter company name exactly: ").strip()
    
    if company_name not in available_companies:
        print("Invalid company name.")
    else:
        predict_stock(company_name)