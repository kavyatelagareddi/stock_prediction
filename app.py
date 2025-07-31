from flask import Flask, render_template, request, redirect, url_for
import os
import numpy as np
import pandas as pd
import yfinance as yf
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error
import io
import base64

app = Flask(__name__, template_folder='myfiles')

# Available companies and their tickers
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

@app.route('/')
def home():
    return render_template('home.html', companies=list(available_companies.keys()))

@app.route('/select_stock', methods=['GET', 'POST'])
def select_stock():
    if request.method == 'POST':
        company_name = request.form['company_name']
        if company_name not in available_companies:
            return "Invalid company selected", 404
        return redirect(url_for('predict', company_name=company_name))
    return render_template('select_stock.html', companies=list(available_companies.keys()))

@app.route('/predict/<company_name>')
def predict(company_name):
    if company_name not in available_companies:
        return "Invalid company selected", 404

    ticker = available_companies[company_name]
    
    df = yf.download(ticker, start='2015-01-01', end='2024-12-31')
    close_prices = df['Close'].values.reshape(-1, 1)
    dates = df.index

    scaler = joblib.load(f"models/{company_name}/scaler.pkl")
    data_scaled = scaler.transform(close_prices)

    X, y = create_dataset(data_scaled)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    lstm_model = load_model(f"models/{company_name}/lstm_model.h5")
    gru_model = load_model(f"models/{company_name}/gru_model.h5")

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

    last_60_days = data_scaled[-60:]
    last_60_days = last_60_days.reshape(1, 60, 1)

    lstm_next = lstm_model.predict(last_60_days)
    gru_next = gru_model.predict(last_60_days)

    lstm_next_price = scaler.inverse_transform(lstm_next)[0][0]
    gru_next_price = scaler.inverse_transform(gru_next)[0][0]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(dates, close_prices, label='Actual Close Price', color='blue')
    ax.set_title(f"{company_name} Stock Price Trend")
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (INR)')
    ax.legend()
    ax.grid()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)

    return render_template('company.html',
                           company_name=company_name,
                           lstm_rmse=f"{lstm_rmse:.4f}",
                           lstm_accuracy=f"{lstm_accuracy:.2f}",
                           gru_rmse=f"{gru_rmse:.4f}",
                           gru_accuracy=f"{gru_accuracy:.2f}",
                           lstm_next_price=f"{lstm_next_price:.2f}",
                           gru_next_price=f"{gru_next_price:.2f}",
                           graph=image_base64)

if __name__ == "__main__":
    app.run(debug=True)


