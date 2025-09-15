
# Stock Prediction

A stock price prediction web application built using machine learning and Flask.

## 🚀 Project Overview

This project predicts stock prices based on historical data using machine learning models such as LSTM and GRU.  
It provides a web interface where users can enter stock symbols, and view predicted stock prices along with visualizations.

---

## ⚙️ Features

- Predict future stock prices using trained ML models
- Visualize historical vs predicted stock trends
- User-friendly Flask web interface
- Fetch live stock data using `yfinance`

---

## 🧱 Technologies Used

- Python
- Flask
- TensorFlow (Keras)
- yfinance
- Pandas, NumPy, scikit-learn
- Matplotlib

---

## 🚀 How to Run

1. Clone the repository:
    ```bash
    git clone https://github.com/kavyatelagareddi/stock_prediction.git
    cd stock_prediction
    ```

2. Create and activate virtual environment:
    ```bash
    python -m venv venv
    .\venv\Scripts\activate   # Windows
    # or
    source venv/bin/activate  # macOS/Linux
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Run the Flask app:
    ```bash
    python app.py
    ```

5. Open the browser at:
    ```
    http://127.0.0.1:5000/
    ```

---

## 📁 Project Structure

stock_prediction/
├── app.py # Flask application
├── ml.ipynb # Machine Learning workflow notebook
├── pred.py # Prediction logic
├── models/ # Saved ML models
├── myfiles/ # Additional files (e.g., datasets)
├── requirements.txt # Dependencies
├── templates/ # HTML templates for Flask
└── static/ # Static files (CSS, JS)


---## 🎬 Demo

Here is how the stock prediction app looks in action:

![App Demo](https://github.com/kavyatelagareddi/stock_prediction/raw/main/assets/demo_ss.png)  
![App Demo](https://github.com/kavyatelagareddi/stock_prediction/raw/main/assets/demo2.png)  
![App Demo](https://github.com/kavyatelagareddi/stock_prediction/raw/main/assets/demo3.png)  
![App Demo](https://github.com/kavyatelagareddi/stock_prediction/raw/main/assets/demo4.png)


## ⚡ Author

Kavya Telagareddi

---

Feel free to ⭐ the repository if you find it useful!
