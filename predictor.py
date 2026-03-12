import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from flask import Blueprint, render_template, request
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import yfinance as yf
import warnings
import logging
import time

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

predict_bp = Blueprint('predict', __name__)

_cache = {}
CACHE_EXPIRY = 3600

def fetch_stock_data(ticker, period="2y"):
    cache_key = f"{ticker}_{period}"
    current_time = time.time()

    if cache_key in _cache:
        cached_data, cached_time = _cache[cache_key]
        if current_time - cached_time < CACHE_EXPIRY:
            logger.info(f"Returning cached data for {ticker}")
            return cached_data.copy()

    logger.info(f"Fetching data for {ticker} from yfinance...")
    df = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=True)

    if df.empty:
        raise Exception(f"No data found for ticker '{ticker}'. Please verify the symbol is correct.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    for col in df.columns:
        df[col] = df[col].astype(float)

    _cache[cache_key] = (df, current_time)
    logger.info(f"Successfully fetched {len(df)} days of data for {ticker}")
    return df

def create_lstm_model(input_shape):
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(100, return_sequences=True),
        Dropout(0.3),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

@predict_bp.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template("home.html")

    ticker = request.form.get("ticker", "AAPL").upper().strip()

    if not ticker or len(ticker) > 10 or not ticker.replace('.', '').replace('-', '').isalnum():
        return render_template("home.html",
            error="Please enter a valid stock ticker (e.g., AAPL, MSFT, GOOGL)")

    try:
        num_days = int(request.form.get("num_days", 30))
        if num_days < 1 or num_days > 365:
            return render_template("home.html",
                error="Number of days must be between 1 and 365")
    except ValueError:
        return render_template("home.html", error="Invalid number of days")

    try:
        logger.info(f"Starting prediction for {ticker}")
        data = fetch_stock_data(ticker, period="2y")

        if len(data) < 100:
            return render_template("home.html",
                error=f"Insufficient historical data for {ticker}. Found {len(data)} days, need at least 100.")

        data = data.dropna()

        # Extract close as clean numpy array
        close_prices = data["Close"].values.astype(float).reshape(-1, 1)

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(close_prices)

        def create_dataset(arr, time_step=60):
            X, y = [], []
            for i in range(len(arr) - time_step - 1):
                X.append(arr[i:(i + time_step), 0])
                y.append(arr[i + time_step, 0])
            return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

        time_step = min(60, len(scaled_data) // 4)
        X, y = create_dataset(scaled_data, time_step)

        if len(X) < 50:
            return render_template("home.html",
                error=f"Not enough data to train model. Found: {len(data)} days.")

        X = X.reshape(X.shape[0], X.shape[1], 1)

        split = int(len(X) * 0.8)
        X_train, y_train = X[:split], y[:split]
        X_test,  y_test  = X[split:], y[split:]

        logger.info(f"Training LSTM model on {len(X_train)} samples...")
        model = create_lstm_model((X_train.shape[1], 1))
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        history = model.fit(
            X_train, y_train,
            epochs=30,
            batch_size=32,
            verbose=0,
            validation_split=0.1,
            callbacks=[early_stop]
        )
        logger.info(f"Training done. Final loss: {history.history['loss'][-1]:.6f}")

        # Test predictions
        predictions   = scaler.inverse_transform(model.predict(X_test, verbose=0).reshape(-1, 1))
        y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

        # Metrics as plain Python floats
        mse  = float(np.mean((predictions - y_test_actual) ** 2))
        rmse = float(np.sqrt(mse))
        mae  = float(np.mean(np.abs(predictions - y_test_actual)))
        mape = float(np.mean(np.abs((y_test_actual - predictions) / (y_test_actual + 1e-8))) * 100)

        # Future predictions
        last_seq = scaled_data[-time_step:].reshape(1, time_step, 1).astype(np.float32)
        future_scaled = []
        for _ in range(num_days):
            next_val = float(model.predict(last_seq, verbose=0)[0][0])
            future_scaled.append(next_val)
            last_seq = np.append(last_seq[:, 1:, :], [[[next_val]]], axis=1).astype(np.float32)

        future_preds_inv = scaler.inverse_transform(
            np.array(future_scaled, dtype=np.float32).reshape(-1, 1)
        )
        future_dates = [data.index[-1] + timedelta(days=i + 1) for i in range(num_days)]

        current_price = float(data["Close"].iloc[-1])
        last_date     = data.index[-1].strftime('%Y-%m-%d')
        volatility    = float(data["Close"].pct_change().std() * 100)

        if volatility < 2:
            confidence, confidence_score = "High", 85
        elif volatility < 4:
            confidence, confidence_score = "Medium", 70
        else:
            confidence, confidence_score = "Low", 55

        # All template values as plain Python lists/floats
        actual_list       = [round(float(v), 2) for v in y_test_actual.flatten()]
        predicted_list    = [round(float(v), 2) for v in predictions.flatten()]
        future_preds_list = [round(float(v), 2) for v in future_preds_inv.flatten()]
        dates_list        = data.index[-len(y_test_actual):].strftime('%Y-%m-%d').tolist()
        future_dates_list = [d.strftime('%Y-%m-%d') for d in future_dates]

        logger.info(f"Prediction complete for {ticker}. Returning {len(future_preds_list)} future points.")

        return render_template("home.html",
            ticker=ticker,
            current_price=round(current_price, 2),
            last_date=last_date,
            actual=actual_list,
            predicted=predicted_list,
            dates=dates_list,
            future_preds=future_preds_list,
            future_dates=future_dates_list,
            rmse=round(rmse, 2),
            mae=round(mae, 2),
            mape=round(mape, 2),
            confidence=confidence,
            confidence_score=confidence_score,
            data_points=len(data),
            volatility=round(volatility, 2)
        )

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        return render_template("home.html", error=f"Error: {str(e)}")