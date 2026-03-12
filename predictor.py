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

def fetch_stock_data(ticker, period="1y"):  # 1 year is enough, faster to train
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
    logger.info(f"Fetched {len(df)} days of data for {ticker}")
    return df

def create_lstm_model(input_shape):
    # Smaller/faster model: 1 LSTM layer, fewer units
    model = Sequential([
        LSTM(32, return_sequences=False, input_shape=input_shape),
        Dropout(0.2),
        Dense(16),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
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
            return render_template("home.html", error="Number of days must be between 1 and 365")
    except ValueError:
        return render_template("home.html", error="Invalid number of days")

    try:
        logger.info(f"Starting prediction for {ticker}")
        data = fetch_stock_data(ticker, period="1y")

        if len(data) < 60:
            return render_template("home.html",
                error=f"Insufficient data for {ticker}: {len(data)} days found, need 60+.")

        data = data.dropna()
        close_prices = data["Close"].values.astype(float).reshape(-1, 1)

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(close_prices)

        # Smaller time_step = fewer sequences = faster training
        time_step = 30

        def create_dataset(arr, time_step):
            X, y = [], []
            for i in range(len(arr) - time_step - 1):
                X.append(arr[i:(i + time_step), 0])
                y.append(arr[i + time_step, 0])
            return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

        X, y = create_dataset(scaled_data, time_step)

        if len(X) < 30:
            return render_template("home.html",
                error=f"Not enough data to train. Found {len(data)} days.")

        X = X.reshape(X.shape[0], X.shape[1], 1)
        split = int(len(X) * 0.8)
        X_train, y_train = X[:split], y[:split]
        X_test,  y_test  = X[split:], y[split:]

        logger.info(f"Training LSTM model on {len(X_train)} samples...")
        model = create_lstm_model((time_step, 1))

        # Aggressive early stopping + fewer epochs = fast
        early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        model.fit(
            X_train, y_train,
            epochs=15,          # max 15 epochs, usually stops at 5-8
            batch_size=16,      # smaller batch = faster per epoch
            verbose=0,
            validation_split=0.1,
            callbacks=[early_stop]
        )
        logger.info("Training complete.")

        predictions   = scaler.inverse_transform(model.predict(X_test, verbose=0).reshape(-1, 1))
        y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

        rmse = float(np.sqrt(np.mean((predictions - y_test_actual) ** 2)))
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
        volatility    = float(data["Close"].pct_change().std() * 100)

        if volatility < 2:
            confidence, confidence_score = "High", 85
        elif volatility < 4:
            confidence, confidence_score = "Medium", 70
        else:
            confidence, confidence_score = "Low", 55

        logger.info(f"Prediction complete for {ticker}.")

        return render_template("home.html",
            ticker=ticker,
            current_price=round(current_price, 2),
            last_date=data.index[-1].strftime('%Y-%m-%d'),
            actual=[round(float(v), 2) for v in y_test_actual.flatten()],
            predicted=[round(float(v), 2) for v in predictions.flatten()],
            dates=data.index[-len(y_test_actual):].strftime('%Y-%m-%d').tolist(),
            future_preds=[round(float(v), 2) for v in future_preds_inv.flatten()],
            future_dates=[d.strftime('%Y-%m-%d') for d in future_dates],
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