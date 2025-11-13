from flask import Blueprint, render_template, request, flash
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import time
from functools import lru_cache
import requests
import warnings
import os
load_dotenv()
warnings.filterwarnings('ignore')

predict_bp = Blueprint('predict', __name__)

# Alpha Vantage API Key
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

# Cache for stock data to prevent repeated API calls
@lru_cache(maxsize=100)
def fetch_stock_data_alpha(ticker, outputsize='full'):
    """Fetch daily stock data from Alpha Vantage"""
    try:
        url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&outputsize={outputsize}&apikey={ALPHA_VANTAGE_API_KEY}'
        response = requests.get(url, timeout=10)
        data = response.json()
        
        if "Time Series (Daily)" not in data:
            if "Note" in data:
                raise Exception("API rate limit reached. Please wait a minute and try again.")
            elif "Error Message" in data:
                raise Exception(f"Invalid ticker symbol: {ticker}")
            else:
                raise Exception(f"Unable to fetch data: {data.get('Information', 'Unknown error')}")
        
        # Convert to DataFrame
        time_series = data["Time Series (Daily)"]
        df = pd.DataFrame.from_dict(time_series, orient='index')
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        # Convert to float
        for col in df.columns:
            df[col] = pd.to_numeric(df[col])
        
        return df
    except Exception as e:
        raise Exception(f"Error fetching data: {str(e)}")

def fetch_stock_data_with_retry(ticker, max_retries=3):
    """Fetch stock data with retry logic"""
    for attempt in range(max_retries):
        try:
            time.sleep(1)  # Rate limiting delay
            data = fetch_stock_data_alpha(ticker, outputsize='full')
            if data.empty:
                raise ValueError(f"No data found for ticker {ticker}")
            return data
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) * 2
                time.sleep(wait_time)
                continue
            else:
                raise Exception(f"Failed after {max_retries} attempts. Please try again in a few minutes.")

@predict_bp.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template("home.html")
    
    ticker = request.form.get("ticker", "AAPL").upper().strip()
    
    # Validate ticker input
    if not ticker or len(ticker) > 10:
        return render_template("home.html", error="Please enter a valid stock ticker (e.g., AAPL, MSFT)")
    
    try:
        num_days = int(request.form.get("num_days", 30))
        if num_days < 1 or num_days > 365:
            return render_template("home.html", error="Number of days must be between 1 and 365")
    except ValueError:
        return render_template("home.html", error="Invalid number of days")

    try:
        # Fetch stock data from Alpha Vantage
        data = fetch_stock_data_with_retry(ticker)
        
        # Filter to last 2 years for faster processing
        two_years_ago = datetime.today() - timedelta(days=730)
        data = data[data.index >= two_years_ago]

        # Check if we have enough data
        if len(data) < 100:
            return render_template("home.html", 
                error=f"Not enough historical data for {ticker}. Need at least 100 days of data.")

        # Normalize data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data[["Close"]])

        def create_dataset(data, time_step=60):
            X, y = [], []
            for i in range(len(data) - time_step - 1):
                X.append(data[i:(i + time_step), 0])
                y.append(data[i + time_step, 0])
            return np.array(X), np.array(y)

        time_step = 60
        X, y = create_dataset(scaled_data, time_step)
        
        if len(X) == 0:
            return render_template("home.html", 
                error="Not enough data to train the model. Try a different ticker.")
        
        X = X.reshape(X.shape[0], X.shape[1], 1)
        
        # Train-test split
        split = int(len(X) * 0.8)
        X_train, y_train = X[:split], y[:split]
        X_test, y_test = X[split:], y[split:]

        # Build and train LSTM model
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer="adam", loss="mean_squared_error")
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

        # Make predictions on test set
        predictions = model.predict(X_test, verbose=0)
        predictions = scaler.inverse_transform(predictions)
        y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

        # Calculate accuracy metrics
        mse = np.mean((predictions - y_test_actual) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - y_test_actual))

        # Future predictions
        last_data = scaled_data[-time_step:].reshape(1, time_step, 1)
        future_preds = []
        
        for _ in range(num_days):
            next_pred = model.predict(last_data, verbose=0)
            future_preds.append(next_pred[0][0])
            last_data = np.append(last_data[:, 1:, :], [[next_pred[0]]], axis=1)

        future_preds = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))
        future_dates = [data.index[-1] + timedelta(days=i+1) for i in range(num_days)]

        # Get the last available date from data
        last_date = data.index[-1].strftime('%Y-%m-%d')
        current_price = float(data["Close"].iloc[-1])

        return render_template("home.html", 
            ticker=ticker,
            current_price=current_price,
            last_date=last_date,
            actual=y_test_actual.flatten().tolist(),
            predicted=predictions.flatten().tolist(),
            dates=data.index[-len(y_test_actual):].strftime('%Y-%m-%d').tolist(),
            future_preds=future_preds.flatten().tolist(),
            future_dates=[d.strftime('%Y-%m-%d') for d in future_dates],
            rmse=round(rmse, 2),
            mae=round(mae, 2)
        )
        
    except ValueError as ve:
        return render_template("home.html", error=f"Invalid ticker or data: {str(ve)}")
    except Exception as e:
        error_msg = str(e)
        if "rate limit" in error_msg.lower():
            return render_template("home.html", 
                error="Alpha Vantage rate limit reached (25 requests/day). Please try again later.")
        return render_template("home.html", error=f"Error: {error_msg}")