from flask import Blueprint, render_template, request
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import time
import requests
import warnings
from dotenv import load_dotenv
import os
import hashlib
load_dotenv()
warnings.filterwarnings('ignore')

predict_bp = Blueprint('predict', __name__)

# Alpha Vantage API Key
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

# Cache with expiration tracking
_cache = {}
CACHE_EXPIRY = 3600  # 1 hour in seconds

def get_cache_key(ticker, outputsize):
    """Generate cache key"""
    return hashlib.md5(f"{ticker}_{outputsize}".encode()).hexdigest()

def fetch_stock_data_alpha(ticker, outputsize='full'):
    """Fetch daily stock data from Alpha Vantage with proper caching"""
    cache_key = get_cache_key(ticker, outputsize)
    current_time = time.time()
    
    # Check cache
    if cache_key in _cache:
        cached_data, cached_time = _cache[cache_key]
        if current_time - cached_time < CACHE_EXPIRY:
            return cached_data.copy()
    
    try:
        url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&outputsize={outputsize}&apikey={ALPHA_VANTAGE_API_KEY}'
        response = requests.get(url, timeout=15)
        data = response.json()
        
        if "Time Series (Daily)" not in data:
            if "Note" in data:
                raise Exception("API rate limit reached. Please wait a minute and try again.")
            elif "Error Message" in data:
                raise Exception(f"Invalid ticker symbol: {ticker}")
            elif "Information" in data:
                raise Exception("API call frequency exceeded. Please wait and try again.")
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
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Store in cache
        _cache[cache_key] = (df, current_time)
        
        return df
    except requests.exceptions.Timeout:
        raise Exception(f"Request timeout for {ticker}. Please try again.")
    except requests.exceptions.RequestException as e:
        raise Exception(f"Network error: {str(e)}")
    except Exception as e:
        raise Exception(f"Error fetching data: {str(e)}")

def fetch_stock_data_with_retry(ticker, max_retries=3):
    """Fetch stock data with retry logic"""
    for attempt in range(max_retries):
        try:
            # Check if data is in cache first (no delay needed)
            cache_key = get_cache_key(ticker, 'full')
            if cache_key in _cache:
                cached_data, cached_time = _cache[cache_key]
                if time.time() - cached_time < CACHE_EXPIRY:
                    return cached_data.copy()
            
            # Not in cache, add delay for rate limiting
            if attempt > 0:
                wait_time = (2 ** attempt) * 3  # Exponential backoff: 3, 6, 12 seconds
                time.sleep(wait_time)
            
            data = fetch_stock_data_alpha(ticker, outputsize='full')
            if data.empty:
                raise ValueError(f"No data found for ticker {ticker}")
            return data
        except Exception as e:
            if attempt < max_retries - 1:
                continue
            else:
                raise Exception(f"Failed after {max_retries} attempts: {str(e)}")

@predict_bp.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template("home.html")
    
    ticker = request.form.get("ticker", "AAPL").upper().strip()
    
    # Validate ticker input
    if not ticker or len(ticker) > 10 or not ticker.replace('.', '').replace('-', '').isalnum():
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
        
        if data.empty:
            return render_template("home.html", 
                error=f"No data available for ticker {ticker}")
        
        # Filter to last 2 years for faster processing
        two_years_ago = datetime.today() - timedelta(days=730)
        data = data[data.index >= two_years_ago]

        # Check if we have enough data
        if len(data) < 100:
            return render_template("home.html", 
                error=f"Not enough historical data for {ticker}. Need at least 100 days of data. Found: {len(data)} days.")

        # Remove any NaN values
        data = data.dropna()
        
        if len(data) < 100:
            return render_template("home.html", 
                error=f"Not enough valid data after cleaning for {ticker}.")

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
        
        if len(X) == 0 or len(X) < 50:
            return render_template("home.html", 
                error=f"Not enough data to train the model. Need at least {time_step + 50} days. Try a different ticker or date range.")
        
        X = X.reshape(X.shape[0], X.shape[1], 1)
        
        # Train-test split
        split = int(len(X) * 0.8)
        if split < 20:
            return render_template("home.html", 
                error="Insufficient data for training. Please try a ticker with more historical data.")
        
        X_train, y_train = X[:split], y[:split]
        X_test, y_test = X[split:], y[split:]

        # Build and train LSTM model
        try:
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(1)
            ])
            model.compile(optimizer="adam", loss="mean_squared_error")
            
            # Train with validation split and early stopping patience
            model.fit(X_train, y_train, epochs=15, batch_size=32, verbose=0, validation_split=0.1)
        except Exception as e:
            return render_template("home.html", 
                error=f"Error training model: {str(e)}")

        # Make predictions on test set
        predictions = model.predict(X_test, verbose=0)
        predictions = scaler.inverse_transform(predictions)
        y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

        # Calculate accuracy metrics
        mse = np.mean((predictions - y_test_actual) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - y_test_actual))
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_test_actual - predictions) / y_test_actual)) * 100

        # Future predictions
        last_data = scaled_data[-time_step:].reshape(1, time_step, 1)
        future_preds = []
        
        for _ in range(num_days):
            next_pred = model.predict(last_data, verbose=0)
            future_preds.append(next_pred[0][0])
            # Update the sequence for next prediction
            last_data = np.append(last_data[:, 1:, :], [[next_pred[0]]], axis=1)

        future_preds = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))
        future_dates = [data.index[-1] + timedelta(days=i+1) for i in range(num_days)]

        # Get the last available date from data
        last_date = data.index[-1].strftime('%Y-%m-%d')
        current_price = float(data["Close"].iloc[-1])
        
        # Calculate prediction confidence based on volatility
        volatility = data["Close"].pct_change().std() * 100
        if volatility < 2:
            confidence = "High"
        elif volatility < 4:
            confidence = "Medium"
        else:
            confidence = "Low"

        return render_template("home.html", 
            ticker=ticker,
            current_price=round(current_price, 2),
            last_date=last_date,
            actual=y_test_actual.flatten().tolist(),
            predicted=predictions.flatten().tolist(),
            dates=data.index[-len(y_test_actual):].strftime('%Y-%m-%d').tolist(),
            future_preds=future_preds.flatten().tolist(),
            future_dates=[d.strftime('%Y-%m-%d') for d in future_dates],
            rmse=round(rmse, 2),
            mae=round(mae, 2),
            mape=round(mape, 2),
            confidence=confidence,
            data_points=len(data)
        )
        
    except ValueError as ve:
        return render_template("home.html", error=f"Invalid ticker or data: {str(ve)}")
    except Exception as e:
        error_msg = str(e)
        if "rate limit" in error_msg.lower() or "frequency" in error_msg.lower():
            return render_template("home.html", 
                error="Alpha Vantage rate limit reached (25 requests/day, 5 requests/minute). Please try again later.")
        return render_template("home.html", error=f"Error: {error_msg}")