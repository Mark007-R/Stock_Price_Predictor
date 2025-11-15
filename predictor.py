from flask import Blueprint, render_template, request, jsonify
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import time
import requests
import warnings
from dotenv import load_dotenv
import os
import hashlib
import logging

load_dotenv()
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

predict_bp = Blueprint('predict', __name__)

# Alpha Vantage API Key
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

# Enhanced caching system
_cache = {}
CACHE_EXPIRY = 3600  # 1 hour

def get_cache_key(ticker, outputsize):
    """Generate unique cache key"""
    return hashlib.md5(f"{ticker}_{outputsize}".encode()).hexdigest()

def validate_api_key():
    """Validate that API key is configured"""
    if not ALPHA_VANTAGE_API_KEY or ALPHA_VANTAGE_API_KEY == "":
        raise Exception("Alpha Vantage API key not configured. Please add ALPHA_VANTAGE_API_KEY to your .env file")
    return True

def fetch_stock_data_alpha(ticker, outputsize='full'):
    """Fetch daily stock data from Alpha Vantage with robust error handling"""
    validate_api_key()
    
    cache_key = get_cache_key(ticker, outputsize)
    current_time = time.time()
    
    # Check cache first
    if cache_key in _cache:
        cached_data, cached_time = _cache[cache_key]
        if current_time - cached_time < CACHE_EXPIRY:
            logger.info(f"Returning cached data for {ticker}")
            return cached_data.copy()
    
    try:
        url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&outputsize={outputsize}&apikey={ALPHA_VANTAGE_API_KEY}'
        logger.info(f"Fetching data for {ticker} from Alpha Vantage...")
        
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        # Comprehensive error checking
        if "Time Series (Daily)" not in data:
            if "Note" in data:
                raise Exception("API rate limit reached (5 calls/min, 25 calls/day). Please wait and try again.")
            elif "Error Message" in data:
                raise Exception(f"Invalid ticker symbol: {ticker}. Please verify the symbol is correct.")
            elif "Information" in data:
                raise Exception("API call frequency exceeded. Please wait 1 minute and try again.")
            else:
                error_msg = data.get('Information', data.get('Note', 'Unknown error'))
                raise Exception(f"Unable to fetch data: {error_msg}")
        
        # Convert to DataFrame
        time_series = data["Time Series (Daily)"]
        df = pd.DataFrame.from_dict(time_series, orient='index')
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        # Convert to numeric and handle any conversion errors
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove any rows with NaN values
        df = df.dropna()
        
        if df.empty:
            raise Exception(f"No valid data available for {ticker}")
        
        # Store in cache
        _cache[cache_key] = (df, current_time)
        logger.info(f"Successfully fetched {len(df)} days of data for {ticker}")
        
        return df
        
    except requests.exceptions.Timeout:
        raise Exception(f"Request timeout for {ticker}. The API server is not responding. Please try again.")
    except requests.exceptions.RequestException as e:
        raise Exception(f"Network error: Unable to connect to Alpha Vantage. Check your internet connection.")
    except Exception as e:
        raise e

def fetch_stock_data_with_retry(ticker, max_retries=3):
    """Fetch stock data with exponential backoff retry logic"""
    for attempt in range(max_retries):
        try:
            # Check cache first (no delay needed)
            cache_key = get_cache_key(ticker, 'full')
            if cache_key in _cache:
                cached_data, cached_time = _cache[cache_key]
                if time.time() - cached_time < CACHE_EXPIRY:
                    return cached_data.copy()
            
            # Add delay only for retries (not first attempt)
            if attempt > 0:
                wait_time = (2 ** attempt) * 5  # 5, 10, 20 seconds
                logger.info(f"Retry attempt {attempt + 1} after {wait_time}s delay...")
                time.sleep(wait_time)
            
            data = fetch_stock_data_alpha(ticker, outputsize='full')
            
            if data.empty:
                raise ValueError(f"No data found for ticker {ticker}")
                
            return data
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Attempt {attempt + 1} failed: {error_msg}")
            
            if attempt < max_retries - 1:
                # Don't retry on certain errors
                if "Invalid ticker" in error_msg or "API key" in error_msg:
                    raise e
                continue
            else:
                raise Exception(f"Failed after {max_retries} attempts: {error_msg}")

def create_lstm_model(input_shape):
    """Create LSTM model with improved architecture"""
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
    
    # Validate ticker input
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
        # Fetch stock data
        logger.info(f"Starting prediction for {ticker}")
        data = fetch_stock_data_with_retry(ticker)
        
        if data.empty:
            return render_template("home.html", 
                error=f"No data available for ticker {ticker}")
        
        # Use last 2 years for training (faster and usually sufficient)
        two_years_ago = datetime.today() - timedelta(days=730)
        data = data[data.index >= two_years_ago]

        # Validate sufficient data
        if len(data) < 100:
            return render_template("home.html", 
                error=f"Insufficient historical data for {ticker}. Found {len(data)} days, need at least 100 days.")

        # Remove any remaining NaN values
        data = data.dropna()
        
        if len(data) < 100:
            return render_template("home.html", 
                error=f"Insufficient valid data after cleaning for {ticker}.")

        # Prepare data for LSTM
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data[["Close"]])

        def create_dataset(data, time_step=60):
            X, y = [], []
            for i in range(len(data) - time_step - 1):
                X.append(data[i:(i + time_step), 0])
                y.append(data[i + time_step, 0])
            return np.array(X), np.array(y)

        time_step = min(60, len(scaled_data) // 4)  # Adaptive time step
        X, y = create_dataset(scaled_data, time_step)
        
        if len(X) < 50:
            return render_template("home.html", 
                error=f"Not enough data to train model. Need at least {time_step + 50} days. Found: {len(data)} days.")
        
        X = X.reshape(X.shape[0], X.shape[1], 1)
        
        # Train-test split (80-20)
        split = int(len(X) * 0.8)
        if split < 20:
            return render_template("home.html", 
                error="Insufficient data for training. Please try a ticker with more historical data.")
        
        X_train, y_train = X[:split], y[:split]
        X_test, y_test = X[split:], y[split:]

        # Build and train LSTM model
        logger.info(f"Training LSTM model on {len(X_train)} samples...")
        
        try:
            model = create_lstm_model((X_train.shape[1], 1))
            
            # Early stopping to prevent overfitting
            early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            
            # Train with validation
            history = model.fit(
                X_train, y_train,
                epochs=30,
                batch_size=32,
                verbose=0,
                validation_split=0.1,
                callbacks=[early_stop]
            )
            
            logger.info(f"Model training completed. Final loss: {history.history['loss'][-1]:.6f}")
            
        except Exception as e:
            logger.error(f"Model training error: {str(e)}")
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
        mape = np.mean(np.abs((y_test_actual - predictions) / y_test_actual)) * 100

        # Future predictions
        last_data = scaled_data[-time_step:].reshape(1, time_step, 1)
        future_preds = []
        
        logger.info(f"Generating {num_days} days of future predictions...")
        
        for _ in range(num_days):
            next_pred = model.predict(last_data, verbose=0)
            future_preds.append(next_pred[0][0])
            # Update sequence for next prediction
            last_data = np.append(last_data[:, 1:, :], [[next_pred[0]]], axis=1)

        future_preds = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))
        future_dates = [data.index[-1] + timedelta(days=i+1) for i in range(num_days)]

        # Get current stats
        last_date = data.index[-1].strftime('%Y-%m-%d')
        current_price = float(data["Close"].iloc[-1])
        
        # Calculate prediction confidence
        volatility = data["Close"].pct_change().std() * 100
        if volatility < 2:
            confidence = "High"
            confidence_score = 85
        elif volatility < 4:
            confidence = "Medium"
            confidence_score = 70
        else:
            confidence = "Low"
            confidence_score = 55

        # Prepare response data
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
            confidence_score=confidence_score,
            data_points=len(data),
            volatility=round(volatility, 2)
        )
        
    except ValueError as ve:
        logger.error(f"ValueError: {str(ve)}")
        return render_template("home.html", error=f"Invalid ticker or data: {str(ve)}")
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Prediction error: {error_msg}")
        
        if "rate limit" in error_msg.lower() or "frequency" in error_msg.lower():
            return render_template("home.html", 
                error="Alpha Vantage rate limit reached (5 calls/min, 25 calls/day). Please wait and try again.")
        elif "API key" in error_msg:
            return render_template("home.html",
                error="API key not configured. Please add ALPHA_VANTAGE_API_KEY to your .env file.")
        
        return render_template("home.html", error=f"Error: {error_msg}")