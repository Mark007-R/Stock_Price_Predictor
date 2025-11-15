from flask import Blueprint, render_template, request
import pandas as pd
from datetime import datetime, timedelta
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

historical_bp = Blueprint('historical', __name__)

# Alpha Vantage API Key
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

# Cache with expiration tracking
_cache = {}
CACHE_EXPIRY = 3600  # 1 hour in seconds

def get_cache_key(ticker, outputsize):
    """Generate cache key"""
    return hashlib.md5(f"{ticker}_{outputsize}".encode()).hexdigest()

def validate_api_key():
    """Validate that API key is configured"""
    if not ALPHA_VANTAGE_API_KEY or ALPHA_VANTAGE_API_KEY == "":
        raise Exception("Alpha Vantage API key not configured. Please add ALPHA_VANTAGE_API_KEY to your .env file")
    return True

def fetch_historical_data_alpha(ticker, outputsize='full'):
    """Fetch historical data from Alpha Vantage with proper caching"""
    validate_api_key()
    
    cache_key = get_cache_key(ticker, outputsize)
    current_time = time.time()
    
    # Check cache
    if cache_key in _cache:
        cached_data, cached_time = _cache[cache_key]
        if current_time - cached_time < CACHE_EXPIRY:
            logger.info(f"Returning cached historical data for {ticker}")
            return cached_data.copy()
    
    try:
        url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&outputsize={outputsize}&apikey={ALPHA_VANTAGE_API_KEY}'
        logger.info(f"Fetching historical data for {ticker}...")
        
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if "Time Series (Daily)" not in data:
            if "Note" in data:
                raise Exception("API rate limit reached (5 calls/min, 25 calls/day). Please wait and try again.")
            elif "Error Message" in data:
                raise Exception(f"Invalid ticker symbol: {ticker}")
            elif "Information" in data:
                raise Exception("API call frequency exceeded. Please wait 1 minute and try again.")
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
        
        # Remove NaN values
        df = df.dropna()
        
        # Store in cache
        _cache[cache_key] = (df, current_time)
        logger.info(f"Successfully fetched {len(df)} days of historical data for {ticker}")
        
        return df
    except requests.exceptions.Timeout:
        raise Exception(f"Request timeout for {ticker}. Please try again.")
    except requests.exceptions.RequestException as e:
        raise Exception(f"Network error: Unable to connect to Alpha Vantage.")
    except Exception as e:
        raise e

def fetch_historical_with_retry(ticker, max_retries=3):
    """Fetch historical data with retry logic"""
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
                wait_time = (2 ** attempt) * 5  # 5, 10, 20 seconds
                logger.info(f"Retry attempt {attempt + 1} after {wait_time}s delay...")
                time.sleep(wait_time)
            
            df = fetch_historical_data_alpha(ticker, outputsize='full')
            if df.empty:
                raise ValueError(f"No data found for ticker {ticker}")
            return df
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

def calculate_technical_indicators(df):
    """Calculate all technical indicators"""
    # Simple Moving Averages
    if len(df) >= 20:
        df["SMA20"] = df["Close"].rolling(window=20).mean()
    else:
        df["SMA20"] = None
        
    if len(df) >= 50:
        df["SMA50"] = df["Close"].rolling(window=50).mean()
    else:
        df["SMA50"] = None
        
    # Exponential Moving Average
    if len(df) >= 20:
        df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    else:
        df["EMA20"] = None
    
    # Daily returns
    df["Daily_Return"] = df["Close"].pct_change() * 100
    
    # RSI (Relative Strength Index)
    if len(df) >= 15:
        delta = df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, 0.0001)
        df["RSI"] = 100 - (100 / (1 + rs))
    else:
        df["RSI"] = None
    
    # Bollinger Bands
    if len(df) >= 20:
        rolling_mean = df["Close"].rolling(window=20).mean()
        rolling_std = df["Close"].rolling(window=20).std()
        df["BB_Upper"] = rolling_mean + (rolling_std * 2)
        df["BB_Lower"] = rolling_mean - (rolling_std * 2)
        df["BB_Middle"] = rolling_mean
    else:
        df["BB_Upper"] = None
        df["BB_Lower"] = None
        df["BB_Middle"] = None
    
    # MACD (Moving Average Convergence Divergence)
    if len(df) >= 26:
        ema12 = df["Close"].ewm(span=12, adjust=False).mean()
        ema26 = df["Close"].ewm(span=26, adjust=False).mean()
        df["MACD"] = ema12 - ema26
        df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    else:
        df["MACD"] = None
        df["MACD_Signal"] = None
    
    return df

@historical_bp.route('/historical', methods=["GET", "POST"])
def analyze():
    if request.method == "POST":
        ticker = request.form.get("hist_ticker", "MSFT").upper().strip()
        period = request.form.get("hist_period", "1y")
        interval = request.form.get("hist_interval", "1d")
        
        # Validate ticker
        if not ticker or len(ticker) > 10 or not ticker.replace('.', '').replace('-', '').isalnum():
            return render_template("historical.html", 
                error="Please enter a valid stock ticker (e.g., AAPL, MSFT)")
        
        # Map period to days
        period_map = {
            "1mo": 30,
            "3mo": 90,
            "6mo": 180,
            "1y": 365,
            "2y": 730,
            "5y": 1825,
            "10y": 3650,
            "ytd": (datetime.today() - datetime(datetime.today().year, 1, 1)).days,
            "max": None
        }
        
        if period not in period_map:
            return render_template("historical.html", 
                error="Invalid period selected")
        
        try:
            # Fetch data from Alpha Vantage
            logger.info(f"Fetching historical data for {ticker} ({period})")
            df = fetch_historical_with_retry(ticker)
            
            if df.empty:
                return render_template("historical.html", 
                    error=f"No data available for ticker {ticker}")
            
            # Filter based on period
            if period_map[period] is not None:
                start_date = datetime.today() - timedelta(days=period_map[period])
                df = df[df.index >= start_date]
            
            # Check if we have data after filtering
            if df.empty:
                return render_template("historical.html", 
                    error=f"No data available for the selected period ({period})")
            
            # Resample if needed
            if interval == "1wk":
                df = df.resample('W').agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum'
                }).dropna()
            elif interval == "1mo":
                df = df.resample('M').agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum'
                }).dropna()
            
            if df.empty:
                return render_template("historical.html", 
                    error="No data available after resampling for the selected period")
            
            # Calculate technical indicators
            df = calculate_technical_indicators(df)
            
            # Summary statistics
            price_change = ((df["Close"].iloc[-1] / df["Close"].iloc[0]) - 1) * 100
            
            summary = {
                "current_price": round(df["Close"].iloc[-1], 2),
                "high": round(df["High"].max(), 2),
                "low": round(df["Low"].min(), 2),
                "avg_volume": int(df["Volume"].mean()),
                "total_return": round(price_change, 2),
                "volatility": round(df["Daily_Return"].std(), 2) if len(df) > 1 else 0,
                "avg_return": round(df["Daily_Return"].mean(), 2) if len(df) > 1 else 0,
                "data_points": len(df),
                "sharpe_ratio": round((df["Daily_Return"].mean() / df["Daily_Return"].std()) * (252 ** 0.5), 2) if df["Daily_Return"].std() > 0 else 0
            }
            
            # Convert DataFrame to records for template
            df_reset = df.reset_index()
            df_reset['Date'] = df_reset['Date'].dt.strftime('%Y-%m-%d')
            
            # Round numeric columns
            numeric_cols = ['Open', 'High', 'Low', 'Close', 'SMA20', 'SMA50', 
                           'EMA20', 'Daily_Return', 'RSI', 'BB_Upper', 'BB_Lower', 
                           'BB_Middle', 'MACD', 'MACD_Signal']
            for col in numeric_cols:
                if col in df_reset.columns:
                    df_reset[col] = df_reset[col].round(2)
            
            # Convert Volume to int
            df_reset['Volume'] = df_reset['Volume'].astype(int)
            
            return render_template(
                "historical.html",
                df=df_reset.to_dict("records"),
                ticker=ticker,
                period=period,
                interval=interval,
                summary=summary
            )
            
        except ValueError as ve:
            logger.error(f"ValueError: {str(ve)}")
            return render_template("historical.html", 
                error=f"Invalid ticker or no data available: {str(ve)}")
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Historical analysis error: {error_msg}")
            
            if "rate limit" in error_msg.lower() or "frequency" in error_msg.lower():
                return render_template("historical.html", 
                    error="Alpha Vantage rate limit reached (5 calls/min, 25 calls/day). Please wait and try again.")
            elif "API key" in error_msg:
                return render_template("historical.html",
                    error="API key not configured. Please add ALPHA_VANTAGE_API_KEY to your .env file.")
            
            return render_template("historical.html", error=f"Error: {error_msg}")
    
    return render_template("historical.html")