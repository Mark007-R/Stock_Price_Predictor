from flask import Blueprint, render_template, request
import pandas as pd
from datetime import datetime, timedelta
import time
from functools import lru_cache
import requests
import warnings
import os
load_dotenv()
warnings.filterwarnings('ignore')

historical_bp = Blueprint('historical', __name__)

# Alpha Vantage API Key
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

# Cache historical data to prevent repeated API calls
@lru_cache(maxsize=50)
def fetch_historical_data_alpha(ticker, outputsize='full'):
    """Fetch historical data from Alpha Vantage with caching"""
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
        raise Exception(f"Failed to fetch data: {str(e)}")

def fetch_historical_with_retry(ticker, max_retries=3):
    """Fetch historical data with retry logic"""
    for attempt in range(max_retries):
        try:
            time.sleep(1)  # Rate limiting delay
            df = fetch_historical_data_alpha(ticker, outputsize='full')
            if df.empty:
                raise ValueError(f"No data found for ticker {ticker}")
            return df
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) * 2
                time.sleep(wait_time)
                continue
            else:
                raise Exception(f"Failed after {max_retries} attempts. Please try again later.")

@historical_bp.route('/historical', methods=["GET", "POST"])
def analyze():
    if request.method == "POST":
        ticker = request.form.get("hist_ticker", "MSFT").upper().strip()
        period = request.form.get("hist_period", "1y")
        interval = request.form.get("hist_interval", "1d")
        
        # Validate ticker
        if not ticker or len(ticker) > 10:
            return render_template("historical.html", 
                error="Please enter a valid stock ticker")
        
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
            "max": None  # Get all available data
        }
        
        if period not in period_map:
            return render_template("historical.html", 
                error="Invalid period selected")
        
        try:
            # Fetch data from Alpha Vantage
            df = fetch_historical_with_retry(ticker)
            
            # Filter based on period
            if period_map[period] is not None:
                start_date = datetime.today() - timedelta(days=period_map[period])
                df = df[df.index >= start_date]
            
            # Note: Alpha Vantage only provides daily data in the free tier
            # If user selected weekly or monthly, we'll resample
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
                    error="No data available for the selected period")
            
            # Calculate technical indicators
            df["SMA20"] = df["Close"].rolling(window=20).mean()
            df["SMA50"] = df["Close"].rolling(window=50).mean()
            df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
            
            # Calculate daily returns
            df["Daily_Return"] = df["Close"].pct_change() * 100
            
            # Calculate RSI (Relative Strength Index)
            delta = df["Close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df["RSI"] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            rolling_mean = df["Close"].rolling(window=20).mean()
            rolling_std = df["Close"].rolling(window=20).std()
            df["BB_Upper"] = rolling_mean + (rolling_std * 2)
            df["BB_Lower"] = rolling_mean - (rolling_std * 2)
            
            # Summary statistics
            summary = {
                "current_price": round(df["Close"].iloc[-1], 2),
                "high": round(df["High"].max(), 2),
                "low": round(df["Low"].min(), 2),
                "avg_volume": int(df["Volume"].mean()),
                "total_return": round(((df["Close"].iloc[-1] / df["Close"].iloc[0]) - 1) * 100, 2),
                "volatility": round(df["Daily_Return"].std(), 2),
                "avg_return": round(df["Daily_Return"].mean(), 2)
            }
            
            # Convert DataFrame to records for template
            df_reset = df.reset_index()
            df_reset['Date'] = df_reset['Date'].dt.strftime('%Y-%m-%d')
            
            # Round numeric columns
            numeric_cols = ['Open', 'High', 'Low', 'Close', 'SMA20', 'SMA50', 
                           'EMA20', 'Daily_Return', 'RSI', 'BB_Upper', 'BB_Lower']
            for col in numeric_cols:
                if col in df_reset.columns:
                    df_reset[col] = df_reset[col].round(2)
            
            return render_template(
                "historical.html",
                df=df_reset.to_dict("records"),
                ticker=ticker,
                period=period,
                interval=interval,
                summary=summary
            )
            
        except ValueError as ve:
            return render_template("historical.html", 
                error=f"Invalid ticker or no data available: {str(ve)}")
        except Exception as e:
            error_msg = str(e)
            if "rate limit" in error_msg.lower():
                return render_template("historical.html", 
                    error="Alpha Vantage rate limit reached (25 requests/day). Please try again later.")
            return render_template("historical.html", error=f"Error: {error_msg}")
    
    return render_template("historical.html")