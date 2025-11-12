from flask import Blueprint, render_template, request
import yfinance as yf
import pandas as pd
import time
from functools import lru_cache
import warnings
warnings.filterwarnings('ignore')

historical_bp = Blueprint('historical', __name__)

# Cache historical data to prevent repeated API calls
@lru_cache(maxsize=50)
def fetch_historical_data_cached(ticker, period, interval):
    """Fetch historical data with caching"""
    try:
        time.sleep(1)  # Rate limiting delay
        df = yf.download(ticker, period=period, interval=interval, 
                        progress=False, show_errors=False)
        return df
    except Exception as e:
        raise Exception(f"Failed to fetch data: {str(e)}")

def fetch_historical_with_retry(ticker, period, interval, max_retries=3):
    """Fetch historical data with retry logic"""
    for attempt in range(max_retries):
        try:
            df = fetch_historical_data_cached(ticker, period, interval)
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
        
        # Validate period
        valid_periods = ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
        if period not in valid_periods:
            return render_template("historical.html", 
                error="Invalid period selected")
        
        # Validate interval
        valid_intervals = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", 
                          "1d", "5d", "1wk", "1mo", "3mo"]
        if interval not in valid_intervals:
            return render_template("historical.html", 
                error="Invalid interval selected")
        
        try:
            df = fetch_historical_with_retry(ticker, period, interval)
            
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
            if "Too Many Requests" in error_msg or "Rate limit" in error_msg:
                return render_template("historical.html", 
                    error="Rate limit exceeded. Please wait a few minutes and try again.")
            return render_template("historical.html", error=f"Error: {error_msg}")
    
    return render_template("historical.html")