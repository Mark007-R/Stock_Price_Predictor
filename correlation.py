from flask import Blueprint, render_template, request
import pandas as pd
from datetime import datetime, timedelta
import time
from functools import lru_cache
import requests
import warnings
from dotenv import load_dotenv
import os
import hashlib
load_dotenv()
warnings.filterwarnings('ignore')

correlation_bp = Blueprint('correlation', __name__)

# Alpha Vantage API Key
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

# Cache with expiration tracking
_cache = {}
CACHE_EXPIRY = 3600  # 1 hour in seconds

def get_cache_key(ticker, outputsize):
    """Generate cache key"""
    return hashlib.md5(f"{ticker}_{outputsize}".encode()).hexdigest()

def fetch_stock_data_alpha(ticker, outputsize='full'):
    """Fetch stock data from Alpha Vantage with proper caching"""
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
                raise Exception(f"Unable to fetch data for {ticker}")
        
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
        raise Exception(f"Network error for {ticker}: {str(e)}")
    except Exception as e:
        raise Exception(f"Failed to fetch data for {ticker}: {str(e)}")

@correlation_bp.route('/correlation', methods=["GET", "POST"])
def analyze():
    if request.method == "POST":
        tickers_input = request.form.get("tickers", "").upper()
        tickers = [t.strip() for t in tickers_input.split(",") if t.strip()]
        
        # Validation
        if len(tickers) < 2:
            return render_template("correlation.html", 
                error="Please enter at least 2 stock tickers separated by commas")
        
        if len(tickers) > 10:
            return render_template("correlation.html", 
                error="Please enter no more than 10 tickers to avoid rate limiting")
        
        # Validate ticker format
        for ticker in tickers:
            if len(ticker) > 10 or not ticker.replace('.', '').replace('-', '').isalnum():
                return render_template("correlation.html", 
                    error=f"Invalid ticker format: {ticker}")
        
        # Date range selection
        start_date = request.form.get("start_date", "2021-01-01")
        end_date = request.form.get("end_date", datetime.today().strftime('%Y-%m-%d'))
        
        # Validate dates
        try:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            if start_dt >= end_dt:
                return render_template("correlation.html", 
                    error="Start date must be before end date")
        except ValueError:
            return render_template("correlation.html", 
                error="Invalid date format")
        
        try:
            # Fetch data for each ticker
            close_data = pd.DataFrame()
            failed_tickers = []
            
            for i, ticker in enumerate(tickers):
                try:
                    # Fetch data from Alpha Vantage
                    df = fetch_stock_data_alpha(ticker, outputsize='full')
                    
                    # Filter by date range
                    df = df[(df.index >= start_date) & (df.index <= end_date)]
                    
                    if not df.empty and 'Close' in df.columns:
                        close_data[ticker] = df['Close']
                    else:
                        failed_tickers.append(ticker)
                    
                    # Rate limiting: Alpha Vantage free tier allows 5 calls/minute
                    # Only sleep if not the last ticker and not using cache
                    if i < len(tickers) - 1:
                        time.sleep(13)  # 13 seconds = ~4.6 calls/minute (safe margin)
                    
                except Exception as e:
                    failed_tickers.append(ticker)
                    # Continue with other tickers if one fails
                    continue
            
            # Check if we have valid data
            if close_data.empty or len(close_data.columns) < 2:
                error_msg = "Not enough valid data found for correlation analysis."
                if failed_tickers:
                    error_msg += f" Failed tickers: {', '.join(failed_tickers)}"
                return render_template("correlation.html", error=error_msg)
            
            # Drop rows with any NaN values
            initial_rows = len(close_data)
            close_data = close_data.dropna()
            
            if close_data.empty:
                return render_template("correlation.html", 
                    error="No overlapping data found for the selected tickers and date range")
            
            # Warn if too much data was dropped
            if len(close_data) < initial_rows * 0.5:
                # Warning: more than 50% data dropped, but continue
                pass
            
            # Calculate correlation matrix
            corr_matrix = close_data.corr()
            
            # Calculate additional statistics
            stats = {}
            for ticker in close_data.columns:
                ticker_data = close_data[ticker].dropna()
                if len(ticker_data) > 0:
                    stats[ticker] = {
                        "current_price": round(ticker_data.iloc[-1], 2),
                        "price_change": round(((ticker_data.iloc[-1] / ticker_data.iloc[0]) - 1) * 100, 2),
                        "volatility": round(ticker_data.pct_change().std() * 100, 2),
                        "mean_price": round(ticker_data.mean(), 2)
                    }
            
            # Find strongest correlations
            corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    ticker1 = corr_matrix.columns[i]
                    ticker2 = corr_matrix.columns[j]
                    corr_value = corr_matrix.iloc[i, j]
                    corr_pairs.append({
                        'ticker1': ticker1,
                        'ticker2': ticker2,
                        'correlation': round(corr_value, 3)
                    })
            
            # Sort by absolute correlation value
            corr_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)
            
            # Add warning message if some tickers failed
            warning = None
            if failed_tickers:
                warning = f"Warning: Could not fetch data for: {', '.join(failed_tickers)}"
            
            return render_template(
                "correlation.html",
                tickers=list(close_data.columns),
                corr_matrix=corr_matrix.round(3).values.tolist(),
                labels=corr_matrix.columns.tolist(),
                stats=stats,
                corr_pairs=corr_pairs[:5],  # Top 5 correlations
                start_date=start_date,
                end_date=end_date,
                warning=warning,
                data_points=len(close_data)
            )
            
        except ValueError as ve:
            return render_template("correlation.html", 
                error=f"Invalid ticker or no data available: {str(ve)}")
        except Exception as e:
            error_msg = str(e)
            if "rate limit" in error_msg.lower() or "frequency" in error_msg.lower():
                return render_template("correlation.html", 
                    error="Alpha Vantage rate limit reached (25 requests/day, 5 requests/minute). Please try again later.")
            return render_template("correlation.html", error=f"Error: {error_msg}")
    
    return render_template("correlation.html")