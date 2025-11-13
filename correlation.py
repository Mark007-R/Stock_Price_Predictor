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

correlation_bp = Blueprint('correlation', __name__)

# Alpha Vantage API Key
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

# Cache correlation data to prevent repeated API calls
@lru_cache(maxsize=50)
def fetch_stock_data_alpha(ticker, outputsize='full'):
    """Fetch stock data from Alpha Vantage"""
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
                raise Exception(f"Unable to fetch data")
        
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
            if len(ticker) > 10 or not ticker.isalnum():
                return render_template("correlation.html", 
                    error=f"Invalid ticker format: {ticker}")
        
        # Date range selection
        start_date = request.form.get("start_date", "2021-01-01")
        end_date = request.form.get("end_date", datetime.today().strftime('%Y-%m-%d'))
        
        try:
            # Fetch data for each ticker
            close_data = pd.DataFrame()
            
            for ticker in tickers:
                try:
                    # Fetch data from Alpha Vantage
                    df = fetch_stock_data_alpha(ticker, outputsize='full')
                    
                    # Filter by date range
                    df = df[(df.index >= start_date) & (df.index <= end_date)]
                    
                    if not df.empty:
                        close_data[ticker] = df['Close']
                    
                    # Rate limiting: Alpha Vantage free tier allows 5 calls/minute
                    time.sleep(12)  # Wait 12 seconds between calls
                    
                except Exception as e:
                    # Continue with other tickers if one fails
                    return render_template("correlation.html", 
                        error=f"Failed to fetch data for {ticker}: {str(e)}")
            
            # Check if we have valid data
            if close_data.empty or len(close_data.columns) < 2:
                return render_template("correlation.html", 
                    error="Not enough valid data found for correlation analysis")
            
            # Drop rows with any NaN values
            close_data = close_data.dropna()
            
            if close_data.empty:
                return render_template("correlation.html", 
                    error="No overlapping data found for the selected tickers and date range")
            
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
            
            return render_template(
                "correlation.html",
                tickers=list(close_data.columns),
                corr_matrix=corr_matrix.round(3).values.tolist(),
                labels=corr_matrix.columns.tolist(),
                stats=stats,
                corr_pairs=corr_pairs[:5],  # Top 5 correlations
                start_date=start_date,
                end_date=end_date
            )
            
        except ValueError as ve:
            return render_template("correlation.html", 
                error=f"Invalid ticker or no data available: {str(ve)}")
        except Exception as e:
            error_msg = str(e)
            if "rate limit" in error_msg.lower():
                return render_template("correlation.html", 
                    error="Alpha Vantage rate limit reached (25 requests/day, 5 requests/minute). Please try again later.")
            return render_template("correlation.html", error=f"Error: {error_msg}")
    
    return render_template("correlation.html")