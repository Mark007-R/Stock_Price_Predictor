from flask import Blueprint, render_template, request
import yfinance as yf
import pandas as pd
from datetime import datetime
import time
from functools import lru_cache
import warnings
warnings.filterwarnings('ignore')

correlation_bp = Blueprint('correlation', __name__)

# Cache correlation data to prevent repeated API calls
@lru_cache(maxsize=50)
def fetch_correlation_data_cached(tickers_str, start_date, end_date):
    """Fetch correlation data with caching"""
    try:
        tickers = tickers_str.split(',')
        time.sleep(1)  # Rate limiting delay
        data = yf.download(tickers, start=start_date, end=end_date, 
                          progress=False, show_errors=False)
        return data
    except Exception as e:
        raise Exception(f"Failed to fetch data: {str(e)}")

def fetch_correlation_with_retry(tickers_str, start_date, end_date, max_retries=3):
    """Fetch correlation data with retry logic"""
    for attempt in range(max_retries):
        try:
            data = fetch_correlation_data_cached(tickers_str, start_date, end_date)
            if data.empty:
                raise ValueError("No data found for the provided tickers")
            return data
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) * 2
                time.sleep(wait_time)
                continue
            else:
                raise Exception(f"Failed after {max_retries} attempts. Please try again later.")

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
            # Convert tickers list to string for caching
            tickers_str = ','.join(tickers)
            
            data = fetch_correlation_with_retry(tickers_str, start_date, end_date)
            
            # Handle multi-index columns for multiple tickers
            if len(tickers) == 1:
                close_data = data[["Close"]]
                close_data.columns = [tickers[0]]
            else:
                close_data = data["Close"]
            
            # Check if we have valid data
            if close_data.empty:
                return render_template("correlation.html", 
                    error="No valid data found for the selected tickers and date range")
            
            # Calculate correlation matrix
            corr_matrix = close_data.corr()
            
            # Calculate additional statistics
            stats = {}
            for ticker in tickers:
                if ticker in close_data.columns:
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
                tickers=tickers,
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
            if "Too Many Requests" in error_msg or "Rate limit" in error_msg:
                return render_template("correlation.html", 
                    error="Rate limit exceeded. Please wait a few minutes and try again.")
            return render_template("correlation.html", error=f"Error: {error_msg}")
    
    return render_template("correlation.html")