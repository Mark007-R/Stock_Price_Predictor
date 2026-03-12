from flask import Blueprint, render_template, request
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
import warnings
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

historical_bp = Blueprint('historical', __name__)

# In-memory cache
_cache = {}
CACHE_EXPIRY = 3600  # 1 hour

def fetch_historical_data(ticker, period="max"):
    """Fetch historical data using yfinance with caching"""
    import time
    cache_key = f"{ticker}_{period}"
    current_time = time.time()

    if cache_key in _cache:
        cached_data, cached_time = _cache[cache_key]
        if current_time - cached_time < CACHE_EXPIRY:
            logger.info(f"Returning cached historical data for {ticker}")
            return cached_data.copy()

    logger.info(f"Fetching historical data for {ticker}...")
    df = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=True)

    if df.empty:
        raise Exception(f"No data found for ticker '{ticker}'. Please verify the symbol is correct.")

    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    _cache[cache_key] = (df, current_time)
    logger.info(f"Successfully fetched {len(df)} days of data for {ticker}")
    return df

def calculate_technical_indicators(df):
    """Calculate all technical indicators"""
    if len(df) >= 20:
        df["SMA20"] = df["Close"].rolling(window=20).mean()
    else:
        df["SMA20"] = None

    if len(df) >= 50:
        df["SMA50"] = df["Close"].rolling(window=50).mean()
    else:
        df["SMA50"] = None

    if len(df) >= 20:
        df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    else:
        df["EMA20"] = None

    df["Daily_Return"] = df["Close"].pct_change() * 100

    if len(df) >= 15:
        delta = df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, 0.0001)
        df["RSI"] = 100 - (100 / (1 + rs))
    else:
        df["RSI"] = None

    if len(df) >= 20:
        rolling_mean = df["Close"].rolling(window=20).mean()
        rolling_std = df["Close"].rolling(window=20).std()
        df["BB_Upper"] = rolling_mean + (rolling_std * 2)
        df["BB_Lower"] = rolling_mean - (rolling_std * 2)
        df["BB_Middle"] = rolling_mean
    else:
        df["BB_Upper"] = df["BB_Lower"] = df["BB_Middle"] = None

    if len(df) >= 26:
        ema12 = df["Close"].ewm(span=12, adjust=False).mean()
        ema26 = df["Close"].ewm(span=26, adjust=False).mean()
        df["MACD"] = ema12 - ema26
        df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    else:
        df["MACD"] = df["MACD_Signal"] = None

    return df

@historical_bp.route('/historical', methods=["GET", "POST"])
def analyze():
    if request.method == "POST":
        ticker = request.form.get("hist_ticker", "MSFT").upper().strip()
        period = request.form.get("hist_period", "1y")
        interval = request.form.get("hist_interval", "1d")

        if not ticker or len(ticker) > 10 or not ticker.replace('.', '').replace('-', '').isalnum():
            return render_template("historical.html",
                error="Please enter a valid stock ticker (e.g., AAPL, MSFT)")

        valid_periods = {"1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"}
        if period not in valid_periods:
            return render_template("historical.html", error="Invalid period selected")

        try:
            logger.info(f"Fetching historical data for {ticker} ({period})")
            df = fetch_historical_data(ticker, period=period)

            if df.empty:
                return render_template("historical.html",
                    error=f"No data available for ticker {ticker}")

            # Resample if needed
            if interval == "1wk":
                df = df.resample('W').agg({
                    'Open': 'first', 'High': 'max',
                    'Low': 'min', 'Close': 'last', 'Volume': 'sum'
                }).dropna()
            elif interval == "1mo":
                df = df.resample('ME').agg({
                    'Open': 'first', 'High': 'max',
                    'Low': 'min', 'Close': 'last', 'Volume': 'sum'
                }).dropna()

            if df.empty:
                return render_template("historical.html",
                    error="No data available after resampling for the selected period")

            df = calculate_technical_indicators(df)

            price_change = ((df["Close"].iloc[-1] / df["Close"].iloc[0]) - 1) * 100
            summary = {
                "current_price": round(float(df["Close"].iloc[-1]), 2),
                "high": round(float(df["High"].max()), 2),
                "low": round(float(df["Low"].min()), 2),
                "avg_volume": int(df["Volume"].mean()),
                "total_return": round(float(price_change), 2),
                "volatility": round(float(df["Daily_Return"].std()), 2) if len(df) > 1 else 0,
                "avg_return": round(float(df["Daily_Return"].mean()), 2) if len(df) > 1 else 0,
                "data_points": len(df),
                "sharpe_ratio": round(
                    (df["Daily_Return"].mean() / df["Daily_Return"].std()) * (252 ** 0.5), 2
                ) if df["Daily_Return"].std() > 0 else 0
            }

            df_reset = df.reset_index()
            df_reset['Date'] = df_reset['Date'].dt.strftime('%Y-%m-%d')

            numeric_cols = ['Open', 'High', 'Low', 'Close', 'SMA20', 'SMA50',
                            'EMA20', 'Daily_Return', 'RSI', 'BB_Upper', 'BB_Lower',
                            'BB_Middle', 'MACD', 'MACD_Signal']
            for col in numeric_cols:
                if col in df_reset.columns:
                    df_reset[col] = df_reset[col].round(2)

            df_reset['Volume'] = df_reset['Volume'].astype(int)

            return render_template("historical.html",
                df=df_reset.to_dict("records"),
                ticker=ticker,
                period=period,
                interval=interval,
                summary=summary
            )

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Historical analysis error: {error_msg}")
            return render_template("historical.html", error=f"Error: {error_msg}")

    return render_template("historical.html")