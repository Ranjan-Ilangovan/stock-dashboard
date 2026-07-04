import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

def fetch_stock_data(ticker, period="2y"):
    """Download stock data from Yahoo Finance"""
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    df = df[['Close', 'Volume', 'High', 'Low', 'Open']]
    df.dropna(inplace=True)
    return df

def calculate_rsi(prices, period=14):
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_moving_averages(df):
    """Add moving averages to dataframe"""
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['RSI'] = calculate_rsi(df['Close'])
    return df

def predict_next_7_days(ticker):
    """Fetch data and predict 7 days using ML"""
    print(f"📈 Fetching data for {ticker}...")
    df = fetch_stock_data(ticker)
    df = calculate_moving_averages(df)

    # Prepare features
    close_prices = df['Close'].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(close_prices.reshape(-1, 1))

    # Create sequences (60 day lookback)
    lookback = 60
    X, y = [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i-lookback:i, 0])
        y.append(scaled[i, 0])

    X = np.array(X)
    y = np.array(y)

    # Train linear regression on sequences
    # Flatten X for sklearn
    X_flat = X.reshape(X.shape[0], -1)

    # Train on 80% of data
    split = int(len(X_flat) * 0.8)
    X_train = X_flat[:split]
    y_train = y[:split]

    model = LinearRegression()
    model.fit(X_train, y_train)
    print("✅ Model trained!")

    # Predict next 7 days iteratively
    predictions = []
    current_seq = scaled[-lookback:].flatten()

    for _ in range(7):
        pred = model.predict(current_seq.reshape(1, -1))[0]
        predictions.append(pred)
        current_seq = np.append(current_seq[1:], pred)

    # Inverse transform
    predictions = scaler.inverse_transform(
        np.array(predictions).reshape(-1, 1)
    ).flatten()

    print(f"🔮 Predictions: {predictions.round(2)}")
    return df, predictions
