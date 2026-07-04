import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import warnings
warnings.filterwarnings('ignore')

def fetch_stock_data(ticker, period="2y"):
    """Download stock data from Yahoo Finance"""
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    df = df[['Close', 'Volume', 'High', 'Low']]
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

def prepare_lstm_data(data, lookback=60):
    """Prepare data for LSTM model"""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))

    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i, 0])
        y.append(scaled_data[i, 0])

    X = np.array(X)
    y = np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    return X, y, scaler

def build_lstm_model(lookback=60):
    """Build LSTM neural network"""
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(lookback, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def predict_next_7_days(ticker):
    """Main function — fetch data, train model, predict 7 days"""
    print(f"📈 Fetching data for {ticker}...")
    df = fetch_stock_data(ticker)
    df = calculate_moving_averages(df)

    # Prepare training data
    close_prices = df['Close'].values
    lookback = 60

    print(f"🧠 Training LSTM model...")
    X, y, scaler = prepare_lstm_data(close_prices, lookback)

    # Train on 80% of data
    split = int(len(X) * 0.8)
    X_train, y_train = X[:split], y[:split]

    # Build and train model
    model = build_lstm_model(lookback)
    model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=32,
        verbose=0
    )
    print(f"✅ Model trained!")

    # Predict next 7 days
    last_60_days = close_prices[-lookback:]
    predictions = []

    current_batch = scaler.transform(
        last_60_days.reshape(-1, 1)
    ).reshape(1, lookback, 1)

    for _ in range(7):
        pred = model.predict(current_batch, verbose=0)[0]
        predictions.append(pred[0])
        current_batch = np.append(
            current_batch[:, 1:, :],
            [[pred]],
            axis=1
        )

    # Inverse transform predictions
    predictions = scaler.inverse_transform(
        np.array(predictions).reshape(-1, 1)
    ).flatten()

    print(f"🔮 7-day predictions: {predictions.round(2)}")

    return df, predictions

if __name__ == "__main__":
    df, predictions = predict_next_7_days("AAPL")
    print(f"\n✅ Done!")
    print(f"Last close: ${df['Close'].iloc[-1]:.2f}")
    print(f"Predicted prices: {predictions.round(2)}")