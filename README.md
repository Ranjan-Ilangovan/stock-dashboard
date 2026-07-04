# 📈 Real-Time Stock Market AI Dashboard
### Live price charts, RSI indicator and LSTM-powered 7-day price predictions

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-LSTM-orange?logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Live%20App-red?logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-Interactive-blue)
![Status](https://img.shields.io/badge/Status-Live-brightgreen)

---

## 🌐 Live Demo

👉 **[Click here to use the app live]https://stock-dashboard-cdztxdm2tscufekjecyvcn.streamlit.app/**

> Type any stock ticker (AAPL, TSLA, GOOGL, MSFT, NVDA) and get instant AI-powered analysis

---

## 🔍 Overview

A fully deployed, real-time stock market analytics dashboard that combines live market data with deep learning predictions. Built with TensorFlow LSTM neural networks, the app fetches live stock prices via the yfinance API, trains a model on 2 years of historical data, and predicts the next 7 days of price movement — all in real time.

**No API key needed. No cost. Works on any stock ticker worldwide.**

---

## 🖥️ App Preview

![alt text](<Screenshot 1.png>)
![alt text](<Screenshot 2.png>)
![alt text](<Screenshot 3.png>)
---

## ✨ Features

| Feature | Description |
|---|---|
| 📊 Live Candlestick Chart | Real-time OHLC price data for any ticker |
| 📈 Moving Averages | 20-day and 50-day MA overlaid on price chart |
| 🔴 RSI Indicator | Momentum signal with overbought/oversold levels |
| 🤖 LSTM Prediction | 7-day price forecast using deep learning |
| 🎯 Confidence Band | ±2% prediction uncertainty band |
| 📉 Volume Chart | Daily trading volume bar chart |
| 🔮 Prediction Table | Day-by-day price forecast with % change |
| ⚙️ Interactive Controls | Toggle MA, RSI, predictions on/off |

---

## ⚙️ How It Works — LSTM Pipeline

```
┌─────────────────────────────────────────────────────────┐
│                   LSTM PIPELINE                          │
│                                                          │
│  yfinance API                                            │
│  Downloads 2 years of daily OHLCV data                  │
│      ↓                                                   │
│  scikit-learn MinMaxScaler                               │
│  Normalises prices to 0-1 range                         │
│      ↓                                                   │
│  TensorFlow LSTM (2 layers, 50 units each)               │
│  Trained on 80% of historical data                       │
│  60-day lookback window                                  │
│      ↓                                                   │
│  Predicts next 7 days iteratively                        │
│  Each prediction feeds back as input                     │
│      ↓                                                   │
│  Inverse transform back to real prices                   │
│      ↓                                                   │
│  Plotly chart + confidence band displayed                │
└─────────────────────────────────────────────────────────┘
```

### Technical Indicators Explained

**Moving Averages (MA20, MA50)**
- MA20 = average closing price over last 20 days (short-term trend)
- MA50 = average closing price over last 50 days (long-term trend)
- When MA20 crosses above MA50 = bullish signal

**RSI (Relative Strength Index)**
- Measures momentum on a 0-100 scale
- Above 70 = Overbought (potential sell signal)
- Below 30 = Oversold (potential buy signal)
- Between 30-70 = Neutral

**LSTM (Long Short-Term Memory)**
- Type of recurrent neural network designed for time-series data
- Remembers patterns from 60 days of price history
- Predicts one day at a time, feeding each prediction back as input

---

## 📁 Project Structure

```
stock-dashboard/
│
├── app.py              # Streamlit web app (UI + charts)
├── model.py            # LSTM model + data pipeline
├── requirements.txt    # Python dependencies
├── screenshot1.png     # Dashboard preview
├── screenshot2.png     # Prediction table preview
└── README.md
```

---

## 🛠️ Tech Stack

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.9+ | Core language |
| yfinance | Latest | Live stock data — free, no API key |
| TensorFlow/Keras | 2.x | LSTM neural network |
| scikit-learn | 1.3+ | Data normalisation (MinMaxScaler) |
| Plotly | Latest | Interactive candlestick + line charts |
| Streamlit | Latest | Web app framework + cloud deployment |
| Pandas/NumPy | Latest | Data manipulation |

---

## 🚀 Run Locally

### Prerequisites
- Python 3.9+
- Anaconda (recommended)

### Steps
```bash
# Clone the repo
git clone https://github.com/Ranjan-Ilangovan/stock-dashboard.git
cd stock-dashboard

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

Open browser at `http://localhost:8501`

---

## 💡 Example Tickers to Try

| Ticker | Company |
|---|---|
| AAPL | Apple Inc |
| TSLA | Tesla Inc |
| GOOGL | Alphabet (Google) |
| MSFT | Microsoft |
| AMZN | Amazon |
| NVDA | NVIDIA |
| RELIANCE.NS | Reliance Industries (India) |
| TCS.NS | Tata Consultancy Services (India) |

> **Tip:** Add `.NS` for NSE India stocks, `.BO` for BSE India stocks

---

## 📊 Model Performance

- **Architecture:** 2-layer LSTM with Dropout (0.2)
- **Training data:** 80% of 2-year historical prices
- **Lookback window:** 60 days
- **Epochs:** 10
- **Loss function:** Mean Squared Error (MSE)
- **Optimizer:** Adam

> Note: Stock price prediction is inherently uncertain. This model demonstrates deep learning concepts and should not be used for actual investment decisions.

---

## 🔮 Future Improvements

- [ ] Add sentiment analysis from financial news
- [ ] Include fundamental data (P/E ratio, EPS)
- [ ] Add portfolio tracker for multiple tickers
- [ ] Implement more advanced models (Transformer, Prophet)
- [ ] Add email alerts for RSI signals

---

## ⚠️ Disclaimer

This dashboard is built for **educational and portfolio purposes only**. LSTM predictions are not financial advice. Past performance does not guarantee future results. Always consult a qualified financial advisor before making investment decisions.

---

## 👤 Author

**Ranjan Ilangovan**
MSc Information Science (Data Analytics) — Northumbria University

🔗 [LinkedIn](https://www.linkedin.com/in/ranjan-ilangovan/)
🔗 [GitHub](https://github.com/Ranjan-Ilangovan)
🔗 [Churn Dashboard](https://github.com/Ranjan-Ilangovan/customer-churn-dashboard)
🔗 [RAG Chatbot](https://github.com/Ranjan-Ilangovan/rag-chatbot)
🔗 [Sales NLP Dashboard](https://github.com/Ranjan-Ilangovan/sales-nlp-dashboard)

---

## 📌 Project Status

✅ LSTM model built and tested
✅ Live stock data integration working
✅ All technical indicators implemented
✅ Interactive Streamlit app complete
✅ Deployed live on Streamlit Cloud
✅ Available for any global stock ticker
