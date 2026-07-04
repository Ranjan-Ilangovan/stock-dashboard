import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from model import predict_next_7_days, fetch_stock_data, calculate_moving_averages

st.set_page_config(
    page_title="Stock Market AI Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Real-Time Stock Market AI Dashboard")
st.markdown("Live price charts, technical indicators and LSTM-powered 7-day predictions")

with st.sidebar:
    st.header("⚙️ Settings")
    ticker = st.text_input(
        "Enter Stock Ticker",
        value="AAPL",
        help="e.g. AAPL, TSLA, GOOGL, MSFT",
        key="ticker_input"
    ).upper()
    period = st.selectbox(
        "Select Time Period",
        ["6mo", "1y", "2y"],
        index=1,
        key="period_select"
    )
    show_prediction = st.checkbox("Show LSTM Prediction", value=True, key="show_pred")
    show_ma = st.checkbox("Show Moving Averages", value=True, key="show_ma")
    show_rsi = st.checkbox("Show RSI Indicator", value=True, key="show_rsi")
    st.divider()
    st.markdown("**Popular Tickers:**")
    st.markdown("AAPL · TSLA · GOOGL · MSFT · AMZN · NVDA")
    run_button = st.button("🚀 Analyse Stock", use_container_width=True, key="run_btn")

if run_button:
    with st.spinner(f"Fetching data and training LSTM for {ticker}..."):
        try:
            if show_prediction:
                df, predictions = predict_next_7_days(ticker)
            else:
                df = fetch_stock_data(ticker, period)
                df = calculate_moving_averages(df)
                predictions = None

            current_price = df['Close'].iloc[-1]
            prev_price = df['Close'].iloc[-2]
            price_change = current_price - prev_price
            price_change_pct = (price_change / prev_price) * 100
            high_52w = df['Close'].tail(252).max()
            low_52w = df['Close'].tail(252).min()
            current_rsi = df['RSI'].iloc[-1]

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(
                    label=f"{ticker} Price",
                    value=f"${current_price:.2f}",
                    delta=f"{price_change_pct:.2f}%"
                )
            with col2:
                st.metric(label="52-Week High", value=f"${high_52w:.2f}")
            with col3:
                st.metric(label="52-Week Low", value=f"${low_52w:.2f}")
            with col4:
                rsi_signal = "Overbought ⚠️" if current_rsi > 70 else "Oversold 🟢" if current_rsi < 30 else "Neutral ✅"
                st.metric(label="RSI Signal", value=f"{current_rsi:.1f}", delta=rsi_signal)

            if show_rsi:
                fig = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.05,
                    row_heights=[0.7, 0.3],
                    subplot_titles=(f"{ticker} Price Chart", "RSI Indicator")
                )
            else:
                fig = make_subplots(rows=1, cols=1)

            fig.add_trace(
                go.Candlestick(
                    x=df.index,
                    open=df['Open'] if 'Open' in df.columns else df['Close'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'],
                    name="Price",
                    increasing_line_color='#00B050',
                    decreasing_line_color='#FF0000'
                ),
                row=1, col=1
            )

            if show_ma:
                fig.add_trace(
                    go.Scatter(
                        x=df.index, y=df['MA20'],
                        name="MA20",
                        line=dict(color='#FFA500', width=1.5)
                    ),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(
                        x=df.index, y=df['MA50'],
                        name="MA50",
                        line=dict(color='#2E75B6', width=1.5)
                    ),
                    row=1, col=1
                )

            if show_prediction and predictions is not None:
                future_dates = [df.index[-1] + timedelta(days=i) for i in range(1, 8)]
                fig.add_trace(
                    go.Scatter(
                        x=future_dates,
                        y=predictions,
                        name="LSTM Prediction",
                        line=dict(color='#FF6B6B', width=2, dash='dash'),
                        mode='lines+markers',
                        marker=dict(size=8)
                    ),
                    row=1, col=1
                )
                upper_band = predictions * 1.02
                lower_band = predictions * 0.98
                fig.add_trace(
                    go.Scatter(
                        x=future_dates + future_dates[::-1],
                        y=list(upper_band) + list(lower_band[::-1]),
                        fill='toself',
                        fillcolor='rgba(255,107,107,0.1)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name="Confidence Band"
                    ),
                    row=1, col=1
                )

            if show_rsi:
                fig.add_trace(
                    go.Scatter(
                        x=df.index, y=df['RSI'],
                        name="RSI",
                        line=dict(color='#9B59B6', width=1.5)
                    ),
                    row=2, col=1
                )
                fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)

            fig.update_layout(
                height=700,
                template="plotly_dark",
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                xaxis_rangeslider_visible=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )

            st.plotly_chart(fig, use_container_width=True, key="main_chart")

            if show_prediction and predictions is not None:
                st.subheader("🔮 7-Day Price Predictions")
                future_dates_str = [
                    (df.index[-1] + timedelta(days=i)).strftime('%Y-%m-%d')
                    for i in range(1, 8)
                ]
                pred_df = pd.DataFrame({
                    'Date': future_dates_str,
                    'Predicted Price': [f"${p:.2f}" for p in predictions],
                    'Change from Today': [
                        f"{((p - current_price) / current_price * 100):+.2f}%"
                        for p in predictions
                    ],
                    'Signal': ['📈 Up' if p > current_price else '📉 Down' for p in predictions]
                })
                st.dataframe(pred_df, use_container_width=True, hide_index=True)

            st.subheader("📊 Trading Volume")
            vol_fig = go.Figure()
            vol_fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df['Volume'],
                    name="Volume",
                    marker_color='#2E75B6',
                    opacity=0.7
                )
            )
            vol_fig.update_layout(
                height=200,
                template="plotly_dark",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                showlegend=False
            )
            st.plotly_chart(vol_fig, use_container_width=True, key="vol_chart")

            st.divider()
            st.caption(
                "⚠️ Disclaimer: This dashboard is for educational purposes only. "
                "LSTM predictions are not financial advice. "
                "Always do your own research before making investment decisions."
            )

        except Exception as e:
            st.error(f"❌ Error fetching data for {ticker}: {str(e)}")
            st.info("Please check the ticker symbol and try again. Examples: AAPL, TSLA, GOOGL, MSFT")