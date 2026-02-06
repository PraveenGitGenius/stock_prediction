

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objects as go
from datetime import date


st.set_page_config(page_title="STOCK PRICE PREDICTION", page_icon="📈", layout="wide")


st.markdown("""
<style>
[data-testid="stAppViewContainer"]{
    background: linear-gradient(135deg,#141e30,#243b55);
    color:white;
}
h1{color:#00FFE0;text-align:center;}

section[data-testid="stSidebar"]{
    background:#111827;
}

/* metric cards */
.stMetric{
    background:rgba(255,255,255,0.10);
    padding:18px;
    border-radius:12px;
    text-align:center;
}

/* big numbers */
.stMetric > div:nth-child(1){
    font-size:28px !important;
    font-weight:700 !important;
    color:#00FFE0 !important;
}

/* labels */
.stMetric label{
    color:white !important;
}
</style>
""", unsafe_allow_html=True)

st.title("STOCK PREDICTION USING MACHINE LEARNING")


START = "2018-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

stock = st.sidebar.text_input("Stock Symbol", "AAPL").upper().strip()
years = st.sidebar.slider("Forecast Years", 1, 3)
period = years * 365

st.cache_data
def load_data(ticker):
    try:
        df = yf.download(ticker, start=START, end=TODAY)

        if df is None or df.empty:
            return None

        df.reset_index(inplace=True)
        return df
    except:
        return None


data = load_data(stock)

if data is None:
    st.error(" Invalid symbol or no data. Try AAPL, MSFT, TSLA, TCS.NS, RELIANCE.NS")
    st.stop()


data['MA20'] = data['Close'].rolling(20).mean()
data['MA50'] = data['Close'].rolling(50).mean()

delta = data['Close'].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()

rs = avg_gain / avg_loss
data['RSI'] = 100 - (100 / (1 + rs))

# BUY/SELL SIGNAL (safe vectorized)
data['Signal'] = np.where(
    data['RSI'] < 30, "BUY",
    np.where(data['RSI'] > 70, "SELL", "HOLD")
)

latest_signal = data['Signal'].iloc[-1]



st.subheader(" Market Snapshot")

c1, c2, c3, c4 = st.columns(4)

current = data['Close'].iloc[-1].item()
previous = data['Close'].iloc[-2].item()

change = round(current - previous, 2)

ma20 = float(data['MA20'].iloc[-1]) if not pd.isna(data['MA20'].iloc[-1]) else 0
ma50 = float(data['MA50'].iloc[-1]) if not pd.isna(data['MA50'].iloc[-1]) else 0


c1.metric("Current Price", round(current,2), change, delta_color="normal")
c2.metric("MA20", round(ma20,2))
c3.metric("MA50", round(ma50,2))
c4.metric("Signal", latest_signal)


st.subheader(" Price Chart")

fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=data['Date'],
    open=data['Open'],
    high=data['High'],
    low=data['Low'],
    close=data['Close'],
    name="Price"
))

fig.add_trace(go.Scatter(x=data['Date'], y=data['MA20'], name="MA20"))
fig.add_trace(go.Scatter(x=data['Date'], y=data['MA50'], name="MA50"))

fig.update_layout(template="plotly_dark", height=500)

st.plotly_chart(fig, use_container_width=True)


st.subheader("RSI Indicator")

fig_rsi = go.Figure()
fig_rsi.add_trace(go.Scatter(x=data['Date'], y=data['RSI'], name="RSI"))

fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")

fig_rsi.update_layout(template="plotly_dark", height=250)

st.plotly_chart(fig_rsi, use_container_width=True)


st.subheader(" AI Forecast")

df_train = data[['Date','Close']].copy()
df_train.columns = ['ds','y']

df_train['ds'] = pd.to_datetime(df_train['ds'])
df_train['y'] = df_train['y'].astype(float)

model = Prophet()
model.fit(df_train)

future = model.make_future_dataframe(periods=period)
forecast = model.predict(future)

fig2 = plot_plotly(model, forecast)
fig2.update_layout(template="plotly_dark", height=500)

st.plotly_chart(fig2, use_container_width=True)



csv = forecast.to_csv(index=False).encode()
st.download_button("⬇️ Download Forecast CSV", csv, "forecast.csv")

st.markdown("---")
st.caption("DESIGN BY TEAM KALI LINUX  |  PRAVEEN   |  HARSH  |  SUMIT  ")
