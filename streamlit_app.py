import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import date

st.set_page_config(layout="wide")
sns.set(style="whitegrid")

# ----------------------------
# 1. Index Definitions
# ----------------------------
indices = {
    "^GSPC": "S&P 500 (US)",
    "^IXIC": "NASDAQ Composite (US)",
    "^DJI": "Dow Jones (US)",
    "^RUT": "Russell 2000 (US)",
    "VTI": "Wilshire 5000 ETF (US)",
    "^GSPTSE": "S&P/TSX Composite (Canada)",
    "^FTSE": "FTSE 100 (UK)",
    "^GDAXI": "DAX (Germany)",
    "^FCHI": "CAC 40 (France)",
    "^STOXX50E": "Euro Stoxx 50 (Eurozone)",
    "^IBEX": "IBEX 35 (Spain)",
    "^N225": "Nikkei 225 (Japan)",
    "^HSI": "Hang Seng (Hong Kong)",
    "GXC": "SPDR S&P China ETF (proxy for Shanghai Composite)",
    "ASHR": "CSI 300 ETF (China)",
    "^KS11": "KOSPI (South Korea)",
    "^BSESN": "SENSEX (India)",
    "^NSEI": "NIFTY 50 (India)",
    "^AXJO": "ASX 200 (Australia)",
    "^BVSP": "Bovespa (Brazil)",
    "^MXX": "IPC (Mexico)",
    "^NYA": "NYSE Composite (US)",
    "^SP500-45": "S&P 500 Energy (proxy)",
    "ESGU": "S&P 500 ESG ETF",
    "EEM": "MSCI Emerging Markets ETF",
     "^VIX": "CBOE VOLATILITY (FEAR)",
}

st.title("📈 Global Market Index Dashboard")

today = date.today()
start_date = st.date_input("Start date", pd.to_datetime("2022-01-01"))
end_date = st.date_input("End date", today)


selected_names = st.multiselect("Select indices:", list(indices.values()), default=["S&P 500 (US)", "NASDAQ Composite (US)"])

# Reverse lookup
ticker_map = {v: k for k, v in indices.items()}
tickers = [ticker_map[name] for name in selected_names]

# ----------------------------
# 2. Cached Data Fetch
# ----------------------------
@st.cache_data(show_spinner="Downloading index data...")
def fetch_data(tickers, start, end):
    return yf.download(tickers, start=start, end=end, group_by="ticker", auto_adjust=True)

raw_data = fetch_data(tickers, start_date, end_date)

# ----------------------------
# 3. Build Close Prices
# ----------------------------
close_prices = {}
for name in selected_names:
    ticker = ticker_map[name]
    try:
        df = raw_data[ticker]
        if isinstance(df, pd.DataFrame) and 'Close' in df:
            series = df['Close'].dropna()
            if not series.empty:
                close_prices[name] = series
    except Exception as e:
        st.warning(f"Failed to load: {name} — {e}")

df_close = pd.DataFrame(close_prices)
df_close = df_close.fillna(method="ffill").fillna(method="bfill")

if not df_close.empty:
    # Normalize
    norm = df_close / df_close.iloc[0] * 100

    # Plot normalized performance
    st.subheader("📊 Normalized Index Performance (Base 100)")
    st.line_chart(norm)

    # Correlation matrix
    st.subheader("🔗 Correlation Matrix")
    returns = df_close.pct_change().dropna()
    corr = returns.corr()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=True, ax=ax)
    st.pyplot(fig)

    # Strongest and weakest correlations
    mask = np.triu(np.ones(corr.shape), k=1).astype(bool)
    corr_values = corr.where(mask).stack()

    if not corr_values.empty:
        strongest = corr_values.idxmax()
        weakest = corr_values.idxmin()
        st.markdown(f"**Strongest correlation:** {strongest[0]} ↔ {strongest[1]} = {corr_values.max():.2f}")
        st.markdown(f"**Weakest correlation:** {weakest[0]} ↔ {weakest[1]} = {corr_values.min():.2f}")
else:
    st.error("No valid data to display.")
