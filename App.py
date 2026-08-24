import streamlit as st
import requests
import pandas as pd
import numpy as np
import time

st.set_page_config(page_title="Delta Scanner", page_icon="🔥", layout="wide")
st.title("🔥 Delta Reversal Scanner (Simple)")

BASE_URL = "https://api.india.delta.exchange"
HEADERS = {"Accept": "application/json"}

def api_get(path, params=None):
    try:
        r = requests.get(BASE_URL + path, params=params or {}, headers=HEADERS, timeout=12)
        if r.status_code != 200:
            return None
        data = r.json()
        return data.get("result") if data.get("success", True) else None
    except:
        return None

@st.cache_data(ttl=30)
def load_market():
    products = api_get("/v2/products", {"contract_types": "perpetual_futures"})
    tickers = api_get("/v2/tickers", {"contract_types": "perpetual_futures"})
    
    if not products or not tickers:
        return pd.DataFrame()
    
    prod_rows = []
    for p in products:
        if p.get("state") != "live" or p.get("trading_status") != "operational":
            continue
        symbol = p.get("symbol")
        if not symbol:
            continue
        try:
            default_lev = float(p.get("default_leverage") or 20)
        except:
            default_lev = 20
        max_lev = default_lev
        try:
            im = float(p.get("initial_margin") or 0)
            if im > 0:
                max_lev = max(round(100 / im), default_lev)
        except:
            pass
        prod_rows.append({"Coin": symbol, "Max Leverage": int(max_lev)})
    
    prod_df = pd.DataFrame(prod_rows).drop_duplicates("Coin")
    
    tick_rows = []
    for t in tickers:
        symbol = t.get("symbol")
        if not symbol:
            continue
        try:
            price = float(t.get("close") or t.get("mark_price") or 0)
            volume = float(t.get("volume_24h") or 0)
            oi = float(t.get("open_interest") or 0)
        except:
            continue
        if price <= 0:
            continue
        tick_rows.append({
            "Coin": symbol,
            "Price": price,
            "24H Volume": volume,
            "OI": oi,
            "Vol/OI": volume / oi if oi > 0 else 0
        })
    
    tick_df = pd.DataFrame(tick_rows)
    market = prod_df.merge(tick_df, on="Coin", how="inner")
    return market.sort_values("24H Volume", ascending=False).reset_index(drop=True)

st.write("Loading market data...")
market = load_market()

if market.empty:
    st.error("Data nahi aaya. Refresh try karo.")
    st.stop()

st.success(f"Loaded {len(market)} coins")

# Sidebar
with st.sidebar:
    st.header("Settings")
    min_vol = st.number_input("Min Vol/OI (only for ≤20x)", 0.0, 20.0, 1.5, 0.5)
    top_n = st.slider("Show top coins", 10, 50, 30)
    if st.button("Refresh"):
        st.cache_data.clear()
        st.rerun()

# Filter
high = market[market["Max Leverage"] > 20]
low = market[(market["Max Leverage"] <= 20) & (market["Vol/OI"] >= min_vol)]

st.write(f"**> 20x coins:** {len(high)} (free from filter)")
st.write(f"**≤ 20x coins (filtered):** {len(low)}")

tab1, tab2, tab3 = st.tabs(["All", "> 20x Leverage", "≤ 20x Leverage"])

with tab1:
    st.dataframe(market.head(top_n)[["Coin", "Max Leverage", "Price", "24H Volume", "Vol/OI"]], use_container_width=True, hide_index=True)

with tab2:
    st.subheader("High Leverage (>20x)")
    if high.empty:
        st.info("No high leverage coins")
    else:
        st.dataframe(high.head(top_n)[["Coin", "Max Leverage", "Price", "24H Volume", "Vol/OI"]], use_container_width=True, hide_index=True)

with tab3:
    st.subheader("Low Leverage (≤20x)")
    if low.empty:
        st.info("No coins after filter")
    else:
        st.dataframe(low.head(top_n)[["Coin", "Max Leverage", "Price", "24H Volume", "Vol/OI"]], use_container_width=True, hide_index=True)

st.caption("Simple version | Data from Delta Exchange India")
