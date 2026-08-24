import streamlit as st
import requests
import pandas as pd
import numpy as np
import time

st.set_page_config(page_title="Delta Scanner", page_icon="🔥", layout="wide")
st.title("🔥 Delta Reversal Scanner")

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
            funding = float(t.get("funding_rate") or 0)
        except:
            continue
        if price <= 0:
            continue
        tick_rows.append({
            "Coin": symbol,
            "Price": price,
            "24H Volume": volume,
            "OI": oi,
            "Vol/OI": volume / oi if oi > 0 else 0,
            "Funding": funding
        })
    
    tick_df = pd.DataFrame(tick_rows)
    market = prod_df.merge(tick_df, on="Coin", how="inner")
    return market.sort_values("24H Volume", ascending=False).reset_index(drop=True)

@st.cache_data(ttl=60)
def get_candles(symbol, resolution="15m", hours=24):
    end = int(time.time())
    start = end - hours * 3600
    result = api_get("/v2/history/candles", {
        "resolution": resolution,
        "symbol": symbol,
        "start": start,
        "end": end
    })
    if not result:
        return pd.DataFrame()
    df = pd.DataFrame(result)
    if df.empty:
        return df
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "time" in df.columns:
        df = df.sort_values("time")
    return df.dropna(subset=["close"]).reset_index(drop=True)

def get_trend(df):
    if len(df) < 20:
        return "UNKNOWN"
    close = df["close"]
    ema9 = close.ewm(span=9).mean().iloc[-1]
    ema21 = close.ewm(span=21).mean().iloc[-1]
    last = close.iloc[-1]
    if last > ema9 > ema21:
        return "BULL"
    if last < ema9 < ema21:
        return "BEAR"
    return "MIXED"

def simple_score(row, trend):
    score = 0
    if trend == "BULL":
        score += 3
    elif trend == "BEAR":
        score += 3
    if row["Vol/OI"] > 3:
        score += 2
    elif row["Vol/OI"] > 1.5:
        score += 1
    if abs(row.get("Funding", 0)) > 0.0003:
        score += 1
    signal = "NO SIGNAL"
    if score >= 6 and trend == "BULL":
        signal = "STRONG LONG"
    elif score >= 5 and trend == "BULL":
        signal = "LONG WATCH"
    elif score >= 6 and trend == "BEAR":
        signal = "STRONG SHORT"
    elif score >= 5 and trend == "BEAR":
        signal = "SHORT WATCH"
    return score, signal, trend

# ========== MAIN ==========
st.write("Loading market...")
market = load_market()

if market.empty:
    st.error("Data nahi aaya")
    st.stop()

st.success(f"Loaded {
