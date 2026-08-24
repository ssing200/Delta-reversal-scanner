import streamlit as st
import requests
import pandas as pd
import numpy as np
import time

BASE_URL = "https://api.india.delta.exchange"
HEADERS = {"Accept": "application/json", "User-Agent": "Delta-Scanner/Final"}

st.set_page_config(page_title="Delta Reversal Scanner", page_icon="🔥", layout="wide")

CACHE_TTL = 20
DEFAULT_DEPTH = 15

def api_get(path, params=None, timeout=12):
    try:
        r = requests.get(BASE_URL + path, params=params or {}, headers=HEADERS, timeout=timeout)
        if r.status_code != 200:
            return None
        data = r.json()
        if not data.get("success", True):
            return None
        return data.get("result")
    except Exception:
        return None

@st.cache_data(ttl=CACHE_TTL)
def get_perpetual_products():
    result = api_get("/v2/products", {"contract_types": "perpetual_futures"})
    if not result:
        return pd.DataFrame(columns=["Coin", "ID", "Max Leverage"])
    rows = []
    for item in result:
        if item.get("contract_type") != "perpetual_futures":
            continue
        if item.get("state") != "live" or item.get("trading_status") != "operational":
            continue
        symbol = item.get("symbol")
        if not symbol:
            continue
        try:
            default_lev = float(item.get("default_leverage") or 20)
        except:
            default_lev = 20.0
        max_lev = default_lev
        try:
            im = float(item.get("initial_margin") or 0)
            if im > 0:
                calculated = round(100 / im)
                max_lev = max(calculated, default_lev)
        except:
            max_lev = default_lev
        rows.append({"Coin": symbol, "ID": item.get("id"), "Max Leverage": int(max_lev)})
    if not rows:
        return pd.DataFrame(columns=["Coin", "ID", "Max Leverage"])
    return pd.DataFrame(rows).drop_duplicates("Coin").reset_index(drop=True)

@st.cache_data(ttl=CACHE_TTL)
def get_tickers():
    result = api_get("/v2/tickers", {"contract_types": "perpetual_futures"})
    if not result:
        return pd.DataFrame()
    rows = []
    for item in result:
        symbol = item.get("symbol")
        if not symbol:
            continue
        try:
            price = float(item.get("close") or item.get("mark_price") or 0)
            volume = float(item.get("volume_24h") or item.get("volume") or 0)
            oi = float(item.get("open_interest") or item.get("oi") or 0)
        except:
            continue
        if price <= 0:
            continue
        funding = np.nan
        try:
            fr = item.get("funding_rate")
            if fr is not None:
                funding = float(fr)
        except:
            pass
        rows.append({"Coin": symbol, "Price": price, "24H Volume": volume, "OI": oi, "Funding": funding})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["Vol/OI"] = df["24H Volume"] / df["OI"].replace(0, np.nan)
    return df

@st.cache_data(ttl=CACHE_TTL)
def get_history(symbol, resolution="5m", hours=48):
    end = int(time.time())
    start = end - int(hours * 3600)
    result = api_get("/v2/history/candles", {"
