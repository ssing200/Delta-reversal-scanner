import streamlit as st
import requests
import pandas as pd
import numpy as np
import time

BASE_URL = "https://api.india.delta.exchange"
HEADERS = {
    "Accept": "application/json",
    "User-Agent": "Delta-Reversal-Scanner/1.1",
}

st.set_page_config(
    page_title="Delta Reversal Scanner",
    page_icon="🔥",
    layout="wide",
)

CACHE_TTL = 20
TOP_COINS = 25
DEFAULT_DEPTH = 15


def api_get(path, params=None, timeout=12):
    try:
        response = requests.get(
            BASE_URL + path,
            params=params or {},
            headers=HEADERS,
            timeout=timeout,
        )
        if response.status_code != 200:
            return None
        payload = response.json()
        if not payload.get("success", True):
            return None
        return payload.get("result")
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
        if item.get("state") != "live":
            continue
        if item.get("trading_status") != "operational":
            continue

        symbol = item.get("symbol")
        if not symbol:
            continue

        try:
            default_lev = float(item.get("default_leverage") or 20)
        except Exception:
            default_lev = 20.0

        max_lev = default_lev
        try:
            im = float(item.get("initial_margin") or 0)
            if im > 0:
                calculated = round(100 / im)
                max_lev = max(calculated, default_lev)
        except Exception:
            max_lev = default_lev

        rows.append({
            "Coin": symbol,
            "ID": item.get("id"),
            "Max Leverage": int(max_lev),
        })

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
        except Exception:
            continue
        if price <= 0:
            continue

        funding = np.nan
        try:
            fr = item.get("funding_rate")
            if fr is not None:
                funding = float(fr)
        except Exception:
            pass

        rows.append({
            "Coin": symbol,
            "Price": price,
            "24H Volume": volume,
            "OI": oi,
            "Funding": funding,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["Vol/OI"] = df["24H Volume"] / df["OI"].replace(0, np.nan)
    return df


@st.cache_data(ttl=CACHE_TTL)
def get_history(symbol, resolution="5m", hours=48):
    end = int(time.time())
    start = end - int(hours * 3600)
    result = api_get("/v2/history/candles", {
        "resolution": resolution,
        "symbol": symbol,
        "start": start,
        "end": end,
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
        df["time"] = pd.to_numeric(df["time"], errors="coerce")
        df = df.sort_values("time")
    required = ["open", "high", "low", "close"]
    if not all(c in df.columns for c in required):
        return pd.DataFrame()
    df = df.dropna(subset=required)
    if "time" in df.columns:
        df = df.drop_duplicates("time")
    return df.reset_index(drop=True)


@st.cache_data(ttl=CACHE_TTL)
def get_oi_history(symbol, hours=48):
    end = int(time.time())
    start = end - int(hours * 3600)
    result = api_get("/v2/history/candles", {
        "resolution": "15m",
        "symbol": f"OI:{symbol}",
        "start": start,
        "end": end,
    })
    if not result:
        return pd.DataFrame()
    df = pd.DataFrame(result)
    if df.empty or "close" not in df.columns:
        return pd.DataFrame()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    if "time" in df.columns:
        df["time"] = pd.to_numeric(df["time"], errors="coerce")
        df = df.sort_values("time")
    return df.dropna(subset=["close"]).reset_index(drop=True)


@st.cache_data(ttl=10)
def get_orderbook(symbol, depth=15):
    result = api_get(f"/v2/l2orderbook/{symbol}", {"depth": int(depth)})
    return result if isinstance(result, dict) else None


def parse_orderbook(symbol, depth=15):
    data = get_orderbook(symbol, depth)
    if not data:
        return None
    bids = data.get("buy") or []
    asks = data.get("sell") or []
    bid_rows = []
    ask_rows = []
    for row in bids:
        try:
            bid_rows.append({"Price": float(row["price"]), "Size": float(row["size"])})
        except Exception:
            pass
    for row in asks:
        try:
            ask_rows.append({"Price": float(row["price"]), "Size": float(row["size"])})
        except Exception:
            pass
    if not bid_rows or not ask_rows:
        return None
    bid_df = pd.DataFrame(bid_rows).sort_values("Price", ascending=False).reset_index(drop=True)
    ask_df = pd.DataFrame(ask_rows).sort_values("Price", ascending=True).reset_index(drop=True)
    bid_depth = float(bid_df["Size"].sum())
    ask_depth = float(ask_df["Size"].sum())
    total = bid_depth + ask_depth
    imbalance = (bid_depth - ask_depth) / total * 100 if total > 0 else 0.0
    best_bid = float(bid_df["Price"].max())
    best_ask = float(ask_df["Price"].min())
    mid = (best_bid + best_ask) / 2
    return {
        "symbol": symbol,
        "bid_df": bid_df,
        "ask_df": ask_df,
        "bid_depth": bid_depth,
        "ask_depth": ask_depth,
        "imbalance": imbalance,
        "best_bid": best_bid,
        "
