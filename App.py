import time

import numpy as np
import pandas as pd
import requests
import streamlit as st


BASE_URL = "https://api.india.delta.exchange"

HEADERS = {
    "Accept": "application/json",
    "User-Agent": "Delta-Reversal-Scanner/2.0",
}

CACHE_TTL = 20
TOP_COINS = 25
DEFAULT_DEPTH = 15
LEVERAGE_LIMIT = 20.0


st.set_page_config(
    page_title="Delta Reversal Scanner",
    page_icon="🔥",
    layout="wide",
)


def api_get(path, params=None, timeout=15):
    try:
        response = requests.get(
            BASE_URL + path,
            params=params or {},
            headers=HEADERS,
            timeout=timeout,
        )

        response.raise_for_status()
        payload = response.json()

        if not isinstance(payload, dict):
            return None

        if payload.get("success") is False:
            return None

        return payload.get("result")

    except (requests.RequestException, ValueError, TypeError):
        return None


def to_float(value, default=np.nan):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@st.cache_data(ttl=CACHE_TTL)
def get_perpetual_products():
    columns = [
        "Coin",
        "ID",
        "Max Leverage",
        "Contract Value",
        "Tick Size",
    ]

    result = api_get("/v2/products")

    if not isinstance(result, list):
        return pd.DataFrame(columns=columns)

    rows = []

    for item in result:
        if not isinstance(item, dict):
            continue

        if item.get("contract_type") != "perpetual_futures":
            continue

        if item.get("state") != "live":
            continue

        if item.get("trading_status") != "operational":
            continue

        symbol = item.get("symbol")

        if not symbol:
            continue

        # Delta response commonly provides default_leverage.
        # Some responses may provide max_leverage.
        leverage_value = item.get("max_leverage")

        if leverage_value is None:
            leverage_value = item.get("default_leverage")

        if leverage_value is None:
            leverage_value = item.get("leverage")

        rows.append(
            {
                "Coin": symbol,
                "ID": item.get("id"),
                "Max Leverage": to_float(leverage_value),
                "Contract Value": item.get("contract_value"),
                "Tick Size": item.get("tick_size"),
            }
        )

    if not rows:
        return pd.DataFrame(columns=columns)

    df = pd.DataFrame(rows)

    df["Max Leverage"] = pd.to_numeric(
        df["Max Leverage"],
        errors="coerce",
    )

    return (
        df.drop_duplicates("Coin")
        .reset_index(drop=True)
    )


@st.cache_data(ttl=CACHE_TTL)
def get_tickers():
    result = api_get("/v2/tickers")

    if not isinstance(result, list):
        return pd.DataFrame()

    rows = []

    for item in result:
        if not isinstance(item, dict):
            continue

        symbol = item.get("symbol")

        if not symbol:
            continue

        price = to_float(
            item.get("close")
            or item.get("mark_price")
        )

        volume = to_float(
            item.get("volume_24h")
            or item.get("volume"),
            default=0.0,
        )

        oi = to_float(
            item.get("open_interest")
            or item.get("oi"),
            default=0.0,
        )

        funding = to_float(
            item.get("funding_rate")
        )

        if not np.isfinite(price) or price <= 0:
            continue

        rows.append(
            {
                "Coin": symbol,
                "Price": price,
                "24H Volume": volume,
                "OI": oi,
                "Funding": funding,
            }
        )

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    df["Vol/OI"] = (
        df["24H Volume"]
        / df["OI"].replace(0, np.nan)
    )

    return df


@st.cache_data(ttl=CACHE_TTL)
def get_history(symbol, resolution="5m", hours=48):
    end = int(time.time())
    start = end - int(hours * 3600)

    result = api_get(
        "/v2/history/candles",
        {
            "resolution": resolution,
            "symbol": symbol,
            "start": start,
            "end": end,
        },
    )

    if not isinstance(result, list):
        return pd.DataFrame()

    df = pd.DataFrame(result)

    if df.empty:
        return df

    for column in [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]:
        if column in df.columns:
            df[column] = pd.to_nume
