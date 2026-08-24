import streamlit as st
import requests
import pandas as pd
import numpy as np
import time

# ============================================================
# DELTA REVERSAL SCANNER - FINAL
# >20x completely free from Vol/OI filter
# ============================================================

BASE_URL = "https://api.india.delta.exchange"
HEADERS = {
    "Accept": "application/json",
    "User-Agent": "Delta-Reversal-Scanner-Final/2.3",
}

st.set_page_config(
    page_title="Delta Reversal Scanner",
    page_icon="🔥",
    layout="wide",
)

CACHE_TTL = 20
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
    except (requests.RequestException, ValueError, TypeError):
        return None


@st.cache_data(ttl=CACHE_TTL)
def get_perpetual_products():
    result = api_get("/v2/products", {"contract_types": "perpetual_futures"})
    if not result:
        return pd.DataFrame(columns=["Coin", "ID", "Max Leverage", "Default Leverage"])

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

        default_lev = 20.0
        try:
            default_lev = float(item.get("default_leverage") or 20)
        except:
            pass

        max_lev = default_lev
        try:
            im = float(item.get("initial_margin") or 0)
            if im > 0:
                calculated = round(100 /
