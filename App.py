import time
import requests
import pandas as pd
import numpy as np
import streamlit as st


BASE_URL = "https://api.india.delta.exchange"

HEADERS = {
    "Accept": "application/json",
    "User-Agent": "Delta-Scanner/1.0",
}

CACHE_TTL = 30
LEVERAGE_LIMIT = 20.0


st.set_page_config(
    page_title="Delta Leverage Scanner",
    page_icon="⚡",
    layout="wide",
)


def api_get(path, params=None, timeout=10):
    try:
        response = requests.get(
            BASE_URL + path,
            params=params or {},
            headers=HEADERS,
            timeout=timeout,
        )

        response.raise_for_status()
        data = response.json()

        if not isinstance(data, dict):
            return None

        if data.get("success") is False:
            return None

        return data.get("result")

    except Exception:
        return None


def number(value, default=np.nan):
    try:
        return float(value)
    except Exception:
        return default


@st.cache_data(ttl=CACHE_TTL)
def get_products():
    result = api_get(
        "/v2/products",
        {
            "page_size": 100,
        },
    )

    if not isinstance(result, list):
        return pd.DataFrame()

    rows = []

    for item in result:
        if not isinstance(item, dict):
            continue

        symbol = item.get("symbol")

        if not symbol:
            continue

        contract_type = str(
            item.get("contract_type", "")
        ).lower()

        state = str(
            item.get("state", "")
        ).lower()

        trading_status = str(
            item.get("trading_status", "")
        ).lower()

        if contract_type != "perpetual_futures":
            continue

        if state and state != "live":
            continue

        if (
            trading_status
            and trading_status != "operational"
        ):
            continue

        leverage = item.get("max_leverage")

        if leverage is None:
            leverage = item.get("default_leverage")

        if leverage is None:
            leverage = item.get("leverage")

        rows.append(
            {
                "Coin": symbol,
                "Product ID": item.get("id"),
                "Max Leverage": number(leverage),
                "Contract Value": item.get(
                    "contract_value"
                ),
                "Tick Size": item.get(
                    "tick_size"
                ),
            }
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    return (
        df.drop_duplicates("Coin")
        .sort_values(
            "Max Leverage",
            ascending=False,
            na_position="last",
        )
        .reset_index(drop=True)
    )


@st.cache_data(ttl=CACHE_TTL)
def get_tickers():
    result = api_get(
        "/v2/tickers",
        {
            "page_size": 100,
        },
    )

    if not isinstance(result, list):
        return pd.DataFrame()

    rows = []

    for item in result:
        if not isinstance(item, dict):
            continue

        symbol = item.get("symbol")

        if not symbol:
            continue

        price = number(
            item.get("close")
            or item.get("mark_price")
        )

        volume = number(
            item.get("volume_24h")
            or item.get("volume"),
            default=0.0,
        )

        oi = number(
            item.get("open_interest")
            or item.get("oi"),
            default=0.0,
        )

        funding = number(
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

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    df["Vol/OI"] = (
        df["24H Volume"]
        / df["OI"].replace(0, np.nan)
    )

    return df


st.title("⚡ Delta Leverage Scanner")

st.caption(
    "Fast version: leverage data loads first. "
    "Deep scanner button ke baad chalega."
)


with st.sidebar:
    st.header("Settings")

    min_vol_oi = st.number_input(
        "Minimum Vol/OI",
        min_value=0.0,
        value=0.0,
        step=1.0,
    )

    show_debug = st.checkbox(
        "Show API debug information",
        value=False,
    )

    refresh = st.button(
        "🔄 Refresh Data"
    )

    if refresh:
        st.cache_data.clear()
        st.rerun()


with st.spinner("Loading product and ticker data..."):
    products = get_products()
    tickers = get_tickers()


if products.empty:
    st.error(
        "Products API se data nahi mila. "
        "Internet/API response check karein."
    )

    if show_debug:
        st.write("Products response empty hai.")

    st.stop()


if tickers.empty:
    st.error(
        "Tickers API se data nahi mila."
    )

    if show_debug:
        st.write("Tickers response empty hai.")

    st.stop()


market = products.merge(
    tickers,
    on="Coin",
    how="left",
)

market = market.dropna(
    subset=["Price"]
)

if min_vol_oi > 0:
    market = market[
        market["Vol/OI"].fillna(0)
        > min_vol_oi
    ]

st.metric(
    "Loaded contracts",
    len(market),
)

st.metric(
    "Leverage above 20x",
    int(
        (
            market["Max Leverage"]
            > LEVERAGE_LIMIT
        ).sum()
    ),
)


tab1, tab2, tab3 = st.tabs(
    [
        "⚡ Leverage Table",
        "📊 Market Data",
        "🔍 Deep Scanner",
    ]
)


with tab1:
    st.subheader(
        "Contracts With Maximum Leverage Above 20x"
    )

    leverage_df = market[
        pd.to_numeric(
            market["Max Leverage"],
            errors="coerce",
        )
        > LEVERAGE_LIMIT
    ].copy()

    if leverage_df.empty:
        st.warning(
            "20x se zyada leverage data nahi mila."
        )

        st.info(
            "Agar Max Leverage column blank hai, "
            "to API response me leverage field "
            "available nahi hai."
        )
    else:
        columns = [
            "Coin",
            "Price",
            "Max Leverage",
            "24H Volume",
            "OI",
            "Vol/OI",
            "Funding",
        ]

        columns = [
            column
            for column in columns
            if column in leverage_df.columns
        ]

        st.dataframe(
            leverage_df[
                columns
            ].sort_values(
                "Max Leverage",
                ascending=False,
            ),
            use_container_width=True,
            hide_index=True,
        )


with tab2:
    st.subheader("📊 Complete Market Metadata")

    st.dataframe(
        market.sort_values(
            "24H Volume",
            ascending=False,
        ),
        use_container_width=True,
        hide_index=True,
    )


with tab3:
    st.subheader("🔍 Deep Scanner")

    st.warning(
        "Deep scanner multiple API calls karta hai. "
        "Page load ke time automatically run nahi hoga."
    )

    scan_limit = st.slider(
        "Coins to scan",
        min_value=1,
        max_value=min(10, len(market)),
        value=min(3, len(market)),
    )

    if st.button("▶️ Start Deep Scan"):
        st.info(
            f"{scan_limit} c
