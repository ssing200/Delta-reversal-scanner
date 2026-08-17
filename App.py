import streamlit as st
import requests
import pandas as pd

BASE_URL = "https://api.india.delta.exchange"

st.set_page_config(
    page_title="Delta Coin Scanner",
    layout="wide"
)

st.title("🔥 Delta Coin Scanner")
st.caption("Price + Volume + OI + Reversal Watch")


def get_data():

    try:
        response = requests.get(
            BASE_URL + "/v2/tickers",
            timeout=15
        )

        if response.status_code != 200:
            return pd.DataFrame()

        data = response.json()

        if not data.get("success"):
            return pd.DataFrame()

        rows = []

        for p in data.get("result", []):

            symbol = p.get("symbol")

            if not symbol:
                continue

            # Perpetual futures only
            contract = str(
                p.get("contract_type", "")
            ).lower()

            if contract != "perpetual_futures":
                continue

            price = p.get(
                "close",
                p.get("mark_price", 0)
            )

            volume = p.get(
                "volume_24h",
                p.get("volume", 0)
            )

            oi = p.get(
                "open_interest",
                p.get("oi", 0)
            )

            try:
                price = float(price or 0)
                volume = float(volume or 0)
                oi = float(oi or 0)
            except:
                continue

            if price <= 0:
                continue

            rows.append({
                "Coin": symbol,
                "Price": price,
                "24H Volume": volume,
                "OI": oi
            })

        df = pd.DataFrame(rows)

        if df.empty:
            return df

        # Volume / OI
        df["Vol/OI"] = (
            df["24H Volume"] /
            df["OI"].replace(0, pd.NA)
        )

        df = df.dropna(
            subset=["Vol/OI"]
        )

        # Highest volume first
        df = df.sort_values(
            "24H Volume",
            ascending=False
        )

        return df

    except Exception as e:

        st.error(
            "API connection error: " + str(e)
        )

        return pd.DataFrame()


# -----------------------------
# GET MARKET DATA
# -----------------------------

df = get_data()


if df.empty:

    st.error(
        "❌ Delta se data nahi mil raha."
    )

    st.stop()


# -----------------------------
# TOP COINS
# -----------------------------

st.subheader("📊 Top Perpetual Coins")

st.dataframe(
    df.head(50),
    use_container_width=True,
    hide_index=True
)


# -----------------------------
# REVERSAL WATCH
# -----------------------------

st.subheader("🎯 Reversal Watch")


# Vol/OI ज्यादा होने पर market activity ज्यादा है
# यह सिर्फ WATCH signal है, guaranteed reversal नहीं।

watch = df[
    df["Vol/OI"] >= 1
].copy()


if watch.empty:

    st.info(
        "Abhi koi strong Vol/OI watch setup nahi mila."
    )

else:

    watch["Signal"] = "👀 WATCH"

    watch["Reason"] = (
        "High trading activity relative to OI"
    )

    st.dataframe(
        watch[
            [
                "Coin",
                "Price",
                "24H Volume",
                "OI",
                "Vol/OI",
                "Signal",
                "Reason"
            ]
        ].head(30),
        use_container_width=True,
        hide_index=True
    )


# -----------------------------
# REFRESH
# -----------------------------

st.divider()

if st.button("🔄 Refresh Data"):

    st.rerun()


st.warning(
    "⚠️ WATCH signal reversal की guarantee नहीं देता। "
    "Trade लेने से पहले price action, structure break, "
    "liquidity और risk management से confirmation करें."
)
