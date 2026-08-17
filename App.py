
import streamlit as st
import requests
import pandas as pd
import numpy as np
import time

# =========================
# SETTINGS
# =========================

BASE_URL = "https://api.india.delta.exchange"

st.set_page_config(
    page_title="Delta Reversal Scanner",
    layout="wide"
)

st.title("🔥 Delta Reversal Scanner")
st.caption("Live market-data scanner | Read-only")

# =========================
# API FUNCTIONS
# =========================

@st.cache_data(ttl=30)
def get_perpetuals():

    url = BASE_URL + "/v2/products"

    r = requests.get(
        url,
        headers={"Accept": "application/json"},
        timeout=15
    )

    r.raise_for_status()

    products = r.json().get("result", [])

    rows = []

    for p in products:

        if (
            p.get("contract_type") == "perpetual_futures"
            and p.get("state") == "live"
            and p.get("trading_status") == "operational"
        ):

            rows.append({
                "id": p.get("id"),
                "symbol": p.get("symbol"),
                "underlying": p.get(
                    "underlying_asset", {}
                ).get("symbol")
            })

    return pd.DataFrame(rows)


@st.cache_data(ttl=15)
def get_tickers():

    url = BASE_URL + "/v2/tickers"

    r = requests.get(
        url,
        params={
            "contract_types": "perpetual_futures"
        },
        headers={"Accept": "application/json"},
        timeout=20
    )

    r.raise_for_status()

    result = r.json().get("result", [])

    rows = []

    for x in result:

        rows.append({
            "symbol": x.get("symbol"),
            "price": x.get("close"),
            "volume_24h": x.get("volume"),
            "oi": x.get("oi"),
            "mark_price": x.get("mark_price")
        })

    return pd.DataFrame(rows)


@st.cache_data(ttl=20)
def get_candles(symbol, resolution="15m", limit=100):

    end_time = int(time.time())

    # 15 minute candles
    start_time = end_time - (limit * 15 * 60)

    url = BASE_URL + "/v2/history/candles"

    r = requests.get(
        url,
        params={
            "resolution": resolution,
            "symbol": symbol,
            "start": start_time,
            "end": end_time
        },
        headers={"Accept": "application/json"},
        timeout=15
    )

    if r.status_code != 200:
        return pd.DataFrame()

    data = r.json().get("result", [])

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)

    if "time" in df.columns:
        df["time"] = pd.to_datetime(
            df["time"],
            unit="s",
            utc=True
        )

    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    df = df.sort_values("time").reset_index(drop=True)

    return df


# =========================
# STRUCTURE BREAK
# =========================

def structure_signal(df):

    if len(df) < 20:
        return "WAIT", 0

    recent = df.iloc[-1]

    previous_high = df["high"].iloc[-10:-2].max()
    previous_low = df["low"].iloc[-10:-2].min()

    bullish_bos = recent["close"] > previous_high
    bearish_bos = recent["close"] < previous_low

    if bullish_bos:
        return "BULLISH BOS", 2

    if bearish_bos:
        return "BEARISH BOS", -2

    return "WAIT", 0


# =========================
# REVERSAL ANALYSIS
# =========================

def analyze_coin(symbol):

    try:

        candles = get_candles(symbol)

        if candles.empty or len(candles) < 30:
            return None

        ticker_df = ticker_data[
            ticker_data["symbol"] == symbol
        ]

        if ticker_df.empty:
            return None

        t = ticker_df.iloc[0]

        price = float(t.get("price") or 0)
        volume = float(t.get("volume_24h") or 0)
        oi = float(t.get("oi") or 0)

        # -------------------------
        # Volume/OI
        # -------------------------

        if oi > 0:
            volume_oi = volume / oi
        else:
            volume_oi = 0

        # -------------------------
        # Candle data
        # -------------------------

        last = candles.iloc[-1]
        prev = candles.iloc[-2]

        avg_volume = candles["volume"].iloc[-21:-1].mean()

        if avg_volume > 0:
            volume_spike = last["volume"] / avg_volume
        else:
            volume_spike = 0

        # -------------------------
        # Price movement
        # -------------------------

        price_change = (
            (last["close"] - prev["close"])
            / prev["close"]
        ) * 100

        # -------------------------
        # Structure
        # -------------------------

        bos_signal, bos_score = structure_signal(candles)

        # -------------------------
        # Reversal score
        # -------------------------

        score = 0

        # Volume spike
        if volume_spike >= 2:
            score += 2

        elif volume_spike >= 1.5:
            score += 1

        # Bullish / bearish candle
        if last["close"] > last["open"]:
            score += 1
        else:
            score -= 1

        # Structure
        score += bos_score

        # -------------------------
        # Signal
        # -------------------------

        if score >= 3:
            signal = "🟢 BULLISH REVERSAL"

        elif score <= -3:
            signal = "🔴 BEARISH REVERSAL"

        else:
            signal = "⚪ WAIT"

        return {
            "Symbol": symbol,
            "Price": round(price, 6),
            "24H Volume": round(volume, 2),
            "OI": round(oi, 2),
            "Vol/OI": round(volume_oi, 3),
            "Vol Spike": round(volume_spike, 2),
            "Price %": round(price_change, 2),
            "Structure": bos_signal,
            "Score": score,
            "Signal": signal
        }

    except Exception:
        return None


# =========================
# LOAD DATA
# =========================

try:

    products_df = get_perpetuals()

    ticker_data = get_tickers()

except Exception as e:

    st.error("Delta API connection error")
    st.code(str(e))
    st.stop()


# =========================
# CONTROLS
# =========================

st.sidebar.header("Scanner Settings")

max_coins = st.sidebar.slider(
    "Coins to scan",
    10,
    220,
    30
)

min_volume_oi = st.sidebar.number_input(
    "Minimum Volume/OI",
    value=1.0,
    step=0.1
)

if st.sidebar.button("🔄 Refresh"):

    st.cache_data.clear()
    st.rerun()


# =========================
# SCAN
# =========================

symbols = products_df["symbol"].tolist()

# ticker available symbols only
available = set(
    ticker_data["symbol"].dropna().tolist()
)

symbols = [
    s for s in symbols
    if s in available
]

symbols = symbols[:max_coins]

results = []

progress = st.progress(0)

for i, symbol in enumerate(symbols):

    result = analyze_coin(symbol)

    if result is not None:
        results.append(result)

    progress.progress(
        (i + 1) / len(symbols)
    )


progress.empty()


# =========================
# RESULTS
# =========================

if not results:

    st.warning("No data available.")
    st.stop()


df = pd.DataFrame(results)

# Volume/OI filter
df = df[
    df["Vol/OI"] >= min_volume_oi
]

# Highest score first
df = df.sort_values(
    "Score",
    ascending=False
)


st.subheader("🔥 Reversal Scanner")

st.dataframe(
    df,
    use_container_width=True,
    hide_index=True
)


# =========================
# STRONG SIGNALS
# =========================

st.subheader("🎯 Strong Signals")

strong = df[
    (df["Score"] >= 3)
    | (df["Score"] <= -3)
]

if strong.empty:

    st.info("Abhi strong reversal signal nahi mila.")

else:

    st.dataframe(
        strong,
        use_container_width=True,
        hide_index=True
    )


# =========================
# DISCLAIMER
# =========================

st.caption(
    "⚠️ This is a market scanner, not a guaranteed-profit system. "
    "Signals should be confirmed with price action, liquidity and risk management."
)
