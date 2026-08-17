import streamlit as st
import requests
import pandas as pd
import time

BASE_URL = "https://api.india.delta.exchange"

st.set_page_config(
    page_title="Delta Reversal Scanner",
    layout="wide"
)

st.title("🔥 Delta Reversal Scanner")
st.caption("Price + Volume + OI based reversal watch")

HEADERS = {
    "Accept": "application/json",
    "User-Agent": "Delta-Reversal-Scanner/1.0"
}


def api_get(path, params=None):
    try:
        r = requests.get(
            BASE_URL + path,
            params=params,
            headers=HEADERS,
            timeout=10
        )

        if r.status_code != 200:
            return None

        data = r.json()

        if data.get("success") is False:
            return None

        return data.get("result", [])

    except Exception:
        return None


@st.cache_data(ttl=30)
def get_tickers():

    result = api_get("/v2/tickers")

    if not result:
        return pd.DataFrame()

    rows = []

    for x in result:

        symbol = x.get("symbol")

        if not symbol:
            continue

        # Perpetual futures only
        product_type = str(
            x.get("contract_type",
            x.get("product_type", ""))
        ).lower()

        if product_type and "perpetual" not in product_type:
            continue

        price = x.get(
            "close",
            x.get("mark_price",
            x.get("spot_price", 0))
        )

        volume = x.get(
            "volume_24h",
            x.get("volume", 0)
        )

        oi = x.get(
            "open_interest",
            x.get("oi", 0)
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
            "Symbol": symbol,
            "Price": price,
            "24H Volume": volume,
            "OI": oi
        })

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    df["Vol/OI"] = (
        df["24H Volume"] /
        df["OI"].replace(0, pd.NA)
    )

    df = df.dropna(subset=["Vol/OI"])

    # Highest volume first
    df = df.sort_values(
        "24H Volume",
        ascending=False
    )

    return df


def get_candles(symbol, resolution="15m", hours=12):

    end = int(time.time())
    start = end - hours * 60 * 60

    result = api_get(
        "/v2/history/candles",
        {
            "resolution": resolution,
            "symbol": symbol,
            "start": start,
            "end": end
        }
    )

    if not result:
        return pd.DataFrame()

    df = pd.DataFrame(result)

    if df.empty:
        return df

    for col in [
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    df = df.dropna(
        subset=["open", "high", "low", "close"]
    )

    # API can return newest first
    df = df.sort_values("time")

    return df


def get_oi_history(symbol):

    end = int(time.time())
    start = end - 12 * 60 * 60

    result = api_get(
        "/v2/history/candles",
        {
            "resolution": "15m",
            "symbol": "OI:" + symbol,
            "start": start,
            "end": end
        }
    )

    if not result:
        return pd.DataFrame()

    df = pd.DataFrame(result)

    if df.empty:
        return df

    # OI history is represented through candle values.
    # Usually close is the useful latest OI value.
    if "close" in df.columns:
        df["close"] = pd.to_numeric(
            df["close"],
            errors="coerce"
        )

    df = df.dropna(subset=["close"])
    df = df.sort_values("time")

    return df


def analyze_symbol(symbol):

    candles = get_candles(symbol)

    if candles.empty or len(candles) < 8:
        return None

    # Last closed candle
    last = candles.iloc[-1]

    # Previous 3 candles
    previous = candles.iloc[-4]

    price_now = float(last["close"])
    price_previous = float(previous["close"])

    price_change = (
        (price_now - price_previous)
        / price_previous
    ) * 100

    # Average volume of previous candles
    previous_volumes = candles[
        "volume"
    ].iloc[-8:-1]

    avg_volume = previous_volumes.mean()

    if avg_volume <= 0:
        return None

    volume_ratio = (
        float(last["volume"])
        / float(avg_volume)
    )

    # Candle rejection
    bullish_candle = (
        float(last["close"])
        > float(last["open"])
    )

    bearish_candle = (
        float(last["close"])
        < float(last["open"])
    )

    # OI history
    oi = get_oi_history(symbol)

    oi_change = 0.0

    if not oi.empty and len(oi) >= 5:

        oi_now = float(
            oi["close"].iloc[-1]
        )

        oi_old = float(
            oi["close"].iloc[-5]
        )

        if oi_old != 0:
            oi_change = (
                (oi_now - oi_old)
                / abs(oi_old)
            ) * 100

    # -------------------------
    # SIGNAL LOGIC
    # -------------------------

    score = 0
    signal = "⚪ NO SIGNAL"
    reason = "No strong reversal confirmation"

    # LONG REVERSAL
    #
    # Price fell
    # Volume increased
    # OI decreased
    # Last candle bullish
    #
    if (
        price_change <= -0.80
        and volume_ratio >= 1.30
        and oi_change <= -0.50
        and bullish_candle
    ):

        score += 4
        signal = "🟢 LONG REVERSAL WATCH"

        reason = (
            "Price fell + volume spike + "
            "OI decreased + bullish rejection"
        )

    # SHORT REVERSAL
    #
    # Price rose
    # Volume increased
    # OI decreased
    # Last candle bearish
    #
    elif (
        price_change >= 0.80
        and volume_ratio >= 1.30
        and oi_change <= -0.50
        and bearish_candle
    ):

        score += 4
        signal = "🔴 SHORT REVERSAL WATCH"

        reason = (
            "Price rose + volume spike + "
            "OI decreased + bearish rejection"
        )

    # EARLY LONG
    elif (
        price_change <= -0.80
        and volume_ratio >= 1.50
        and oi_change <= -0.50
    ):

        score = 2
        signal = "🟡 EARLY LONG WATCH"

        reason = (
            "Price down + volume high + "
            "OI falling"
        )

    # EARLY SHORT
    elif (
        price_change >= 0.80
        and volume_ratio >= 1.50
        and oi_change <= -0.50
    ):

        score = 2
        signal = "🟠 EARLY SHORT WATCH"

        reason = (
            "Price up + volume high + "
            "OI falling"
        )

    return {
        "Symbol": symbol,
        "Price Change %": round(price_change, 2),
        "Volume Ratio": round(volume_ratio, 2),
        "OI Change %": round(oi_change, 2),
        "Signal": signal,
        "Score": score,
        "Reason": reason
    }


# =====================================
# MAIN DATA
# =====================================

df = get_tickers()

if df.empty:

    st.error(
        "Delta API se data nahi aa raha."
    )

    st.stop()


# =====================================
# MARKET OVERVIEW
# =====================================

st.subheader("📊 Market Overview")

display_df = df.head(30).copy()

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)


# =====================================
# REVERSAL SCANNER
# =====================================

st.subheader("🎯 Reversal Scanner")

st.info(
    "Scanner top-volume contracts ko 15-minute "
    "price, volume aur OI behaviour ke basis par check karta hai."
)

# Limit API calls
scan_list = df.head(10)["Symbol"].tolist()

signals = []

progress = st.progress(0)

for i, symbol in enumerate(scan_list):

    result = analyze_symbol(symbol)

    if result:
        signals.append(result)

    progress.progress(
        int((i + 1) / len(scan_list) * 100)
    )

progress.empty()


signals_df = pd.DataFrame(signals)


if signals_df.empty:

    st.warning(
        "Abhi strong reversal data available nahi hai."
    )

else:

    # Strong signals first
    signals_df = signals_df.sort_values(
        ["Score", "Price Change %"],
        ascending=[False, False]
    )

    st.dataframe(
        signals_df,
        use_container_width=True,
        hide_index=True
    )


# =====================================
# STRONG SIGNALS
# =====================================

st.subheader("🔥 Strong Signals")

if not signals_df.empty:

    strong = signals_df[
        signals_df["Score"] >= 4
    ]

    if strong.empty:

        st.info(
            "Abhi koi confirmed-style reversal watch nahi mila."
        )

    else:

        st.dataframe(
            strong,
            use_container_width=True,
            hide_index=True
        )


# =====================================
# EXPLANATION
# =====================================

st.subheader("🧠 Signal Logic")

st.write(
    """
    🟢 LONG REVERSAL WATCH:
    Price down + volume spike + OI falling +
    bullish candle rejection.

    🔴 SHORT REVERSAL WATCH:
    Price up + volume spike + OI falling +
    bearish candle rejection.

    🟡 / 🟠 EARLY WATCH:
    Price and volume/OI conditions support a
    possible reversal, but candle confirmation
    is missing.
    """
)

st.warning(
    "⚠️ Ye prediction ya guaranteed-profit system nahi hai. "
    "Signal ko price action, liquidity, market structure "
    "aur risk management se confirm karein."
)

st.caption(
    "Data source: Delta Exchange India public market API"
)
