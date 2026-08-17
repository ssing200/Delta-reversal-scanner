import streamlit as st
import requests
import pandas as pd
import time

BASE_URL = "https://api.india.delta.exchange"

st.set_page_config(
    page_title="Delta 220 Reversal Scanner",
    layout="wide"
)

st.title("🔥 Delta 220 Reversal Scanner")
st.caption(
    "220 Coins → 1H Trend → 15m Liquidity Sweep → "
    "5m BOS → OI Displacement → Entry Zone"
)

# =========================================================
# SETTINGS
# =========================================================

HEADERS = {
    "Accept": "application/json",
    "User-Agent": "Delta-220-Reversal-Scanner/3.0"
}

# Cache keeps API load under control
CACHE_SECONDS = 120

# Deep analysis only for promising candidates
DEEP_SCAN_LIMIT = 30


# =========================================================
# API FUNCTION
# =========================================================

def api_get(path, params=None):

    try:

        response = requests.get(
            BASE_URL + path,
            params=params,
            headers=HEADERS,
            timeout=12
        )

        if response.status_code != 200:
            return None

        data = response.json()

        if data.get("success") is False:
            return None

        return data.get("result", [])

    except Exception:

        return None


# =========================================================
# GET ALL LIVE PERPETUALS
# =========================================================

@st.cache_data(ttl=CACHE_SECONDS)
def get_all_perpetuals():

    result = api_get("/v2/products")

    if not result:
        return pd.DataFrame()

    rows = []

    for p in result:

        if p.get("contract_type") != "perpetual_futures":
            continue

        if p.get("state") != "live":
            continue

        if p.get("trading_status") != "operational":
            continue

        symbol = p.get("symbol")

        if not symbol:
            continue

        rows.append({
            "Coin": symbol,
            "ID": p.get("id"),
            "Underlying": p.get(
                "underlying_asset",
                {}
            ).get("symbol", "")
        })

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    return df.drop_duplicates(
        subset=["Coin"]
    )


# =========================================================
# TICKER DATA
# =========================================================

@st.cache_data(ttl=CACHE_SECONDS)
def get_tickers():

    result = api_get("/v2/tickers")

    if not result:
        return pd.DataFrame()

    rows = []

    for p in result:

        symbol = p.get("symbol")

        if not symbol:
            continue

        try:

            price = float(
                p.get(
                    "close",
                    p.get("mark_price", 0)
                ) or 0
            )

            volume = float(
                p.get(
                    "volume_24h",
                    p.get("volume", 0)
                ) or 0
            )

            oi = float(
                p.get(
                    "open_interest",
                    p.get("oi", 0)
                ) or 0
            )

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

    df["Vol/OI"] = (
        df["24H Volume"] /
        df["OI"].replace(0, pd.NA)
    )

    return df


# =========================================================
# CANDLES
# =========================================================

@st.cache_data(ttl=CACHE_SECONDS)
def get_candles(symbol, resolution, hours):

    end = int(time.time())

    start = end - (
        hours * 60 * 60
    )

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
        subset=[
            "open",
            "high",
            "low",
            "close"
        ]
    )

    return df.sort_values("time")


# =========================================================
# OI HISTORY
# =========================================================

@st.cache_data(ttl=CACHE_SECONDS)
def get_oi_history(symbol):

    end = int(time.time())

    start = end - (
        6 * 60 * 60
    )

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

    if "close" in df.columns:

        df["close"] = pd.to_numeric(
            df["close"],
            errors="coerce"
        )

    return df.dropna(
        subset=["close"]
    ).sort_values("time")


# =========================================================
# 1H TREND
# =========================================================

def analyze_1h(symbol):

    df = get_candles(
        symbol,
        "1h",
        72
    )

    if df.empty or len(df) < 20:

        return {
            "trend": "UNKNOWN",
            "trend_score": 0
        }

    close = df["close"]

    ema_fast = close.ewm(
        span=9,
        adjust=False
    ).mean()

    ema_slow = close.ewm(
        span=21,
        adjust=False
    ).mean()

    last_price = float(
        close.iloc[-1]
    )

    fast = float(
        ema_fast.iloc[-1]
    )

    slow = float(
        ema_slow.iloc[-1]
    )

    if fast > slow and last_price > fast:

        return {
            "trend": "🟢 BULLISH",
            "trend_score": 2
        }

    if fast < slow and last_price < fast:

        return {
            "trend": "🔴 BEARISH",
            "trend_score": 2
        }

    return {
        "trend": "⚪ NEUTRAL",
        "trend_score": 0
    }


# =========================================================
# 15 MIN LIQUIDITY SWEEP
# =========================================================

def analyze_15m(symbol):

    df = get_candles(
        symbol,
        "15m",
        24
    )

    if df.empty or len(df) < 10:

        return {
            "bull_sweep": False,
            "bear_sweep": False,
            "liquidity": "⚪ None"
        }

    last = df.iloc[-1]

    previous = df.iloc[-7:-1]

    previous_high = float(
        previous["high"].max()
    )

    previous_low = float(
        previous["low"].min()
    )

    high = float(
        last["high"]
    )

    low = float(
        last["low"]
    )

    close = float(
        last["close"]
    )

    bull_sweep = (
        low < previous_low
        and close > previous_low
    )

    bear_sweep = (
        high > previous_high
        and close < previous_high
    )

    if bull_sweep:

        liquidity = "🟢 BULL SWEEP"

    elif bear_sweep:

        liquidity = "🔴 BEAR SWEEP"

    else:

        liquidity = "⚪ None"

    return {
        "bull_sweep": bull_sweep,
        "bear_sweep": bear_sweep,
        "liquidity": liquidity
    }


# =========================================================
# 5 MIN STRUCTURE BREAK
# =========================================================

def analyze_5m(symbol):

    df = get_candles(
        symbol,
        "5m",
        12
    )

    if df.empty or len(df) < 15:

        return {
            "bull_bos": False,
            "bear_bos": False,
            "structure": "⚪ None"
        }

    last = df.iloc[-1]

    structure = df.iloc[-8:-1]

    previous_high = float(
        structure["high"].max()
    )

    previous_low = float(
        structure["low"].min()
    )

    close = float(
        last["close"]
    )

    bull_bos = (
        close > previous_high
    )

    bear_bos = (
        close < previous_low
    )

    if bull_bos:

        structure_name = "🟢 BULL BOS"

    elif bear_bos:

        structure_name = "🔴 BEAR BOS"

    else:

        structure_name = "⚪ None"

    return {
        "bull_bos": bull_bos,
        "bear_bos": bear_bos,
        "structure": structure_name
    }


# =========================================================
# VOLUME
# =========================================================

def analyze_volume(symbol):

    df = get_candles(
        symbol,
        "5m",
        8
    )

    if df.empty or len(df) < 6:

        return {
            "volume_ratio": 0
        }

    last_volume = float(
        df["volume"].iloc[-1]
    )

    avg_volume = float(
        df["volume"].iloc[-6:-1].mean()
    )

    if avg_volume <= 0:

        return {
            "volume_ratio": 0
        }

    return {
        "volume_ratio":
            last_volume / avg_volume
    }


# =========================================================
# OI DISPLACEMENT
# =========================================================

def analyze_oi(symbol):

    df = get_oi_history(symbol)

    if df.empty or len(df) < 5:

        return {
            "oi_change": 0,
            "oi_signal": "⚪ OI Unknown"
        }

    now = float(
        df["close"].iloc[-1]
    )

    old = float(
        df["close"].iloc[-5]
    )

    if old == 0:

        return {
            "oi_change": 0,
            "oi_signal": "⚪ OI Unknown"
        }

    change = (
        (now - old)
        / abs(old)
    ) * 100

    if change <= -1:

        signal = "🔻 OI DISPLACEMENT DOWN"

    elif change >= 1:

        signal = "🔺 OI DISPLACEMENT UP"

    else:

        signal = "⚪ OI NEUTRAL"

    return {
        "oi_change": change,
        "oi_signal": signal
    }


# =========================================================
# DEEP ANALYSIS
# =========================================================

def deep_analysis(symbol, ticker):

    trend = analyze_1h(symbol)

    sweep = analyze_15m(symbol)

    bos = analyze_5m(symbol)

    volume = analyze_volume(symbol)

    oi = analyze_oi(symbol)

    long_score = 0

    short_score = 0

    long_reason = []

    short_reason = []


    # --------------------------------
    # 1H TREND
    # --------------------------------

    if trend["trend"] == "🟢 BULLISH":

        long_score += 2

        long_reason.append(
            "1H bullish trend"
        )

    elif trend["trend"] == "🔴 BEARISH":

        short_score += 2

        short_reason.append(
            "1H bearish trend"
        )


    # --------------------------------
    # 15M LIQUIDITY
    # --------------------------------

    if sweep["bull_sweep"]:

        long_score += 2

        long_reason.append(
            "15m liquidity sweep"
        )

    if sweep["bear_sweep"]:

        short_score += 2

        short_reason.append(
            "15m liquidity sweep"
        )


    # --------------------------------
    # 5M BOS
    # --------------------------------

    if bos["bull_bos"]:

        long_score += 3

        long_reason.append(
            "5m bullish BOS"
        )

    if bos["bear_bos"]:

        short_score += 3

        short_reason.append(
            "5m bearish BOS"
        )


    # --------------------------------
    # VOLUME
    # --------------------------------

    vr = volume["volume_ratio"]

    if vr >= 2:

        long_score += 2
        short_score += 2

        long_reason.append(
            "5m volume spike"
        )

        short_reason.append(
            "5m volume spike"
        )

    elif vr >= 1.3:

        long_score += 1
        short_score += 1


    # --------------------------------
    # OI
    # --------------------------------

    oi_change = oi["oi_change"]

    if oi_change <= -1:

        long_score += 2
        short_score += 2

        long_reason.append(
            "OI displacement down"
        )

        short_reason.append(
            "OI displacement down"
        )

    elif oi_change >= 1:

        long_score += 1
        short_score += 1


    # =================================================
    # SIGNAL
    # =================================================

    if (
        long_score >= 8
        and long_score > short_score
    ):

        signal = "🟢 STRONG LONG"

        score = long_score

        reason = " + ".join(
            long_reason
        )

    elif (
        short_score >= 8
        and short_score > long_score
    ):

        signal = "🔴 STRONG SHORT"

        score = short_score

        reason = " + ".join(
            short_reason
        )

    elif (
        long_score >= 5
        and long_score > short_score
    ):

        signal = "🟡 LONG WATCH"

        score = long_score

        reason = " + ".join(
            long_reason
        )

    elif (
        short_score >= 5
        and short_score > long_score
    ):

        signal = "🟠 SHORT WATCH"

        score = short_score

        reason = " + ".join(
            short_reason
        )

    else:

        signal = "⚪ NO SIGNAL"

        score = max(
            long_score,
            short_score
        )

        reason = "Conditions mixed"


    # =================================================
    # ENTRY ZONE
    # =================================================

    entry_zone = "Wait"

    candles = get_candles(
        symbol,
        "5m",
        6
    )

    if not candles.empty:

        last = candles.iloc[-1]

        low = float(
            last["low"]
        )

        high = float(
            last["high"]
        )

        close = float(
            last["close"]
        )

        if "LONG" in signal:

            entry_low = low

            entry_high = (
                low +
                (high - low) * 0.50
            )

            entry_zone = (
                f"{entry_low:.6g} - "
                f"{entry_high:.6g}"
            )

        elif "SHORT" in signal:

            entry_low = (
                high -
                (high - low) * 0.50
            )

            entry_high = high

            entry_zone = (
                f"{entry_low:.6g} - "
                f"{entry_high:.6g}"
            )


    return {

        "Coin": symbol,

        "Price": round(
            float(ticker["Price"]),
            8
        ),

        "1H Trend":
            trend["trend"],

        "15m Liquidity":
            sweep["liquidity"],

        "5m BOS":
            bos["structure"],

        "Volume Ratio":
            round(vr, 2),

        "OI Change %":
            round(oi_change, 2),

        "OI":
            oi["oi_signal"],

        "Score":
            score,

        "Signal":
            signal,

        "Entry Zone":
            entry_zone,

        "Reason":
            reason
    }


# =========================================================
# LOAD DATA
# =========================================================

all_coins = get_all_perpetuals()

tickers = get_tickers()


if all_coins.empty:

    st.error(
        "❌ Perpetual contracts load nahi hue."
    )

    st.stop()


if tickers.empty:

    st.error(
        "❌ Ticker data load nahi hua."
    )

    st.stop()


# =========================================================
# MERGE
# =========================================================

market = all_coins.merge(
    tickers,
    on="Coin",
    how="left"
)

market = market.dropna(
    subset=["Price"]
)

market = market.sort_values(
    "24H Volume",
    ascending=False
)


# =========================================================
# HEADER
# =========================================================

c1, c2, c3 = st.columns(3)

with c1:

    st.metric(
        "Live Perpetuals",
        len(market)
    )

with c2:

    st.metric(
        "Coins with Market Data",
        market["Price"].notna().sum()
    )

with c3:

    st.metric(
        "Deep Scan",
        DEEP_SCAN_LIMIT
    )


# =========================================================
# ALL 220 COINS
# =========================================================

st.subheader(
    "📊 All Live Perpetual Contracts"
)

st.dataframe(
    market[
        [
            "Coin",
            "Price",
            "24H Volume",
            "OI",
            "Vol/OI"
        ]
    ].head(250),
    use_container_width=True,
    hide_index=True
)


# =========================================================
# LIGHTWEIGHT FILTER
# =========================================================

st.subheader(
    "🔎 Candidate Selection"
)

# All coins are considered here.
# Deep analysis is applied to the highest-activity
# contracts to prevent API overload.

candidate_market = market.copy()

candidate_market["Activity"] = (
    candidate_market["24H Volume"].rank(
        pct=True
    )
)

candidate_market = candidate_market.sort_values(
    [
        "Activity",
        "Vol/OI"
    ],
    ascending=False
)

candidates = candidate_market.head(
    DEEP_SCAN_LIMIT
)


st.info(
    f"All {len(market)} perpetuals considered. "
    f"Deep multi-timeframe analysis is being applied "
    f"to the top {len(candidates)} active candidates "
    f"to control API load."
)


# =========================================================
# DEEP SCAN
# =========================================================

st.subheader(
    "🎯 Multi-Timeframe Reversal Scanner"
)

results = []

progress = st.progress(0)

for i, (_, row) in enumerate(
    candidates.iterrows()
):

    result = deep_analysis(
        row["Coin"],
        row
    )

    if result:

        results.append(result)

    progress.progress(
        int(
            ((i + 1) /
             len(candidates)) * 100
        )
    )

progress.empty()


signals = pd.DataFrame(
    results
)


# =========================================================
# RESULTS
# =========================================================

if signals.empty:

    st.warning(
        "Abhi deep-analysis data available nahi hai."
    )

else:

    signals = signals.sort_values(
        "Score",
        ascending=False
    )

    st.dataframe(
        signals,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# STRONG SIGNALS
# =========================================================

st.subheader(
    "🔥 Strong Reversal Signals"
)

if not signals.empty:

    strong = signals[
        signals["Score"] >= 8
    ]

    if strong.empty:

        st.info(
            "Abhi 8+ score wala setup nahi mila."
        )

    else:

        st.dataframe(
            strong,
            use_container_width=True,
            hide_index=True
        )


# =========================================================
# LONG / SHORT
# =========================================================

st.subheader(
    "🟢 Long / 🔴 Short Watch"
)

if not signals.empty:

    long_watch = signals[
        signals["Signal"].str.contains(
            "LONG"
        )
    ]

    short_watch = signals[
        signals["Signal"].str.contains(
            "SHORT"
        )
    ]

    col1, col2 = st.columns(2)

    with col1:

        st.write("🟢 LONG")

        st.dataframe(
            long_watch,
            use_container_width=True,
            hide_index=True
        )

    with col2:

        st.write("🔴 SHORT")

        st.dataframe(
            short_watch,
            use_container_width=True,
            hide_index=True
        )


# =========================================================
# STRATEGY
# =========================================================

st.divider()

st.subheader(
    "🧠 Scanner Strategy"
)

st.write(
    """
1️⃣ 220 live perpetual contracts

2️⃣ 1H trend:
Bullish / Bearish / Neutral

3️⃣ 15m:
Liquidity sweep

4️⃣ 5m:
Break of Structure (BOS)

5️⃣ OI:
Displacement up/down

6️⃣ Volume:
Volume expansion confirmation

7️⃣ Score:
Multiple confirmations ko combine karke
LONG / SHORT WATCH generate hota hai.

8️⃣ Entry Zone:
5m candle ke basis par possible zone dikhaya jata hai.
"""
)

st.warning(
    "⚠️ Entry Zone guaranteed entry nahi hai. "
    "Ye scanner research/confirmation tool hai. "
    "Real trade se pehle stop-loss aur risk management "
    "zaroor use karein."
)


# =========================================================
# REFRESH
# =========================================================

st.divider()

if st.button(
    "🔄 Refresh Scanner"
):

    st.cache_data.clear()

    st.rerun()
