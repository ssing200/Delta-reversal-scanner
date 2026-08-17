import streamlit as st
import requests
import pandas as pd
import time

BASE_URL = "https://api.india.delta.exchange"

HEADERS = {
    "Accept": "application/json",
    "User-Agent": "Delta-Reversal-Scanner/5.0"
}

CACHE_SECONDS = 120
DEEP_SCAN_LIMIT = 30

st.set_page_config(
    page_title="Delta Reversal Scanner",
    layout="wide"
)

st.title("🔥 Delta Reversal Scanner")

st.caption(
    "1H Trend → 15m Liquidity Sweep → "
    "5m BOS → OI → Funding → Volume → Score"
)


# =========================================================
# API
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
# ALL PERPETUALS
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
                "underlying_asset", {}
            ).get("symbol", "")
        })

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    return df.drop_duplicates(
        subset=["Coin"]
    )


# =========================================================
# TICKERS
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

        except Exception:
            continue

        if price <= 0:
            continue

        # -------------------------------
        # FUNDING
        # -------------------------------

        funding_raw = p.get(
            "funding_rate",
            p.get("funding", None)
        )

        try:

            funding = (
                float(funding_raw)
                if funding_raw is not None
                else None
            )

        except Exception:

            funding = None

        rows.append({
            "Coin": symbol,
            "Price": price,
            "24H Volume": volume,
            "OI": oi,
            "Funding": funding
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

    return df.dropna(
        subset=[
            "open",
            "high",
            "low",
            "close"
        ]
    ).sort_values("time")


# =========================================================
# OI HISTORY
# =========================================================

@st.cache_data(ttl=CACHE_SECONDS)
def get_oi_history(symbol):

    end = int(time.time())

    start = end - 6 * 60 * 60

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
            "trend": "⚪ UNKNOWN"
        }

    close = df["close"]

    ema9 = close.ewm(
        span=9,
        adjust=False
    ).mean()

    ema21 = close.ewm(
        span=21,
        adjust=False
    ).mean()

    price = float(
        close.iloc[-1]
    )

    fast = float(
        ema9.iloc[-1]
    )

    slow = float(
        ema21.iloc[-1]
    )

    if fast > slow and price > fast:

        return {
            "trend": "🟢 BULLISH"
        }

    if fast < slow and price < fast:

        return {
            "trend": "🔴 BEARISH"
        }

    return {
        "trend": "⚪ NEUTRAL"
    }


# =========================================================
# 15M LIQUIDITY SWEEP
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

    bull_sweep = (
        float(last["low"]) < previous_low
        and
        float(last["close"]) > previous_low
    )

    bear_sweep = (
        float(last["high"]) > previous_high
        and
        float(last["close"]) < previous_high
    )

    if bull_sweep:

        name = "🟢 BULL SWEEP"

    elif bear_sweep:

        name = "🔴 BEAR SWEEP"

    else:

        name = "⚪ None"

    return {
        "bull_sweep": bull_sweep,
        "bear_sweep": bear_sweep,
        "liquidity": name
    }


# =========================================================
# 5M BOS
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

    previous = df.iloc[-8:-1]

    previous_high = float(
        previous["high"].max()
    )

    previous_low = float(
        previous["low"].min()
    )

    close = float(
        last["close"]
    )

    bull_bos = close > previous_high

    bear_bos = close < previous_low

    if bull_bos:

        name = "🟢 BULL BOS"

    elif bear_bos:

        name = "🔴 BEAR BOS"

    else:

        name = "⚪ None"

    return {
        "bull_bos": bull_bos,
        "bear_bos": bear_bos,
        "structure": name
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

    current_volume = float(
        df["volume"].iloc[-1]
    )

    average_volume = float(
        df["volume"].iloc[-6:-1].mean()
    )

    if average_volume <= 0:

        return {
            "volume_ratio": 0
        }

    return {
        "volume_ratio":
            current_volume /
            average_volume
    }


# =========================================================
# OI
# =========================================================

def analyze_oi(symbol):

    df = get_oi_history(symbol)

    if df.empty or len(df) < 5:

        return {
            "oi_change": None,
            "oi_signal": "⚪ Unknown"
        }

    current = float(
        df["close"].iloc[-1]
    )

    previous = float(
        df["close"].iloc[-5]
    )

    if previous == 0:

        return {
            "oi_change": None,
            "oi_signal": "⚪ Unknown"
        }

    change = (
        (current - previous)
        / abs(previous)
    ) * 100

    if change >= 1:

        signal = "🔺 OI UP"

    elif change <= -1:

        signal = "🔻 OI DOWN"

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


    # =====================================================
    # FUNDING
    # =====================================================

    funding = ticker.get(
        "Funding",
        None
    )

    if funding is not None:

        try:

            funding = float(funding)

            funding_pct = (
                funding * 100
            )

        except Exception:

            funding = None
            funding_pct = None

    else:

        funding_pct = None


    # =====================================================
    # SEPARATE LONG / SHORT SCORES
    # =====================================================

    long_score = 0
    short_score = 0

    long_reason = []
    short_reason = []


    # =====================================================
    # 1H TREND
    # =====================================================

    if trend["trend"] == "🟢 BULLISH":

        long_score += 2

        long_reason.append(
            "1H bullish"
        )

    elif trend["trend"] == "🔴 BEARISH":

        short_score += 2

        short_reason.append(
            "1H bearish"
        )


    # =====================================================
    # LIQUIDITY
    # =====================================================

    if sweep["bull_sweep"]:

        long_score += 2

        long_reason.append(
            "15m bull sweep"
        )

    if sweep["bear_sweep"]:

        short_score += 2

        short_reason.append(
            "15m bear sweep"
        )


    # =====================================================
    # BOS
    # =====================================================

    if bos["bull_bos"]:

        long_score += 3

        long_reason.append(
            "5m bull BOS"
        )

    if bos["bear_bos"]:

        short_score += 3

        short_reason.append(
            "5m bear BOS"
        )


    # =====================================================
    # VOLUME
    # =====================================================

    volume_ratio = volume[
        "volume_ratio"
    ]

    if volume_ratio >= 2:

        long_score += 2
        short_score += 2

        long_reason.append(
            "Volume spike"
        )

        short_reason.append(
            "Volume spike"
        )

    elif volume_ratio >= 1.3:

        long_score += 1
        short_score += 1


    # =====================================================
    # OI
    # =====================================================

    oi_change = oi[
        "oi_change"
    ]

    if oi_change is not None:

        if oi_change >= 1:

            # OI increasing is confirmation
            # for the direction already supported
            # by structure.

            if trend["trend"] == "🟢 BULLISH":

                long_score += 1

                long_reason.append(
                    "OI increasing"
                )

            if trend["trend"] == "🔴 BEARISH":

                short_score += 1

                short_reason.append(
                    "OI increasing"
                )

        elif oi_change <= -1:

            # OI falling = possible position closing.
            # Give only a small contextual point.

            if sweep["bull_sweep"]:

                long_score += 1

                long_reason.append(
                    "OI falling after sweep"
                )

            if sweep["bear_sweep"]:

                short_score += 1

                short_reason.append(
                    "OI falling after sweep"
                )


    # =====================================================
    # FUNDING
    # =====================================================

    if funding_pct is None:

        funding_signal = (
            "⚪ Funding unavailable"
        )

    else:

        # -------------------------------------------------
        # POSITIVE FUNDING
        # -------------------------------------------------

        if funding_pct >= 0.05:

            short_score += 2

            short_reason.append(
                "High positive funding"
            )

            funding_signal = (
                "🔴 Longs crowded"
            )

        # -------------------------------------------------
        # NEGATIVE FUNDING
        # -------------------------------------------------

        elif funding_pct <= -0.05:

            long_score += 2

            long_reason.append(
                "High negative funding"
            )

            funding_signal = (
                "🟢 Shorts crowded"
            )

        else:

            funding_signal = (
                "⚪ Funding neutral"
            )


    # =====================================================
    # LONG SIGNAL
    # =====================================================

    if long_score >= 8:

        long_signal = (
            "🟢 STRONG LONG"
        )

    elif long_score >= 5:

        long_signal = (
            "🟡 LONG WATCH"
        )

    else:

        long_signal = (
            "⚪ NO LONG"
        )


    # =====================================================
    # SHORT SIGNAL
    # =====================================================

    if short_score >= 8:

        short_signal = (
            "🔴 STRONG SHORT"
        )

    elif short_score >= 5:

        short_signal = (
            "🟠 SHORT WATCH"
        )

    else:

        short_signal = (
            "⚪ NO SHORT"
        )


    # =====================================================
    # DOMINANT SIGNAL
    # =====================================================

    if (
        long_score >= 8
        and
        long_score > short_score
    ):

        signal = "🟢 STRONG LONG"

        dominant_score = long_score

        dominant_reason = (
            " + ".join(
                long_reason
            )
        )

    elif (
        short_score >= 8
        and
        short_score > long_score
    ):

        signal = "🔴 STRONG SHORT"

        dominant_score = short_score

        dominant_reason = (
            " + ".join(
                short_reason
            )
        )

    elif (
        long_score >= 5
        and
        long_score > short_score
    ):

        signal = "🟡 LONG WATCH"

        dominant_score = long_score

        dominant_reason = (
            " + ".join(
                long_reason
            )
        )

    elif (
        short_score >= 5
        and
        short_score > long_score
    ):

        signal = "🟠 SHORT WATCH"

        dominant_score = short_score

        dominant_reason = (
            " + ".join(
                short_reason
            )
        )

    else:

        signal = "⚪ NO SIGNAL"

        dominant_score = max(
            long_score,
            short_score
        )

        dominant_reason = (
            "Conditions mixed"
        )


    # =====================================================
    # ENTRY ZONE
    # =====================================================

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


    # =====================================================
    # RESULT
    # =====================================================

    return {

        "Coin":
            symbol,

        "Price":
            round(
                float(
                    ticker["Price"]
                ),
                8
            ),

        "1H Trend":
            trend["trend"],

        "15m Liquidity":
            sweep["liquidity"],

        "5m BOS":
            bos["structure"],

        "Volume x":
            round(
                volume_ratio,
                2
            ),

        "OI Change %":
            (
                round(
                    oi_change,
                    2
                )
                if oi_change is not None
                else None
            ),

        "OI":
            oi["oi_signal"],

        "Funding %":
            (
                round(
                    funding_pct,
                    4
                )
                if funding_pct is not None
                else None
            ),

        "Funding":
            funding_signal,

        # Separate scores
        "Long Score":
            long_score,

        "Long Signal":
            long_signal,

        "Short Score":
            short_score,

        "Short Signal":
            short_signal,

        # Dominant
        "Score":
            dominant_score,

        "Signal":
            signal,

        "Entry Zone":
            entry_zone,

        "Long Reason":
            " + ".join(
                long_reason
            ),

        "Short Reason":
            " + ".join(
                short_reason
            ),

        "Reason":
            dominant_reason
    }


# =========================================================
# LOAD
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
# METRICS
# =========================================================

c1, c2, c3 = st.columns(3)

with c1:

    st.metric(
        "Live Perpetuals",
        len(market)
    )

with c2:

    st.metric(
        "Market Data",
        market["Price"].notna().sum()
    )

with c3:

    st.metric(
        "Deep Scan",
        DEEP_SCAN_LIMIT
    )


# =========================================================
# ALL COINS
# =========================================================

st.subheader(
    "📊 All Live Perpetuals"
)

st.dataframe(
    market[
        [
            "Coin",
            "Price",
            "24H Volume",
            "OI",
            "Funding"
        ]
    ].head(250),
    use_container_width=True,
    hide_index=True
)


# =========================================================
# CANDIDATES
# =========================================================

candidate_market = market.copy()

candidate_market[
    "Activity"
] = candidate_market[
    "24H Volume"
].rank(
    pct=True
)

candidate_market = (
    candidate_market.sort_values(
        [
            "Activity",
            "Vol/OI"
        ],
        ascending=False
    )
)

candidates = (
    candidate_market.head(
        DEEP_SCAN_LIMIT
    )
)


st.info(
    f"All {len(market)} live perpetuals "
    f"considered. Deep analysis top "
    f"{len(candidates)} active coins par "
    f"kiya ja raha hai."
)


# =========================================================
# SCAN
# =========================================================

st.subheader(
    "🎯 Scanner Results"
)

results = []

progress = st.progress(0)

total = len(candidates)

for i, (_, row) in enumerate(
    candidates.iterrows()
):

    result = deep_analysis(
        row["Coin"],
        row
    )

    if result:

        results.append(
            result
        )

    progress.progress(
        int(
            ((i + 1) / total) * 100
        )
    )

progress.empty()


signals = pd.DataFrame(
    results
)


# =========================================================
# COMPLETE RESULTS
# =========================================================

if signals.empty:

    st.warning(
        "Analysis data available nahi hai."
    )

else:

    signals = signals.sort_values(
        [
            "Score",
            "Long Score",
            "Short Score"
        ],
        ascending=False
    )

    st.dataframe(
        signals,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# LONG TABLE
# =========================================================

st.subheader(
    "🟢 LONG — Separate Score & Funding"
)

if not signals.empty:

    long_table = signals[
        [
            "Coin",
            "Price",
            "1H Trend",
            "15m Liquidity",
            "5m BOS",
            "Volume x",
            "OI Change %",
            "OI",
            "Funding %",
            "Funding",
            "Long Score",
            "Long Signal",
            "Long Reason"
        ]
    ].sort_values(
        "Long Score",
        ascending=False
    )

    st.dataframe(
        long_table,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# SHORT TABLE
# =========================================================

st.subheader(
    "🔴 SHORT — Separate Score & Funding"
)

if not signals.empty:

    short_table = signals[
        [
            "Coin",
            "Price",
            "1H Trend",
            "15m Liquidity",
            "5m BOS",
            "Volume x",
            "OI Change %",
            "OI",
            "Funding %",
            "Funding",
            "Short Score",
            "Short Signal",
            "Short Reason"
        ]
    ].sort_values(
        "Short Score",
        ascending=False
    )

    st.dataframe(
        short_table,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# STRONG LONG
# =========================================================

st.subheader(
    "🔥 STRONG LONG — 8+"
)

if not signals.empty:

    strong_long = signals[
        signals["Long Score"] >= 8
    ].sort_values(
        "Long Score",
        ascending=False
    )

    if strong_long.empty:

        st.info(
            "Abhi Strong Long nahi mila."
        )

    else:

        st.dataframe(
            strong_long[
                [
                    "Coin",
                    "Price",
                    "Long Score",
                    "Long Signal",
                    "Funding %",
                    "Funding",
                    "OI Change %",
                    "Volume x",
                    "Entry Zone",
                    "Long Reason"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )


# =========================================================
# STRONG SHORT
# =========================================================

st.subheader(
    "🔥 STRONG SHORT — 8+"
)

if not signals.empty:

    strong_short = signals[
        signals["Short Score"] >= 8
    ].sort_values(
        "Short Score",
        ascending=False
    )

    if strong_short.empty:

        st.info(
            "Abhi Strong Short nahi mila."
        )

    else:

        st.dataframe(
            strong_short[
                [
                    "Coin",
                    "Price",
                    "Short Score",
                    "Short Signal",
                    "Funding %",
                    "Funding",
                    "OI Change %",
                    "Volume x",
                    "Entry Zone",
                    "Short Reason"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )


# =========================================================
# FUNDING EXPLANATION
# =========================================================

st.divider()

st.subheader(
    "💰 Funding Logic"
)

st.write(
    """
Positive Funding ≥ +0.05%
→ Longs crowded
→ Short Score +2

Negative Funding ≤ -0.05%
→ Shorts crowded
→ Long Score +2

Between -0.05% and +0.05%
→ Funding neutral
→ No score

Funding ko अकेले signal nahi maana gaya hai.
Liquidity + BOS + OI + Volume ke saath
confirmation ke रूप में use kiya gaya hai.
"""
)


# =========================================================
# STRATEGY
# =========================================================

st.subheader(
    "🧠 Scanner Structure"
)

st.write(
    """
220 live perpetuals
↓
1H trend
↓
15m liquidity sweep
↓
5m BOS
↓
OI displacement
↓
Volume
↓
Funding
↓
Separate LONG / SHORT score
↓
8+ = Strong setup
"""
)


# =========================================================
# WARNING
# =========================================================

st.warning(
    "⚠️ Scanner confirmation tool hai, "
    "guaranteed trade signal nahi. "
    "Real position se pehle risk management "
    "aur stop-loss zaroor check karein."
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
