import streamlit as st
import requests
import pandas as pd
import numpy as np
import time

# =========================================================
# CONFIG
# =========================================================

BASE_URL = "https://api.india.delta.exchange"

HEADERS = {
    "Accept": "application/json",
    "User-Agent": "Delta-Advanced-Scanner/8.0"
}

CACHE_SECONDS = 120

# API load control
DEEP_SCAN_LIMIT = 30

# Hard filter
MIN_VOL_OI = 6.0

# Score levels
WATCH_SCORE = 50
GOOD_SCORE = 65
HIGH_SCORE = 80
A_PLUS_SCORE = 90

st.set_page_config(
    page_title="Delta Advanced Scanner",
    layout="wide"
)

st.title("🔥 Delta Advanced Reversal Scanner")

st.caption(
    "Volume/OI → 1H Regime → Swing Liquidity → "
    "15m Sweep → FVG → 5m BOS → OI Displacement → "
    "ATR → Funding → MTF Confirmation → Quality Score"
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
        df["OI"].replace(0, np.nan)
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

    required = [
        "open",
        "high",
        "low",
        "close"
    ]

    df = df.dropna(
        subset=required
    )

    return df.sort_values("time").reset_index(
        drop=True
    )


# =========================================================
# OI HISTORY
# =========================================================

@st.cache_data(ttl=CACHE_SECONDS)
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

    if "close" in df.columns:

        df["close"] = pd.to_numeric(
            df["close"],
            errors="coerce"
        )

    return df.dropna(
        subset=["close"]
    ).sort_values("time").reset_index(
        drop=True
    )


# =========================================================
# ATR
# =========================================================

def calculate_atr(df, period=14):

    if df.empty or len(df) < period + 2:
        return None, None

    high = df["high"]
    low = df["low"]
    close = df["close"]

    previous_close = close.shift(1)

    tr1 = high - low

    tr2 = (
        high -
        previous_close
    ).abs()

    tr3 = (
        low -
        previous_close
    ).abs()

    tr = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    atr = tr.rolling(
        period
    ).mean()

    current = float(
        atr.iloc[-1]
    )

    previous = float(
        atr.iloc[-4]
    )

    if previous <= 0:
        return current, None

    change = (
        (current - previous)
        / previous
    ) * 100

    return current, change


# =========================================================
# 1H MARKET REGIME
# =========================================================

def analyze_1h(symbol):

    df = get_candles(
        symbol,
        "1h",
        120
    )

    if df.empty or len(df) < 30:

        return {
            "trend": "⚪ UNKNOWN",
            "regime": "⚪ UNCERTAIN",
            "trend_strength": 0
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

    ema50 = close.ewm(
        span=50,
        adjust=False
    ).mean()

    price = float(
        close.iloc[-1]
    )

    e9 = float(
        ema9.iloc[-1]
    )

    e21 = float(
        ema21.iloc[-1]
    )

    e50 = float(
        ema50.iloc[-1]
    )

    atr, atr_change = calculate_atr(
        df,
        14
    )

    if e9 > e21 > e50 and price > e9:

        trend = "🟢 BULLISH"

    elif e9 < e21 < e50 and price < e9:

        trend = "🔴 BEARISH"

    else:

        trend = "⚪ NEUTRAL"

    recent = df.tail(24)

    range_high = float(
        recent["high"].max()
    )

    range_low = float(
        recent["low"].min()
    )

    range_size = (
        range_high -
        range_low
    )

    if price > 0:

        range_pct = (
            range_size /
            price
        ) * 100

    else:

        range_pct = 0

    # EMA separation
    ema_gap = (
        abs(e9 - e21)
        / price
    ) * 100

    if trend != "⚪ NEUTRAL" and ema_gap >= 0.35:

        regime = "➡️ DIRECTIONAL"

    elif range_pct < 5 and ema_gap < 0.25:

        regime = "↔️ RANGE"

    else:

        regime = "⚠️ UNCERTAIN"

    trend_strength = round(
        ema_gap,
        2
    )

    return {

        "trend": trend,

        "regime": regime,

        "trend_strength":
            trend_strength

    }


# =========================================================
# SWING HIGH / LOW
# =========================================================

def analyze_swings(symbol):

    df = get_candles(
        symbol,
        "15m",
        48
    )

    if df.empty or len(df) < 15:

        return {

            "swing_high": None,

            "swing_low": None

        }

    highs = []

    lows = []

    # Confirmed swing points
    for i in range(
        2,
        len(df) - 2
    ):

        h = float(
            df["high"].iloc[i]
        )

        l = float(
            df["low"].iloc[i]
        )

        left_high = df[
            "high"
        ].iloc[i-2:i]

        right_high = df[
            "high"
        ].iloc[i+1:i+3]

        left_low = df[
            "low"
        ].iloc[i-2:i]

        right_low = df[
            "low"
        ].iloc[i+1:i+3]

        if (
            h >= left_high.max()
            and
            h >= right_high.max()
        ):

            highs.append(
                h
            )

        if (
            l <= left_low.min()
            and
            l <= right_low.min()
        ):

            lows.append(
                l
            )

    swing_high = (
        highs[-1]
        if highs
        else None
    )

    swing_low = (
        lows[-1]
        if lows
        else None
    )

    return {

        "swing_high":
            swing_high,

        "swing_low":
            swing_low

    }


# =========================================================
# LIQUIDITY SWEEP
# =========================================================

def analyze_15m(symbol):

    df = get_candles(
        symbol,
        "15m",
        36
    )

    if df.empty or len(df) < 12:

        return {

            "bull_sweep": False,

            "bear_sweep": False,

            "liquidity":
                "⚪ NONE"

        }

    swings = analyze_swings(
        symbol
    )

    last = df.iloc[-1]

    swing_high = swings[
        "swing_high"
    ]

    swing_low = swings[
        "swing_low"
    ]

    bull_sweep = False

    bear_sweep = False

    if swing_low is not None:

        bull_sweep = (
            float(last["low"])
            < swing_low
            and
            float(last["close"])
            > swing_low
        )

    if swing_high is not None:

        bear_sweep = (
            float(last["high"])
            > swing_high
            and
            float(last["close"])
            < swing_high
        )

    if bull_sweep:

        liquidity = "🟢 BULL SWEEP"

    elif bear_sweep:

        liquidity = "🔴 BEAR SWEEP"

    else:

        liquidity = "⚪ NONE"

    return {

        "bull_sweep":
            bull_sweep,

        "bear_sweep":
            bear_sweep,

        "liquidity":
            liquidity

    }


# =========================================================
# FVG
# =========================================================

def analyze_fvg(symbol):

    df = get_candles(
        symbol,
        "15m",
        36
    )

    if df.empty or len(df) < 5:

        return {

            "bull_fvg": False,

            "bear_fvg": False,

            "fvg_type":
                "⚪ NONE",

            "fvg_size_pct":
                0

        }

    bull_fvg = False
    bear_fvg = False

    bull_size = 0
    bear_size = 0

    # Search recent candles
    start = max(
        2,
        len(df) - 12
    )

    for i in range(
        start,
        len(df)
    ):

        first = df.iloc[i-2]

        middle = df.iloc[i-1]

        third = df.iloc[i]

        # Bullish FVG
        if (
            float(third["low"])
            >
            float(first["high"])
        ):

            bull_fvg = True

            bull_size = (
                float(third["low"])
                -
                float(first["high"])
            )

        # Bearish FVG
        if (
            float(third["high"])
            <
            float(first["low"])
        ):

            bear_fvg = True

            bear_size = (
                float(first["low"])
                -
                float(third["high"])
            )

    price = float(
        df["close"].iloc[-1]
    )

    if bull_fvg and price > 0:

        bull_pct = (
            bull_size /
            price
        ) * 100

    else:

        bull_pct = 0

    if bear_fvg and price > 0:

        bear_pct = (
            bear_size /
            price
        ) * 100

    else:

        bear_pct = 0

    if bull_fvg and not bear_fvg:

        fvg_type = "🟢 BULL FVG"

        size_pct = bull_pct

    elif bear_fvg and not bull_fvg:

        fvg_type = "🔴 BEAR FVG"

        size_pct = bear_pct

    elif bull_fvg and bear_fvg:

        if bull_pct >= bear_pct:

            fvg_type = "🟢 BULL FVG"

            size_pct = bull_pct

        else:

            fvg_type = "🔴 BEAR FVG"

            size_pct = bear_pct

    else:

        fvg_type = "⚪ NONE"

        size_pct = 0

    return {

        "bull_fvg":
            bull_fvg,

        "bear_fvg":
            bear_fvg,

        "fvg_type":
            fvg_type,

        "fvg_size_pct":
            round(
                size_pct,
                3
            )

    }


# =========================================================
# 5M BOS
# =========================================================

def analyze_5m(symbol):

    df = get_candles(
        symbol,
        "5m",
        16
    )

    if df.empty or len(df) < 15:

        return {

            "bull_bos": False,

            "bear_bos": False,

            "structure":
                "⚪ NONE"

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

    bull_bos = (
        close >
        previous_high
    )

    bear_bos = (
        close <
        previous_low
    )

    if bull_bos:

        structure = "🟢 BULL BOS"

    elif bear_bos:

        structure = "🔴 BEAR BOS"

    else:

        structure = "⚪ NONE"

    return {

        "bull_bos":
            bull_bos,

        "bear_bos":
            bear_bos,

        "structure":
            structure

    }


# =========================================================
# 5M FVG
# =========================================================

def analyze_5m_fvg(symbol):

    df = get_candles(
        symbol,
        "5m",
        12
    )

    if df.empty or len(df) < 5:

        return {

            "bull":
                False,

            "bear":
                False

        }

    bull = False
    bear = False

    start = max(
        2,
        len(df) - 8
    )

    for i in range(
        start,
        len(df)
    ):

        first = df.iloc[i-2]

        third = df.iloc[i]

        if (
            float(third["low"])
            >
            float(first["high"])
        ):

            bull = True

        if (
            float(third["high"])
            <
            float(first["low"])
        ):

            bear = True

    return {

        "bull":
            bull,

        "bear":
            bear

    }


# =========================================================
# VOLUME
# =========================================================

def analyze_volume(symbol):

    df = get_candles(
        symbol,
        "5m",
        12
    )

    if df.empty or len(df) < 7:

        return {

            "volume_ratio":
                0

        }

    current_volume = float(
        df["volume"].iloc[-1]
    )

    average_volume = float(
        df["volume"].iloc[-7:-1].mean()
    )

    if average_volume <= 0:

        return {

            "volume_ratio":
                0

        }

    return {

        "volume_ratio":
            current_volume /
            average_volume

    }


# =========================================================
# OI DISPLACEMENT
# =========================================================

def analyze_oi(symbol):

    df = get_oi_history(
        symbol
    )

    if df.empty or len(df) < 5:

        return {

            "oi_change":
                None,

            "oi_signal":
                "⚪ UNKNOWN"

        }

    current = float(
        df["close"].iloc[-1]
    )

    previous = float(
        df["close"].iloc[-5]
    )

    if previous == 0:

        return {

            "oi_change":
                None,

            "oi_signal":
                "⚪ UNKNOWN"

        }

    change = (
        (current - previous)
        /
        abs(previous)
    ) * 100

    if change >= 1:

        signal = "🔺 OI UP"

    elif change <= -1:

        signal = "🔻 OI DOWN"

    else:

        signal = "⚪ OI FLAT"

    return {

        "oi_change":
            change,

        "oi_signal":
            signal

    }


# =========================================================
# PRICE + OI DISPLACEMENT
# =========================================================

def price_oi_displacement(
    symbol,
    oi_change
):

    df = get_candles(
        symbol,
        "15m",
        4
    )

    if (
        df.empty
        or
        len(df) < 2
        or
        oi_change is None
    ):

        return {
            "signal":
                "⚪ UNKNOWN"
        }

    price_now = float(
        df["close"].iloc[-1]
    )

    price_prev = float(
        df["close"].iloc[-2]
    )

    if price_prev == 0:

        return {
            "signal":
                "⚪ UNKNOWN"
        }

    price_change = (
        (price_now - price_prev)
        /
        price_prev
    ) * 100

    if (
        price_change > 0
        and
        oi_change > 1
    ):

        signal = "🟢 LONG BUILD"

    elif (
        price_change < 0
        and
        oi_change > 1
    ):

        signal = "🔴 SHORT BUILD"

    elif (
        price_change > 0
        and
        oi_change < -1
    ):

        signal = "🟡 SHORT COVER"

    elif (
        price_change < 0
        and
        oi_change < -1
    ):

        signal = "🟠 LONG EXIT"

    else:

        signal = "⚪ MIXED"

    return {

        "signal":
            signal

    }


# =========================================================
# ATR
# =========================================================

def analyze_atr(symbol):

    df = get_candles(
        symbol,
        "15m",
        36
    )

    atr, change = calculate_atr(
        df,
        14
    )

    if atr is None:

        return {

            "atr":
                None,

            "atr_change":
                None,

            "atr_regime":
                "⚪ UNKNOWN"

        }

    if change is not None:

        if change >= 8:

            regime = "🔥 EXPANDING"

        elif change <= -8:

            regime = "❄️ CONTRACTING"

        else:

            regime = "⚪ STABLE"

    else:

        regime = "⚪ UNKNOWN"

    return {

        "atr":
            atr,

        "atr_change":
            change,

        "atr_regime":
            regime

    }


# =========================================================
# QUALITY LABEL
# =========================================================

def quality_label(score):

    if score >= A_PLUS_SCORE:

        return "🔥 A+"

    if score >= HIGH_SCORE:

        return "🟢 HIGH QUALITY"

    if score >= GOOD_SCORE:

        return "🟡 GOOD"

    if score >= WATCH_SCORE:

        return "👀 WATCH"

    return "❌ REJECT"


# =========================================================
# DEEP ANALYSIS
# =========================================================

def deep_analysis(
    symbol,
    ticker
):

    trend = analyze_1h(
        symbol
    )

    sweep = analyze_15m(
        symbol
    )

    swings = analyze_swings(
        symbol
    )

    fvg = analyze_fvg(
        symbol
    )

    bos = analyze_5m(
        symbol
    )

    fvg5 = analyze_5m_fvg(
        symbol
    )

    volume = analyze_volume(
        symbol
    )

    oi = analyze_oi(
        symbol
    )

    atr = analyze_atr(
        symbol
    )

    displacement = price_oi_displacement(
        symbol,
        oi["oi_change"]
    )


    # =====================================================
    # FUNDING
    # =====================================================

    funding = ticker.get(
        "Funding",
        None
    )

    if funding is not None:

        try:

            funding_pct = (
                float(funding)
                * 100
            )

        except Exception:

            funding_pct = None

    else:

        funding_pct = None


    # =====================================================
    # SCORES
    # =====================================================

    long_score = 0
    short_score = 0

    long_reason = []
    short_reason = []


    # =====================================================
    # 1H TREND
    # =====================================================

    if trend["trend"] == "🟢 BULLISH":

        long_score += 10

        long_reason.append(
            "1H bullish"
        )

    elif trend["trend"] == "🔴 BEARISH":

        short_score += 10

        short_reason.append(
            "1H bearish"
        )


    # =====================================================
    # MARKET REGIME
    # =====================================================

    if trend["regime"] == "➡️ DIRECTIONAL":

        if trend["trend"] == "🟢 BULLISH":

            long_score += 6

            long_reason.append(
                "Directional bullish regime"
            )

        elif trend["trend"] == "🔴 BEARISH":

            short_score += 6

            short_reason.append(
                "Directional bearish regime"
            )

    elif trend["regime"] == "↔️ RANGE":

        long_score -= 2
        short_score -= 2

    elif trend["regime"] == "⚠️ UNCERTAIN":

        long_score -= 3
        short_score -= 3


    # =====================================================
    # LIQUIDITY SWEEP
    # =====================================================

    if sweep["bull_sweep"]:

        long_score += 12

        long_reason.append(
            "15m swing-low liquidity sweep"
        )

    if sweep["bear_sweep"]:

        short_score += 12

        short_reason.append(
            "15m swing-high liquidity sweep"
        )


    # =====================================================
    # FVG
    # =====================================================

    if fvg["bull_fvg"]:

        long_score += 8

        long_reason.append(
            "15m Bull FVG"
        )

    if fvg["bear_fvg"]:

        short_score += 8

        short_reason.append(
            "15m Bear FVG"
        )


    # =====================================================
    # 5M BOS
    # =====================================================

    if bos["bull_bos"]:

        long_score += 15

        long_reason.append(
            "5m Bull BOS"
        )

    if bos["bear_bos"]:

        short_score += 15

        short_reason.append(
            "5m Bear BOS"
        )


    # =====================================================
    # 5M FVG CONFIRMATION
    # =====================================================

    if fvg5["bull"]:

        long_score += 5

        long_reason.append(
            "5m Bull FVG"
        )

    if fvg5["bear"]:

        short_score += 5

        short_reason.append(
            "5m Bear FVG"
        )


    # =====================================================
    # VOLUME
    # =====================================================

    volume_ratio = volume[
        "volume_ratio"
    ]

    if volume_ratio >= 2:

        long_score += 7
        short_score += 7

        long_reason.append(
            "5m volume spike"
        )

        short_reason.append(
            "5m volume spike"
        )

    elif volume_ratio >= 1.3:

        long_score += 3
        short_score += 3


    # =====================================================
    # OI
    # =====================================================

    oi_change = oi[
        "oi_change"
    ]

    if displacement["signal"] == "🟢 LONG BUILD":

        long_score += 8

        long_reason.append(
            "Price + OI long build"
        )

    elif displacement["signal"] == "🔴 SHORT BUILD":

        short_score += 8

        short_reason.append(
            "Price + OI short build"
        )

    elif displacement["signal"] == "🟡 SHORT COVER":

        long_score += 3

        long_reason.append(
            "Short covering"
        )

    elif displacement["signal"] == "🟠 LONG EXIT":

        short_score += 3

        short_reason.append(
            "Long exit pressure"
        )


    # =====================================================
    # ATR
    # =====================================================

    if atr["atr_regime"] == "🔥 EXPANDING":

        if (
            trend["trend"]
            == "🟢 BULLISH"
        ):

            long_score += 5

            long_reason.append(
                "ATR expanding"
            )

        elif (
            trend["trend"]
            == "🔴 BEARISH"
        ):

            short_score += 5

            short_reason.append(
                "ATR expanding"
            )

    elif atr["atr_regime"] == "❄️ CONTRACTING":

        long_score -= 2
        short_score -= 2


    # =====================================================
    # FUNDING
    # =====================================================

    if funding_pct is None:

        funding_signal = (
            "⚪ UNAVAILABLE"
        )

    elif funding_pct >= 0.05:

        short_score += 6

        short_reason.append(
            "Positive funding / longs crowded"
        )

        funding_signal = (
            "🔴 LONG CROWDED"
        )

    elif funding_pct <= -0.05:

        long_score += 6

        long_reason.append(
            "Negative funding / shorts crowded"
        )

        funding_signal = (
            "🟢 SHORT CROWDED"
        )

    else:

        funding_signal = (
            "⚪ NEUTRAL"
        )


    # =====================================================
    # SWEEP + BOS SEQUENCE BONUS
    # =====================================================

    if (
        sweep["bull_sweep"]
        and
        bos["bull_bos"]
    ):

        long_score += 8

        long_reason.append(
            "Sweep → BOS sequence"
        )

    if (
        sweep["bear_sweep"]
        and
        bos["bear_bos"]
    ):

        short_score += 8

        short_reason.append(
            "Sweep → BOS sequence"
        )


    # =====================================================
    # SWEEP + FVG + BOS BONUS
    # =====================================================

    if (
        sweep["bull_sweep"]
        and
        fvg["bull_fvg"]
        and
        bos["bull_bos"]
    ):

        long_score += 7

        long_reason.append(
            "Sweep + FVG + BOS alignment"
        )

    if (
        sweep["bear_sweep"]
        and
        fvg["bear_fvg"]
        and
        bos["bear_bos"]
    ):

        short_score += 7

        short_reason.append(
            "Sweep + FVG + BOS alignment"
        )


    # =====================================================
    # CONFLICT PENALTY
    # =====================================================

    if (
        trend["trend"]
        == "🔴 BEARISH"
        and
        long_score >= 20
    ):

        long_score -= 8

        long_reason.append(
            "Counter-trend penalty"
        )

    if (
        trend["trend"]
        == "🟢 BULLISH"
        and
        short_score >= 20
    ):

        short_score -= 8

        short_reason.append(
            "Counter-trend penalty"
        )


    # Never negative
    long_score = max(
        0,
        long_score
    )

    short_score = max(
        0,
        short_score
    )


    # =====================================================
    # QUALITY
    # =====================================================

    long_quality = quality_label(
        long_score
    )

    short_quality = quality_label(
        short_score
    )


    # =====================================================
    # SIGNAL
    # =====================================================

    if (
        long_score >= HIGH_SCORE
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
        short_score >= HIGH_SCORE
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
        long_score >= GOOD_SCORE
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
        short_score >= GOOD_SCORE
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

    entry_zone = "WAIT"

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
                (
                    high -
                    low
                ) * 0.50
            )

            entry_zone = (
                f"{entry_low:.6g} - "
                f"{entry_high:.6g}"
            )

        elif "SHORT" in signal:

            entry_low = (
                high -
                (
                    high -
                    low
                ) * 0.50
            )

            entry_high = high

            entry_zone = (
                f"{entry_low:.6g} - "
                f"{entry_high:.6g}"
            )


    # =====================================================
    # LEVERAGE REFERENCE
    # =====================================================

    # Ye recommendation nahi hai.
    # Sirf price move sensitivity reference hai.

    leverage_20x = (
        f"{dominant_score}% score"
    )

    leverage_group = (
        "20x Reference | "
        "50x / 100x / 200x"
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

        "24H Volume":
            ticker.get(
                "24H Volume"
            ),

        "OI":
            ticker.get(
                "OI"
            ),

        "Vol/OI":
            round(
                float(
                    ticker.get(
                        "Vol/OI",
                        0
                    )
                ),
                2
            ),

        "1H Trend":
            trend["trend"],

        "Market Regime":
            trend["regime"],

        "Trend Strength":
            trend["trend_strength"],

        "Swing High":
            swings[
                "swing_high"
            ],

        "Swing Low":
            swings[
                "swing_low"
            ],

        "15m Liquidity":
            sweep["liquidity"],

        "15m FVG":
            fvg["fvg_type"],

        "FVG Size %":
            fvg["fvg_size_pct"],

        "5m BOS":
            bos["structure"],

        "5m FVG":
            (
                "🟢 BULL"
                if fvg5["bull"]
                else
                "🔴 BEAR"
                if fvg5["bear"]
                else
                "⚪ NONE"
            ),

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

        "OI Displacement":
            displacement[
                "signal"
            ],

        "ATR":
            (
                round(
                    atr["atr"],
                    8
                )
                if atr["atr"]
                is not None
                else None
            ),

        "ATR Change %":
            (
                round(
                    atr["atr_change"],
                    2
                )
                if atr["atr_change"]
                is not None
                else None
            ),

        "ATR Regime":
            atr["atr_regime"],

        "Funding %":
            (
                round(
                    funding_pct,
                    4
                )
                if funding_pct
                is not None
                else None
            ),

        "Funding":
            funding_signal,

        "Long Score":
            long_score,

        "Long Quality":
            long_quality,

        "Long Signal":
            (
                "🟢 STRONG LONG"
                if long_score >= HIGH_SCORE
                else
                "🟡 LONG WATCH"
                if long_score >= GOOD_SCORE
                else
                "⚪ NO LONG"
            ),

        "Short Score":
            short_score,

        "Short Quality":
            short_quality,

        "Short Signal":
            (
                "🔴 STRONG SHORT"
                if short_score >= HIGH_SCORE
                else
                "🟠 SHORT WATCH"
                if short_score >= GOOD_SCORE
                else
                "⚪ NO SHORT"
            ),

        "Score":
            dominant_score,

        "Signal":
            signal,

        "Quality":
            quality_label(
                dominant_score
            ),

        "Entry Zone":
            entry_zone,

        "20x Reference":
            leverage_20x,

        "50x-200x":
            leverage_group,

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
# LOAD MARKET
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

market = market[
    market["Vol/OI"] >=
    MIN_VOL_OI
].copy()

market = market.sort_values(
    "24H Volume",
    ascending=False
)


# =========================================================
# METRICS
# =========================================================

c1, c2, c3, c4 = st.columns(4)

with c1:

    st.metric(
        "Qualified Coins",
        len(market)
    )

with c2:

    st.metric(
        "Vol/OI Filter",
        f"> {MIN_VOL_OI}"
    )

with c3:

    st.metric(
        "Deep Scan",
        min(
            DEEP_SCAN_LIMIT,
            len(market)
        )
    )

with c4:

    st.metric(
        "API Mode",
        "Controlled"
    )


# =========================================================
# ALL QUALIFIED COINS
# =========================================================

st.subheader(
    "📊 Qualified Market — Volume/OI > 6"
)

if market.empty:

    st.warning(
        "Koi coin Volume/OI > 6 filter pass nahi kar raha."
    )

else:

    st.dataframe(

        market[
            [
                "Coin",
                "Price",
                "24H Volume",
                "OI",
                "Vol/OI",
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

if not candidate_market.empty:

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

else:

    candidates = pd.DataFrame()


st.info(
    f"Volume/OI > {MIN_VOL_OI} "
    f"pass karne wale {len(market)} coins me se "
    f"top {len(candidates)} active coins ka "
    f"deep MTF scan kiya ja raha hai."
)


# =========================================================
# SCAN
# =========================================================

st.subheader(
    "🎯 Advanced Scanner Results"
)

results = []

progress = st.progress(0)

total = len(candidates)

if total > 0:

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
                (
                    (i + 1)
                    /
                    total
                ) * 100
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
# HIGH QUALITY SETUPS
# =========================================================

st.subheader(
    "🔥 HIGH QUALITY SETUPS — 80+"
)

if not signals.empty:

    high_quality = signals[
        signals["Score"] >= HIGH_SCORE
    ].sort_values(
        "Score",
        ascending=False
    )

    if high_quality.empty:

        st.info(
            "Abhi 80+ quality setup nahi mila."
        )

    else:

        st.dataframe(

            high_quality[
                [
                    "Coin",
                    "Price",
                    "Signal",
                    "Score",
                    "Quality",
                    "Market Regime",
                    "15m Liquidity",
                    "15m FVG",
                    "5m BOS",
                    "OI Displacement",
                    "ATR Regime",
                    "Funding",
                    "Vol/OI",
                    "Volume x",
                    "Entry Zone",
                    "Reason"
                ]
            ],

            use_container_width=True,

            hide_index=True

        )


# =========================================================
# A+ SETUPS
# =========================================================

st.subheader(
    "🚀 A+ SETUPS — 90+"
)

if not signals.empty:

    a_plus = signals[
        signals["Score"] >= A_PLUS_SCORE
    ].sort_values(
        "Score",
        ascending=False
    )

    if a_plus.empty:

        st.info(
            "Abhi A+ setup nahi mila."
        )

    else:

        st.dataframe(

            a_plus[
                [
                    "Coin",
                    "Price",
                    "Signal",
                    "Score",
                    "Quality",
                    "1H Trend",
                    "Market Regime",
                    "15m Liquidity",
                    "15m FVG",
                    "5m BOS",
                    "5m FVG",
                    "OI Displacement",
                    "ATR Regime",
                    "Funding",
                    "Vol/OI",
                    "Entry Zone",
                    "Reason"
                ]
            ],

            use_container_width=True,

            hide_index=True

        )


# =========================================================
# LONG
# =========================================================

st.subheader(
    "🟢 LONG — Quality Ranking"
)

if not signals.empty:

    long_table = signals[
        [
            "Coin",
            "Price",
            "1H Trend",
            "Market Regime",
            "15m Liquidity",
            "15m FVG",
            "5m BOS",
            "5m FVG",
            "OI Displacement",
            "ATR Regime",
            "Funding %",
            "Funding",
            "Vol/OI",
            "Volume x",
            "Long Score",
            "Long Quality",
            "Long Signal",
            "Entry Zone",
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
# SHORT
# =========================================================

st.subheader(
    "🔴 SHORT — Quality Ranking"
)

if not signals.empty:

    short_table = signals[
        [
            "Coin",
            "Price",
            "1H Trend",
            "Market Regime",
            "15m Liquidity",
            "15m FVG",
            "5m BOS",
            "5m FVG",
            "OI Displacement",
            "ATR Regime",
            "Funding %",
            "Funding",
            "Vol/OI",
            "Volume x",
            "Short Score",
            "Short Quality",
            "Short Signal",
            "Entry Zone",
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
# SCORING EXPLANATION
# =========================================================

st.divider()

st.subheader(
    "🧠 Scoring Framework"
)

st.write(
    """
1H Trend              → 10
Directional Regime    → 6
Liquidity Sweep       → 12
15m FVG               → 8
5m BOS                → 15
5m FVG                → 5
Volume Spike          → 7
OI Displacement       → 8
ATR Expansion         → 5
Funding Confirmation  → 6
Sweep + BOS           → 8
Sweep + FVG + BOS     → 7

Counter-trend setup par penalty bhi lagti hai.

50+  → Watch
65+  → Good
80+  → High Quality
90+  → A+
"""
)


# =========================================================
# VOLUME/OI EXPLANATION
# =========================================================

st.subheader(
    "📊 Volume / OI Filter"
)

st.write(
    """
24H Volume / Open Interest ratio < 6
→ Scanner se reject.

6–10
→ Normal qualifying activity.

10–20
→ Strong activity.

20+
→ Very high activity; lekin ise automatic
strong signal nahi maana gaya hai.

Volume/OI sirf liquidity/activity filter hai,
direction ka standalone signal nahi.
"""
)


# =========================================================
# OI EXPLANATION
# =========================================================

st.subheader(
    "📈 Price + OI Displacement"
)

st.write(
    """
Price ↑ + OI ↑
→ LONG BUILD

Price ↓ + OI ↑
→ SHORT BUILD

Price ↑ + OI ↓
→ SHORT COVER

Price ↓ + OI ↓
→ LONG EXIT

Isliye sirf OI UP/DOWN ke bajay
price ke saath OI ko interpret kiya gaya hai.
"""
)


# =========================================================
# MTF LOGIC
# =========================================================

st.subheader(
    "⏱️ Multi-Timeframe Logic"
)

st.write(
    """
1H
↓
Trend + Market Regime

15m
↓
Swing High / Swing Low
↓
Liquidity Sweep
↓
FVG

5m
↓
BOS
↓
FVG confirmation

Then
↓
OI Displacement
↓
ATR
↓
Volume
↓
Funding
↓
Final Quality Score
"""
)


# =========================================================
# LEVERAGE WARNING
# =========================================================

st.subheader(
    "⚠️ Leverage Reference"
)

st.write(
    """
20x / 50x / 100x / 200x columns
sirf sensitivity/reference ke liye hain.

High scanner score ka matlab
high leverage use karna nahi hai.

20x, 50x, 100x aur 200x par
same market move ka account impact
bahut alag ho sakta hai.

Scanner leverage ko automatically
recommend nahi karta.
"""
)


# =========================================================
# IMPORTANT
# =========================================================

st.warning(
    """
⚠️ Ye scanner probability/quality filter hai,
guaranteed trade signal nahi.

A+ score bhi guaranteed winning trade nahi hai.

Sabse important next step historical backtest hai:
kaunse score range ke setups actual me
kitni baar target hit karte hain,
ye data se verify karna hoga.
"""
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
