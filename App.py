import streamlit as st
import requests
import pandas as pd
import time
import numpy as np

BASE_URL = "https://api.india.delta.exchange"

HEADERS = {
    "Accept": "application/json",
    "User-Agent": "Delta-Reversal-Scanner/6.0"
}

CACHE_SECONDS = 120

# Server load ko control karne ke liye
DEEP_SCAN_LIMIT = 30

# OI history
OI_HISTORY_HOURS = 6

# ATR settings
ATR_PERIOD = 14
ATR_COMPARE_PERIODS = 5

st.set_page_config(
    page_title="Delta Advanced Scanner",
    layout="wide"
)

st.title("🔥 Delta Advanced Reversal Scanner")

st.caption(
    "1H Trend → 15m Liquidity → 5m BOS → "
    "MTF Confirmation → OI → Volume → ATR → Funding → Score"
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

        # -------------------------------------------------
        # LEVERAGE
        # -------------------------------------------------

        max_leverage = (
            p.get("max_leverage")
            or p.get("max_leverage_ratio")
            or p.get("leverage")
        )

        try:

            if max_leverage is not None:
                max_leverage = float(max_leverage)

        except Exception:

            max_leverage = None

        rows.append({

            "Coin": symbol,

            "ID": p.get("id"),

            "Underlying": p.get(
                "underlying_asset", {}
            ).get("symbol", ""),

            "Max Leverage": max_leverage

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

        # -------------------------------------------------
        # FUNDING
        # -------------------------------------------------

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

    return df.dropna(
        subset=[
            "open",
            "high",
            "low",
            "close"
        ]
    ).sort_values("time").reset_index(drop=True)


# =========================================================
# OI HISTORY
# =========================================================

@st.cache_data(ttl=CACHE_SECONDS)
def get_oi_history(symbol):

    end = int(time.time())

    start = (
        end -
        OI_HISTORY_HOURS * 60 * 60
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
    ).sort_values("time").reset_index(drop=True)


# =========================================================
# GENERIC TREND
# =========================================================

def timeframe_trend(df):

    if df.empty or len(df) < 20:

        return "⚪ UNKNOWN"

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

    # Strong directional
    if fast > slow and price > fast:

        return "🟢 BULLISH"

    if fast < slow and price < fast:

        return "🔴 BEARISH"

    # EMA distance very small = range
    ema_distance = (
        abs(fast - slow)
        / max(abs(slow), 1e-9)
    ) * 100

    if ema_distance < 0.15:

        return "⚪ RANGE"

    return "🟡 UNCERTAIN"


# =========================================================
# 1H TREND
# =========================================================

def analyze_1h(symbol):

    df = get_candles(
        symbol,
        "1h",
        72
    )

    return {
        "trend":
            timeframe_trend(df)
    }


# =========================================================
# 15M TREND + LIQUIDITY
# =========================================================

def analyze_15m(symbol):

    df = get_candles(
        symbol,
        "15m",
        24
    )

    if df.empty or len(df) < 20:

        return {

            "bull_sweep": False,

            "bear_sweep": False,

            "liquidity": "⚪ None",

            "trend": "⚪ UNKNOWN"

        }

    trend = timeframe_trend(df)

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

        liquidity = "🟢 BULL SWEEP"

    elif bear_sweep:

        liquidity = "🔴 BEAR SWEEP"

    else:

        liquidity = "⚪ None"

    return {

        "bull_sweep":
            bull_sweep,

        "bear_sweep":
            bear_sweep,

        "liquidity":
            liquidity,

        "trend":
            trend

    }


# =========================================================
# 5M BOS + TREND
# =========================================================

def analyze_5m(df):

    if df.empty or len(df) < 20:

        return {

            "bull_bos": False,

            "bear_bos": False,

            "structure": "⚪ None",

            "trend": "⚪ UNKNOWN"

        }

    trend = timeframe_trend(df)

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
        close > previous_high
    )

    bear_bos = (
        close < previous_low
    )

    if bull_bos:

        structure = "🟢 BULL BOS"

    elif bear_bos:

        structure = "🔴 BEAR BOS"

    else:

        structure = "⚪ None"

    return {

        "bull_bos":
            bull_bos,

        "bear_bos":
            bear_bos,

        "structure":
            structure,

        "trend":
            trend

    }


# =========================================================
# MULTI TIMEFRAME
# =========================================================

def analyze_mtf(
    trend_1h,
    trend_15m,
    trend_5m
):

    trends = [
        trend_1h,
        trend_15m,
        trend_5m
    ]

    bullish = trends.count(
        "🟢 BULLISH"
    )

    bearish = trends.count(
        "🔴 BEARISH"
    )

    ranges = trends.count(
        "⚪ RANGE"
    )

    unknown = trends.count(
        "⚪ UNKNOWN"
    )

    uncertain = trends.count(
        "🟡 UNCERTAIN"
    )

    # -----------------------------------------------------
    # DIRECTIONAL UP
    # -----------------------------------------------------

    if bullish >= 2 and bearish == 0:

        state = "🟢 DIRECTIONAL UP"

        long_points = 2

        short_points = 0

    # -----------------------------------------------------
    # DIRECTIONAL DOWN
    # -----------------------------------------------------

    elif bearish >= 2 and bullish == 0:

        state = "🔴 DIRECTIONAL DOWN"

        long_points = 0

        short_points = 2

    # -----------------------------------------------------
    # RANGE
    # -----------------------------------------------------

    elif ranges >= 2:

        state = "⚪ RANGE BOUND"

        long_points = 0

        short_points = 0

    # -----------------------------------------------------
    # UNCERTAINTY
    # -----------------------------------------------------

    else:

        state = "🟡 UNCERTAINTY"

        long_points = 0

        short_points = 0

    return {

        "state":
            state,

        "long_points":
            long_points,

        "short_points":
            short_points,

        "confirmation":
            f"1H {trend_1h} | "
            f"15m {trend_15m} | "
            f"5m {trend_5m}"

    }


# =========================================================
# ATR
# =========================================================

def calculate_atr(df, period=14):

    if df.empty or len(df) < period + 2:

        return pd.Series(
            dtype=float
        )

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

    true_range = pd.concat(
        [
            tr1,
            tr2,
            tr3
        ],
        axis=1
    ).max(axis=1)

    atr = true_range.rolling(
        period
    ).mean()

    return atr


# =========================================================
# ATR ANALYSIS
# =========================================================

def analyze_atr(df):

    if df.empty or len(df) < 30:

        return {

            "atr": None,

            "atr_average": None,

            "atr_change": None,

            "atr_signal":
                "⚪ ATR Unknown",

            "atr_state":
                "Unknown"

        }

    atr_series = calculate_atr(
        df,
        ATR_PERIOD
    )

    atr_series = atr_series.dropna()

    if len(atr_series) < (
        ATR_COMPARE_PERIODS * 2
    ):

        return {

            "atr": None,

            "atr_average": None,

            "atr_change": None,

            "atr_signal":
                "⚪ ATR Unknown",

            "atr_state":
                "Unknown"

        }

    current_atr = float(
        atr_series.iloc[-1]
    )

    previous_atr_average = float(
        atr_series.iloc[
            -(
                ATR_COMPARE_PERIODS + 1
            ):
            -1
        ].mean()
    )

    if previous_atr_average <= 0:

        return {

            "atr": current_atr,

            "atr_average":
                previous_atr_average,

            "atr_change": None,

            "atr_signal":
                "⚪ ATR Neutral",

            "atr_state":
                "Neutral"

        }

    change = (
        (
            current_atr -
            previous_atr_average
        )
        /
        previous_atr_average
    ) * 100

    # -----------------------------------------------------
    # Expansion
    # -----------------------------------------------------

    if change >= 5:

        signal = "🔥 ATR EXPANDING"

        state = "EXPANSION"

    # -----------------------------------------------------
    # Compression
    # -----------------------------------------------------

    elif change <= -5:

        signal = "🧊 ATR CONTRACTING"

        state = "COMPRESSION"

    else:

        signal = "⚪ ATR STABLE"

        state = "STABLE"

    return {

        "atr":
            current_atr,

        "atr_average":
            previous_atr_average,

        "atr_change":
            change,

        "atr_signal":
            signal,

        "atr_state":
            state

    }


# =========================================================
# VOLUME
# =========================================================

def analyze_volume(df):

    if df.empty or len(df) < 10:

        return {

            "volume_ratio": 0,

            "volume_signal":
                "⚪ Volume Unknown",

            "current_volume":
                0,

            "average_volume":
                0

        }

    current_volume = float(
        df["volume"].iloc[-1]
    )

    average_volume = float(
        df["volume"].iloc[-6:-1].mean()
    )

    if average_volume <= 0:

        return {

            "volume_ratio": 0,

            "volume_signal":
                "⚪ Volume Unknown",

            "current_volume":
                current_volume,

            "average_volume":
                average_volume

        }

    ratio = (
        current_volume /
        average_volume
    )

    if ratio >= 2:

        signal = "🔥 VOLUME SPIKE"

    elif ratio >= 1.3:

        signal = "🟢 VOLUME HIGH"

    elif ratio <= 0.7:

        signal = "🧊 VOLUME LOW"

    else:

        signal = "⚪ VOLUME NORMAL"

    return {

        "volume_ratio":
            ratio,

        "volume_signal":
            signal,

        "current_volume":
            current_volume,

        "average_volume":
            average_volume

    }


# =========================================================
# OI
# =========================================================

def analyze_oi(symbol):

    df = get_oi_history(
        symbol
    )

    if df.empty or len(df) < 5:

        return {

            "oi_current":
                None,

            "oi_average":
                None,

            "oi_change":
                None,

            "oi_ratio":
                None,

            "oi_signal":
                "⚪ OI Unknown"

        }

    current = float(
        df["close"].iloc[-1]
    )

    average = float(
        df["close"].iloc[:-1].mean()
    )

    previous = float(
        df["close"].iloc[-5]
    )

    if previous == 0:

        change = None

    else:

        change = (
            (
                current -
                previous
            )
            /
            abs(previous)
        ) * 100

    if average > 0:

        ratio = (
            current /
            average
        )

    else:

        ratio = None

    if change is None:

        signal = "⚪ OI Unknown"

    elif change >= 1:

        signal = "🔺 OI UP"

    elif change <= -1:

        signal = "🔻 OI DOWN"

    else:

        signal = "⚪ OI NEUTRAL"

    return {

        "oi_current":
            current,

        "oi_average":
            average,

        "oi_change":
            change,

        "oi_ratio":
            ratio,

        "oi_signal":
            signal

    }


# =========================================================
# LEVERAGE CATEGORY
# =========================================================

def leverage_category(max_leverage):

    if max_leverage is None:

        return (
            "⚪ Unknown",
            "Unknown"
        )

    try:

        lev = float(
            max_leverage
        )

    except Exception:

        return (
            "⚪ Unknown",
            "Unknown"
        )

    # 20x specifically
    if lev >= 20 and lev < 50:

        return (
            "20x+",
            "20x"
        )

    # 50x - 200x
    if lev >= 50 and lev <= 200:

        return (
            "50x–200x",
            "50x–200x"
        )

    # Above 200x
    if lev > 200:

        return (
            f"{lev:.0f}x+",
            "200x+"
        )

    return (
        f"{lev:.0f}x max",
        "<20x"
    )


# =========================================================
# DEEP ANALYSIS
# =========================================================

def deep_analysis(
    symbol,
    ticker,
    max_leverage,
    market_avg_volume,
    market_avg_oi
):

    # -----------------------------------------------------
    # TIMEFRAMES
    # -----------------------------------------------------

    df_1h = get_candles(
        symbol,
        "1h",
        72
    )

    df_15m = get_candles(
        symbol,
        "15m",
        24
    )

    # 5m ko ek baar fetch kar rahe hain
    df_5m = get_candles(
        symbol,
        "5m",
        12
    )

    # -----------------------------------------------------
    # TREND
    # -----------------------------------------------------

    trend_1h = timeframe_trend(
        df_1h
    )

    trend_15m = timeframe_trend(
        df_15m
    )

    bos = analyze_5m(
        df_5m
    )

    trend_5m = bos[
        "trend"
    ]

    # -----------------------------------------------------
    # LIQUIDITY
    # -----------------------------------------------------

    if df_15m.empty or len(df_15m) < 10:

        sweep = {

            "bull_sweep": False,

            "bear_sweep": False,

            "liquidity": "⚪ None"

        }

    else:

        last = df_15m.iloc[-1]

        previous = df_15m.iloc[-7:-1]

        previous_high = float(
            previous["high"].max()
        )

        previous_low = float(
            previous["low"].min()
        )

        bull_sweep = (

            float(last["low"])
            < previous_low

            and

            float(last["close"])
            > previous_low

        )

        bear_sweep = (

            float(last["high"])
            > previous_high

            and

            float(last["close"])
            < previous_high

        )

        if bull_sweep:

            liquidity_name = (
                "🟢 BULL SWEEP"
            )

        elif bear_sweep:

            liquidity_name = (
                "🔴 BEAR SWEEP"
            )

        else:

            liquidity_name = (
                "⚪ None"
            )

        sweep = {

            "bull_sweep":
                bull_sweep,

            "bear_sweep":
                bear_sweep,

            "liquidity":
                liquidity_name

        }

    # -----------------------------------------------------
    # MTF
    # -----------------------------------------------------

    mtf = analyze_mtf(
        trend_1h,
        trend_15m,
        trend_5m
    )

    # -----------------------------------------------------
    # VOLUME
    # -----------------------------------------------------

    volume = analyze_volume(
        df_5m
    )

    volume_ratio = volume[
        "volume_ratio"
    ]

    current_volume = volume[
        "current_volume"
    ]

    average_volume = volume[
        "average_volume"
    ]

    # -----------------------------------------------------
    # OI
    # -----------------------------------------------------

    oi = analyze_oi(
        symbol
    )

    oi_current = oi[
        "oi_current"
    ]

    oi_average = oi[
        "oi_average"
    ]

    oi_change = oi[
        "oi_change"
    ]

    oi_ratio = oi[
        "oi_ratio"
    ]

    # -----------------------------------------------------
    # ATR
    # -----------------------------------------------------

    atr = analyze_atr(
        df_5m
    )

    atr_current = atr[
        "atr"
    ]

    atr_average = atr[
        "atr_average"
    ]

    atr_change = atr[
        "atr_change"
    ]

    atr_signal = atr[
        "atr_signal"
    ]

    # -----------------------------------------------------
    # MARKET AVERAGES
    # -----------------------------------------------------

    if market_avg_volume > 0:

        market_volume_ratio = (
            current_volume /
            market_avg_volume
        )

    else:

        market_volume_ratio = None

    if (
        oi_current is not None
        and
        market_avg_oi > 0
    ):

        market_oi_ratio = (
            oi_current /
            market_avg_oi
        )

    else:

        market_oi_ratio = None

    # -----------------------------------------------------
    # FUNDING
    # -----------------------------------------------------

    funding = ticker.get(
        "Funding",
        None
    )

    if funding is not None:

        try:

            funding = float(
                funding
            )

            funding_pct = (
                funding * 100
            )

        except Exception:

            funding_pct = None

    else:

        funding_pct = None

    # -----------------------------------------------------
    # SCORES
    # -----------------------------------------------------

    long_score = 0

    short_score = 0

    long_reason = []

    short_reason = []

    # =====================================================
    # 1H
    # =====================================================

    if trend_1h == "🟢 BULLISH":

        long_score += 2

        long_reason.append(
            "1H bullish"
        )

    elif trend_1h == "🔴 BEARISH":

        short_score += 2

        short_reason.append(
            "1H bearish"
        )

    # =====================================================
    # MTF
    # =====================================================

    if mtf["long_points"]:

        long_score += (
            mtf["long_points"]
        )

        long_reason.append(
            "MTF directional up"
        )

    if mtf["short_points"]:

        short_score += (
            mtf["short_points"]
        )

        short_reason.append(
            "MTF directional down"
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

    if volume_ratio >= 2:

        long_score += 2

        short_score += 2

        long_reason.append(
            "5m volume spike"
        )

        short_reason.append(
            "5m volume spike"
        )

    elif volume_ratio >= 1.3:

        long_score += 1

        short_score += 1

    # =====================================================
    # MARKET VOLUME AVERAGE
    # =====================================================

    if (
        market_volume_ratio is not None
        and
        market_volume_ratio >= 2
    ):

        long_score += 1

        short_score += 1

        long_reason.append(
            "Volume > market average"
        )

        short_reason.append(
            "Volume > market average"
        )

    # =====================================================
    # OI
    # =====================================================

    if oi_change is not None:

        if oi_change >= 1:

            if trend_1h == "🟢 BULLISH":

                long_score += 1

                long_reason.append(
                    "OI increasing"
                )

            elif trend_1h == "🔴 BEARISH":

                short_score += 1

                short_reason.append(
                    "OI increasing"
                )

        elif oi_change <= -1:

            if sweep["bull_sweep"]:

                long_score += 1

                long_reason.append(
                    "OI falling after bull sweep"
                )

            if sweep["bear_sweep"]:

                short_score += 1

                short_reason.append(
                    "OI falling after bear sweep"
                )

    # =====================================================
    # OI ABOVE MARKET AVERAGE
    # =====================================================

    if (
        market_oi_ratio is not None
        and
        market_oi_ratio >= 1.5
    ):

        if trend_1h == "🟢 BULLISH":

            long_score += 1

            long_reason.append(
                "OI > market average"
            )

        elif trend_1h == "🔴 BEARISH":

            short_score += 1

            short_reason.append(
                "OI > market average"
            )

    # =====================================================
    # ATR
    # =====================================================

    if atr_change is not None:

        # Expansion
        if atr_change >= 5:

            if (
                mtf["state"]
                ==
                "🟢 DIRECTIONAL UP"
            ):

                long_score += 1

                long_reason.append(
                    "ATR expanding + bullish"
                )

            elif (
                mtf["state"]
                ==
                "🔴 DIRECTIONAL DOWN"
            ):

                short_score += 1

                short_reason.append(
                    "ATR expanding + bearish"
                )

            else:

                # Expansion without direction
                # gets no directional point.
                pass

        # Compression
        elif atr_change <= -5:

            # Compression means range/
            # reduced volatility.
            # No directional score.

            pass

    # =====================================================
    # FUNDING
    # =====================================================

    if funding_pct is None:

        funding_signal = (
            "⚪ Funding unavailable"
        )

    else:

        # Positive funding
        if funding_pct >= 0.05:

            short_score += 2

            short_reason.append(
                "High positive funding"
            )

            funding_signal = (
                "🔴 Longs crowded"
            )

        # Negative funding
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
    # RANGE PENALTY
    # =====================================================

    if mtf["state"] == "⚪ RANGE BOUND":

        # Range-bound market me
        # strong directional score ko
        # artificially strong nahi hone dena.

        long_score = max(
            0,
            long_score - 1
        )

        short_score = max(
            0,
            short_score - 1
        )

    # =====================================================
    # UNCERTAINTY
    # =====================================================

    if mtf["state"] == "🟡 UNCERTAINTY":

        # No bonus
        pass

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
    # DOMINANT
    # =====================================================

    if (
        long_score >= 8
        and
        long_score > short_score
    ):

        signal = (
            "🟢 STRONG LONG"
        )

        dominant_score = (
            long_score
        )

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

        signal = (
            "🔴 STRONG SHORT"
        )

        dominant_score = (
            short_score
        )

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

        signal = (
            "🟡 LONG WATCH"
        )

        dominant_score = (
            long_score
        )

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

        signal = (
            "🟠 SHORT WATCH"
        )

        dominant_score = (
            short_score
        )

        dominant_reason = (
            " + ".join(
                short_reason
            )
        )

    else:

        signal = (
            "⚪ NO SIGNAL"
        )

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

    if not df_5m.empty:

        last = df_5m.iloc[-1]

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
                    high - low
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
                    high - low
                ) * 0.50
            )

            entry_high = high

            entry_zone = (
                f"{entry_low:.6g} - "
                f"{entry_high:.6g}"
            )

    # =====================================================
    # LEVERAGE
    # =====================================================

    leverage_display, leverage_group = (
        leverage_category(
            max_leverage
        )
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

        # -----------------------------------------------
        # MTF
        # -----------------------------------------------

        "1H Trend":
            trend_1h,

        "15m Trend":
            trend_15m,

        "5m Trend":
            trend_5m,

        "MTF State":
            mtf["state"],

        "MTF Confirmation":
            mtf["confirmation"],

        # -----------------------------------------------
        # STRUCTURE
        # -----------------------------------------------

        "15m Liquidity":
            sweep["liquidity"],

        "5m BOS":
            bos["structure"],

        # -----------------------------------------------
        # VOLUME
        # -----------------------------------------------

        "24H Volume":
            round(
                float(
                    ticker["24H Volume"]
                ),
                2
            ),

        "Volume Avg":
            round(
                average_volume,
                2
            ),

        "Volume x":
            round(
                volume_ratio,
                2
            ),

        "Market Vol x":
            (
                round(
                    market_volume_ratio,
                    2
                )
                if market_volume_ratio
                is not None
                else None
            ),

        "Volume Signal":
            volume[
                "volume_signal"
            ],

        # -----------------------------------------------
        # OI
        # -----------------------------------------------

        "OI Current":
            (
                round(
                    oi_current,
                    2
                )
                if oi_current
                is not None
                else None
            ),

        "OI Average":
            (
                round(
                    oi_average,
                    2
                )
                if oi_average
                is not None
                else None
            ),

        "OI Avg x":
            (
                round(
                    oi_ratio,
                    2
                )
                if oi_ratio
                is not None
                else None
            ),

        "OI Change %":
            (
                round(
                    oi_change,
                    2
                )
                if oi_change
                is not None
                else None
            ),

        "OI":
            oi[
                "oi_signal"
            ],

        # -----------------------------------------------
        # ATR
        # -----------------------------------------------

        "ATR":
            (
                round(
                    atr_current,
                    6
                )
                if atr_current
                is not None
                else None
            ),

        "ATR Avg":
            (
                round(
                    atr_average,
                    6
                )
                if atr_average
                is not None
                else None
            ),

        "ATR Change %":
            (
                round(
                    atr_change,
                    2
                )
                if atr_change
                is not None
                else None
            ),

        "ATR Signal":
            atr_signal,

        # -----------------------------------------------
        # FUNDING
        # -----------------------------------------------

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

        # -----------------------------------------------
        # LEVERAGE
        # -----------------------------------------------

        "Max Leverage":
            max_leverage,

        "Leverage":
            leverage_display,

        "Leverage Group":
            leverage_group,

        # -----------------------------------------------
        # SCORES
        # -----------------------------------------------

        "Long Score":
            long_score,

        "Long Signal":
            long_signal,

        "Short Score":
            short_score,

        "Short Signal":
            short_signal,

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
# MARKET AVERAGES
# =========================================================

market_avg_volume = float(
    market["24H Volume"].mean()
)

market_avg_oi = float(
    market["OI"].mean()
)


# =========================================================
# METRICS
# =========================================================

c1, c2, c3, c4 = st.columns(4)

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
        "Market Avg 24H Volume",
        f"{market_avg_volume:,.0f}"
    )

with c4:

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

st.caption(
    "Market Average = सभी live perpetual contracts "
    "का current average. Deep historical analysis "
    "top active coins पर किया जाता है ताकि API load "
    "control में रहे."
)

all_display = market.copy()

all_display["Vol / Market Avg"] = (
    all_display["24H Volume"] /
    market_avg_volume
    if market_avg_volume > 0
    else np.nan
)

all_display["OI / Market Avg"] = (
    all_display["OI"] /
    market_avg_oi
    if market_avg_oi > 0
    else np.nan
)

all_display["Leverage"] = (
    all_display["Max Leverage"]
    .apply(
        lambda x:
        leverage_category(x)[0]
    )
)

st.dataframe(
    all_display[
        [
            "Coin",
            "Price",
            "24H Volume",
            "Vol / Market Avg",
            "OI",
            "OI / Market Avg",
            "Funding",
            "Max Leverage",
            "Leverage"
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
    "🎯 Advanced Scanner Results"
)

results = []

progress = st.progress(0)

total = len(candidates)

for i, (_, row) in enumerate(
    candidates.iterrows()
):

    result = deep_analysis(

        row["Coin"],

        row,

        row.get(
            "Max Leverage"
        ),

        market_avg_volume,

        market_avg_oi

    )

    if result:

        results.append(
            result
        )

    if total > 0:

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
    "🟢 LONG — Complete Confirmation"
)

if not signals.empty:

    long_table = signals[
        [
            "Coin",
            "Price",
            "MTF State",
            "1H Trend",
            "15m Trend",
            "5m Trend",
            "15m Liquidity",
            "5m BOS",
            "24H Volume",
            "Volume Avg",
            "Volume x",
            "OI Current",
            "OI Average",
            "OI Avg x",
            "OI Change %",
            "ATR",
            "ATR Avg",
            "ATR Change %",
            "ATR Signal",
            "Funding %",
            "Funding",
            "Leverage",
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
    "🔴 SHORT — Complete Confirmation"
)

if not signals.empty:

    short_table = signals[
        [
            "Coin",
            "Price",
            "MTF State",
            "1H Trend",
            "15m Trend",
            "5m Trend",
            "15m Liquidity",
            "5m BOS",
            "24H Volume",
            "Volume Avg",
            "Volume x",
            "OI Current",
            "OI Average",
            "OI Avg x",
            "OI Change %",
            "ATR",
            "ATR Avg",
            "ATR Change %",
            "ATR Signal",
            "Funding %",
            "Funding",
            "Leverage",
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
                    "MTF State",
                    "Funding %",
                    "Funding",
                    "OI Change %",
                    "OI Avg x",
                    "Volume x",
                    "ATR Change %",
                    "ATR Signal",
                    "Leverage",
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
                    "MTF State",
                    "Funding %",
                    "Funding",
                    "OI Change %",
                    "OI Avg x",
                    "Volume x",
                    "ATR Change %",
                    "ATR Signal",
                    "Leverage",
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
Liquidity + BOS + OI + Volume + MTF ke saath
confirmation ke रूप में use kiya gaya hai.
"""
)


# =========================================================
# ATR EXPLANATION
# =========================================================

st.subheader(
    "📐 ATR Logic"
)

st.write(
    """
ATR ↑ 5% या उससे ज्यादा
→ Volatility expansion

ATR ↓ 5% या उससे ज्यादा
→ Volatility compression

ATR expansion + Directional MTF
→ संबंधित LONG/SHORT को +1

ATR compression
→ Directional score नहीं दिया जाता।

इससे सिर्फ volatility बढ़ने को
automatically bullish या bearish नहीं माना जाता।
"""
)


# =========================================================
# MULTI TIMEFRAME
# =========================================================

st.subheader(
    "🧭 Multi-Timeframe Logic"
)

st.write(
    """
1H + 15m + 5m को साथ देखा जाता है।

🟢 DIRECTIONAL UP
→ कम से कम 2 timeframes bullish

🔴 DIRECTIONAL DOWN
→ कम से कम 2 timeframes bearish

⚪ RANGE BOUND
→ कम से कम 2 timeframes range

🟡 UNCERTAINTY
→ बाकी mixed conditions

Directional MTF confirmation
→ संबंधित direction को +2
"""
)


# =========================================================
# LEVERAGE
# =========================================================

st.subheader(
    "⚡ Leverage Information"
)

st.write(
    """
20x category
→ 20x से 50x से कम maximum leverage

50x–200x
→ अलग category

200x से ऊपर
→ 200x+ category

Leverage को score में शामिल नहीं किया गया है।
यह केवल risk/contract information है।
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
Activity / Volume ranking
↓
Top active coins
↓
1H Trend
↓
15m Trend + Liquidity Sweep
↓
5m Trend + BOS
↓
Multi-Timeframe State
↓
OI Current + OI Average + OI Change
↓
24H Volume + Average + Ratio
↓
ATR + ATR Average + ATR Direction
↓
Funding
↓
Separate LONG / SHORT Score
↓
8+ = Strong setup
"""
)


# =========================================================
# IMPORTANT NOTE
# =========================================================

st.info(
    """
एक जरूरी बात:

इस version में अभी scanner top active
coins को deep scan करता है। सभी 220 coins
पर candles + OI history एक साथ नहीं लिया जाता,
क्योंकि API requests बहुत बढ़ जाएँगी और
server problem दोबारा हो सकती है।

All 220 coins की current ticker information
ऊपर दिखाई जाती है।

Deep historical analysis फिलहाल top
active coins पर किया जाता है।

अगले चरण में चाहें तो:

1H trend
→ 15m liquidity sweep
→ 5m BOS
→ OI displacement
→ Funding
→ ATR expansion
→ Entry zone

को और ज्यादा strictly filter करके
signal quality test किया जा सकता है।
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
