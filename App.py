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
    "User-Agent": "Delta-Advanced-Reversal-Scanner/8.0"
}

CACHE_SECONDS = 120

# API/server protection
DEEP_SCAN_LIMIT = 30

# Main filter
MIN_VOL_OI_RATIO = 6.0

# Signal thresholds
STRONG_SCORE = 10
WATCH_SCORE = 6

# Funding
FUNDING_EXTREME = 0.05
FUNDING_CHANGE_EXTREME = 0.02

# OI
OI_THRESHOLD = 1.0

# Volume
VOLUME_EXPANSION = 1.30
VOLUME_STRONG = 2.00

# ATR
ATR_PERIOD = 14
ATR_RISING_THRESHOLD = 5.0

# FVG
FVG_LOOKBACK = 20
FVG_MAX_AGE = 12

# R:R
MIN_RR = 1.5

st.set_page_config(
    page_title="Delta Advanced Scanner",
    layout="wide"
)

st.title("🔥 Delta Advanced Reversal Scanner")

st.caption(
    "BTC Regime → 1H Trend → Swing → Liquidity Sweep → "
    "5m BOS → Displacement → FVG → OI → Funding → "
    "Volume → ATR → R:R"
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

        # Funding
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
    ).sort_values("time")


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
    ).sort_values("time")


# =========================================================
# BTC REGIME
# =========================================================

def btc_regime():

    df = get_candles(
        "BTCUSD",
        "1h",
        72
    )

    if df.empty or len(df) < 25:

        return {
            "regime": "⚪ UNKNOWN",
            "score": 0
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

    price = float(close.iloc[-1])

    fast = float(ema9.iloc[-1])

    slow = float(ema21.iloc[-1])

    if fast > slow and price > fast:

        return {
            "regime": "🟢 BTC BULLISH",
            "score": 1
        }

    if fast < slow and price < fast:

        return {
            "regime": "🔴 BTC BEARISH",
            "score": -1
        }

    return {
        "regime": "⚪ BTC NEUTRAL",
        "score": 0
    }


# =========================================================
# 1H TREND
# =========================================================

def analyze_1h(symbol):

    df = get_candles(
        symbol,
        "1h",
        72
    )

    if df.empty or len(df) < 25:

        return {
            "trend": "⚪ UNKNOWN",
            "direction": "UNCERTAIN"
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

    price = float(close.iloc[-1])

    fast = float(ema9.iloc[-1])

    slow = float(ema21.iloc[-1])

    if fast > slow and price > fast:

        return {
            "trend": "🟢 BULLISH",
            "direction": "LONG"
        }

    if fast < slow and price < fast:

        return {
            "trend": "🔴 BEARISH",
            "direction": "SHORT"
        }

    return {
        "trend": "⚪ NEUTRAL",
        "direction": "UNCERTAIN"
    }


# =========================================================
# 15M SWING + LIQUIDITY
# =========================================================

def analyze_15m(symbol):

    df = get_candles(
        symbol,
        "15m",
        36
    )

    if df.empty or len(df) < 15:

        return {
            "bull_sweep": False,
            "bear_sweep": False,
            "liquidity": "⚪ None",
            "swing_high": None,
            "swing_low": None
        }

    last = df.iloc[-1]

    previous = df.iloc[-8:-1]

    swing_high = float(
        previous["high"].max()
    )

    swing_low = float(
        previous["low"].min()
    )

    bull_sweep = (
        float(last["low"]) < swing_low
        and
        float(last["close"]) > swing_low
    )

    bear_sweep = (
        float(last["high"]) > swing_high
        and
        float(last["close"]) < swing_high
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
        "liquidity": liquidity,
        "swing_high": swing_high,
        "swing_low": swing_low
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
            "structure": "⚪ None",
            "displacement": False
        }

    last = df.iloc[-1]

    previous = df.iloc[-8:-1]

    previous_high = float(
        previous["high"].max()
    )

    previous_low = float(
        previous["low"].min()
    )

    close = float(last["close"])

    open_price = float(last["open"])

    candle_range = (
        float(last["high"]) -
        float(last["low"])
    )

    body = abs(
        close - open_price
    )

    body_ratio = (
        body / candle_range
        if candle_range > 0
        else 0
    )

    bull_bos = (
        close > previous_high
    )

    bear_bos = (
        close < previous_low
    )

    displacement = (
        body_ratio >= 0.60
    )

    if bull_bos:

        structure = "🟢 BULL BOS"

    elif bear_bos:

        structure = "🔴 BEAR BOS"

    else:

        structure = "⚪ None"

    return {
        "bull_bos": bull_bos,
        "bear_bos": bear_bos,
        "structure": structure,
        "displacement": displacement
    }


# =========================================================
# FVG
# =========================================================

def analyze_fvg(symbol):

    df = get_candles(
        symbol,
        "5m",
        12
    )

    if df.empty or len(df) < 5:

        return {
            "bull_fvg": False,
            "bear_fvg": False,
            "fvg": "⚪ None",
            "fresh": False,
            "retest": False,
            "zone": None
        }

    bull_fvg = False
    bear_fvg = False

    zone_low = None
    zone_high = None

    age = None

    # Search recent FVG
    start_index = max(
        2,
        len(df) - FVG_LOOKBACK
    )

    for i in range(
        len(df) - 1,
        start_index - 1,
        -1
    ):

        c1 = df.iloc[i - 2]
        c3 = df.iloc[i]

        # Bullish FVG
        if float(c3["low"]) > float(c1["high"]):

            bull_fvg = True

            zone_low = float(
                c1["high"]
            )

            zone_high = float(
                c3["low"]
            )

            age = (
                len(df) - 1 - i
            )

            break

        # Bearish FVG
        if float(c3["high"]) < float(c1["low"]):

            bear_fvg = True

            zone_low = float(
                c3["high"]
            )

            zone_high = float(
                c1["low"]
            )

            age = (
                len(df) - 1 - i
            )

            break

    if age is None:

        return {
            "bull_fvg": False,
            "bear_fvg": False,
            "fvg": "⚪ None",
            "fresh": False,
            "retest": False,
            "zone": None
        }

    last_price = float(
        df["close"].iloc[-1]
    )

    fresh = (
        age <= FVG_MAX_AGE
    )

    retest = (
        zone_low <= last_price <= zone_high
    )

    if bull_fvg:

        fvg_name = "🟢 Bull FVG"

    elif bear_fvg:

        fvg_name = "🔴 Bear FVG"

    else:

        fvg_name = "⚪ None"

    return {
        "bull_fvg": bull_fvg,
        "bear_fvg": bear_fvg,
        "fvg": fvg_name,
        "fresh": fresh,
        "retest": retest,
        "zone": (
            f"{zone_low:.6g} - "
            f"{zone_high:.6g}"
        )
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
            "volume_ratio": 0,
            "volume_signal": "⚪ Unknown"
        }

    current = float(
        df["volume"].iloc[-1]
    )

    average = float(
        df["volume"].iloc[-6:-1].mean()
    )

    if average <= 0:

        return {
            "volume_ratio": 0,
            "volume_signal": "⚪ Unknown"
        }

    ratio = current / average

    if ratio >= VOLUME_STRONG:

        signal = "🔥 Volume Expansion"

    elif ratio >= VOLUME_EXPANSION:

        signal = "🟢 Volume Rising"

    else:

        signal = "⚪ Volume Normal"

    return {
        "volume_ratio": ratio,
        "volume_signal": signal
    }


# =========================================================
# ATR
# =========================================================

def analyze_atr(symbol):

    df = get_candles(
        symbol,
        "5m",
        8
    )

    if df.empty or len(df) < ATR_PERIOD + 3:

        return {
            "atr_pct": 0,
            "atr_change": 0,
            "atr_signal": "⚪ Unknown"
        }

    high = df["high"]
    low = df["low"]
    close = df["close"]

    previous_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs()
        ],
        axis=1
    ).max(axis=1)

    atr = tr.rolling(
        ATR_PERIOD
    ).mean()

    current_atr = float(
        atr.iloc[-1]
    )

    previous_atr = float(
        atr.iloc[-2]
    )

    price = float(
        close.iloc[-1]
    )

    if price <= 0:

        return {
            "atr_pct": 0,
            "atr_change": 0,
            "atr_signal": "⚪ Unknown"
        }

    atr_pct = (
        current_atr /
        price
    ) * 100

    if previous_atr > 0:

        atr_change = (
            (
                current_atr -
                previous_atr
            )
            /
            previous_atr
        ) * 100

    else:

        atr_change = 0

    if atr_change >= ATR_RISING_THRESHOLD:

        signal = "🔥 ATR EXPANDING"

    elif atr_change <= -ATR_RISING_THRESHOLD:

        signal = "🔻 ATR FALLING"

    else:

        signal = "⚪ ATR STABLE"

    return {
        "atr_pct": atr_pct,
        "atr_change": atr_change,
        "atr_signal": signal
    }


# =========================================================
# OI
# =========================================================

def analyze_oi(symbol, price_change=None):

    df = get_oi_history(symbol)

    if df.empty or len(df) < 5:

        return {
            "oi_change": None,
            "oi_signal": "⚪ Unknown",
            "oi_state": "UNKNOWN"
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
            "oi_signal": "⚪ Unknown",
            "oi_state": "UNKNOWN"
        }

    change = (
        (current - previous)
        /
        abs(previous)
    ) * 100

    if change >= OI_THRESHOLD:

        oi_signal = "🔺 OI UP"

    elif change <= -OI_THRESHOLD:

        oi_signal = "🔻 OI DOWN"

    else:

        oi_signal = "⚪ OI NEUTRAL"

    # -----------------------------------------------------
    # PRICE + OI FOUR STATES
    # -----------------------------------------------------

    if price_change is None:

        state = "UNKNOWN"

    elif price_change > 0 and change > 0:

        state = "🟢 LONG BUILDUP"

    elif price_change < 0 and change > 0:

        state = "🔴 SHORT BUILDUP"

    elif price_change > 0 and change < 0:

        state = "🟡 SHORT COVERING"

    elif price_change < 0 and change < 0:

        state = "🟠 LONG LIQUIDATION"

    else:

        state = "⚪ NEUTRAL"

    return {
        "oi_change": change,
        "oi_signal": oi_signal,
        "oi_state": state
    }


# =========================================================
# MULTI TIMEFRAME REGIME
# =========================================================

def analyze_market_regime(symbol):

    d1 = get_candles(
        symbol,
        "1h",
        48
    )

    d15 = get_candles(
        symbol,
        "15m",
        24
    )

    d5 = get_candles(
        symbol,
        "5m",
        8
    )

    if (
        d1.empty
        or d15.empty
        or d5.empty
    ):

        return "⚪ UNCERTAIN"

    def direction(df):

        if len(df) < 10:
            return 0

        ema9 = df["close"].ewm(
            span=9,
            adjust=False
        ).mean().iloc[-1]

        ema21 = df["close"].ewm(
            span=21,
            adjust=False
        ).mean().iloc[-1]

        if ema9 > ema21:
            return 1

        if ema9 < ema21:
            return -1

        return 0

    d1v = direction(d1)
    d15v = direction(d15)
    d5v = direction(d5)

    total = d1v + d15v + d5v

    # Strong directional
    if total >= 3:

        return "🟢 DIRECTIONAL LONG"

    if total <= -3:

        return "🔴 DIRECTIONAL SHORT"

    # Mixed
    if abs(total) <= 1:

        # Check range
        recent = d15["close"].tail(20)

        if len(recent) >= 10:

            high = recent.max()
            low = recent.min()

            if low > 0:

                range_pct = (
                    (high - low)
                    /
                    low
                ) * 100

                if range_pct < 3:

                    return "🟡 RANGE BOUND"

        return "⚪ UNCERTAIN"

    return "⚪ UNCERTAIN"


# =========================================================
# FUNDING
# =========================================================

def analyze_funding(current_funding):

    if current_funding is None:

        return {
            "funding_pct": None,
            "funding_signal": "⚪ Unavailable"
        }

    funding_pct = (
        current_funding * 100
    )

    if funding_pct >= FUNDING_EXTREME:

        signal = "🔴 LONGS CROWDED"

    elif funding_pct <= -FUNDING_EXTREME:

        signal = "🟢 SHORTS CROWDED"

    else:

        signal = "⚪ FUNDING NEUTRAL"

    return {
        "funding_pct": funding_pct,
        "funding_signal": signal
    }


# =========================================================
# RISK / REWARD
# =========================================================

def calculate_rr(
    signal,
    candles
):

    if candles.empty:

        return {
            "entry": None,
            "sl": None,
            "tp": None,
            "rr": None
        }

    last = candles.iloc[-1]

    high = float(last["high"])
    low = float(last["low"])
    close = float(last["close"])

    candle_range = high - low

    if candle_range <= 0:

        return {
            "entry": close,
            "sl": None,
            "tp": None,
            "rr": None
        }

    # LONG
    if "LONG" in signal:

        entry = (
            low +
            candle_range * 0.50
        )

        sl = low - (
            candle_range * 0.20
        )

        risk = entry - sl

        tp = entry + (
            risk * 2
        )

    # SHORT
    elif "SHORT" in signal:

        entry = (
            high -
            candle_range * 0.50
        )

        sl = high + (
            candle_range * 0.20
        )

        risk = sl - entry

        tp = entry - (
            risk * 2
        )

    else:

        return {
            "entry": None,
            "sl": None,
            "tp": None,
            "rr": None
        }

    if risk <= 0:

        rr = None

    else:

        rr = abs(
            (tp - entry)
            /
            risk
        )

    return {
        "entry": entry,
        "sl": sl,
        "tp": tp,
        "rr": rr
    }


# =========================================================
# DEEP ANALYSIS
# =========================================================

def deep_analysis(
    symbol,
    ticker,
    btc
):

    trend = analyze_1h(symbol)

    sweep = analyze_15m(symbol)

    bos = analyze_5m(symbol)

    fvg = analyze_fvg(symbol)

    volume = analyze_volume(symbol)

    atr = analyze_atr(symbol)

    regime = analyze_market_regime(
        symbol
    )

    # -----------------------------------------------------
    # PRICE CHANGE
    # -----------------------------------------------------

    candles_15 = get_candles(
        symbol,
        "15m",
        4
    )

    if (
        not candles_15.empty
        and len(candles_15) >= 2
    ):

        current_price = float(
            candles_15["close"].iloc[-1]
        )

        previous_price = float(
            candles_15["close"].iloc[-2]
        )

        if previous_price != 0:

            price_change = (
                (
                    current_price -
                    previous_price
                )
                /
                previous_price
            ) * 100

        else:

            price_change = None

    else:

        price_change = None

    oi = analyze_oi(
        symbol,
        price_change
    )

    funding = analyze_funding(
        ticker.get("Funding")
    )

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

    if trend["direction"] == "LONG":

        long_score += 2

        long_reason.append(
            "1H bullish"
        )

    elif trend["direction"] == "SHORT":

        short_score += 2

        short_reason.append(
            "1H bearish"
        )

    # =====================================================
    # BTC REGIME
    # =====================================================

    if btc["score"] > 0:

        long_score += 2

        long_reason.append(
            "BTC bullish"
        )

        short_score -= 1

        short_reason.append(
            "BTC bullish penalty"
        )

    elif btc["score"] < 0:

        short_score += 2

        short_reason.append(
            "BTC bearish"
        )

        long_score -= 1

        long_reason.append(
            "BTC bearish penalty"
        )

    # =====================================================
    # LIQUIDITY
    # =====================================================

    if sweep["bull_sweep"]:

        long_score += 3

        long_reason.append(
            "15m liquidity sweep"
        )

    if sweep["bear_sweep"]:

        short_score += 3

        short_reason.append(
            "15m liquidity sweep"
        )

    # =====================================================
    # BOS
    # =====================================================

    if bos["bull_bos"]:

        long_score += 3

        long_reason.append(
            "5m bullish BOS"
        )

        if bos["displacement"]:

            long_score += 2

            long_reason.append(
                "Strong displacement"
            )

    if bos["bear_bos"]:

        short_score += 3

        short_reason.append(
            "5m bearish BOS"
        )

        if bos["displacement"]:

            short_score += 2

            short_reason.append(
                "Strong displacement"
            )

    # =====================================================
    # FVG
    # =====================================================

    if fvg["bull_fvg"]:

        long_score += 2

        long_reason.append(
            "Bull FVG"
        )

        if fvg["fresh"]:

            long_score += 1

            long_reason.append(
                "Fresh FVG"
            )

        if fvg["retest"]:

            long_score += 2

            long_reason.append(
                "FVG retest"
            )

    if fvg["bear_fvg"]:

        short_score += 2

        short_reason.append(
            "Bear FVG"
        )

        if fvg["fresh"]:

            short_score += 1

            short_reason.append(
                "Fresh FVG"
            )

        if fvg["retest"]:

            short_score += 2

            short_reason.append(
                "FVG retest"
            )

    # =====================================================
    # VOLUME
    # =====================================================

    volume_ratio = volume[
        "volume_ratio"
    ]

    if volume_ratio >= VOLUME_STRONG:

        long_score += 2
        short_score += 2

        long_reason.append(
            "Strong volume"
        )

        short_reason.append(
            "Strong volume"
        )

    elif volume_ratio >= VOLUME_EXPANSION:

        long_score += 1
        short_score += 1

    # =====================================================
    # ATR
    # =====================================================

    if (
        atr["atr_change"]
        >= ATR_RISING_THRESHOLD
    ):

        long_score += 1
        short_score += 1

        long_reason.append(
            "ATR expanding"
        )

        short_reason.append(
            "ATR expanding"
        )

    elif (
        atr["atr_change"]
        <= -ATR_RISING_THRESHOLD
    ):

        long_score -= 1
        short_score -= 1

    # =====================================================
    # OI
    # =====================================================

    oi_change = oi[
        "oi_change"
    ]

    if oi_change is not None:

        if oi["oi_state"] == "🟢 LONG BUILDUP":

            long_score += 2

            long_reason.append(
                "Price + OI long buildup"
            )

        elif oi["oi_state"] == "🔴 SHORT BUILDUP":

            short_score += 2

            short_reason.append(
                "Price down + OI buildup"
            )

        elif oi["oi_state"] == "🟡 SHORT COVERING":

            long_score += 1

            long_reason.append(
                "Short covering"
            )

        elif oi["oi_state"] == "🟠 LONG LIQUIDATION":

            short_score += 1

            short_reason.append(
                "Long liquidation"
            )

    # =====================================================
    # FUNDING
    # =====================================================

    funding_pct = funding[
        "funding_pct"
    ]

    if funding_pct is not None:

        if funding_pct >= FUNDING_EXTREME:

            short_score += 2

            short_reason.append(
                "Positive funding crowding"
            )

        elif funding_pct <= -FUNDING_EXTREME:

            long_score += 2

            long_reason.append(
                "Negative funding crowding"
            )

    # =====================================================
    # MARKET REGIME
    # =====================================================

    if regime == "🟢 DIRECTIONAL LONG":

        long_score += 2

        long_reason.append(
            "MTF directional long"
        )

    elif regime == "🔴 DIRECTIONAL SHORT":

        short_score += 2

        short_reason.append(
            "MTF directional short"
        )

    elif regime == "🟡 RANGE BOUND":

        long_score -= 1
        short_score -= 1

    else:

        long_score -= 1
        short_score -= 1

    # =====================================================
    # SIGNAL
    # =====================================================

    if (
        long_score >= STRONG_SCORE
        and
        long_score > short_score
    ):

        signal = "🟢 STRONG LONG"

    elif (
        short_score >= STRONG_SCORE
        and
        short_score > long_score
    ):

        signal = "🔴 STRONG SHORT"

    elif (
        long_score >= WATCH_SCORE
        and
        long_score > short_score
    ):

        signal = "🟡 LONG WATCH"

    elif (
        short_score >= WATCH_SCORE
        and
        short_score > long_score
    ):

        signal = "🟠 SHORT WATCH"

    else:

        signal = "⚪ NO SIGNAL"

    # =====================================================
    # R:R
    # =====================================================

    candles_5 = get_candles(
        symbol,
        "5m",
        6
    )

    risk = calculate_rr(
        signal,
        candles_5
    )

    rr = risk["rr"]

    # R:R filter only for strong signal
    if (
        "STRONG" in signal
        and
        rr is not None
        and
        rr < MIN_RR
    ):

        signal = "⚪ R:R NOT GOOD"

    # =====================================================
    # LEVERAGE DISTANCE
    # =====================================================

    if risk["entry"] is not None:

        entry = risk["entry"]

        sl = risk["sl"]

        sl_distance_pct = (
            abs(entry - sl)
            /
            entry
        ) * 100

        lev20 = sl_distance_pct * 20
        lev50 = sl_distance_pct * 50
        lev100 = sl_distance_pct * 100
        lev200 = sl_distance_pct * 200

    else:

        sl_distance_pct = None
        lev20 = None
        lev50 = None
        lev100 = None
        lev200 = None

    # =====================================================
    # DOMINANT SCORE
    # =====================================================

    dominant_score = max(
        long_score,
        short_score
    )

    if (
        long_score >
        short_score
    ):

        dominant_reason = (
            " + ".join(long_reason)
        )

    elif (
        short_score >
        long_score
    ):

        dominant_reason = (
            " + ".join(short_reason)
        )

    else:

        dominant_reason = (
            "Mixed conditions"
        )

    # =====================================================
    # RESULT
    # =====================================================

    return {

        "Coin": symbol,

        "Price": round(
            float(ticker["Price"]),
            8
        ),

        "Vol/OI": round(
            float(ticker["Vol/OI"]),
            2
        ),

        "1H Trend":
            trend["trend"],

        "BTC Regime":
            btc["regime"],

        "MTF Regime":
            regime,

        "15m Liquidity":
            sweep["liquidity"],

        "Swing High":
            sweep["swing_high"],

        "Swing Low":
            sweep["swing_low"],

        "5m BOS":
            bos["structure"],

        "Displacement":
            "🔥 YES"
            if bos["displacement"]
            else "⚪ NO",

        "FVG":
            fvg["fvg"],

        "FVG Fresh":
            "YES"
            if fvg["fresh"]
            else "NO",

        "FVG Retest":
            "YES"
            if fvg["retest"]
            else "NO",

        "FVG Zone":
            fvg["zone"],

        "Volume x":
            round(
                volume_ratio,
                2
            ),

        "Volume":
            volume["volume_signal"],

        "ATR %":
            round(
                atr["atr_pct"],
                3
            ),

        "ATR Change %":
            round(
                atr["atr_change"],
                2
            ),

        "ATR":
            atr["atr_signal"],

        "Price Change %":
            (
                round(
                    price_change,
                    3
                )
                if price_change is not None
                else None
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

        "OI State":
            oi["oi_state"],

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
            funding["funding_signal"],

        "Long Score":
            long_score,

        "Long Signal":
            (
                "🟢 STRONG LONG"
                if long_score >= STRONG_SCORE
                else
                "🟡 LONG WATCH"
                if long_score >= WATCH_SCORE
                else
                "⚪ NO LONG"
            ),

        "Short Score":
            short_score,

        "Short Signal":
            (
                "🔴 STRONG SHORT"
                if short_score >= STRONG_SCORE
                else
                "🟠 SHORT WATCH"
                if short_score >= WATCH_SCORE
                else
                "⚪ NO SHORT"
            ),

        "Score":
            dominant_score,

        "Signal":
            signal,

        "Entry":
            (
                round(
                    risk["entry"],
                    8
                )
                if risk["entry"] is not None
                else None
            ),

        "SL":
            (
                round(
                    risk["sl"],
                    8
                )
                if risk["sl"] is not None
                else None
            ),

        "TP":
            (
                round(
                    risk["tp"],
                    8
                )
                if risk["tp"] is not None
                else None
            ),

        "R:R":
            (
                round(
                    rr,
                    2
                )
                if rr is not None
                else None
            ),

        "SL Distance %":
            (
                round(
                    sl_distance_pct,
                    3
                )
                if sl_distance_pct is not None
                else None
            ),

        "20x SL Impact %":
            (
                round(
                    lev20,
                    2
                )
                if lev20 is not None
                else None
            ),

        "50x SL Impact %":
            (
                round(
                    lev50,
                    2
                )
                if lev50 is not None
                else None
            ),

        "100x SL Impact %":
            (
                round(
                    lev100,
                    2
                )
                if lev100 is not None
                else None
            ),

        "200x SL Impact %":
            (
                round(
                    lev200,
                    2
                )
                if lev200 is not None
                else None
            ),

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
# BTC
# =========================================================

BTC = btc_regime()

st.info(
    f"BTC Market Regime: {BTC['regime']}"
)


# =========================================================
# MERGE
# =========================================================

market = all_coins.merge(
    tickers,
    on="Coin",
    how="left"
)

market = market.dropna(
    subset=[
        "Price",
        "Vol/OI"
    ]
)


# =========================================================
# IMPORTANT FILTER
# =========================================================

market = market[
    market["Vol/OI"] >
    MIN_VOL_OI_RATIO
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
        "Eligible Coins",
        len(market)
    )

with c2:

    st.metric(
        "Vol/OI > 6",
        len(market)
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
        "BTC",
        BTC["regime"]
    )


# =========================================================
# MARKET DATA
# =========================================================

st.subheader(
    "📊 Volume / OI Filtered Market"
)

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

candidates = market.head(
    DEEP_SCAN_LIMIT
)


st.info(
    f"Vol/OI > {MIN_VOL_OI_RATIO} "
    f"wale {len(market)} coins me se "
    f"top {len(candidates)} active coins "
    f"deep scan ho rahe hain."
)


# =========================================================
# SCAN
# =========================================================

st.subheader(
    "🎯 Advanced Scanner"
)

results = []

progress = st.progress(0)

total = len(candidates)

if total == 0:

    st.warning(
        "Vol/OI > 6 wala coin abhi nahi mila."
    )

else:

    for i, (_, row) in enumerate(
        candidates.iterrows()
    ):

        try:

            result = deep_analysis(
                row["Coin"],
                row,
                BTC
            )

            if result:

                results.append(
                    result
                )

        except Exception as e:

            pass

        progress.progress(
            int(
                ((i + 1) / total) *
                100
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
        "❌ Analysis result available nahi hai."
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
# STRONG LONG
# =========================================================

st.subheader(
    f"🟢 STRONG LONG — {STRONG_SCORE}+"
)

if not signals.empty:

    strong_long = signals[
        signals["Long Score"] >=
        STRONG_SCORE
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
                    "Vol/OI",
                    "BTC Regime",
                    "1H Trend",
                    "MTF Regime",
                    "15m Liquidity",
                    "5m BOS",
                    "FVG",
                    "FVG Fresh",
                    "FVG Retest",
                    "Volume x",
                    "ATR %",
                    "ATR Change %",
                    "OI State",
                    "OI Change %",
                    "Funding %",
                    "Long Score",
                    "Entry",
                    "SL",
                    "TP",
                    "R:R",
                    "20x SL Impact %",
                    "50x SL Impact %",
                    "100x SL Impact %",
                    "200x SL Impact %",
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
    f"🔴 STRONG SHORT — {STRONG_SCORE}+"
)

if not signals.empty:

    strong_short = signals[
        signals["Short Score"] >=
        STRONG_SCORE
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
                    "Vol/OI",
                    "BTC Regime",
                    "1H Trend",
                    "MTF Regime",
                    "15m Liquidity",
                    "5m BOS",
                    "FVG",
                    "FVG Fresh",
                    "FVG Retest",
                    "Volume x",
                    "ATR %",
                    "ATR Change %",
                    "OI State",
                    "OI Change %",
                    "Funding %",
                    "Short Score",
                    "Entry",
                    "SL",
                    "TP",
                    "R:R",
                    "20x SL Impact %",
                    "50x SL Impact %",
                    "100x SL Impact %",
                    "200x SL Impact %",
                    "Short Reason"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )


# =========================================================
# WATCH LIST
# =========================================================

st.subheader(
    "👀 Watchlist"
)

if not signals.empty:

    watch = signals[
        (
            signals["Long Score"] >=
            WATCH_SCORE
        )
        |
        (
            signals["Short Score"] >=
            WATCH_SCORE
        )
    ]

    if not watch.empty:

        st.dataframe(
            watch[
                [
                    "Coin",
                    "Price",
                    "Vol/OI",
                    "1H Trend",
                    "MTF Regime",
                    "Liquidity",
                    "5m BOS",
                    "FVG",
                    "FVG Retest",
                    "OI State",
                    "Funding",
                    "Long Score",
                    "Short Score",
                    "Signal"
                ]
            ]
            if "Liquidity" in watch.columns
            else watch[
                [
                    "Coin",
                    "Price",
                    "Vol/OI",
                    "1H Trend",
                    "MTF Regime",
                    "15m Liquidity",
                    "5m BOS",
                    "FVG",
                    "FVG Retest",
                    "OI State",
                    "Funding",
                    "Long Score",
                    "Short Score",
                    "Signal"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )


# =========================================================
# LOGIC EXPLANATION
# =========================================================

st.divider()

st.subheader(
    "🧠 Scanner Logic"
)

st.write(
    """
Volume/OI > 6
        ↓
BTC Market Regime
        ↓
1H Trend
        ↓
15m Swing High / Swing Low
        ↓
Liquidity Sweep
        ↓
5m BOS
        ↓
Displacement
        ↓
Fresh FVG
        ↓
FVG Retest
        ↓
Price + OI relationship
        ↓
Funding crowding
        ↓
Volume expansion
        ↓
ATR expansion
        ↓
Multi-Timeframe Direction
        ↓
Entry / SL / TP
        ↓
R:R filter
        ↓
Long / Short Score
"""
)


# =========================================================
# OI EXPLANATION
# =========================================================

st.subheader(
    "📈 Price + OI Logic"
)

st.write(
    """
Price ↑ + OI ↑
→ LONG BUILDUP

Price ↓ + OI ↑
→ SHORT BUILDUP

Price ↑ + OI ↓
→ SHORT COVERING

Price ↓ + OI ↓
→ LONG LIQUIDATION
"""
)


# =========================================================
# FUNDING
# =========================================================

st.subheader(
    "💰 Funding Logic"
)

st.write(
    """
Funding ≥ +0.05%
→ Longs crowded
→ Short score बढ़ता है

Funding ≤ -0.05%
→ Shorts crowded
→ Long score बढ़ता है

Funding बीच में
→ Neutral
"""
)


# =========================================================
# LEVERAGE WARNING
# =========================================================

st.warning(
    """
⚠️ 20x/50x/100x/200x columns केवल यह दिखाते हैं कि
price movement का leveraged P&L impact कितना हो सकता है।

ये liquidation-price calculator नहीं हैं और leverage को
risk limit नहीं माना जाना चाहिए।
"""
)


# =========================================================
# REFRESH
# =========================================================

st.divider()

if st.button(
    "🔄 Refresh Advanced Scanner"
):

    st.cache_data.clear()

    st.rerun()
