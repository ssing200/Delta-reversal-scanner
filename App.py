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
    "User-Agent": "Delta-MTF-Scanner/7.0"
}

CACHE_SECONDS = 120

# Server/API load control
DEEP_SCAN_LIMIT = 30

# =========================================================
# PAGE
# =========================================================

st.set_page_config(
    page_title="Delta MTF Smart Scanner",
    layout="wide"
)

st.title("🔥 Delta MTF Smart Scanner")

st.caption(
    "4H Bias → 1H Trend → 15m Liquidity Sweep → "
    "5m BOS → OI → Volume → ATR → Funding → "
    "Long/Short Score"
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

        # =================================================
        # FUNDING
        # =================================================

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
# ATR
# =========================================================

def calculate_atr(df, period=14):

    if df.empty or len(df) < period + 2:

        return None

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

    if atr.dropna().empty:
        return None

    return atr


def analyze_atr(symbol):

    df = get_candles(
        symbol,
        "5m",
        24
    )

    if df.empty or len(df) < 20:

        return {
            "atr": None,
            "atr_previous": None,
            "atr_change": None,
            "atr_state": "⚪ Unknown",
            "atr_pct": None
        }

    atr = calculate_atr(
        df,
        14
    )

    if atr is None:
        return {
            "atr": None,
            "atr_previous": None,
            "atr_change": None,
            "atr_state": "⚪ Unknown",
            "atr_pct": None
        }

    current = float(
        atr.iloc[-1]
    )

    previous = float(
        atr.iloc[-2]
    )

    price = float(
        df["close"].iloc[-1]
    )

    if previous == 0:

        change = None

    else:

        change = (
            (current - previous)
            / previous
        ) * 100

    if change is None:

        state = "⚪ Unknown"

    elif change >= 10:

        state = "🔥 ATR EXPANDING"

    elif change >= 3:

        state = "🟢 ATR RISING"

    elif change <= -10:

        state = "🔵 ATR CONTRACTING"

    elif change <= -3:

        state = "🟡 ATR FALLING"

    else:

        state = "⚪ ATR FLAT"

    atr_pct = (
        current /
        price *
        100
    )

    return {

        "atr": current,

        "atr_previous": previous,

        "atr_change": change,

        "atr_state": state,

        "atr_pct": atr_pct
    }


# =========================================================
# 4H TREND
# =========================================================

def analyze_4h(symbol):

    df = get_candles(
        symbol,
        "4h",
        240
    )

    if df.empty or len(df) < 30:

        return {
            "trend": "⚪ UNKNOWN"
        }

    close = df["close"]

    ema20 = close.ewm(
        span=20,
        adjust=False
    ).mean()

    ema50 = close.ewm(
        span=50,
        adjust=False
    ).mean()

    price = float(
        close.iloc[-1]
    )

    fast = float(
        ema20.iloc[-1]
    )

    slow = float(
        ema50.iloc[-1]
    )

    if fast > slow and price > fast:

        trend = "🟢 BULLISH"

    elif fast < slow and price < fast:

        trend = "🔴 BEARISH"

    else:

        trend = "⚪ NEUTRAL"

    return {
        "trend": trend
    }


# =========================================================
# 1H TREND
# =========================================================

def analyze_1h(symbol):

    df = get_candles(
        symbol,
        "1h",
        120
    )

    if df.empty or len(df) < 30:

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

    if (
        e9 > e21
        and
        e21 > e50
        and
        price > e9
    ):

        trend = "🟢 BULLISH"

    elif (
        e9 < e21
        and
        e21 < e50
        and
        price < e9
    ):

        trend = "🔴 BEARISH"

    else:

        trend = "⚪ NEUTRAL"

    return {
        "trend": trend
    }


# =========================================================
# MARKET REGIME
# =========================================================

def analyze_regime(symbol):

    df = get_candles(
        symbol,
        "1h",
        120
    )

    if df.empty or len(df) < 30:

        return {
            "regime": "⚪ UNCERTAINTY"
        }

    close = df["close"]

    ema20 = close.ewm(
        span=20,
        adjust=False
    ).mean()

    ema50 = close.ewm(
        span=50,
        adjust=False
    ).mean()

    current_price = float(
        close.iloc[-1]
    )

    e20 = float(
        ema20.iloc[-1]
    )

    e50 = float(
        ema50.iloc[-1]
    )

    separation = (
        abs(e20 - e50)
        / current_price
    ) * 100

    recent = close.iloc[-20:]

    recent_high = float(
        recent.max()
    )

    recent_low = float(
        recent.min()
    )

    range_pct = (
        (recent_high - recent_low)
        / current_price
    ) * 100

    # Strong directional separation
    if separation >= 0.35:

        if e20 > e50:

            regime = "🟢 DIRECTIONAL BULLISH"

        else:

            regime = "🔴 DIRECTIONAL BEARISH"

    # Tight range / compression
    elif range_pct <= 2.5:

        regime = "🟡 RANGE BOUND"

    else:

        regime = "⚪ UNCERTAINTY"

    return {
        "regime": regime
    }


# =========================================================
# 15M LIQUIDITY SWEEP
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

    last_high = float(
        last["high"]
    )

    last_low = float(
        last["low"]
    )

    last_close = float(
        last["close"]
    )

    # Bullish liquidity sweep:
    # low breaks previous liquidity
    # but candle closes back above it

    bull_sweep = (
        last_low < previous_low
        and
        last_close > previous_low
    )

    # Bearish liquidity sweep:
    # high breaks previous liquidity
    # but candle closes back below it

    bear_sweep = (
        last_high > previous_high
        and
        last_close < previous_high
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
# 5M BOS
# =========================================================

def analyze_5m(symbol):

    df = get_candles(
        symbol,
        "5m",
        18
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

        "bull_bos": bull_bos,

        "bear_bos": bear_bos,

        "structure": structure
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

            "volume_ratio": 0,

            "volume_state": "⚪ Unknown",

            "avg_volume": None
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

            "volume_state": "⚪ Unknown",

            "avg_volume": average_volume
        }

    ratio = (
        current_volume /
        average_volume
    )

    if ratio >= 2:

        state = "🔥 VOLUME SPIKE"

    elif ratio >= 1.3:

        state = "🟢 VOLUME HIGH"

    elif ratio <= 0.7:

        state = "🔵 VOLUME LOW"

    else:

        state = "⚪ VOLUME NORMAL"

    return {

        "volume_ratio": ratio,

        "volume_state": state,

        "avg_volume": average_volume
    }


# =========================================================
# OI
# =========================================================

def analyze_oi(symbol):

    df = get_oi_history(symbol)

    if df.empty or len(df) < 6:

        return {

            "oi_change": None,

            "oi_current": None,

            "oi_average": None,

            "oi_signal": "⚪ Unknown"
        }

    current = float(
        df["close"].iloc[-1]
    )

    previous = float(
        df["close"].iloc[-5]
    )

    average_oi = float(
        df["close"].iloc[:-1].mean()
    )

    if previous == 0:

        change = None

    else:

        change = (
            (current - previous)
            / abs(previous)
        ) * 100

    if change is None:

        signal = "⚪ Unknown"

    elif change >= 2:

        signal = "🔥 OI STRONG UP"

    elif change >= 1:

        signal = "🟢 OI UP"

    elif change <= -2:

        signal = "🔻 OI STRONG DOWN"

    elif change <= -1:

        signal = "🔵 OI DOWN"

    else:

        signal = "⚪ OI NEUTRAL"

    return {

        "oi_change": change,

        "oi_current": current,

        "oi_average": average_oi,

        "oi_signal": signal
    }


# =========================================================
# PRICE + OI INTERPRETATION
# =========================================================

def price_oi_logic(symbol):

    candles = get_candles(
        symbol,
        "15m",
        6
    )

    oi = analyze_oi(symbol)

    if candles.empty:

        return "⚪ Unknown"

    if oi["oi_change"] is None:

        return "⚪ Unknown"

    current_price = float(
        candles["close"].iloc[-1]
    )

    previous_price = float(
        candles["close"].iloc[-5]
    )

    price_change = (
        (
            current_price -
            previous_price
        )
        /
        abs(previous_price)
    ) * 100

    oi_change = oi["oi_change"]

    # Price UP + OI UP
    if (
        price_change > 0.3
        and
        oi_change > 1
    ):

        return "🟢 LONG BUILDUP"

    # Price DOWN + OI UP
    if (
        price_change < -0.3
        and
        oi_change > 1
    ):

        return "🔴 SHORT BUILDUP"

    # Price UP + OI DOWN
    if (
        price_change > 0.3
        and
        oi_change < -1
    ):

        return "🟡 SHORT COVERING"

    # Price DOWN + OI DOWN
    if (
        price_change < -0.3
        and
        oi_change < -1
    ):

        return "🔵 LONG LIQUIDATION"

    return "⚪ MIXED"


# =========================================================
# FUNDING
# =========================================================

def analyze_funding(ticker):

    funding = ticker.get(
        "Funding",
        None
    )

    if funding is None:

        return {

            "funding_pct": None,

            "funding_signal":
                "⚪ Funding unavailable",

            "funding_score_long": 0,

            "funding_score_short": 0
        }

    try:

        funding_pct = (
            float(funding)
            * 100
        )

    except Exception:

        return {

            "funding_pct": None,

            "funding_signal":
                "⚪ Funding unavailable",

            "funding_score_long": 0,

            "funding_score_short": 0
        }

    # Extreme positive funding
    if funding_pct >= 0.10:

        return {

            "funding_pct": funding_pct,

            "funding_signal":
                "🔴 EXTREME LONG CROWDING",

            "funding_score_long": 0,

            "funding_score_short": 3
        }

    if funding_pct >= 0.05:

        return {

            "funding_pct": funding_pct,

            "funding_signal":
                "🔴 LONG CROWDING",

            "funding_score_long": 0,

            "funding_score_short": 2
        }

    # Extreme negative funding
    if funding_pct <= -0.10:

        return {

            "funding_pct": funding_pct,

            "funding_signal":
                "🟢 EXTREME SHORT CROWDING",

            "funding_score_long": 3,

            "funding_score_short": 0
        }

    if funding_pct <= -0.05:

        return {

            "funding_pct": funding_pct,

            "funding_signal":
                "🟢 SHORT CROWDING",

            "funding_score_long": 2,

            "funding_score_short": 0
        }

    return {

        "funding_pct": funding_pct,

        "funding_signal":
            "⚪ FUNDING NEUTRAL",

        "funding_score_long": 0,

        "funding_score_short": 0
    }


# =========================================================
# MULTI-TIMEFRAME CONFIRMATION
# =========================================================

def mtf_confirmation(
    trend_4h,
    trend_1h,
    sweep,
    bos
):

    long_confirmations = 0
    short_confirmations = 0

    # 4H
    if trend_4h == "🟢 BULLISH":
        long_confirmations += 1

    if trend_4h == "🔴 BEARISH":
        short_confirmations += 1

    # 1H
    if trend_1h == "🟢 BULLISH":
        long_confirmations += 1

    if trend_1h == "🔴 BEARISH":
        short_confirmations += 1

    # 15m
    if sweep["bull_sweep"]:
        long_confirmations += 1

    if sweep["bear_sweep"]:
        short_confirmations += 1

    # 5m
    if bos["bull_bos"]:
        long_confirmations += 1

    if bos["bear_bos"]:
        short_confirmations += 1

    if (
        long_confirmations >= 3
        and
        long_confirmations > short_confirmations
    ):

        direction = "🟢 LONG CONFIRMED"

    elif (
        short_confirmations >= 3
        and
        short_confirmations > long_confirmations
    ):

        direction = "🔴 SHORT CONFIRMED"

    elif (
        long_confirmations ==
        short_confirmations
    ):

        direction = "⚪ UNCERTAINTY"

    else:

        direction = "🟡 WEAK/MIXED"

    return {

        "long_confirmations":
            long_confirmations,

        "short_confirmations":
            short_confirmations,

        "mtf": direction
    }


# =========================================================
# LEVERAGE IMPACT
# =========================================================

def leverage_impact():

    # Approximate gross exposure effect
    # for a 1% underlying price move

    return {

        "20x": "±20%",

        "50x": "±50%",

        "100x": "±100%",

        "200x": "±200%"
    }


# =========================================================
# ENTRY / SL / TP
# =========================================================

def calculate_trade_levels(
    symbol,
    signal
):

    df = get_candles(
        symbol,
        "5m",
        12
    )

    if df.empty:

        return {

            "entry": "Wait",

            "sl": "N/A",

            "tp1": "N/A",

            "tp2": "N/A",

            "rr": None
        }

    last = df.iloc[-1]

    high = float(
        last["high"]
    )

    low = float(
        last["low"]
    )

    close = float(
        last["close"]
    )

    candle_range = (
        high - low
    )

    if candle_range <= 0:

        return {

            "entry": "Wait",

            "sl": "N/A",

            "tp1": "N/A",

            "tp2": "N/A",

            "rr": None
        }

    # LONG
    if "LONG" in signal:

        entry_low = low

        entry_high = (
            low +
            candle_range * 0.50
        )

        entry = (
            entry_low +
            entry_high
        ) / 2

        sl = (
            low -
            candle_range * 0.50
        )

        risk = (
            entry - sl
        )

        tp1 = (
            entry +
            risk * 1.5
        )

        tp2 = (
            entry +
            risk * 2.5
        )

    # SHORT
    elif "SHORT" in signal:

        entry_low = (
            high -
            candle_range * 0.50
        )

        entry_high = high

        entry = (
            entry_low +
            entry_high
        ) / 2

        sl = (
            high +
            candle_range * 0.50
        )

        risk = (
            sl - entry
        )

        tp1 = (
            entry -
            risk * 1.5
        )

        tp2 = (
            entry -
            risk * 2.5
        )

    else:

        return {

            "entry": "Wait",

            "sl": "N/A",

            "tp1": "N/A",

            "tp2": "N/A",

            "rr": None
        }

    if risk <= 0:

        rr = None

    else:

        rr = 2.5

    return {

        "entry":
            f"{entry_low:.6g} - "
            f"{entry_high:.6g}",

        "sl":
            f"{sl:.6g}",

        "tp1":
            f"{tp1:.6g}",

        "tp2":
            f"{tp2:.6g}",

        "rr":
            rr
    }


# =========================================================
# DEEP ANALYSIS
# =========================================================

def deep_analysis(symbol, ticker):

    trend4 = analyze_4h(symbol)

    trend1 = analyze_1h(symbol)

    regime = analyze_regime(symbol)

    sweep = analyze_15m(symbol)

    bos = analyze_5m(symbol)

    volume = analyze_volume(symbol)

    oi = analyze_oi(symbol)

    price_oi = price_oi_logic(symbol)

    atr = analyze_atr(symbol)

    funding = analyze_funding(ticker)

    mtf = mtf_confirmation(
        trend4["trend"],
        trend1["trend"],
        sweep,
        bos
    )

    long_score = 0

    short_score = 0

    long_reason = []

    short_reason = []

    # =====================================================
    # 4H
    # =====================================================

    if trend4["trend"] == "🟢 BULLISH":

        long_score += 2

        long_reason.append(
            "4H bullish"
        )

    elif trend4["trend"] == "🔴 BEARISH":

        short_score += 2

        short_reason.append(
            "4H bearish"
        )

    # =====================================================
    # 1H
    # =====================================================

    if trend1["trend"] == "🟢 BULLISH":

        long_score += 2

        long_reason.append(
            "1H bullish"
        )

    elif trend1["trend"] == "🔴 BEARISH":

        short_score += 2

        short_reason.append(
            "1H bearish"
        )

    # =====================================================
    # REGIME
    # =====================================================

    if regime["regime"] == "🟢 DIRECTIONAL BULLISH":

        long_score += 2

        long_reason.append(
            "Directional bullish"
        )

    elif regime["regime"] == "🔴 DIRECTIONAL BEARISH":

        short_score += 2

        short_reason.append(
            "Directional bearish"
        )

    elif regime["regime"] == "🟡 RANGE BOUND":

        # Do not add score
        # Instead mark as caution

        long_reason.append(
            "Range bound caution"
        )

        short_reason.append(
            "Range bound caution"
        )

    else:

        long_reason.append(
            "Market uncertainty"
        )

        short_reason.append(
            "Market uncertainty"
        )

    # =====================================================
    # LIQUIDITY SWEEP
    # =====================================================

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

    # =====================================================
    # BOS
    # =====================================================

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

    # =====================================================
    # MTF CONFIRMATION
    # =====================================================

    if (
        mtf["long_confirmations"] >= 3
        and
        mtf["long_confirmations"] >
        mtf["short_confirmations"]
    ):

        long_score += 2

        long_reason.append(
            "MTF confirmed"
        )

    if (
        mtf["short_confirmations"] >= 3
        and
        mtf["short_confirmations"] >
        mtf["long_confirmations"]
    ):

        short_score += 2

        short_reason.append(
            "MTF confirmed"
        )

    # =====================================================
    # VOLUME
    # =====================================================

    vr = volume["volume_ratio"]

    if vr >= 2:

        long_score += 2

        short_score += 2

        long_reason.append(
            "Volume spike"
        )

        short_reason.append(
            "Volume spike"
        )

    elif vr >= 1.3:

        long_score += 1

        short_score += 1

    # =====================================================
    # OI
    # =====================================================

    oi_change = oi["oi_change"]

    if price_oi == "🟢 LONG BUILDUP":

        long_score += 2

        long_reason.append(
            "Price + OI long buildup"
        )

    elif price_oi == "🔴 SHORT BUILDUP":

        short_score += 2

        short_reason.append(
            "Price + OI short buildup"
        )

    elif price_oi == "🟡 SHORT COVERING":

        long_score += 1

        long_reason.append(
            "Short covering"
        )

    elif price_oi == "🔵 LONG LIQUIDATION":

        short_score += 1

        short_reason.append(
            "Long liquidation"
        )

    # =====================================================
    # ATR
    # =====================================================

    if atr["atr_change"] is not None:

        if atr["atr_change"] >= 5:

            # Volatility expansion
            # confirms an already existing direction

            if (
                long_score >
                short_score
            ):

                long_score += 2

                long_reason.append(
                    "ATR expanding"
                )

            elif (
                short_score >
                long_score
            ):

                short_score += 2

                short_reason.append(
                    "ATR expanding"
                )

        elif atr["atr_change"] >= 2:

            if (
                long_score >
                short_score
            ):

                long_score += 1

            elif (
                short_score >
                long_score
            ):

                short_score += 1

    # =====================================================
    # FUNDING
    # =====================================================

    long_score += (
        funding["funding_score_long"]
    )

    short_score += (
        funding["funding_score_short"]
    )

    if funding["funding_score_long"] > 0:

        long_reason.append(
            "Negative funding"
        )

    if funding["funding_score_short"] > 0:

        short_reason.append(
            "Positive funding"
        )

    # =====================================================
    # RANGE PENALTY
    # =====================================================

    if regime["regime"] == "🟡 RANGE BOUND":

        if long_score > 0:

            long_score = max(
                0,
                long_score - 2
            )

        if short_score > 0:

            short_score = max(
                0,
                short_score - 2
            )

    # =====================================================
    # SIGNALS
    # =====================================================

    if long_score >= 10:

        long_signal = "🟢 STRONG LONG"

    elif long_score >= 7:

        long_signal = "🟡 LONG WATCH"

    else:

        long_signal = "⚪ NO LONG"

    if short_score >= 10:

        short_signal = "🔴 STRONG SHORT"

    elif short_score >= 7:

        short_signal = "🟠 SHORT WATCH"

    else:

        short_signal = "⚪ NO SHORT"

    # =====================================================
    # DOMINANT
    # =====================================================

    if (
        long_score >= 10
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
        short_score >= 10
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
        long_score >= 7
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
        short_score >= 7
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
    # SETUP QUALITY
    # =====================================================

    if (
        dominant_score >= 12
        and
        regime["regime"]
        not in [
            "🟡 RANGE BOUND",
            "⚪ UNCERTAINTY"
        ]
    ):

        quality = "🔥 A+ SETUP"

    elif dominant_score >= 10:

        quality = "🟢 A SETUP"

    elif dominant_score >= 7:

        quality = "🟡 B SETUP"

    elif dominant_score >= 5:

        quality = "🟠 C SETUP"

    else:

        quality = "⚪ AVOID"

    # =====================================================
    # ENTRY
    # =====================================================

    levels = calculate_trade_levels(
        symbol,
        signal
    )

    lev = leverage_impact()

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

        "4H":
            trend4["trend"],

        "1H":
            trend1["trend"],

        "Market Regime":
            regime["regime"],

        "MTF":
            mtf["mtf"],

        "15m Liquidity":
            sweep["liquidity"],

        "5m BOS":
            bos["structure"],

        "Volume x":
            round(
                vr,
                2
            ),

        "Avg 5m Volume":
            (
                round(
                    volume["avg_volume"],
                    2
                )
                if volume["avg_volume"]
                is not None
                else None
            ),

        "OI":
            oi["oi_signal"],

        "OI Current":
            (
                round(
                    oi["oi_current"],
                    2
                )
                if oi["oi_current"]
                is not None
                else None
            ),

        "Avg OI":
            (
                round(
                    oi["oi_average"],
                    2
                )
                if oi["oi_average"]
                is not None
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

        "Price + OI":
            price_oi,

        "ATR":
            (
                round(
                    atr["atr"],
                    6
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

        "ATR State":
            atr["atr_state"],

        "ATR %":
            (
                round(
                    atr["atr_pct"],
                    3
                )
                if atr["atr_pct"]
                is not None
                else None
            ),

        "Funding %":
            (
                round(
                    funding["funding_pct"],
                    4
                )
                if funding["funding_pct"]
                is not None
                else None
            ),

        "Funding":
            funding["funding_signal"],

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

        "Quality":
            quality,

        "Entry Zone":
            levels["entry"],

        "SL":
            levels["sl"],

        "TP1":
            levels["tp1"],

        "TP2":
            levels["tp2"],

        "R:R":
            levels["rr"],

        "20x 1% Move":
            lev["20x"],

        "50x 1% Move":
            lev["50x"],

        "100x 1% Move":
            lev["100x"],

        "200x 1% Move":
            lev["200x"],

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

market = market.sort_values(
    "24H Volume",
    ascending=False
)


# =========================================================
# MARKET METRICS
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
        "Deep Scan",
        DEEP_SCAN_LIMIT
    )

with c4:

    st.metric(
        "Funding Data",
        market["Funding"].notna().sum()
    )


# =========================================================
# ALL MARKET DATA
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
            "Funding",
            "Vol/OI"
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
    f"market activity ke liye considered hain. "
    f"Deep analysis top {len(candidates)} "
    f"active coins par ho raha hai."
)


# =========================================================
# SCAN
# =========================================================

st.subheader(
    "🎯 MTF Scanner Results"
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
# LONG
# =========================================================

st.subheader(
    "🟢 LONG — Smart Score"
)

if not signals.empty:

    long_table = signals[
        [
            "Coin",
            "Price",
            "4H",
            "1H",
            "Market Regime",
            "MTF",
            "15m Liquidity",
            "5m BOS",
            "Volume x",
            "Price + OI",
            "OI Change %",
            "ATR Change %",
            "ATR State",
            "Funding %",
            "Funding",
            "Long Score",
            "Long Signal",
            "Quality",
            "Entry Zone",
            "SL",
            "TP1",
            "TP2",
            "R:R"
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
    "🔴 SHORT — Smart Score"
)

if not signals.empty:

    short_table = signals[
        [
            "Coin",
            "Price",
            "4H",
            "1H",
            "Market Regime",
            "MTF",
            "15m Liquidity",
            "5m BOS",
            "Volume x",
            "Price + OI",
            "OI Change %",
            "ATR Change %",
            "ATR State",
            "Funding %",
            "Funding",
            "Short Score",
            "Short Signal",
            "Quality",
            "Entry Zone",
            "SL",
            "TP1",
            "TP2",
            "R:R"
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
    "🔥 STRONG LONG — 10+"
)

if not signals.empty:

    strong_long = signals[
        signals["Long Score"] >= 10
    ].sort_values(
        "Long Score",
        ascending=False
    )

    if strong_long.empty:

        st.info(
            "Abhi 10+ Strong Long nahi mila."
        )

    else:

        st.dataframe(

            strong_long[
                [
                    "Coin",
                    "Price",
                    "Long Score",
                    "Long Signal",
                    "Quality",
                    "4H",
                    "1H",
                    "Market Regime",
                    "MTF",
                    "15m Liquidity",
                    "5m BOS",
                    "Price + OI",
                    "OI Change %",
                    "Volume x",
                    "ATR State",
                    "Funding %",
                    "Entry Zone",
                    "SL",
                    "TP1",
                    "TP2",
                    "R:R",
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
    "🔥 STRONG SHORT — 10+"
)

if not signals.empty:

    strong_short = signals[
        signals["Short Score"] >= 10
    ].sort_values(
        "Short Score",
        ascending=False
    )

    if strong_short.empty:

        st.info(
            "Abhi 10+ Strong Short nahi mila."
        )

    else:

        st.dataframe(

            strong_short[
                [
                    "Coin",
                    "Price",
                    "Short Score",
                    "Short Signal",
                    "Quality",
                    "4H",
                    "1H",
                    "Market Regime",
                    "MTF",
                    "15m Liquidity",
                    "5m BOS",
                    "Price + OI",
                    "OI Change %",
                    "Volume x",
                    "ATR State",
                    "Funding %",
                    "Entry Zone",
                    "SL",
                    "TP1",
                    "TP2",
                    "R:R",
                    "Short Reason"
                ]
            ],

            use_container_width=True,

            hide_index=True
        )


# =========================================================
# TOP QUALITY SETUPS
# =========================================================

st.subheader(
    "🏆 Best Quality Setups"
)

if not signals.empty:

    best = signals[
        signals["Quality"].isin(
            [
                "🔥 A+ SETUP",
                "🟢 A SETUP"
            ]
        )
    ].sort_values(
        "Score",
        ascending=False
    )

    if best.empty:

        st.info(
            "Abhi A/A+ setup available nahi hai."
        )

    else:

        st.dataframe(

            best[
                [
                    "Coin",
                    "Price",
                    "Score",
                    "Signal",
                    "Quality",
                    "4H",
                    "1H",
                    "Market Regime",
                    "MTF",
                    "15m Liquidity",
                    "5m BOS",
                    "Price + OI",
                    "OI Change %",
                    "Volume x",
                    "ATR State",
                    "Funding %",
                    "Entry Zone",
                    "SL",
                    "TP1",
                    "TP2",
                    "R:R"
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
Funding ko standalone signal nahi maana gaya hai.

Positive Funding:
→ Long crowding
→ Short side ko contrarian confirmation

Negative Funding:
→ Short crowding
→ Long side ko contrarian confirmation

≥ +0.05% → Short score +2
≥ +0.10% → Short score +3

≤ -0.05% → Long score +2
≤ -0.10% → Long score +3

Beech mein → Neutral
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

Isko BOS, liquidity aur trend ke saath
confirmation ke roop mein use kiya gaya hai.
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
ATR rising
→ volatility expansion
→ existing directional setup ko confirmation

ATR falling
→ volatility compression

ATR contracting market mein
late breakout ko blindly chase nahi karna chahiye.
"""
)


# =========================================================
# LEVERAGE
# =========================================================

st.subheader(
    "⚠️ Leverage Reference"
)

st.write(
    """
Approximate gross exposure effect for a 1% underlying move:

20x  → ±20%
50x  → ±50%
100x → ±100%
200x → ±200%

Ye liquidation calculation nahi hai.
Fees, funding, maintenance margin aur
actual exchange liquidation mechanics alag hote hain.
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
4H Bias
↓
1H Trend
↓
Market Regime
↓
15m Liquidity Sweep
↓
5m BOS
↓
Price + OI
↓
Volume
↓
ATR Expansion
↓
Funding
↓
MTF Confirmation
↓
LONG / SHORT Score
↓
A+ / A / B / C Quality
↓
Entry / SL / TP / R:R
"""
)


# =========================================================
# IMPORTANT LIMITATION
# =========================================================

st.info(
    """
एक जरूरी बात:

इस version में भी scanner top active coins
को deep scan करता है। सभी 220+ coins पर
एक साथ 4H + 1H + 15m + 5m candles और OI
history लेना API requests बहुत बढ़ा सकता है।

इसलिए पहले सभी live perpetuals को market activity
से rank किया जाता है और फिर top coins का deep scan
होता है।

इससे server/API overload की संभावना कम रहती है।
"""
)


# =========================================================
# CURRENT VS HISTORICAL
# =========================================================

st.subheader(
    "🕒 Data Interpretation"
)

st.write(
    """
CURRENT DATA:
• Current price
• Current 24H volume
• Current OI
• Current funding

HISTORICAL DATA:
• 4H / 1H trend
• 15m liquidity sweep
• 5m BOS
• ATR change
• Average volume
• Average OI
• Price + OI relationship

Isliye scanner historical data ko sirf context,
trend aur confirmation ke liye use karta hai;
final signal latest available market condition
par based hota hai.
"""
)


# =========================================================
# WARNING
# =========================================================

st.warning(
    "⚠️ Ye scanner probability/confirmation tool hai, "
    "guaranteed trade signal nahi. "
    "High leverage par risk bahut rapidly badhta hai. "
    "Real position se pehle entry, invalidation aur "
    "risk management independently check karein."
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
