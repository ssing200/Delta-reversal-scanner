import streamlit as st
import requests
import pandas as pd
import numpy as np
import time

BASE_URL = "https://api.india.delta.exchange"

HEADERS = {
    "Accept": "application/json",
    "User-Agent": "Delta-Reversal-Scanner/7.0"
}

CACHE_SECONDS = 120
DEEP_SCAN_LIMIT = 30

# HARD FILTER
MIN_VOL_OI_RATIO = 6.0

st.set_page_config(
    page_title="Delta Reversal Scanner V7",
    layout="wide"
)

st.title("🔥 Delta Reversal Scanner V7")

st.caption(
    "1H Trend → 15m Swing/Liquidity → 5m BOS → FVG → "
    "OI → Funding → Volume → ATR → Score"
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
    ).sort_values("time").reset_index(drop=True)


# =========================================================
# ATR
# =========================================================

def calculate_atr(df, period=14):

    if df.empty or len(df) < period + 1:
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

    true_range = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    atr = (
        true_range
        .rolling(period)
        .mean()
    )

    current_atr = atr.iloc[-1]

    if pd.isna(current_atr):
        return None, None

    # Compare current ATR with previous ATR
    previous_atr = atr.iloc[-2]

    if pd.isna(previous_atr):
        direction = "⚪ UNKNOWN"

    elif current_atr > previous_atr * 1.05:
        direction = "🔺 ATR RISING"

    elif current_atr < previous_atr * 0.95:
        direction = "🔻 ATR FALLING"

    else:
        direction = "⚪ ATR FLAT"

    return float(current_atr), direction


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
            "market": "⚪ UNKNOWN"
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

    # Slope
    ema21_previous = float(
        ema21.iloc[-4]
    )

    if fast > slow and price > fast:

        trend = "🟢 BULLISH"

    elif fast < slow and price < fast:

        trend = "🔴 BEARISH"

    else:

        trend = "⚪ NEUTRAL"

    # =====================================================
    # RANGE / DIRECTIONAL
    # =====================================================

    recent = df.tail(20)

    highest = float(
        recent["high"].max()
    )

    lowest = float(
        recent["low"].min()
    )

    range_pct = (
        (highest - lowest)
        / price
    ) * 100

    ema_slope_pct = (
        (ema21.iloc[-1] -
         ema21.iloc[-4])
        /
        abs(ema21.iloc[-4])
    ) * 100

    if (
        abs(ema_slope_pct) < 0.15
        and range_pct < 4
    ):

        market = "🟡 RANGE BOUND"

    elif abs(ema_slope_pct) >= 0.15:

        market = "🔵 DIRECTIONAL"

    else:

        market = "⚪ UNCERTAINTY"

    return {

        "trend": trend,

        "market": market

    }


# =========================================================
# SWING HIGH / LOW
# =========================================================

def analyze_swings(df, left=2, right=2):

    if df.empty or len(df) < 10:

        return {
            "swing_high": None,
            "swing_low": None
        }

    highs = []

    lows = []

    for i in range(
        left,
        len(df) - right
    ):

        current_high = df["high"].iloc[i]

        left_high = df["high"].iloc[
            i-left:i
        ].max()

        right_high = df["high"].iloc[
            i+1:i+1+right
        ].max()

        current_low = df["low"].iloc[i]

        left_low = df["low"].iloc[
            i-left:i
        ].min()

        right_low = df["low"].iloc[
            i+1:i+1+right
        ].min()

        if (
            current_high > left_high
            and
            current_high > right_high
        ):

            highs.append(
                float(current_high)
            )

        if (
            current_low < left_low
            and
            current_low < right_low
        ):

            lows.append(
                float(current_low)
            )

    return {

        "swing_high":
            highs[-1]
            if highs
            else None,

        "swing_low":
            lows[-1]
            if lows
            else None

    }


# =========================================================
# 15M LIQUIDITY + SWING
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

            "liquidity":
                "⚪ None",

            "swing_high": None,

            "swing_low": None

        }

    swings = analyze_swings(
        df
    )

    swing_high = swings[
        "swing_high"
    ]

    swing_low = swings[
        "swing_low"
    ]

    last = df.iloc[-1]

    # Use previous candles for confirmation
    previous = df.iloc[-7:-1]

    previous_high = float(
        previous["high"].max()
    )

    previous_low = float(
        previous["low"].min()
    )

    reference_high = (
        swing_high
        if swing_high is not None
        else previous_high
    )

    reference_low = (
        swing_low
        if swing_low is not None
        else previous_low
    )

    bull_sweep = (
        float(last["low"]) < reference_low
        and
        float(last["close"]) > reference_low
    )

    bear_sweep = (
        float(last["high"]) > reference_high
        and
        float(last["close"]) < reference_high
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

        "swing_high":
            swing_high,

        "swing_low":
            swing_low

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

            "structure":
                "⚪ None"

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

        "bull_bos":
            bull_bos,

        "bear_bos":
            bear_bos,

        "structure":
            structure

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

            "fvg_zone": "None"

        }

    # Last 3 completed candles
    c1 = df.iloc[-3]
    c2 = df.iloc[-2]
    c3 = df.iloc[-1]

    # -----------------------------------------------------
    # BULLISH FVG
    #
    # Candle 3 low > Candle 1 high
    # -----------------------------------------------------

    bull_fvg = (
        float(c3["low"])
        >
        float(c1["high"])
    )

    # -----------------------------------------------------
    # BEARISH FVG
    #
    # Candle 3 high < Candle 1 low
    # -----------------------------------------------------

    bear_fvg = (
        float(c3["high"])
        <
        float(c1["low"])
    )

    if bull_fvg:

        fvg_low = float(
            c1["high"]
        )

        fvg_high = float(
            c3["low"]
        )

        fvg_name = "🟢 BULL FVG"

        zone = (
            f"{fvg_low:.6g} - "
            f"{fvg_high:.6g}"
        )

    elif bear_fvg:

        fvg_low = float(
            c3["high"]
        )

        fvg_high = float(
            c1["low"]
        )

        fvg_name = "🔴 BEAR FVG"

        zone = (
            f"{fvg_low:.6g} - "
            f"{fvg_high:.6g}"
        )

    else:

        fvg_name = "⚪ None"

        zone = "None"

    return {

        "bull_fvg":
            bull_fvg,

        "bear_fvg":
            bear_fvg,

        "fvg":
            fvg_name,

        "fvg_zone":
            zone

    }


# =========================================================
# VOLUME
# =========================================================

def analyze_volume(symbol):

    df = get_candles(
        symbol,
        "5m",
        10
    )

    if df.empty or len(df) < 7:

        return {

            "volume_ratio": 0,

            "volume_signal":
                "⚪ Unknown"

        }

    current_volume = float(
        df["volume"].iloc[-1]
    )

    average_volume = float(
        df["volume"].iloc[-7:-1].mean()
    )

    if average_volume <= 0:

        return {

            "volume_ratio": 0,

            "volume_signal":
                "⚪ Unknown"

        }

    ratio = (
        current_volume /
        average_volume
    )

    if ratio >= 2:

        signal = "🔥 VOLUME SPIKE"

    elif ratio >= 1.3:

        signal = "🟢 VOLUME HIGH"

    else:

        signal = "⚪ VOLUME NORMAL"

    return {

        "volume_ratio":
            ratio,

        "volume_signal":
            signal

    }


# =========================================================
# OI
# =========================================================

def analyze_oi(symbol):

    df = get_oi_history(
        symbol
    )

    if df.empty or len(df) < 6:

        return {

            "oi_change": None,

            "oi_signal":
                "⚪ Unknown"

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

            "oi_signal":
                "⚪ Unknown"

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

        signal = "⚪ OI NEUTRAL"

    return {

        "oi_change":
            change,

        "oi_signal":
            signal

    }


# =========================================================
# ATR ANALYSIS
# =========================================================

def analyze_atr(symbol):

    df = get_candles(
        symbol,
        "15m",
        30
    )

    atr, direction = calculate_atr(
        df,
        14
    )

    return {

        "ATR":
            atr,

        "ATR Direction":
            direction

    }


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

    bos = analyze_5m(
        symbol
    )

    fvg = analyze_fvg(
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
                float(funding) * 100
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
    # MARKET CONDITION
    # =====================================================

    if trend["market"] == "🔵 DIRECTIONAL":

        long_score += 1
        short_score += 1

    elif trend["market"] == "🟡 RANGE BOUND":

        # Do not reward directional setup
        pass

    else:

        # uncertainty
        pass


    # =====================================================
    # LIQUIDITY SWEEP
    # =====================================================

    if sweep["bull_sweep"]:

        long_score += 2

        long_reason.append(
            "15m swing-low liquidity sweep"
        )

    if sweep["bear_sweep"]:

        short_score += 2

        short_reason.append(
            "15m swing-high liquidity sweep"
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
    # FVG
    # =====================================================

    # Bull FVG supports LONG
    if fvg["bull_fvg"]:

        long_score += 2

        long_reason.append(
            "Bullish FVG"
        )

    # Bear FVG supports SHORT
    if fvg["bear_fvg"]:

        short_score += 2

        short_reason.append(
            "Bearish FVG"
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

            if (
                trend["trend"]
                ==
                "🟢 BULLISH"
            ):

                long_score += 1

                long_reason.append(
                    "OI increasing with bullish trend"
                )

            if (
                trend["trend"]
                ==
                "🔴 BEARISH"
            ):

                short_score += 1

                short_reason.append(
                    "OI increasing with bearish trend"
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
    # FUNDING
    # =====================================================

    if funding_pct is None:

        funding_signal = (
            "⚪ Funding unavailable"
        )

    elif funding_pct >= 0.05:

        short_score += 2

        short_reason.append(
            "Positive funding / longs crowded"
        )

        funding_signal = (
            "🔴 Longs crowded"
        )

    elif funding_pct <= -0.05:

        long_score += 2

        long_reason.append(
            "Negative funding / shorts crowded"
        )

        funding_signal = (
            "🟢 Shorts crowded"
        )

    else:

        funding_signal = (
            "⚪ Funding neutral"
        )


    # =====================================================
    # ATR
    # =====================================================

    atr_direction = atr[
        "ATR Direction"
    ]

    if atr_direction == "🔺 ATR RISING":

        # Rising ATR means expansion.
        # Reward only the direction
        # already supported by structure.

        if long_score > short_score:

            long_score += 1

            long_reason.append(
                "ATR expanding"
            )

        elif short_score > long_score:

            short_score += 1

            short_reason.append(
                "ATR expanding"
            )


    # =====================================================
    # SIGNAL
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

    if fvg["fvg"] != "⚪ None":

        # Prefer FVG as entry zone
        entry_zone = fvg[
            "fvg_zone"
        ]

    else:

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

                entry_zone = (
                    f"{low:.6g} - "
                    f"{(low + high) / 2:.6g}"
                )

            elif "SHORT" in signal:

                entry_zone = (
                    f"{(low + high) / 2:.6g} - "
                    f"{high:.6g}"
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

        "Vol/OI":
            round(
                float(
                    ticker["Vol/OI"]
                ),
                2
            ),

        "1H Trend":
            trend["trend"],

        "Market":
            trend["market"],

        "Swing High":
            (
                round(
                    sweep["swing_high"],
                    8
                )
                if sweep["swing_high"]
                else None
            ),

        "Swing Low":
            (
                round(
                    sweep["swing_low"],
                    8
                )
                if sweep["swing_low"]
                else None
            ),

        "15m Liquidity":
            sweep["liquidity"],

        "5m BOS":
            bos["structure"],

        "FVG":
            fvg["fvg"],

        "FVG Zone":
            fvg["fvg_zone"],

        "Volume x":
            round(
                volume_ratio,
                2
            ),

        "Volume Signal":
            volume["volume_signal"],

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

        "ATR":
            (
                round(
                    atr["ATR"],
                    8
                )
                if atr["ATR"] is not None
                else None
            ),

        "ATR Direction":
            atr_direction,

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
        "Vol/OI > 6",
        int(
            (
                market["Vol/OI"]
                > MIN_VOL_OI_RATIO
            ).sum()
        )
    )

with c3:

    st.metric(
        "Deep Scan Limit",
        DEEP_SCAN_LIMIT
    )

with c4:

    st.metric(
        "Filter",
        "> 6"
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
            "Vol/OI",
            "Funding"
        ]
    ].head(250),

    use_container_width=True,

    hide_index=True
)


# =========================================================
# HARD FILTER
# =========================================================

candidate_market = market[
    market["Vol/OI"]
    > MIN_VOL_OI_RATIO
].copy()


# =========================================================
# ACTIVITY RANK
# =========================================================

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
    f"{len(market)} live perpetuals me se "
    f"{len(candidate_market)} coins ka "
    f"24H Volume/OI ratio > {MIN_VOL_OI_RATIO} hai. "
    f"Inme se top {len(candidates)} ko deep scan kiya ja raha hai."
)


# =========================================================
# SCAN
# =========================================================

st.subheader(
    "🎯 Scanner Results"
)

results = []

if not candidates.empty:

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
                ((i + 1) / total)
                * 100
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
        "❌ Vol/OI > 6 filter ke baad "
        "analysis data available nahi hai."
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
    "🟢 LONG — FVG + Swing + Funding"
)

if not signals.empty:

    long_table = signals[
        [
            "Coin",
            "Price",
            "Vol/OI",
            "1H Trend",
            "Market",
            "Swing Low",
            "15m Liquidity",
            "5m BOS",
            "FVG",
            "FVG Zone",
            "Volume x",
            "OI Change %",
            "Funding %",
            "Funding",
            "ATR Direction",
            "Long Score",
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
    "🔴 SHORT — FVG + Swing + Funding"
)

if not signals.empty:

    short_table = signals[
        [
            "Coin",
            "Price",
            "Vol/OI",
            "1H Trend",
            "Market",
            "Swing High",
            "15m Liquidity",
            "5m BOS",
            "FVG",
            "FVG Zone",
            "Volume x",
            "OI Change %",
            "Funding %",
            "Funding",
            "ATR Direction",
            "Short Score",
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
                    "Vol/OI",
                    "1H Trend",
                    "Market",
                    "Swing Low",
                    "15m Liquidity",
                    "5m BOS",
                    "FVG",
                    "FVG Zone",
                    "Long Score",
                    "Long Signal",
                    "Funding %",
                    "OI Change %",
                    "Volume x",
                    "ATR Direction",
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
                    "Vol/OI",
                    "1H Trend",
                    "Market",
                    "Swing High",
                    "15m Liquidity",
                    "5m BOS",
                    "FVG",
                    "FVG Zone",
                    "Short Score",
                    "Short Signal",
                    "Funding %",
                    "OI Change %",
                    "Volume x",
                    "ATR Direction",
                    "Entry Zone",
                    "Short Reason"
                ]
            ],

            use_container_width=True,

            hide_index=True
        )


# =========================================================
# FVG EXPLANATION
# =========================================================

st.divider()

st.subheader(
    "📐 FVG Logic"
)

st.write(
    """
Bullish FVG:
3-candle structure me third candle ka LOW
first candle ke HIGH se upar ho.

Bearish FVG:
third candle ka HIGH
first candle ke LOW se neeche ho.

FVG ko standalone entry signal nahi maana gaya.
Sweep + BOS + FVG ka alignment zyada important hai.
"""
)


# =========================================================
# VOL/OI EXPLANATION
# =========================================================

st.subheader(
    "📊 Volume / OI Filter"
)

st.write(
    f"""
24H Volume ÷ Open Interest > {MIN_VOL_OI_RATIO}

Sirf wahi coins deep scanner me jayenge
jinka ratio 6 se zyada hai.

Ye HARD FILTER hai, score point nahi.

Example:

Volume = 600M
OI = 100M

Vol/OI = 6

6 se exactly greater condition chahiye,
isliye 6.00 qualify nahi karega.
6.01 qualify karega.
"""
)


# =========================================================
# MARKET STRUCTURE
# =========================================================

st.subheader(
    "🧠 Multi-Timeframe Structure"
)

st.write(
    """
1H → Direction / Market regime

15m → Swing High / Swing Low
      + Liquidity Sweep

5m → BOS
     + FVG

OI → Position participation

Funding → Long/Short crowding

Volume → Current activity

ATR → Volatility expansion/contraction

Final → Separate LONG / SHORT score
"""
)


# =========================================================
# WARNING
# =========================================================

st.warning(
    "⚠️ Ye scanner probability/confirmation tool hai, "
    "guaranteed trade signal nahi. "
    "FVG, BOS, OI, funding aur volume ko "
    "ek saath confirmation ke liye use karein."
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
