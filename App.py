import streamlit as st
import requests
import pandas as pd
import numpy as np
import time

BASE_URL = "https://api.india.delta.exchange"

HEADERS = {
    "Accept": "application/json",
    "User-Agent": "Delta-Reversal-Scanner/8.0"
}

CACHE_SECONDS = 120
DEEP_SCAN_LIMIT = 30

st.set_page_config(
    page_title="Delta Reversal Scanner",
    layout="wide"
)

st.title("🔥 Delta Reversal Scanner PRO")

st.caption(
    "1H Trend → 15m Sweep → Swing → 5m BOS → FVG → "
    "ATR → OI → Funding → Volume → Score"
)


# =========================================================
# API
# =========================================================

def api_get(path, params=None):

    try:
        r = requests.get(
            BASE_URL + path,
            params=params,
            headers=HEADERS,
            timeout=15
        )

        if r.status_code != 200:
            return None

        data = r.json()

        if data.get("success") is False:
            return None

        return data.get("result", [])

    except Exception:
        return None


# =========================================================
# PRODUCTS
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

    return df.drop_duplicates("Coin")


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
    start = end - hours * 3600

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
# ATR
# =========================================================

def add_atr(df, period=14):

    df = df.copy()

    prev_close = df["close"].shift(1)

    tr1 = df["high"] - df["low"]
    tr2 = abs(df["high"] - prev_close)
    tr3 = abs(df["low"] - prev_close)

    df["TR"] = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    df["ATR"] = (
        df["TR"]
        .rolling(period)
        .mean()
    )

    return df


def atr_direction(df):

    if len(df) < 20:
        return "⚪ UNKNOWN"

    atr_now = df["ATR"].iloc[-1]
    atr_old = df["ATR"].iloc[-6]

    if pd.isna(atr_now) or pd.isna(atr_old):
        return "⚪ UNKNOWN"

    if atr_now > atr_old * 1.10:
        return "🔺 ATR RISING"

    if atr_now < atr_old * 0.90:
        return "🔻 ATR FALLING"

    return "⚪ ATR FLAT"


# =========================================================
# SWING HIGH / LOW
# =========================================================

def find_swings(df, left=2, right=2):

    df = df.copy()

    df["SwingHigh"] = False
    df["SwingLow"] = False

    for i in range(
        left,
        len(df) - right
    ):

        high = df["high"].iloc[i]
        low = df["low"].iloc[i]

        left_high = df["high"].iloc[
            i-left:i
        ].max()

        right_high = df["high"].iloc[
            i+1:i+right+1
        ].max()

        left_low = df["low"].iloc[
            i-left:i
        ].min()

        right_low = df["low"].iloc[
            i+1:i+right+1
        ].min()

        if high > left_high and high > right_high:
            df.loc[i, "SwingHigh"] = True

        if low < left_low and low < right_low:
            df.loc[i, "SwingLow"] = True

    return df


# =========================================================
# FVG
# =========================================================

def detect_fvg(df):

    bull = False
    bear = False

    if len(df) < 4:
        return bull, bear

    a = df.iloc[-3]
    b = df.iloc[-2]
    c = df.iloc[-1]

    # Bullish FVG
    if c["low"] > a["high"]:
        bull = True

    # Bearish FVG
    if c["high"] < a["low"]:
        bear = True

    return bull, bear


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

    price = close.iloc[-1]

    if price > ema9.iloc[-1] > ema21.iloc[-1]:
        return "🟢 BULLISH"

    if price < ema9.iloc[-1] < ema21.iloc[-1]:
        return "🔴 BEARISH"

    return "⚪ NEUTRAL"


# =========================================================
# 15M SWEEP
# =========================================================

def analyze_15m(symbol):

    df = get_candles(
        symbol,
        "15m",
        30
    )

    if df.empty or len(df) < 12:

        return {
            "bull": False,
            "bear": False,
            "name": "⚪ None"
        }

    df = find_swings(df)

    last = df.iloc[-1]

    swing_highs = df[
        df["SwingHigh"]
    ]

    swing_lows = df[
        df["SwingLow"]
    ]

    if swing_highs.empty or swing_lows.empty:

        previous_high = df["high"].iloc[-7:-1].max()
        previous_low = df["low"].iloc[-7:-1].min()

    else:

        previous_high = swing_highs["high"].iloc[-1]
        previous_low = swing_lows["low"].iloc[-1]

    bull = (
        last["low"] < previous_low
        and
        last["close"] > previous_low
    )

    bear = (
        last["high"] > previous_high
        and
        last["close"] < previous_high
    )

    if bull:
        name = "🟢 BULL SWEEP"
    elif bear:
        name = "🔴 BEAR SWEEP"
    else:
        name = "⚪ None"

    return {
        "bull": bull,
        "bear": bear,
        "name": name
    }


# =========================================================
# 5M STRUCTURE + FVG
# =========================================================

def analyze_5m(symbol):

    df = get_candles(
        symbol,
        "5m",
        24
    )

    if df.empty or len(df) < 20:

        return {
            "bull_bos": False,
            "bear_bos": False,
            "structure": "⚪ None",
            "fvg": "⚪ None",
            "bull_fvg": False,
            "bear_fvg": False
        }

    df = find_swings(df)

    last = df.iloc[-1]

    swing_highs = df[
        df["SwingHigh"]
    ]

    swing_lows = df[
        df["SwingLow"]
    ]

    if not swing_highs.empty:
        last_swing_high = swing_highs["high"].iloc[-1]
    else:
        last_swing_high = df["high"].iloc[-8:-1].max()

    if not swing_lows.empty:
        last_swing_low = swing_lows["low"].iloc[-1]
    else:
        last_swing_low = df["low"].iloc[-8:-1].min()

    bull_bos = (
        last["close"] >
        last_swing_high
    )

    bear_bos = (
        last["close"] <
        last_swing_low
    )

    bull_fvg, bear_fvg = detect_fvg(df)

    if bull_bos:
        structure = "🟢 BULL BOS"
    elif bear_bos:
        structure = "🔴 BEAR BOS"
    else:
        structure = "⚪ None"

    if bull_fvg:
        fvg = "🟢 BULL FVG"
    elif bear_fvg:
        fvg = "🔴 BEAR FVG"
    else:
        fvg = "⚪ None"

    return {
        "bull_bos": bull_bos,
        "bear_bos": bear_bos,
        "structure": structure,
        "fvg": fvg,
        "bull_fvg": bull_fvg,
        "bear_fvg": bear_fvg
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

    if df.empty or len(df) < 6:
        return 0

    current = df["volume"].iloc[-1]

    average = df["volume"].iloc[-6:-1].mean()

    if average <= 0:
        return 0

    return current / average


# =========================================================
# OI HISTORY
# =========================================================

@st.cache_data(ttl=CACHE_SECONDS)
def get_oi_history(symbol):

    end = int(time.time())
    start = end - 12 * 3600

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

    if df.empty or "close" not in df.columns:
        return pd.DataFrame()

    df["close"] = pd.to_numeric(
        df["close"],
        errors="coerce"
    )

    return df.dropna(
        subset=["close"]
    ).sort_values("time")


def analyze_oi(symbol):

    df = get_oi_history(symbol)

    if df.empty or len(df) < 6:

        return {
            "change": None,
            "signal": "⚪ Unknown"
        }

    current = df["close"].iloc[-1]
    old = df["close"].iloc[-6]

    if old == 0:
        return {
            "change": None,
            "signal": "⚪ Unknown"
        }

    change = (
        (current - old) /
        abs(old)
    ) * 100

    if change >= 1:
        signal = "🔺 OI UP"
    elif change <= -1:
        signal = "🔻 OI DOWN"
    else:
        signal = "⚪ OI FLAT"

    return {
        "change": change,
        "signal": signal
    }


# =========================================================
# MARKET STATE
# =========================================================

def market_state(trend, sweep, bos, atr_dir):

    bull = 0
    bear = 0

    if trend == "🟢 BULLISH":
        bull += 2

    if trend == "🔴 BEARISH":
        bear += 2

    if sweep["bull"]:
        bull += 2

    if sweep["bear"]:
        bear += 2

    if bos["bull_bos"]:
        bull += 2

    if bos["bear_bos"]:
        bear += 2

    if bull >= bear + 2:
        return "🟢 DIRECTIONAL LONG"

    if bear >= bull + 2:
        return "🔴 DIRECTIONAL SHORT"

    if bull <= 1 and bear <= 1:
        return "🟡 RANGE BOUND"

    return "⚪ UNCERTAINTY"


# =========================================================
# LIVE ANALYSIS
# =========================================================

def deep_analysis(symbol, ticker):

    trend = analyze_1h(symbol)

    sweep = analyze_15m(symbol)

    bos = analyze_5m(symbol)

    volume_ratio = analyze_volume(symbol)

    oi = analyze_oi(symbol)

    candles = get_candles(
        symbol,
        "5m",
        24
    )

    if candles.empty:
        atr_dir = "⚪ UNKNOWN"
        atr_value = None
    else:
        candles = add_atr(candles)
        atr_dir = atr_direction(candles)
        atr_value = candles["ATR"].iloc[-1]

    state = market_state(
        trend,
        sweep,
        bos,
        atr_dir
    )

    funding = ticker.get("Funding")

    funding_pct = None

    if funding is not None:

        try:
            funding_pct = float(funding) * 100
        except:
            funding_pct = None

    long_score = 0
    short_score = 0

    long_reason = []
    short_reason = []

    # -------------------------------
    # TREND
    # -------------------------------

    if trend == "🟢 BULLISH":

        long_score += 2
        long_reason.append("1H bullish")

    if trend == "🔴 BEARISH":

        short_score += 2
        short_reason.append("1H bearish")

    # -------------------------------
    # SWEEP
    # -------------------------------

    if sweep["bull"]:

        long_score += 2
        long_reason.append("15m liquidity sweep")

    if sweep["bear"]:

        short_score += 2
        short_reason.append("15m liquidity sweep")

    # -------------------------------
    # BOS
    # -------------------------------

    if bos["bull_bos"]:

        long_score += 3
        long_reason.append("5m BOS")

    if bos["bear_bos"]:

        short_score += 3
        short_reason.append("5m BOS")

    # -------------------------------
    # FVG
    # -------------------------------

    if bos["bull_fvg"]:

        long_score += 2
        long_reason.append("Bull FVG")

    if bos["bear_fvg"]:

        short_score += 2
        short_reason.append("Bear FVG")

    # -------------------------------
    # VOLUME
    # -------------------------------

    if volume_ratio >= 2:

        long_score += 2
        short_score += 2

        long_reason.append("Volume spike")
        short_reason.append("Volume spike")

    elif volume_ratio >= 1.3:

        long_score += 1
        short_score += 1

    # -------------------------------
    # OI
    # -------------------------------

    oi_change = oi["change"]

    if oi_change is not None:

        if oi_change >= 1:

            if trend == "🟢 BULLISH":

                long_score += 1
                long_reason.append("OI rising")

            if trend == "🔴 BEARISH":

                short_score += 1
                short_reason.append("OI rising")

        elif oi_change <= -1:

            if sweep["bull"]:

                long_score += 1
                long_reason.append("OI falling after sweep")

            if sweep["bear"]:

                short_score += 1
                short_reason.append("OI falling after sweep")

    # -------------------------------
    # FUNDING
    # -------------------------------

    if funding_pct is not None:

        if funding_pct >= 0.05:

            short_score += 2

            short_reason.append(
                "Long crowding / positive funding"
            )

            funding_signal = "🔴 Long crowded"

        elif funding_pct <= -0.05:

            long_score += 2

            long_reason.append(
                "Short crowding / negative funding"
            )

            funding_signal = "🟢 Short crowded"

        else:

            funding_signal = "⚪ Neutral"

    else:

        funding_signal = "⚪ Unavailable"

    # -------------------------------
    # ATR
    # -------------------------------

    if atr_dir == "🔺 ATR RISING":

        long_score += 1
        short_score += 1

    # Falling ATR does NOT add score.
    # It means volatility is contracting.

    # -------------------------------
    # SIGNALS
    # -------------------------------

    if long_score >= 8:
        long_signal = "🟢 STRONG LONG"
    elif long_score >= 5:
        long_signal = "🟡 LONG WATCH"
    else:
        long_signal = "⚪ NO LONG"

    if short_score >= 8:
        short_signal = "🔴 STRONG SHORT"
    elif short_score >= 5:
        short_signal = "🟠 SHORT WATCH"
    else:
        short_signal = "⚪ NO SHORT"

    if long_score > short_score and long_score >= 5:

        signal = long_signal
        score = long_score
        reason = " + ".join(long_reason)

    elif short_score > long_score and short_score >= 5:

        signal = short_signal
        score = short_score
        reason = " + ".join(short_reason)

    else:

        signal = "⚪ NO SIGNAL"
        score = max(long_score, short_score)
        reason = "Mixed conditions"

    return {
        "Coin": symbol,
        "Price": ticker["Price"],
        "24H Volume": ticker["24H Volume"],
        "OI": ticker["OI"],
        "Vol/OI": ticker["Vol/OI"],
        "1H Trend": trend,
        "15m Liquidity": sweep["name"],
        "5m BOS": bos["structure"],
        "FVG": bos["fvg"],
        "ATR": round(atr_value, 8)
        if atr_value is not None else None,
        "ATR Direction": atr_dir,
        "Volume x": round(volume_ratio, 2),
        "OI Change %": round(oi_change, 2)
        if oi_change is not None else None,
        "Funding %": round(funding_pct, 4)
        if funding_pct is not None else None,
        "Funding": funding_signal,
        "Market State": state,
        "Long Score": long_score,
        "Long Signal": long_signal,
        "Short Score": short_score,
        "Short Signal": short_signal,
        "Score": score,
        "Signal": signal,
        "Long Reason": " + ".join(long_reason),
        "Short Reason": " + ".join(short_reason),
        "Reason": reason
    }


# =========================================================
# BACKTEST ENGINE
# =========================================================

def backtest_symbol(
    symbol,
    days=7,
    rr=2.0,
    threshold=8
):

    df = get_candles(
        symbol,
        "5m",
        days * 24
    )

    if df.empty or len(df) < 150:
        return []

    df = add_atr(df)
    df = find_swings(df)

    trades = []

    start_index = 50

    for i in range(
        start_index,
        len(df) - 20
    ):

        current = df.iloc[i]

        # ---------------------------------
        # Historical window
        # ---------------------------------

        hist = df.iloc[
            max(0, i-50):i+1
        ]

        if len(hist) < 20:
            continue

        # ---------------------------------
        # ATR
        # ---------------------------------

        atr_now = hist["ATR"].iloc[-1]

        atr_old = hist["ATR"].iloc[-6]

        if pd.isna(atr_now) or pd.isna(atr_old):
            continue

        if atr_now > atr_old * 1.10:
            atr_rising = True
        else:
            atr_rising = False

        # ---------------------------------
        # Swing levels
        # ---------------------------------

        swing_highs = hist[
            hist["SwingHigh"]
        ]

        swing_lows = hist[
            hist["SwingLow"]
        ]

        if swing_highs.empty or swing_lows.empty:
            continue

        swing_high = swing_highs["high"].iloc[-1]
        swing_low = swing_lows["low"].iloc[-1]

        # ---------------------------------
        # Liquidity sweep
        # ---------------------------------

        bull_sweep = (
            current["low"] < swing_low
            and
            current["close"] > swing_low
        )

        bear_sweep = (
            current["high"] > swing_high
            and
            current["close"] < swing_high
        )

        # ---------------------------------
        # BOS
        # ---------------------------------

        previous_high = hist["high"].iloc[-8:-1].max()
        previous_low = hist["low"].iloc[-8:-1].min()

        bull_bos = (
            current["close"] >
            previous_high
        )

        bear_bos = (
            current["close"] <
            previous_low
        )

        # ---------------------------------
        # FVG
        # ---------------------------------

        if i >= 2:

            a = df.iloc[i-2]
            c = df.iloc[i]

            bull_fvg = (
                c["low"] > a["high"]
            )

            bear_fvg = (
                c["high"] < a["low"]
            )

        else:

            bull_fvg = False
            bear_fvg = False

        # ---------------------------------
        # Volume
        # ---------------------------------

        avg_volume = hist[
            "volume"
        ].iloc[-6:-1].mean()

        if avg_volume <= 0:
            continue

        volume_ratio = (
            current["volume"] /
            avg_volume
        )

        # ---------------------------------
        # Score
        # ---------------------------------

        long_score = 0
        short_score = 0

        if bull_sweep:
            long_score += 2

        if bear_sweep:
            short_score += 2

        if bull_bos:
            long_score += 3

        if bear_bos:
            short_score += 3

        if bull_fvg:
            long_score += 2

        if bear_fvg:
            short_score += 2

        if volume_ratio >= 2:
            long_score += 2
            short_score += 2

        elif volume_ratio >= 1.3:
            long_score += 1
            short_score += 1

        if atr_rising:
            long_score += 1
            short_score += 1

        # ---------------------------------
        # LONG
        # ---------------------------------

        if long_score >= threshold:

            entry = current["close"]

            stop = min(
                current["low"],
                swing_low
            )

            risk = entry - stop

            if risk <= 0:
                continue

            target = (
                entry +
                risk * rr
            )

            result = None
            exit_price = None

            for j in range(
                i + 1,
                min(i + 50, len(df))
            ):

                future = df.iloc[j]

                if future["low"] <= stop:

                    result = "LOSS"
                    exit_price = stop
                    break

                if future["high"] >= target:

                    result = "WIN"
                    exit_price = target
                    break

            if result is None:
                continue

            r = (
                rr
                if result == "WIN"
                else -1
            )

            trades.append({
                "Coin": symbol,
                "Time": current["time"],
                "Side": "LONG",
                "Score": long_score,
                "Entry": entry,
                "SL": stop,
                "TP": target,
                "Exit": exit_price,
                "Result": result,
                "R": r,
                "Volume x": round(
                    volume_ratio,
                    2
                ),
                "ATR Rising": atr_rising,
                "FVG": bull_fvg,
                "Sweep": bull_sweep,
                "BOS": bull_bos
            })

        # ---------------------------------
        # SHORT
        # ---------------------------------

        if short_score >= threshold:

            entry = current["close"]

            stop = max(
                current["high"],
                swing_high
            )

            risk = stop - entry

            if risk <= 0:
                continue

            target = (
                entry -
                risk * rr
            )

            result = None
            exit_price = None

            for j in range(
                i + 1,
                min(i + 50, len(df))
            ):

                future = df.iloc[j]

                if future["high"] >= stop:

                    result = "LOSS"
                    exit_price = stop
                    break

                if future["low"] <= target:

                    result = "WIN"
                    exit_price = target
                    break

            if result is None:
                continue

            r = (
                rr
                if result == "WIN"
                else -1
            )

            trades.append({
                "Coin": symbol,
                "Time": current["time"],
                "Side": "SHORT",
                "Score": short_score,
                "Entry": entry,
                "SL": stop,
                "TP": target,
                "Exit": exit_price,
                "Result": result,
                "R": r,
                "Volume x": round(
                    volume_ratio,
                    2
                ),
                "ATR Rising": atr_rising,
                "FVG": bear_fvg,
                "Sweep": bear_sweep,
                "BOS": bear_bos
            })

    return trades


# =========================================================
# LOAD MARKET
# =========================================================

all_coins = get_all_perpetuals()
tickers = get_tickers()

if all_coins.empty:
    st.error("❌ Perpetual contracts load nahi hue.")
    st.stop()

if tickers.empty:
    st.error("❌ Ticker data load nahi hua.")
    st.stop()


market = all_coins.merge(
    tickers,
    on="Coin",
    how="left"
)

market = market.dropna(
    subset=["Price"]
)

# =========================================================
# IMPORTANT FILTER
# 24H VOLUME / OI > 6
# =========================================================

market = market[
    market["Vol/OI"] > 6
].copy()

market = market.sort_values(
    "24H Volume",
    ascending=False
)


# =========================================================
# MODE
# =========================================================

mode = st.radio(
    "Mode",
    [
        "🔥 Live Scanner",
        "📊 Backtest"
    ],
    horizontal=True
)


# =========================================================
# LIVE SCANNER
# =========================================================

if mode == "🔥 Live Scanner":

    st.info(
        f"24H Volume/OI > 6 filter ke baad "
        f"{len(market)} coins available hain."
    )

    candidates = market.head(
        DEEP_SCAN_LIMIT
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

        results.append(result)

        progress.progress(
            int(
                ((i + 1) /
                 len(candidates)) * 100
            )
        )

    progress.empty()

    signals = pd.DataFrame(results)

    if signals.empty:

        st.warning(
            "Analysis data available nahi hai."
        )

    else:

        signals = signals.sort_values(
            "Score",
            ascending=False
        )

        st.subheader(
            "🎯 Scanner Results"
        )

        st.dataframe(
            signals,
            use_container_width=True,
            hide_index=True
        )

        st.subheader(
            "🟢 LONG"
        )

        st.dataframe(
            signals[
                [
                    "Coin",
                    "Price",
                    "Market State",
                    "1H Trend",
                    "15m Liquidity",
                    "5m BOS",
                    "FVG",
                    "ATR Direction",
                    "Volume x",
                    "OI Change %",
                    "Funding %",
                    "Long Score",
                    "Long Signal",
                    "Long Reason"
                ]
            ].sort_values(
                "Long Score",
                ascending=False
            ),
            use_container_width=True,
            hide_index=True
        )

        st.subheader(
            "🔴 SHORT"
        )

        st.dataframe(
            signals[
                [
                    "Coin",
                    "Price",
                    "Market State",
                    "1H Trend",
                    "15m Liquidity",
                    "5m BOS",
                    "FVG",
                    "ATR Direction",
                    "Volume x",
                    "OI Change %",
                    "Funding %",
                    "Short Score",
                    "Short Signal",
                    "Short Reason"
                ]
            ].sort_values(
                "Short Score",
                ascending=False
            ),
            use_container_width=True,
            hide_index=True
        )

        st.subheader(
            "🔥 STRONG 8+"
        )

        strong = signals[
            signals["Score"] >= 8
        ]

        if strong.empty:

            st.info(
                "Abhi 8+ score ka setup nahi mila."
            )

        else:

            st.dataframe(
                strong[
                    [
                        "Coin",
                        "Price",
                        "Market State",
                        "Score",
                        "Signal",
                        "FVG",
                        "ATR Direction",
                        "Volume x",
                        "OI Change %",
                        "Funding %",
                        "Reason"
                    ]
                ],
                use_container_width=True,
                hide_index=True
            )


# =========================================================
# BACKTEST
# =========================================================

else:

    st.subheader(
        "📊 Historical Backtest"
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        days = st.slider(
            "Historical days",
            2,
            30,
            7
        )

    with c2:
        rr = st.selectbox(
            "Risk : Reward",
            [1.0, 1.5, 2.0, 2.5, 3.0],
            index=2
        )

    with c3:
        score_threshold = st.selectbox(
            "Minimum Score",
            [6, 7, 8, 9, 10],
            index=2
        )

    coin_limit = st.slider(
        "Coins to backtest",
        1,
        min(20, len(market))
        if len(market) > 0 else 1,
        min(10, len(market))
        if len(market) > 0 else 1
    )

    backtest_coins = market.head(
        coin_limit
    )

    st.info(
        f"Backtest mein sirf un coins ko liya "
        f"ja raha hai jinka current 24H Volume/OI > 6 hai. "
        f"Coins: {len(backtest_coins)}"
    )

    if st.button(
        "▶️ Run Backtest"
    ):

        all_trades = []

        progress = st.progress(0)

        total = len(backtest_coins)

        for i, (_, row) in enumerate(
            backtest_coins.iterrows()
        ):

            trades = backtest_symbol(
                row["Coin"],
                days=days,
                rr=rr,
                threshold=score_threshold
            )

            all_trades.extend(trades)

            progress.progress(
                int(
                    ((i + 1) /
                     total) * 100
                )
            )

        progress.empty()

        if not all_trades:

            st.warning(
                "Selected conditions par "
                "historical trades nahi mile."
            )

        else:

            bt = pd.DataFrame(
                all_trades
            )

            # --------------------------------
            # METRICS
            # --------------------------------

            total_trades = len(bt)

            wins = (
                bt["Result"] == "WIN"
            ).sum()

            losses = (
                bt["Result"] == "LOSS"
            ).sum()

            win_rate = (
                wins /
                total_trades *
                100
            )

            total_r = bt["R"].sum()

            gross_profit = bt.loc[
                bt["R"] > 0,
                "R"
            ].sum()

            gross_loss = abs(
                bt.loc[
                    bt["R"] < 0,
                    "R"
                ].sum()
            )

            if gross_loss > 0:
                profit_factor = (
                    gross_profit /
                    gross_loss
                )
            else:
                profit_factor = np.inf

            avg_r = bt["R"].mean()

            # --------------------------------
            # DRAWDOWN
            # --------------------------------

            equity = bt["R"].cumsum()

            peak = equity.cummax()

            drawdown = equity - peak

            max_drawdown = drawdown.min()

            # --------------------------------
            # METRICS UI
            # --------------------------------

            m1, m2, m3, m4, m5 = st.columns(5)

            with m1:
                st.metric(
                    "Trades",
                    total_trades
                )

            with m2:
                st.metric(
                    "Win Rate",
                    f"{win_rate:.2f}%"
                )

            with m3:
                st.metric(
                    "Total R",
                    f"{total_r:.2f}"
                )

            with m4:
                st.metric(
                    "Profit Factor",
                    (
                        "∞"
                        if np.isinf(profit_factor)
                        else f"{profit_factor:.2f}"
                    )
                )

            with m5:
                st.metric(
                    "Max Drawdown",
                    f"{max_drawdown:.2f} R"
                )

            # --------------------------------
            # SCORE ANALYSIS
            # --------------------------------

            st.subheader(
                "🎯 Score-wise Performance"
            )

            score_stats = (
                bt.groupby("Score")
                .agg(
                    Trades=("Result", "count"),
                    Wins=("Result",
                          lambda x:
                          (x == "WIN").sum()),
                    Total_R=("R", "sum"),
                    Avg_R=("R", "mean")
                )
                .reset_index()
            )

            score_stats["Win %"] = (
                score_stats["Wins"] /
                score_stats["Trades"] *
                100
            )

            st.dataframe(
                score_stats,
                use_container_width=True,
                hide_index=True
            )

            # --------------------------------
            # LONG / SHORT
            # --------------------------------

            st.subheader(
                "🟢 LONG vs 🔴 SHORT"
            )

            side_stats = (
                bt.groupby("Side")
                .agg(
                    Trades=("Result", "count"),
                    Wins=("Result",
                          lambda x:
                          (x == "WIN").sum()),
                    Total_R=("R", "sum"),
                    Avg_R=("R", "mean")
                )
                .reset_index()
            )

            side_stats["Win %"] = (
                side_stats["Wins"] /
                side_stats["Trades"] *
                100
            )

            st.dataframe(
                side_stats,
                use_container_width=True,
                hide_index=True
            )

            # --------------------------------
            # CONDITION ANALYSIS
            # --------------------------------

            st.subheader(
                "🔬 Condition Performance"
            )

            condition_rows = []

            for condition in [
                "FVG",
                "Sweep",
                "BOS",
                "ATR Rising"
            ]:

                yes = bt[
                    bt[condition] == True
                ]

                if len(yes) == 0:
                    continue

                condition_rows.append({
                    "Condition": condition,
                    "Trades": len(yes),
                    "Win %":
                        (
                            (yes["Result"] == "WIN")
                            .mean() * 100
                        ),
                    "Total R":
                        yes["R"].sum(),
                    "Avg R":
                        yes["R"].mean()
                })

            condition_stats = pd.DataFrame(
                condition_rows
            )

            if not condition_stats.empty:

                st.dataframe(
                    condition_stats,
                    use_container_width=True,
                    hide_index=True
                )

            # --------------------------------
            # EQUITY CURVE
            # --------------------------------

            st.subheader(
                "📈 Backtest Equity Curve"
            )

            equity_df = pd.DataFrame({
                "Trade": range(
                    1,
                    len(equity) + 1
                ),
                "R": equity.values
            })

            st.line_chart(
                equity_df.set_index("Trade")
            )

            # --------------------------------
            # TRADES
            # --------------------------------

            st.subheader(
                "📋 Trade Log"
            )

            st.dataframe(
                bt.sort_values(
                    "Time",
                    ascending=False
                ),
                use_container_width=True,
                hide_index=True
            )


# =========================================================
# EXPLANATION
# =========================================================

st.divider()

st.subheader(
    "🧠 Scanner ka current structure"
)

st.write(
    """
LIVE:

24H Volume/OI > 6
↓
1H Trend
↓
15m Liquidity Sweep
↓
Swing High / Low
↓
5m BOS
↓
FVG
↓
OI
↓
Funding
↓
Volume
↓
ATR
↓
Market State
↓
Long / Short Score


BACKTEST:

Historical 5m candles
↓
Swing High / Low
↓
Liquidity Sweep
↓
BOS
↓
FVG
↓
Volume
↓
ATR
↓
Score
↓
Entry
↓
SL
↓
TP
↓
Win/Loss
↓
R
↓
Win Rate
↓
Profit Factor
↓
Drawdown
"""
)

st.warning(
    "⚠️ Backtest historical probability batata hai, "
    "future result guarantee nahi karta. "
    "Is version mein current 24H Volume/OI > 6 ko "
    "candidate filter ke रूप में use kiya gaya hai; "
    "historical OI/funding ko future-data leak se bachane "
    "ke liye trade signal mein directly use nahi kiya gaya."
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
