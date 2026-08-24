import streamlit as st
import requests
import pandas as pd
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


# ============================================================
# CONFIG
# ============================================================

BASE_URL = "https://api.india.delta.exchange"

HEADERS = {
    "Accept": "application/json",
    "User-Agent": "Delta-Reversal-Scanner-PRO/11.0"
}

CACHE_SECONDS = 20
ORDERBOOK_CACHE = 5
TRADE_CACHE = 5

TOP_COINS = 20
ORDERBOOK_DEPTH = 20

VOL_OI_MIN = 6.0


st.set_page_config(
    page_title="Delta Reversal Scanner PRO 11",
    page_icon="🔥",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🔥 Delta Reversal Scanner PRO 11")

st.caption(
    "Delta India → MTF → 5D Regime → S/R → Sweep → BOS/CHOCH → "
    "FVG → OI → Funding → Volume → ATR → Order Book → "
    "Trade Flow → Liquidation Pressure Proxy"
)


# ============================================================
# SESSION
# ============================================================

if "last_error" not in st.session_state:
    st.session_state.last_error = None


# ============================================================
# API CORE
# ============================================================

def api_get(path, params=None, timeout=12):

    url = BASE_URL + path

    try:

        response = requests.get(
            url,
            params=params,
            headers=HEADERS,
            timeout=timeout
        )

        if response.status_code != 200:
            st.session_state.last_error = (
                f"{path} → HTTP {response.status_code}"
            )
            return None

        try:
            payload = response.json()
        except Exception:
            return None

        if not isinstance(payload, dict):
            return payload

        if payload.get("success") is False:
            st.session_state.last_error = str(
                payload.get("error", "Delta API error")
            )
            return None

        return payload.get("result")

    except requests.exceptions.Timeout:
        st.session_state.last_error = f"{path} → timeout"
        return None

    except requests.exceptions.RequestException as e:
        st.session_state.last_error = f"{path} → {str(e)}"
        return None

    except Exception as e:
        st.session_state.last_error = f"{path} → {str(e)}"
        return None


# ============================================================
# SAFE FLOAT
# ============================================================

def safe_float(value, default=np.nan):

    try:

        if value is None:
            return default

        if isinstance(value, str):
            value = value.replace(",", "")

        return float(value)

    except Exception:
        return default


# ============================================================
# PRODUCTS
# ============================================================

@st.cache_data(ttl=CACHE_SECONDS)
def get_all_perpetuals():

    result = api_get(
        "/v2/products",
        {"page_size": 100}
    )

    if not result:
        return pd.DataFrame()

    if not isinstance(result, list):
        return pd.DataFrame()

    rows = []

    for p in result:

        if not isinstance(p, dict):
            continue

        contract_type = str(
            p.get("contract_type", "")
        ).lower()

        state = str(
            p.get("state", "")
        ).lower()

        trading_status = str(
            p.get("trading_status", "")
        ).lower()

        if contract_type != "perpetual_futures":
            continue

        if state != "live":
            continue

        if trading_status not in [
            "",
            "operational"
        ]:
            continue

        symbol = p.get("symbol")

        if not symbol:
            continue

        rows.append({
            "Coin": symbol,
            "ID": p.get("id")
        })

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .drop_duplicates("Coin")
        .reset_index(drop=True)
    )


# ============================================================
# TICKERS
# ============================================================

@st.cache_data(ttl=CACHE_SECONDS)
def get_tickers():

    result = api_get("/v2/tickers")

    if not result:
        return pd.DataFrame()

    if not isinstance(result, list):
        return pd.DataFrame()

    rows = []

    for item in result:

        if not isinstance(item, dict):
            continue

        symbol = item.get("symbol")

        if not symbol:
            continue

        price = safe_float(
            item.get(
                "close",
                item.get("mark_price")
            )
        )

        if not np.isfinite(price) or price <= 0:
            continue

        volume = safe_float(
            item.get(
                "volume_24h",
                item.get("volume", 0)
            ),
            0
        )

        oi = safe_float(
            item.get(
                "open_interest",
                item.get("oi", 0)
            ),
            0
        )

        funding = safe_float(
            item.get(
                "funding_rate",
                item.get("funding")
            )
        )

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
        df["24H Volume"]
        /
        df["OI"].replace(0, np.nan)
    )

    return df


# ============================================================
# CANDLES
# ============================================================

@st.cache_data(ttl=CACHE_SECONDS)
def get_candles(symbol, resolution, hours):

    end = int(time.time())

    start = end - int(
        hours * 3600
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

    if not isinstance(result, list):
        return pd.DataFrame()

    df = pd.DataFrame(result)

    if df.empty:
        return df

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

    if "time" in df.columns:

        df["time"] = pd.to_numeric(
            df["time"],
            errors="coerce"
        )

    required = [
        "open",
        "high",
        "low",
        "close"
    ]

    if not all(
        c in df.columns
        for c in required
    ):
        return pd.DataFrame()

    df = df.dropna(
        subset=required
    )

    if "time" in df.columns:

        df = df.sort_values(
            "time"
        )

        df = df.drop_duplicates(
            "time"
        )

    return df.reset_index(
        drop=True
    )


# ============================================================
# OI HISTORY
# ============================================================

@st.cache_data(ttl=CACHE_SECONDS)
def get_oi_history(symbol, hours=24):

    end = int(time.time())

    start = end - (
        hours * 3600
    )

    result = api_get(
        "/v2/history/candles",
        {
            "resolution": "15m",

            # IMPORTANT:
            # Delta OI history format
            "symbol": "OI:" + symbol,

            "start": start,
            "end": end
        }
    )

    if not result:
        return pd.DataFrame()

    if not isinstance(result, list):
        return pd.DataFrame()

    df = pd.DataFrame(result)

    if df.empty:
        return df

    if "close" not in df.columns:
        return pd.DataFrame()

    df["close"] = pd.to_numeric(
        df["close"],
        errors="coerce"
    )

    if "time" in df.columns:

        df["time"] = pd.to_numeric(
            df["time"],
            errors="coerce"
        )

        df = df.sort_values(
            "time"
        )

    return (
        df.dropna(
            subset=["close"]
        )
        .reset_index(drop=True)
    )


# ============================================================
# OI ANALYSIS
# ============================================================

def oi_analysis(symbol):

    df = get_oi_history(
        symbol,
        24
    )

    if len(df) < 7:
        return None, "⚪ UNKNOWN"

    current = safe_float(
        df["close"].iloc[-1]
    )

    previous = safe_float(
        df["close"].iloc[-7]
    )

    if (
        not np.isfinite(current)
        or
        not np.isfinite(previous)
        or
        previous == 0
    ):
        return None, "⚪ UNKNOWN"

    change = (
        (current - previous)
        /
        abs(previous)
        * 100
    )

    if change >= 1:
        signal = "🔺 OI EXPANSION"

    elif change <= -1:
        signal = "🔻 OI UNWIND"

    else:
        signal = "⚪ OI FLAT"

    return change, signal


# ============================================================
# ORDER BOOK
# ============================================================

@st.cache_data(ttl=ORDERBOOK_CACHE)
def get_orderbook(symbol, depth=20):

    result = api_get(
        f"/v2/l2orderbook/{symbol}",
        {
            "depth": depth
        }
    )

    if not result:
        return None

    if not isinstance(result, dict):
        return None

    return result


# ============================================================
# ORDER BOOK ANALYSIS
# ============================================================

def orderbook_analysis(
    symbol,
    depth=20
):

    data = get_orderbook(
        symbol,
        depth
    )

    if not data:
        return None

    bids = data.get(
        "buy",
        []
    )

    asks = data.get(
        "sell",
        []
    )

    if not bids or not asks:
        return None

    bid_rows = []
    ask_rows = []

    for item in bids:

        if not isinstance(
            item,
            dict
        ):
            continue

        price = safe_float(
            item.get("price")
        )

        size = safe_float(
            item.get("size"),
            0
        )

        if (
            np.isfinite(price)
            and
            np.isfinite(size)
        ):

            bid_rows.append({
                "Price": price,
                "Size": size
            })

    for item in asks:

        if not isinstance(
            item,
            dict
        ):
            continue

        price = safe_float(
            item.get("price")
        )

        size = safe_float(
            item.get("size"),
            0
        )

        if (
            np.isfinite(price)
            and
            np.isfinite(size)
        ):

            ask_rows.append({
                "Price": price,
                "Size": size
            })

    if not bid_rows or not ask_rows:
        return None

    bid_df = pd.DataFrame(
        bid_rows
    )

    ask_df = pd.DataFrame(
        ask_rows
    )

    bid_depth = float(
        bid_df["Size"].sum()
    )

    ask_depth = float(
        ask_df["Size"].sum()
    )

    total_depth = (
        bid_depth +
        ask_depth
    )

    if total_depth <= 0:
        imbalance = 0
    else:
        imbalance = (
            (
                bid_depth -
                ask_depth
            )
            /
            total_depth
            *
            100
        )

    best_bid = float(
        bid_df["Price"].max()
    )

    best_ask = float(
        ask_df["Price"].min()
    )

    spread = (
        best_ask -
        best_bid
    )

    mid = (
        best_bid +
        best_ask
    ) / 2

    return {
        "bid_df": bid_df,
        "ask_df": ask_df,

        "bid_depth": bid_depth,
        "ask_depth": ask_depth,

        "imbalance": imbalance,

        "best_bid": best_bid,
        "best_ask": best_ask,

        "spread": spread,
        "mid": mid
    }


# ============================================================
# ORDER BOOK WALLS
# ============================================================

def liquidity_analysis(symbol):

    ob = orderbook_analysis(
        symbol,
        ORDERBOOK_DEPTH
    )

    if not ob:
        return None

    bid_df = ob["bid_df"]
    ask_df = ob["ask_df"]

    largest_bid = bid_df.loc[
        bid_df["Size"].idxmax()
    ]

    largest_ask = ask_df.loc[
        ask_df["Size"].idxmax()
    ]

    mid = ob["mid"]

    bid_distance = (
        mid -
        largest_bid["Price"]
    ) / mid * 100

    ask_distance = (
        largest_ask["Price"] -
        mid
    ) / mid * 100

    return {
        "imbalance": ob["imbalance"],

        "bid_depth":
        ob["bid_depth"],

        "ask_depth":
        ob["ask_depth"],

        "best_bid":
        ob["best_bid"],

        "best_ask":
        ob["best_ask"],

        "spread":
        ob["spread"],

        "largest_bid_price":
        float(largest_bid["Price"]),

        "largest_bid_size":
        float(largest_bid["Size"]),

        "largest_ask_price":
        float(largest_ask["Price"]),

        "largest_ask_size":
        float(largest_ask["Size"]),

        "bid_wall_distance":
        bid_distance,

        "ask_wall_distance":
        ask_distance
    }


# ============================================================
# PUBLIC TRADES
# ============================================================

@st.cache_data(ttl=TRADE_CACHE)
def get_recent_trades(symbol):

    result = api_get(
        f"/v2/trades/{symbol}"
    )

    if not result:
        return pd.DataFrame()

    # Delta normally returns:
    # {"trades":[...]}

    if isinstance(
        result,
        dict
    ):

        trades = result.get(
            "trades",
            []
        )

    elif isinstance(
        result,
        list
    ):

        trades = result

    else:

        return pd.DataFrame()

    if not trades:
        return pd.DataFrame()

    df = pd.DataFrame(
        trades
    )

    if df.empty:
        return df

    if "price" in df.columns:

        df["price"] = pd.to_numeric(
            df["price"],
            errors="coerce"
        )

    if "size" in df.columns:

        df["size"] = pd.to_numeric(
            df["size"],
            errors="coerce"
        )

    if "side" not in df.columns:

        return pd.DataFrame()

    return df.dropna(
        subset=[
            "price",
            "size"
        ]
    )


# ============================================================
# TRADE FLOW
# ============================================================

def trade_flow_analysis(symbol):

    df = get_recent_trades(
        symbol
    )

    if df.empty:
        return None

    sides = (
        df["side"]
        .astype(str)
        .str.lower()
    )

    buy_volume = float(
        df.loc[
            sides == "buy",
            "size"
        ].sum()
    )

    sell_volume = float(
        df.loc[
            sides == "sell",
            "size"
        ].sum()
    )

    total = (
        buy_volume +
        sell_volume
    )

    if total <= 0:
        return None

    delta = (
        buy_volume -
        sell_volume
    )

    delta_pct = (
        delta /
        total *
        100
    )

    return {
        "buy_volume":
        buy_volume,

        "sell_volume":
        sell_volume,

        "total":
        total,

        "delta":
        delta,

        "delta_pct":
        delta_pct,

        "trades":
        len(df)
    }


# ============================================================
# PRICE PRESSURE
# ============================================================

def recent_price_pressure(
    symbol
):

    df = get_candles(
        symbol,
        "1m",
        1
    )

    if len(df) < 2:
        return None

    first = safe_float(
        df["close"].iloc[0]
    )

    last = safe_float(
        df["close"].iloc[-1]
    )

    if (
        not np.isfinite(first)
        or
        not np.isfinite(last)
        or
        first == 0
    ):
        return None

    move = (
        (last - first)
        /
        first
        *
        100
    )

    return move


# ============================================================
# LIQUIDATION PRESSURE PROXY
# ============================================================

def liquidation_pressure(
    symbol
):

    flow = trade_flow_analysis(
        symbol
    )

    ob = orderbook_analysis(
        symbol,
        ORDERBOOK_DEPTH
    )

    price_move = recent_price_pressure(
        symbol
    )

    if not flow:
        return None

    score = 0

    reasons = []

    delta_pct = flow[
        "delta_pct"
    ]

    # --------------------------------------------------------
    # Aggressive trade imbalance
    # --------------------------------------------------------

    if abs(delta_pct) >= 50:

        score += 2

        reasons.append(
            "extreme trade-flow imbalance"
        )

    elif abs(delta_pct) >= 30:

        score += 1

        reasons.append(
            "strong trade-flow imbalance"
        )

    # --------------------------------------------------------
    # Order book
    # --------------------------------------------------------

    ob_imbalance = 0

    if ob:

        ob_imbalance = ob[
            "imbalance"
        ]

        if abs(ob_imbalance) >= 40:

            score += 2

            reasons.append(
                "extreme order-book imbalance"
            )

        elif abs(ob_imbalance) >= 25:

            score += 1

            reasons.append(
                "strong order-book imbalance"
            )

    # --------------------------------------------------------
    # Price move
    # --------------------------------------------------------

    if price_move is not None:

        if abs(price_move) >= 1:

            score += 2

            reasons.append(
                "large short-term price move"
            )

        elif abs(price_move) >= 0.5:

            score += 1

            reasons.append(
                "short-term price move"
            )

    # --------------------------------------------------------
    # Direction
    # --------------------------------------------------------

    if delta_pct > 30:

        direction = (
            "🟢 BUY AGGRESSION"
        )

    elif delta_pct < -30:

        direction = (
            "🔴 SELL AGGRESSION"
        )

    else:

        direction = (
            "⚪ BALANCED"
        )

    # --------------------------------------------------------
    # Signal
    # --------------------------------------------------------

    if score >= 5:

        signal = (
            "🔥 EXTREME LIQUIDATION-LIKE PRESSURE"
        )

    elif score >= 3:

        signal = (
            "🟠 HIGH LIQUIDATION-LIKE PRESSURE"
        )

    elif score >= 1:

        signal = (
            "🟡 WATCH"
        )

    else:

        signal = (
            "⚪ LOW"
        )

    return {

        "score":
        score,

        "signal":
        signal,

        "direction":
        direction,

        "buy_volume":
        flow["buy_volume"],

        "sell_volume":
        flow["sell_volume"],

        "delta_pct":
        delta_pct,

        "ob_imbalance":
        ob_imbalance,

        "price_move":
        price_move,

        "reasons":
        " + ".join(reasons)
        if reasons
        else "None"
    }


# ============================================================
# ATR
# ============================================================

def add_atr(
    df,
    period=14
):

    x = df.copy()

    previous_close = (
        x["close"].shift(1)
    )

    tr = pd.concat(
        [
            x["high"] -
            x["low"],

            (
                x["high"] -
                previous_close
            ).abs(),

            (
                x["low"] -
                previous_close
            ).abs()
        ],
        axis=1
    ).max(axis=1)

    x["ATR"] = (
        tr.rolling(
            period
        ).mean()
    )

    x["ATRpct"] = (
        x["ATR"]
        /
        x["close"]
        *
        100
    )

    return x


def atr_analysis(df):

    if len(df) < 25:
        return None, "⚪ UNKNOWN"

    x = add_atr(df)

    current = x[
        "ATR"
    ].iloc[-1]

    old = x[
        "ATR"
    ].iloc[-7]

    if (
        pd.isna(current)
        or
        pd.isna(old)
        or
        old == 0
    ):
        return None, "⚪ UNKNOWN"

    ratio = (
        current /
        old
    )

    if ratio >= 1.10:

        direction = (
            "🔺 ATR EXPANDING"
        )

    elif ratio <= 0.90:

        direction = (
            "🔻 ATR CONTRACTING"
        )

    else:

        direction = (
            "⚪ ATR FLAT"
        )

    return (
        float(current),
        direction
    )


# ============================================================
# VOLUME
# ============================================================

def volume_analysis(df):

    if len(df) < 10:
        return 0

    current = safe_float(
        df["volume"].iloc[-1],
        0
    )

    average = safe_float(
        df["volume"]
        .iloc[-7:-1]
        .mean(),
        0
    )

    if average <= 0:
        return 0

    return (
        current /
        average
    )


# ============================================================
# SWINGS
# ============================================================

def swings(
    df,
    left=2,
    right=2
):

    x = df.copy()

    x["SwingHigh"] = False
    x["SwingLow"] = False

    if len(x) < (
        left +
        right +
        1
    ):
        return x

    for i in range(
        left,
        len(x) - right
    ):

        current_high = (
            x["high"].iloc[i]
        )

        current_low = (
            x["low"].iloc[i]
        )

        left_high = (
            x["high"]
            .iloc[
                i-left:i
            ]
            .max()
        )

        right_high = (
            x["high"]
            .iloc[
                i+1:i+right+1
            ]
            .max()
        )

        left_low = (
            x["low"]
            .iloc[
                i-left:i
            ]
            .min()
        )

        right_low = (
            x["low"]
            .iloc[
                i+1:i+right+1
            ]
            .min()
        )

        if (
            current_high > left_high
            and
            current_high > right_high
        ):

            x.loc[
                x.index[i],
                "SwingHigh"
            ] = True

        if (
            current_low < left_low
            and
            current_low < right_low
        ):

            x.loc[
                x.index[i],
                "SwingLow"
            ] = True

    return x


# ============================================================
# CLOSED CANDLES
# ============================================================

def closed(df):

    if (
        df.empty
        or
        "time" not in df.columns
    ):
        return df

    if len(df) < 2:
        return df

    try:

        last_time = int(
            df["time"].iloc[-1]
        )

        now = int(
            time.time()
        )

        # candle still forming
        if (
            now -
            last_time
        ) < 60:

            return df.iloc[:-1].copy()

    except Exception:
        pass

    return df


# ============================================================
# TIMEFRAME TREND
# ============================================================

def timeframe_trend(df):

    if len(df) < 30:
        return "⚪ UNKNOWN"

    close = df[
        "close"
    ]

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

    current = close.iloc[-1]

    if (
        current >
        ema9.iloc[-1] >
        ema21.iloc[-1] >
        ema50.iloc[-1]
    ):

        return "🟢 BULL"

    if (
        current <
        ema9.iloc[-1] <
        ema21.iloc[-1] <
        ema50.iloc[-1]
    ):

        return "🔴 BEAR"

    return "🟡 MIXED"


# ============================================================
# MTF
# ============================================================

def mtf_state(
    t5,
    t15,
    t1
):

    states = [
        t5,
        t15,
        t1
    ]

    bulls = sum(
        x == "🟢 BULL"
        for x in states
    )

    bears = sum(
        x == "🔴 BEAR"
        for x in states
    )

    if bulls == 3:
        return "🟢 MTF ALIGNED LONG"

    if bears == 3:
        return "🔴 MTF ALIGNED SHORT"

    if bulls >= 2 and bears == 0:
        return "🟢 MTF LONG BIAS"

    if bears >= 2 and bulls == 0:
        return "🔴 MTF SHORT BIAS"

    if bulls == 0 and bears == 0:
        return "🟡 MTF RANGE"

    return "⚪ MTF CONFLICT"


# ============================================================
# 5D REGIME
# ============================================================

def regime_5d(symbol):

    df = get_candles(
        symbol,
        "1h",
        5 * 24 + 10
    )

    if len(df) < 60:

        return {
            "state":
            "⚪ UNKNOWN",

            "range_pct":
            None
        }

    x = (
        df.iloc[-120:]
        if len(df) > 120
        else df
    )

    first = safe_float(
        x["close"].iloc[0]
    )

    last = safe_float(
        x["close"].iloc[-1]
    )

    high = safe_float(
        x["high"].max()
    )

    low = safe_float(
        x["low"].min()
    )

    if (
        not np.isfinite(first)
        or
        not np.isfinite(last)
        or
        low <= 0
    ):

        return {
            "state":
            "⚪ UNKNOWN",

            "range_pct":
            None
        }

    range_pct = (
        high -
        low
    ) / low * 100

    move_pct = (
        last -
        first
    ) / first * 100

    if move_pct > 4:

        state = (
            "🟢 5D UPTREND"
        )

    elif move_pct < -4:

        state = (
            "🔴 5D DOWNTREND"
        )

    elif (
        range_pct > 12
        and
        abs(move_pct) < 3
    ):

        state = (
            "🟡 5D RANGE"
        )

    else:

        state = (
            "⚪ 5D UNCERTAINTY"
        )

    return {
        "state":
        state,

        "range_pct":
        range_pct
    }


# ============================================================
# SUPPORT RESISTANCE
# ============================================================

def sr_levels(
    df,
    lookback=100
):

    if len(df) < 20:
        return (
            np.nan,
            np.nan,
            0,
            0
        )

    x = swings(
        df.iloc[-lookback:].copy(),
        2,
        2
    )

    highs = x.loc[
        x["SwingHigh"],
        "high"
    ].tolist()

    lows = x.loc[
        x["SwingLow"],
        "low"
    ].tolist()

    price = safe_float(
        df["close"].iloc[-1]
    )

    supports = sorted(
        [
            float(v)
            for v in lows
            if v < price
        ],
        reverse=True
    )

    resistances = sorted(
        [
            float(v)
            for v in highs
            if v > price
        ]
    )

    support = (
        supports[0]
        if supports
        else np.nan
    )

    resistance = (
        resistances[0]
        if resistances
        else np.nan
    )

    return (
        support,
        resistance,
        len(supports),
        len(resistances)
    )


# ============================================================
# SWEEP BOS CHOCH FVG
# ============================================================

def sweep_bos_fvg(df):

    if len(df) < 15:
        return None

    x = swings(
        df.copy(),
        2,
        2
    )

    last = x.iloc[-1]

    swing_highs = x.loc[
        x["SwingHigh"],
        "high"
    ]

    swing_lows = x.loc[
        x["SwingLow"],
        "low"
    ]

    if len(swing_highs):

        previous_high = float(
            swing_highs.iloc[-1]
        )

    else:

        previous_high = float(
            x["high"]
            .iloc[-8:-1]
            .max()
        )

    if len(swing_lows):

        previous_low = float(
            swing_lows.iloc[-1]
        )

    else:

        previous_low = float(
            x["low"]
            .iloc[-8:-1]
            .min()
        )

    bull_sweep = (
        last["low"] <
        previous_low
        and
        last["close"] >
        previous_low
    )

    bear_sweep = (
        last["high"] >
        previous_high
        and
        last["close"] <
        previous_high
    )

    prev_high = float(
        x["high"]
        .iloc[-8:-1]
        .max()
    )

    prev_low = float(
        x["low"]
        .iloc[-8:-1]
        .min()
    )

    bull_bos = (
        last["close"] >
        prev_high
    )

    bear_bos = (
        last["close"] <
        prev_low
    )

    recent = x.iloc[-20:]

    if len(recent) >= 10:

        prior_high = float(
            recent["high"]
            .iloc[:10]
            .max()
        )

        prior_low = float(
            recent["low"]
            .iloc[:10]
            .min()
        )

    else:

        prior_high = prev_high
        prior_low = prev_low

    bull_choch = (
        last["close"] >
        prior_high
        and
        not bull_bos
    )

    bear_choch = (
        last["close"] <
        prior_low
        and
        not bear_bos
    )

    bull_fvg = (
        len(x) >= 3
        and
        x["low"].iloc[-1] >
        x["high"].iloc[-3]
    )

    bear_fvg = (
        len(x) >= 3
        and
        x["high"].iloc[-1] <
        x["low"].iloc[-3]
    )

    return {

        "bull_sweep":
        bull_sweep,

        "bear_sweep":
        bear_sweep,

        "bull_bos":
        bull_bos,

        "bear_bos":
        bear_bos,

        "bull_choch":
        bull_choch,

        "bear_choch":
        bear_choch,

        "bull_fvg":
        bull_fvg,

        "bear_fvg":
        bear_fvg,

        "swing_high":
        previous_high,

        "swing_low":
        previous_low
    }


# ============================================================
# DEEP ANALYSIS
# ============================================================

def deep_analysis(
    symbol,
    ticker
):

    d5 = closed(
        get_candles(
            symbol,
            "5m",
            36
        )
    )

    d15 = closed(
        get_candles(
            symbol,
            "15m",
            72
        )
    )

    d1 = closed(
        get_candles(
            symbol,
            "1h",
            120
        )
    )

    if min(
        len(d5),
        len(d15),
        len(d1)
    ) < 30:

        return None

    t5 = timeframe_trend(
        d5
    )

    t15 = timeframe_trend(
        d15
    )

    t1 = timeframe_trend(
        d1
    )

    mtf = mtf_state(
        t5,
        t15,
        t1
    )

    regime = regime_5d(
        symbol
    )

    support, resistance, support_count, resistance_count = (
        sr_levels(d15)
    )

    structure = sweep_bos_fvg(
        d5
    )

    if structure is None:
        return None

    atr, atr_direction = (
        atr_analysis(d5)
    )

    volume_x =
