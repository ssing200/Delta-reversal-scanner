import streamlit as st
import requests
import pandas as pd
import numpy as np
import time

BASE_URL = "https://api.india.delta.exchange"
HEADERS = {
    "Accept": "application/json",
    "User-Agent": "Delta-Reversal-Scanner/10.0"
}

CACHE_SECONDS = 30
DEEP_SCAN_LIMIT = 30
VOL_OI_MIN = 6.0
RR_DEFAULT = 2.0

st.set_page_config(
    page_title="Delta Reversal Scanner PRO 10",
    layout="wide"
)

st.title("🔥 Delta Reversal Scanner PRO 10")

st.caption(
    "MTF → 5D Regime → S/R → Sweep → BOS/CHOCH → FVG → "
    "OI → Funding → Volume → ATR → Order Book → Liquidity Proxy"
)


# ============================================================
# API
# ============================================================

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

        return data.get("result")

    except Exception:
        return None


# ============================================================
# PRODUCTS
# ============================================================

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

        if symbol:
            rows.append({
                "Coin": symbol,
                "ID": p.get("id")
            })

    return pd.DataFrame(rows).drop_duplicates("Coin")


# ============================================================
# TICKERS
# ============================================================

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
                p.get("close", p.get("mark_price", 0)) or 0
            )

            volume = float(
                p.get("volume_24h", p.get("volume", 0)) or 0
            )

            oi = float(
                p.get("open_interest", p.get("oi", 0)) or 0
            )

        except Exception:
            continue

        if price <= 0:
            continue

        raw_funding = p.get(
            "funding_rate",
            p.get("funding")
        )

        try:
            funding = (
                float(raw_funding)
                if raw_funding is not None
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


# ============================================================
# CANDLES
# ============================================================

@st.cache_data(ttl=CACHE_SECONDS)
def get_candles(symbol, resolution, hours):

    end = int(time.time())
    start = end - int(hours * 3600)

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

    for c in [
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]:

        if c in df.columns:
            df[c] = pd.to_numeric(
                df[c],
                errors="coerce"
            )

    if "time" in df.columns:
        df["time"] = pd.to_numeric(
            df["time"],
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

    return (
        df.sort_values("time")
        .drop_duplicates("time")
        .reset_index(drop=True)
    )


# ============================================================
# OI HISTORY
# ============================================================

@st.cache_data(ttl=CACHE_SECONDS)
def get_oi_history(symbol, hours=24):

    end = int(time.time())
    start = end - hours * 3600

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

    return (
        df.dropna(subset=["close"])
        .sort_values("time")
        .reset_index(drop=True)
    )


# ============================================================
# DELTA L2 ORDER BOOK
# ============================================================

@st.cache_data(ttl=10)
def get_orderbook(symbol, depth=15):

    result = api_get(
        f"/v2/l2orderbook/{symbol}",
        {"depth": depth}
    )

    if not result:
        return None

    return result


def orderbook_analysis(symbol, depth=15):

    data = get_orderbook(symbol, depth)

    if not data:
        return None

    bids = data.get("buy", [])
    asks = data.get("sell", [])

    if not bids or not asks:
        return None

    bid_rows = []
    ask_rows = []

    for x in bids:

        try:
            bid_rows.append({
                "Price": float(x["price"]),
                "Size": float(x["size"])
            })
        except Exception:
            pass

    for x in asks:

        try:
            ask_rows.append({
                "Price": float(x["price"]),
                "Size": float(x["size"])
            })
        except Exception:
            pass

    if not bid_rows or not ask_rows:
        return None

    bid_df = pd.DataFrame(bid_rows)
    ask_df = pd.DataFrame(ask_rows)

    bid_depth = bid_df["Size"].sum()
    ask_depth = ask_df["Size"].sum()

    total = bid_depth + ask_depth

    imbalance = (
        (bid_depth - ask_depth) / total * 100
        if total > 0 else 0
    )

    best_bid = bid_df["Price"].max()
    best_ask = ask_df["Price"].min()

    spread = best_ask - best_bid

    mid = (best_bid + best_ask) / 2

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
# PUBLIC TRADES
# ============================================================

@st.cache_data(ttl=10)
def get_recent_trades(symbol):

    result = api_get(
        f"/v2/trades/{symbol}"
    )

    if not result:
        return pd.DataFrame()

    trades = result.get("trades", [])

    if not trades:
        return pd.DataFrame()

    df = pd.DataFrame(trades)

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

    return df.dropna(
        subset=["price", "size"]
    )


def trade_flow_analysis(symbol):

    df = get_recent_trades(symbol)

    if df.empty:
        return None

    buy_volume = 0
    sell_volume = 0

    if "side" in df.columns:

        buy_volume = df.loc[
            df["side"].astype(str).str.lower() == "buy",
            "size"
        ].sum()

        sell_volume = df.loc[
            df["side"].astype(str).str.lower() == "sell",
            "size"
        ].sum()

    total = buy_volume + sell_volume

    if total <= 0:
        return None

    delta = buy_volume - sell_volume

    delta_pct = delta / total * 100

    return {
        "buy_volume": buy_volume,
        "sell_volume": sell_volume,
        "delta": delta,
        "delta_pct": delta_pct,
        "trades": len(df)
    }


# ============================================================
# LIQUIDITY / ORDER BOOK ANALYSIS
# ============================================================

def liquidity_analysis(symbol):

    ob = orderbook_analysis(symbol, 15)

    if not ob:
        return None

    bid_df = ob["bid_df"]
    ask_df = ob["ask_df"]

    # Largest visible bid/ask walls
    largest_bid = bid_df.loc[
        bid_df["Size"].idxmax()
    ]

    largest_ask = ask_df.loc[
        ask_df["Size"].idxmax()
    ]

    price = ob["mid"]

    bid_wall_distance = (
        (price - largest_bid["Price"])
        / price * 100
    )

    ask_wall_distance = (
        (largest_ask["Price"] - price)
        / price * 100
    )

    return {
        "imbalance": ob["imbalance"],
        "bid_depth": ob["bid_depth"],
        "ask_depth": ob["ask_depth"],
        "best_bid": ob["best_bid"],
        "best_ask": ob["best_ask"],
        "spread": ob["spread"],
        "largest_bid_price": largest_bid["Price"],
        "largest_bid_size": largest_bid["Size"],
        "largest_ask_price": largest_ask["Price"],
        "largest_ask_size": largest_ask["Size"],
        "bid_wall_distance": bid_wall_distance,
        "ask_wall_distance": ask_wall_distance
    }


# ============================================================
# LIQUIDATION PROXY
# ============================================================

def liquidation_proxy(symbol):

    """
    IMPORTANT:

    This is NOT actual liquidation data.

    Delta public API gives public trades and orderbook.
    Actual liquidation reason is exposed through
    account-specific v2/user_trades.

    Therefore this function estimates liquidation-like
    pressure using:

    1. Large aggressive trade flow
    2. Strong buy/sell imbalance
    3. Orderbook imbalance
    4. Recent price movement
    """

    flow = trade_flow_analysis(symbol)

    if not flow:
        return None

    ob = orderbook_analysis(symbol, 15)

    score = 0
    signal = "⚪ LOW"

    if flow["delta_pct"] > 30:
        score += 2

    if flow["delta_pct"] < -30:
        score += 2

    if ob:

        if abs(ob["imbalance"]) > 30:
            score += 1

    if score >= 3:

        if flow["delta_pct"] > 30:
            signal = "🟢 BUY-SIDE LIQUIDATION PROXY"

        elif flow["delta_pct"] < -30:
            signal = "🔴 SELL-SIDE LIQUIDATION PROXY"

        else:
            signal = "🟡 HIGH LIQUIDATION-LIKE PRESSURE"

    elif score >= 1:
        signal = "🟡 WATCH"

    return {
        "score": score,
        "signal": signal,
        "buy_volume": flow["buy_volume"],
        "sell_volume": flow["sell_volume"],
        "delta_pct": flow["delta_pct"]
    }


# ============================================================
# INDICATORS
# ============================================================

def add_atr(df, period=14):

    x = df.copy()

    pc = x["close"].shift(1)

    tr = pd.concat(
        [
            x["high"] - x["low"],
            (x["high"] - pc).abs(),
            (x["low"] - pc).abs()
        ],
        axis=1
    ).max(axis=1)

    x["ATR"] = tr.rolling(period).mean()

    x["ATRpct"] = (
        x["ATR"] /
        x["close"] *
        100
    )

    return x


def swings(df, left=2, right=2):

    x = df.copy()

    x["SwingHigh"] = False
    x["SwingLow"] = False

    for i in range(left, len(x) - right):

        if (
            x["high"].iloc[i]
            >
            x["high"].iloc[
                i-left:i
            ].max()
            and
            x["high"].iloc[i]
            >
            x["high"].iloc[
                i+1:i+right+1
            ].max()
        ):
            x.loc[
                x.index[i],
                "SwingHigh"
            ] = True

        if (
            x["low"].iloc[i]
            <
            x["low"].iloc[
                i-left:i
            ].min()
            and
            x["low"].iloc[i]
            <
            x["low"].iloc[
                i+1:i+right+1
            ].min()
        ):
            x.loc[
                x.index[i],
                "SwingLow"
            ] = True

    return x


def closed(df):

    if df.empty or "time" not in df.columns:
        return df

    now = int(time.time())

    if len(df) >= 2:

        last_t = int(
            df["time"].iloc[-1]
        )

        if now - last_t < 60:
            return df.iloc[:-1].copy()

    return df


def timeframe_trend(df):

    if len(df) < 30:
        return "⚪ UNKNOWN"

    c = df["close"]

    e9 = c.ewm(
        span=9,
        adjust=False
    ).mean()

    e21 = c.ewm(
        span=21,
        adjust=False
    ).mean()

    e50 = c.ewm(
        span=50,
        adjust=False
    ).mean()

    if (
        c.iloc[-1]
        >
        e9.iloc[-1]
        >
        e21.iloc[-1]
        >
        e50.iloc[-1]
    ):
        return "🟢 BULL"

    if (
        c.iloc[-1]
        <
        e9.iloc[-1]
        <
        e21.iloc[-1]
        <
        e50.iloc[-1]
    ):
        return "🔴 BEAR"

    return "🟡 MIXED"


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
            "state": "⚪ UNKNOWN",
            "range_pct": None
        }

    x = (
        df.iloc[-120:]
        if len(df) > 120
        else df
    )

    first = x["close"].iloc[0]
    last = x["close"].iloc[-1]

    hi = x["high"].max()
    lo = x["low"].min()

    rng = (
        (hi - lo) /
        lo *
        100
        if lo else 0
    )

    move = (
        (last - first) /
        first *
        100
        if first else 0
    )

    if move > 4:
        state = "🟢 5D UPTREND"

    elif move < -4:
        state = "🔴 5D DOWNTREND"

    elif rng > 12 and abs(move) < 3:
        state = "🟡 5D RANGE"

    else:
        state = "⚪ 5D UNCERTAINTY"

    return {
        "state": state,
        "range_pct": rng
    }


# ============================================================
# SUPPORT / RESISTANCE
# ============================================================

def sr_levels(
    df,
    lookback=80,
    tol=0.006
):

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

    price = float(
        df["close"].iloc[-1]
    )

    supports = sorted(
        [
            v for v in lows
            if v < price
        ],
        reverse=True
    )

    resistances = sorted(
        [
            v for v in highs
            if v > price
        ]
    )

    support = (
        supports[0]
        if supports else np.nan
    )

    resistance = (
        resistances[0]
        if resistances else np.nan
    )

    return (
        support,
        resistance,
        len(supports),
        len(resistances)
    )


# ============================================================
# SWEEP BOS FVG
# ============================================================

def sweep_bos_fvg(df):

    x = swings(df, 2, 2)

    last = x.iloc[-1]

    sh = x.loc[
        x["SwingHigh"],
        "high"
    ]

    sl = x.loc[
        x["SwingLow"],
        "low"
    ]

    ph = (
        sh.iloc[-1]
        if len(sh)
        else x["high"].iloc[-8:-1].max()
    )

    pl = (
        sl.iloc[-1]
        if len(sl)
        else x["low"].iloc[-8:-1].min()
    )

    bull_sweep = (
        last["low"] < pl
        and
        last["close"] > pl
    )

    bear_sweep = (
        last["high"] > ph
        and
        last["close"] < ph
    )

    prev_h = x["high"].iloc[-8:-1].max()
    prev_l = x["low"].iloc[-8:-1].min()

    bull_bos = last["close"] > prev_h
    bear_bos = last["close"] < prev_l

    recent = x.iloc[-20:]

    if len(recent) >= 10:

        prior_high = (
            recent["high"]
            .iloc[:10]
            .max()
        )

        prior_low = (
            recent["low"]
            .iloc[:10]
            .min()
        )

    else:

        prior_high = prev_h
        prior_low = prev_l

    bull_choch = (
        last["close"] > prior_high
        and
        not bull_bos
    )

    bear_choch = (
        last["close"] < prior_low
        and
        not bear_bos
    )

    bull_fvg = (
        len(x) >= 3
        and
        x["low"].iloc[-1]
        >
        x["high"].iloc[-3]
    )

    bear_fvg = (
        len(x) >= 3
        and
        x["high"].iloc[-1]
        <
        x["low"].iloc[-3]
    )

    return {

        "bull_sweep": bull_sweep,
        "bear_sweep": bear_sweep,

        "bull_bos": bull_bos,
        "bear_bos": bear_bos,

        "bull_choch": bull_choch,
        "bear_choch": bear_choch,

        "bull_fvg": bull_fvg,
        "bear_fvg": bear_fvg,

        "swing_high": float(ph),
        "swing_low": float(pl)
    }


# ============================================================
# OI
# ============================================================

def oi_analysis(symbol):

    df = get_oi_history(
        symbol,
        24
    )

    if len(df) < 7:
        return None, "⚪ UNKNOWN"

    cur = float(
        df["close"].iloc[-1]
    )

    old = float(
        df["close"].iloc[-7]
    )

    if old == 0:
        return None, "⚪ UNKNOWN"

    ch = (
        (cur - old) /
        abs(old) *
        100
    )

    if ch >= 1:
        sig = "🔺 OI UP"

    elif ch <= -1:
        sig = "🔻 OI DOWN"

    else:
        sig = "⚪ OI FLAT"

    return ch, sig


# ============================================================
# VOLUME
# ============================================================

def volume_analysis(df):

    if len(df) < 10:
        return 0

    avg = (
        df["volume"]
        .iloc[-7:-1]
        .mean()
    )

    return (
        float(
            df["volume"].iloc[-1]
            /
            avg
        )
        if avg > 0
        else 0
    )


# ============================================================
# ATR
# ============================================================

def atr_analysis(df):

    x = add_atr(df)

    if (
        len(x) < 25
        or
        pd.isna(
            x["ATR"].iloc[-1]
        )
        or
        pd.isna(
            x["ATR"].iloc[-7]
        )
    ):
        return None, "⚪ UNKNOWN"

    a = float(
        x["ATR"].iloc[-1]
    )

    old = float(
        x["ATR"].iloc[-7]
    )

    d = (
        a / old
        if old
        else 1
    )

    if d >= 1.10:
        direction = "🔺 ATR EXPANDING"

    elif d <= 0.90:
        direction = "🔻 ATR CONTRACTING"

    else:
        direction = "⚪ ATR FLAT"

    return a, direction


# ============================================================
# MTF
# ============================================================

def mtf_state(
    t5,
    t15,
    t1
):

    bulls = sum(
        v == "🟢 BULL"
        for v in [t5, t15, t1]
    )

    bears = sum(
        v == "🔴 BEAR"
        for v in [t5, t15, t1]
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
        return "🟡 MTF RANGE/MIXED"

    return "⚪ MTF CONFLICT"


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
    ) < 25:

        return None

    t5 = timeframe_trend(d5)
    t15 = timeframe_trend(d15)
    t1 = timeframe_trend(d1)

    mtf = mtf_state(
        t5,
        t15,
        t1
    )

    reg = regime_5d(symbol)

    sr_sup, sr_res, sup_count, res_count = (
        sr_levels(d15)
    )

    struct = sweep_bos_fvg(d5)

    atr, atr_dir = atr_analysis(d5)

    volx = volume_analysis(d5)

    oi_ch, oi_sig = oi_analysis(symbol)

    ob = liquidity_analysis(symbol)

    liq = liquidation_proxy(symbol)

    funding = ticker.get("Funding")

    fp = (
        float(funding) * 100
        if funding is not None
        else None
    )

    price = float(
        ticker["Price"]
    )

    long_score = 0
    short_score = 0

    lr = []
    sr = []

    # --------------------------------------------------------
    # MTF
    # --------------------------------------------------------

    if mtf == "🟢 MTF ALIGNED LONG":

        long_score += 4
        lr.append("5m+15m+1H aligned")

    elif mtf == "🔴 MTF ALIGNED SHORT":

        short_score += 4
        sr.append("5m+15m+1H aligned")

    elif mtf == "🟢 MTF LONG BIAS":

        long_score += 2
        lr.append("MTF long bias")

    elif mtf == "🔴 MTF SHORT BIAS":

        short_score += 2
        sr.append("MTF short bias")

    elif mtf == "⚪ MTF CONFLICT":

        long_score -= 2
        short_score -= 2

    # --------------------------------------------------------
    # 5D
    # --------------------------------------------------------

    if "UPTREND" in reg["state"]:

        long_score += 2
        lr.append("5D uptrend")

    elif "DOWNTREND" in reg["state"]:

        short_score += 2
        sr.append("5D downtrend")

    elif "RANGE" in reg["state"]:

        long_score -= 1
        short_score -= 1

    # --------------------------------------------------------
    # STRUCTURE
    # --------------------------------------------------------

    if struct["bull_sweep"]:

        long_score += 2
        lr.append("bull liquidity sweep")

    if struct["bear_sweep"]:

        short_score += 2
        sr.append("bear liquidity sweep")

    if struct["bull_bos"]:

        long_score += 3
        lr.append("bull BOS")

    if struct["bear_bos"]:

        short_score += 3
        sr.append("bear BOS")

    if struct["bull_choch"]:

        long_score += 2
        lr.append("bull CHOCH")

    if struct["bear_choch"]:

        short_score += 2
        sr.append("bear CHOCH")

    if struct["bull_fvg"]:

        long_score += 2
        lr.append("bull FVG")

    if struct["bear_fvg"]:

        short_score += 2
        sr.append("bear FVG")

    # --------------------------------------------------------
    # S/R
    # --------------------------------------------------------

    if not pd.isna(sr_sup):

        dist = abs(
            price - sr_sup
        ) / price

        if dist <= 0.01:

            long_score += 2
            lr.append("near support")

    if not pd.isna(sr_res):

        dist = abs(
            sr_res - price
        ) / price

        if dist <= 0.01:

            short_score += 2
            sr.append("near resistance")

    # --------------------------------------------------------
    # VOLUME
    # --------------------------------------------------------

    if volx >= 2:

        long_score += 2
        short_score += 2

        lr.append("volume spike")
        sr.append("volume spike")

    elif volx >= 1.3:

        long_score += 1
        short_score += 1

    # --------------------------------------------------------
    # OI
    # --------------------------------------------------------

    if oi_ch is not None:

        if oi_ch >= 1:

            if mtf.startswith("🟢"):

                long_score += 2
                lr.append("OI expansion")

            if mtf.startswith("🔴"):

                short_score += 2
                sr.append("OI expansion")

        elif oi_ch <= -1:

            if struct["bull_sweep"]:

                long_score += 1
                lr.append(
                    "OI unwind after bull sweep"
                )

            if struct["bear_sweep"]:

                short_score += 1
                sr.append(
                    "OI unwind after bear sweep"
                )

    # --------------------------------------------------------
    # FUNDING
    # --------------------------------------------------------

    if fp is not None:

        if fp >= 0.05:

            short_score += 2

            sr.append(
                "positive funding crowding"
            )

            funding_signal = (
                "🔴 Long crowded"
            )

        elif fp <= -0.05:

            long_score += 2

            lr.append(
                "negative funding crowding"
            )

            funding_signal = (
                "🟢 Short crowded"
            )

        else:

            funding_signal = "⚪ Neutral"

    else:

        funding_signal = "⚪ Unavailable"

    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

    if atr_dir == "🔺 ATR EXPANDING":

        long_score += 1
        short_score += 1

    elif atr_dir == "🔻 ATR CONTRACTING":

        long_score -= 1
        short_score -= 1

    # --------------------------------------------------------
    # ORDER BOOK
    # --------------------------------------------------------

    if ob:

        imbalance = ob["imbalance"]

        if imbalance >= 25:

            long_score += 2
            lr.append("orderbook bid imbalance")

        elif imbalance <= -25:

            short_score += 2
            sr.append("orderbook ask imbalance")

    # --------------------------------------------------------
    # LIQUIDATION PROXY
    # --------------------------------------------------------

    if liq:

        if (
            "BUY-SIDE" in liq["signal"]
        ):

            long_score += 1
            lr.append(
                "buy-side liquidation proxy"
            )

        elif (
            "SELL-SIDE" in liq["signal"]
        ):

            short_score += 1
            sr.append(
                "sell-side liquidation proxy"
            )

    # --------------------------------------------------------
    # SIGNAL
    # --------------------------------------------------------

    blocked = (
        mtf == "⚪ MTF CONFLICT"
    )

    if blocked:

        signal = "⛔ MTF CONFLICT"

    elif (
        long_score > short_score
        and
        long_score >= 8
    ):

        signal = "🟢 STRONG LONG"

    elif (
        short_score > long_score
        and
        short_score >= 8
    ):

        signal = "🔴 STRONG SHORT"

    elif (
        long_score > short_score
        and
        long_score >= 5
    ):

        signal = "🟡 LONG WATCH"

    elif (
        short_score > long_score
        and
        short_score >= 5
    ):

        signal = "🟠 SHORT WATCH"

    else:

        signal = "⚪ NO SIGNAL"

    score = max(
        long_score,
        short_score
    )

    return {

        "Coin": symbol,
        "Price": price,

        "24H Volume": ticker["24H Volume"],
        "OI": ticker["OI"],
        "Vol/OI": round(
            float(ticker["Vol/OI"]),
            2
        ),

        "5m": t5,
        "15m": t15,
        "1H": t1,

        "MTF": mtf,

        "5D Regime": reg["state"],

        "5D Range %": (
            round(
                reg["range_pct"],
                2
            )
            if reg["range_pct"]
            else None
        ),

        "Support": (
            round(sr_sup, 8)
            if not pd.isna(sr_sup)
            else None
        ),

        "Resistance": (
            round(sr_res, 8)
            if not pd.isna(sr_res)
            else None
        ),

        "S/R Count":
        f"{sup_count}/{res_count}",

        "Liquidity":
        (
            "🟢 BULL SWEEP"
            if struct["bull_sweep"]
            else
            "🔴 BEAR SWEEP"
            if struct["bear_sweep"]
            else
            "⚪ None"
        ),

        "BOS":
        (
            "🟢 BULL BOS"
            if struct["bull_bos"]
            else
            "🔴 BEAR BOS"
            if struct["bear_bos"]
            else
            "⚪ None"
        ),

        "CHOCH":
        (
            "🟢 BULL CHOCH"
            if struct["bull_choch"]
            else
            "🔴 BEAR CHOCH"
            if struct["bear_choch"]
            else
            "⚪ None"
        ),

        "FVG":
        (
            "🟢 BULL FVG"
            if struct["bull_fvg"]
            else
            "🔴 BEAR FVG"
            if struct["bear_fvg"]
            else
            "⚪ None"
        ),

        "ATR":
        round(atr, 8)
        if atr
        else None,

        "ATR Direction":
        atr_dir,

        "Volume x":
        round(volx, 2),

        "OI Change %":
        round(oi_ch, 2)
        if oi_ch is not None
        else None,

        "OI Signal":
        oi_sig,

        "Funding %":
        round(fp, 4)
        if fp is not None
        else None,

        "Funding":
        funding_signal,

        "OB Imbalance %":
        round(
            ob["imbalance"],
            2
        )
        if ob
        else None,

        "Bid Depth":
        round(
            ob["bid_depth"],
            2
        )
        if ob
        else None,

        "Ask Depth":
        round(
            ob["ask_depth"],
            2
        )
        if ob
        else None,

        "Liq Proxy":
        liq["signal"]
        if liq
        else "⚪ Unknown",

        "Liq Proxy Score":
        liq["score"]
        if liq
        else 0,

        "Long Score":
        long_score,

        "Short Score":
        short_score,

        "Score":
        score,

        "Signal":
        signal,

        "Long Reason":
        " + ".join(lr)
        if lr
        else "None",

        "Short Reason":
        " + ".join(sr)
        if sr
        else "None"
    }


# ============================================================
# MAIN MARKET
# ============================================================

coins = get_all_perpetuals()
tickers = get_tickers()

if (
    coins.empty
    or
    tickers.empty
):

    st.error(
        "❌ Market data load nahi hua."
    )

    st.stop()


market = (
    coins
    .merge(
        tickers,
        on="Coin",
        how="left"
    )
    .dropna(
        subset=["Price"]
    )
)

market = market[
    market["Vol/OI"] > VOL_OI_MIN
]

market = market.sort_values(
    "24H Volume",
    ascending=False
)


st.metric(
    "Coins after Vol/OI > 6",
    len(market)
)


# ============================================================
# MODE
# ============================================================

mode = st.radio(
    "Mode",
    [
        "🔥 Live Scanner",
        "📚 Order Book / Liquidity",
        "💥 Liquidation Proxy",
        "📊 Backtest"
    ],
    horizontal=True
)


# ============================================================
# LIVE SCANNER
# ============================================================

if mode == "🔥 Live Scanner":

    candidates = market.head(
        DEEP_SCAN_LIMIT
    )

    st.info(
        f"Vol/OI > 6 wale top "
        f"{len(candidates)} coins deep scan honge."
    )

    results = []

    bar = st.progress(0)

    for i, (_, row) in enumerate(
        candidates.iterrows()
    ):

        r = deep_analysis(
            row["Coin"],
            row
        )

        if r:
            results.append(r)

        bar.progress(
            int(
                (i + 1)
                /
                len(candidates)
                *
                100
            )
        )

    bar.empty()

    sig = pd.DataFrame(results)

    if sig.empty:

        st.warning(
            "Signal data nahi mila."
        )

    else:

        st.subheader(
            "🎯 Complete Scanner"
        )

        st.dataframe(
            sig.sort_values(
                "Score",
                ascending=False
            ),
            use_container_width=True,
            hide_index=True
        )

        st.subheader(
            "🟢 LONG"
        )

        st.dataframe(
            sig[
                [
                    "Coin",
                    "Price",
                    "MTF",
                    "5D Regime",
                    "Support",
                    "Resistance",
                    "Liquidity",
                    "BOS",
                    "CHOCH",
                    "FVG",
                    "ATR Direction",
                    "Volume x",
                    "OI Change %",
                    "Funding %",
                    "OB Imbalance %",
                    "Liq Proxy",
                    "Long Score",
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
            sig[
                [
                    "Coin",
                    "Price",
                    "MTF",
                    "5D Regime",
                    "Support",
                    "Resistance",
                    "Liquidity",
                    "BOS",
                    "CHOCH",
                    "FVG",
                    "ATR Direction",
                    "Volume x",
                    "OI Change %",
                    "Funding %",
                    "OB Imbalance %",
                    "Liq Proxy",
                    "Short Score",
                    "Short Reason"
                ]
            ].sort_values(
                "Short Score",
                ascending=False
            ),
            use_container_width=True,
            hide_index=True
        )

        strong = sig[
            (sig["Score"] >= 8)
            &
            (~sig["Signal"]
             .str.contains(
                 "CONFLICT",
                 na=False
             ))
        ]

        st.subheader(
            "🔥 STRONG 8+"
        )

        if not strong.empty:

            st.dataframe(
                strong,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info(
                "Abhi 8+ aligned setup nahi mila."
            )


# ============================================================
# ORDER BOOK / LIQUIDITY PAGE
# ============================================================

elif mode == "📚 Order Book / Liquidity":

    st.subheader(
        "📚 Delta L2 Order Book + Liquidity"
    )

    available = (
        market["Coin"]
        .head(50)
        .tolist()
    )

    if not available:

        st.warning(
            "Coins available nahi hain."
        )

    else:

        selected = st.selectbox(
            "Coin select karo",
            available
        )

        if st.button(
            "🔍 Analyze Order Book"
        ):

            ob = orderbook_analysis(
                selected,
                15
            )

            if not ob:

                st.error(
                    "Order book data nahi mila."
                )

            else:

                c1, c2, c3, c4 = st.columns(4)

                c1.metric(
                    "Best Bid",
                    f'{ob["best_bid"]:.8f}'
                )

                c2.metric(
                    "Best Ask",
                    f'{ob["best_ask"]:.8f}'
                )

                c3.metric(
                    "Bid Depth",
                    f'{ob["bid_depth"]:,.0f}'
                )

                c4.metric(
                    "Ask Depth",
                    f'{ob["ask_depth"]:,.0f}'
                )

                imbalance = ob["imbalance"]

                if imbalance > 25:

                    st.success(
                        f"🟢 Strong Bid Imbalance: "
                        f"{imbalance:.2f}%"
                    )

                elif imbalance < -25:

                    st.error(
                        f"🔴 Strong Ask Imbalance: "
                        f"{imbalance:.2f}%"
                    )

                else:

                    st.info(
                        f"⚪ Balanced Order Book: "
                        f"{imbalance:.2f}%"
                    )

                left, right = st.columns(2)

                with left:

                    st.subheader(
                        "🟢 BIDS"
                    )

                    bid_display = (
                        ob["bid_df"]
                        .sort_values(
                            "Price",
                            ascending=False
                        )
                    )

                    st.dataframe(
                        bid_display,
                        use_container_width=True,
                        hide_index=True
                    )

                with right:

                    st.subheader(
                        "🔴 ASKS"
                    )

                    ask_display = (
                        ob["ask_df"]
                        .sort_values(
                            "Price",
                            ascending=True
                        )
                    )

                    st.dataframe(
                        ask_display,
                        use_container_width=True,
                        hide_index=True
                    )

                st.subheader(
                    "🧱 Visible Liquidity Walls"
                )

                largest_bid = (
                    ob["bid_df"]
                    .loc[
                        ob["bid_df"]["Size"]
                        .idxmax()
                    ]
                )

                largest_ask = (
                    ob["ask_df"]
                    .loc[
                        ob["ask_df"]["Size"]
                        .idxmax()
                    ]
                )

                w1, w2 = st.columns(2)

                with w1:

                    st.metric(
                        "Largest Bid Wall",
                        f'{largest_bid["Price"]:.8f}'
                    )

                    st.write(
                        f'Size: '
                        f'{largest_bid["Size"]:,.0f}'
                    )

                with w2:

                    st.metric(
                        "Largest Ask Wall",
                        f'{largest_ask["Price"]:.8f}'
                    )

                    st.write(
                        f'Size: '
                        f'{largest_ask["Size"]:,.0f}'
                    )

                st.warning(
                    "⚠️ Order-book wall ko guaranteed "
                    "support/resistance mat samjho. "
                    "Orders cancel bhi ho sakte hain."
                )


# ============================================================
# LIQUIDATION PROXY PAGE
# ============================================================

elif mode == "💥 Liquidation Proxy":

    st.subheader(
        "💥 Liquidation Pressure Scanner"
    )

    st.warning(
        "Important: Delta public API se sab traders ki "
        "actual liquidation feed available nahi hai. "
        "Neeche ka signal PUBLIC TRADES + ORDER BOOK se "
        "liquidation-like pressure ka proxy hai."
    )

    candidates = market.head(30)

    rows = []

    bar = st.progress(0)

    for i, (_, row) in enumerate(
        candidates.iterrows()
    ):

        symbol = row["Coin"]

        liq = liquidation_proxy(
            symbol
        )

        if liq:

            ob = orderbook_analysis(
                symbol,
                15
            )

            rows.append({

                "Coin": symbol,

                "Price":
                row["Price"],

                "Buy Volume":
                round(
                    liq["buy_volume"],
                    2
                ),

                "Sell Volume":
                round(
                    liq["sell_volume"],
                    2
                ),

                "Trade Delta %":
                round(
                    liq["delta_pct"],
                    2
                ),

                "OB Imbalance %":
                round(
                    ob["imbalance"],
                    2
                )
                if ob
                else None,

                "Liquidation Proxy":
                liq["signal"],

                "Proxy Score":
                liq["score"]
            })

        bar.progress(
            int(
                (i + 1)
                /
                len(candidates)
                *
                100
            )
        )

    bar.empty()

    if rows:

        liq_df = pd.DataFrame(
            rows
        )

        liq_df = liq_df.sort_values(
            "Proxy Score",
            ascending=False
        )

        st.dataframe(
            liq_df,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "Liquidation proxy data nahi mila."
        )


# ============================================================
# BACKTEST
# ============================================================

else:

    st.subheader(
        "📊 Historical Backtest"
    )

    st.info(
        "Backtest me order-book aur liquidation proxy "
        "historical snapshots available na hone ki wajah "
        "se include nahi kiye gaye hain."
    )

    st.write(
        "Tumhara original backtest logic yahan "
        "baad me order-book history ke saath "
        "alag se upgrade kiya ja sakta hai."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.write(
    """
### Scanner Logic

Vol/OI > 6  
↓  
5m / 15m / 1H alignment  
↓  
5D regime  
↓  
Support / Resistance  
↓  
Liquidity Sweep  
↓  
BOS / CHOCH  
↓  
FVG  
↓  
OI displacement  
↓  
Funding  
↓  
Volume  
↓  
ATR  
↓  
**Delta L2 Order Book**  
↓  
**Bid/Ask Imbalance**  
↓  
**Visible Liquidity Walls**  
↓  
**Public Trade Flow**  
↓  
**Liquidation Pressure Proxy**
"""
)

st.caption(
    "⚠️ This is an analytical scanner, not a guaranteed "
    "trade signal. Order-book liquidity can disappear, "
    "and liquidation proxy is not actual liquidation data."
)


# ============================================================
# REFRESH
# ============================================================

if st.button(
    "🔄 Refresh Scanner"
):

    st.cache_data.clear()
    st.rerun()
