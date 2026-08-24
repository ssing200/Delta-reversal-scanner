import streamlit as st
import requests
import pandas as pd
import numpy as np
import time

# ============================================================
# DELTA REVERSAL SCANNER PRO
# LEVERAGE SPLIT:
#   1) EXACT 20X  -> Vol/OI > 6 required
#   2) >20X       -> NO Vol/OI filter
# ============================================================

BASE_URL = "https://api.india.delta.exchange"
HEADERS = {
    "Accept": "application/json",
    "User-Agent": "Delta-Reversal-Scanner/11.0"
}

CACHE_SECONDS = 30
ORDERBOOK_CACHE = 8
DEEP_SCAN_LIMIT = 30
VOL_OI_MIN = 6.0

st.set_page_config(
    page_title="Delta Reversal Scanner PRO 11",
    layout="wide"
)

st.title("🔥 Delta Reversal Scanner PRO 11")
st.caption(
    "Leverage Split → MTF → 5D Regime → S/R → Sweep → BOS/CHOCH → "
    "FVG → OI → Funding → Volume → ATR → Delta L2 Order Book"
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
# HELPERS
# ============================================================

def first_number(obj, keys):
    """Return the first usable numeric field from a dict."""
    for key in keys:
        value = obj.get(key)
        if value is None:
            continue
        try:
            n = float(value)
            if np.isfinite(n):
                return n
        except Exception:
            pass
    return None


def extract_leverage(product):
    """
    Delta product schemas can vary by contract/version.
    We check several common public product fields.

    IMPORTANT:
    This is contract/default/max leverage metadata, NOT the
    user's current position leverage.
    """
    direct = first_number(
        product,
        [
            "default_leverage",
            "max_leverage",
            "leverage",
            "default_leverage_value"
        ]
    )

    if direct is not None and direct > 0:
        return direct

    # Sometimes leverage information may be nested.
    for parent_key in [
        "contract",
        "risk_limits",
        "margin",
        "product"
    ]:
        parent = product.get(parent_key)
        if isinstance(parent, dict):
            nested = first_number(
                parent,
                [
                    "default_leverage",
                    "max_leverage",
                    "leverage"
                ]
            )
            if nested is not None and nested > 0:
                return nested

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
        if not symbol:
            continue

        leverage = extract_leverage(p)

        rows.append({
            "Coin": symbol,
            "ID": p.get("id"),
            "Leverage": leverage,
            "Leverage Source": (
                "Product metadata"
                if leverage is not None
                else "Unavailable"
            )
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

    rows = []

    for p in result:
        symbol = p.get("symbol")
        if not symbol:
            continue

        price = first_number(
            p,
            ["close", "mark_price", "spot_price"]
        )

        volume = first_number(
            p,
            ["volume_24h", "volume"]
        )

        oi = first_number(
            p,
            ["open_interest", "oi"]
        )

        if price is None or price <= 0:
            continue

        if volume is None:
            volume = 0.0

        if oi is None:
            oi = 0.0

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

    for c in ["open", "high", "low", "close", "volume"]:
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

    required = ["open", "high", "low", "close"]

    if not all(c in df.columns for c in required):
        return pd.DataFrame()

    df = df.dropna(subset=required)

    if "time" in df.columns:
        df = (
            df.sort_values("time")
            .drop_duplicates("time")
        )

    return df.reset_index(drop=True)


def closed(df):
    if df.empty or "time" not in df.columns:
        return df

    if len(df) < 2:
        return df

    now = int(time.time())
    last_t = int(df["time"].iloc[-1])

    # Delta candle timestamps may be seconds.
    # Remove the current candle only when it is clearly open.
    if now - last_t < 60:
        return df.iloc[:-1].copy()

    return df


# ============================================================
# ORDER BOOK - DELTA L2
# ============================================================

@st.cache_data(ttl=ORDERBOOK_CACHE)
def get_orderbook(symbol, depth=15):
    result = api_get(
        f"/v2/l2orderbook/{symbol}",
        {"depth": depth}
    )

    return result


def _book_rows(items):
    rows = []

    if not isinstance(items, list):
        return pd.DataFrame(columns=["Price", "Size"])

    for x in items:
        try:
            if isinstance(x, dict):
                price = float(x.get("price"))
                size = float(
                    x.get(
                        "size",
                        x.get("quantity", 0)
                    )
                )
            elif isinstance(x, (list, tuple)) and len(x) >= 2:
                price = float(x[0])
                size = float(x[1])
            else:
                continue

            if price > 0 and size >= 0:
                rows.append({
                    "Price": price,
                    "Size": size
                })
        except Exception:
            continue

    return pd.DataFrame(rows)


def orderbook_analysis(symbol, depth=15):
    data = get_orderbook(symbol, depth)

    if not data:
        return None

    # Delta public L2 normally exposes buy/sell.
    # Keep fallbacks for alternate response shapes.
    bids = (
        data.get("buy")
        or data.get("bids")
        or []
    )

    asks = (
        data.get("sell")
        or data.get("asks")
        or []
    )

    bid_df = _book_rows(bids)
    ask_df = _book_rows(asks)

    if bid_df.empty or ask_df.empty:
        return None

    bid_depth = float(bid_df["Size"].sum())
    ask_depth = float(ask_df["Size"].sum())

    total = bid_depth + ask_depth

    imbalance = (
        (bid_depth - ask_depth) / total * 100
        if total > 0 else 0
    )

    best_bid = float(bid_df["Price"].max())
    best_ask = float(ask_df["Price"].min())
    spread = best_ask - best_bid
    mid = (best_bid + best_ask) / 2

    largest_bid = bid_df.loc[
        bid_df["Size"].idxmax()
    ]

    largest_ask = ask_df.loc[
        ask_df["Size"].idxmax()
    ]

    return {
        "bid_df": bid_df,
        "ask_df": ask_df,
        "bid_depth": bid_depth,
        "ask_depth": ask_depth,
        "imbalance": imbalance,
        "best_bid": best_bid,
        "best_ask": best_ask,
        "spread": spread,
        "mid": mid,
        "largest_bid_price": float(largest_bid["Price"]),
        "largest_bid_size": float(largest_bid["Size"]),
        "largest_ask_price": float(largest_ask["Price"]),
        "largest_ask_size": float(largest_ask["Size"])
    }


# ============================================================
# PUBLIC TRADES / FLOW
# ============================================================

@st.cache_data(ttl=ORDERBOOK_CACHE)
def get_recent_trades(symbol):
    result = api_get(f"/v2/trades/{symbol}")

    if not result:
        return pd.DataFrame()

    if isinstance(result, dict):
        trades = result.get("trades", [])
    else:
        trades = result

    if not trades:
        return pd.DataFrame()

    df = pd.DataFrame(trades)

    for c in ["price", "size"]:
        if c in df.columns:
            df[c] = pd.to_numeric(
                df[c],
                errors="coerce"
            )

    if not all(c in df.columns for c in ["price", "size"]):
        return pd.DataFrame()

    return df.dropna(
        subset=["price", "size"]
    )


def trade_flow_analysis(symbol):
    df = get_recent_trades(symbol)

    if df.empty or "side" not in df.columns:
        return None

    side = df["side"].astype(str).str.lower()

    buy_volume = float(
        df.loc[side == "buy", "size"].sum()
    )

    sell_volume = float(
        df.loc[side == "sell", "size"].sum()
    )

    total = buy_volume + sell_volume

    if total <= 0:
        return None

    delta = buy_volume - sell_volume

    return {
        "buy_volume": buy_volume,
        "sell_volume": sell_volume,
        "delta": delta,
        "delta_pct": delta / total * 100,
        "trades": len(df)
    }


# ============================================================
# LIQUIDITY + LIQUIDATION PROXY
# ============================================================

def liquidity_analysis(symbol):
    ob = orderbook_analysis(symbol, 15)

    if not ob:
        return None

    price = ob["mid"]

    return {
        "imbalance": ob["imbalance"],
        "bid_depth": ob["bid_depth"],
        "ask_depth": ob["ask_depth"],
        "best_bid": ob["best_bid"],
        "best_ask": ob["best_ask"],
        "spread": ob["spread"],
        "largest_bid_price": ob["largest_bid_price"],
        "largest_bid_size": ob["largest_bid_size"],
        "largest_ask_price": ob["largest_ask_price"],
        "largest_ask_size": ob["largest_ask_size"],
        "bid_wall_distance": (
            (price - ob["largest_bid_price"]) /
            price * 100
            if price else 0
        ),
        "ask_wall_distance": (
            (ob["largest_ask_price"] - price) /
            price * 100
            if price else 0
        )
    }


def liquidation_proxy(symbol):
    """
    NOT actual liquidation feed.

    Public-trade aggression + L2 imbalance are used only as
    a liquidation-like pressure proxy.
    """
    flow = trade_flow_analysis(symbol)

    if not flow:
        return None

    ob = orderbook_analysis(symbol, 15)

    score = 0

    if abs(flow["delta_pct"]) >= 30:
        score += 2

    if ob and abs(ob["imbalance"]) >= 30:
        score += 1

    if score >= 3:
        if flow["delta_pct"] > 0:
            signal = "🟢 BUY-SIDE LIQUIDATION PROXY"
        else:
            signal = "🔴 SELL-SIDE LIQUIDATION PROXY"
    elif score >= 1:
        signal = "🟡 WATCH"
    else:
        signal = "⚪ LOW"

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
        x["ATR"] / x["close"] * 100
    )

    return x


def swings(df, left=2, right=2):
    x = df.copy()

    x["SwingHigh"] = False
    x["SwingLow"] = False

    if len(x) < left + right + 1:
        return x

    for i in range(left, len(x) - right):
        if (
            x["high"].iloc[i]
            > x["high"].iloc[i-left:i].max()
            and
            x["high"].iloc[i]
            > x["high"].iloc[i+1:i+right+1].max()
        ):
            x.loc[x.index[i], "SwingHigh"] = True

        if (
            x["low"].iloc[i]
            < x["low"].iloc[i-left:i].min()
            and
            x["low"].iloc[i]
            < x["low"].iloc[i+1:i+right+1].min()
        ):
            x.loc[x.index[i], "SwingLow"] = True

    return x


def timeframe_trend(df):
    if len(df) < 30:
        return "⚪ UNKNOWN"

    c = df["close"]

    e9 = c.ewm(span=9, adjust=False).mean()
    e21 = c.ewm(span=21, adjust=False).mean()
    e50 = c.ewm(span=50, adjust=False).mean()

    if c.iloc[-1] > e9.iloc[-1] > e21.iloc[-1] > e50.iloc[-1]:
        return "🟢 BULL"

    if c.iloc[-1] < e9.iloc[-1] < e21.iloc[-1] < e50.iloc[-1]:
        return "🔴 BEAR"

    return "🟡 MIXED"


def regime_5d(symbol):
    df = get_candles(symbol, "1h", 5 * 24 + 10)

    if len(df) < 60:
        return {
            "state": "⚪ UNKNOWN",
            "range_pct": None
        }

    x = df.iloc[-120:] if len(df) > 120 else df

    first = float(x["close"].iloc[0])
    last = float(x["close"].iloc[-1])

    hi = float(x["high"].max())
    lo = float(x["low"].min())

    rng = (hi - lo) / lo * 100 if lo else 0
    move = (last - first) / first * 100 if first else 0

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


def sr_levels(df, lookback=80):
    if df.empty:
        return np.nan, np.nan, 0, 0

    x = swings(
        df.iloc[-lookback:].copy(),
        2,
        2
    )

    highs = x.loc[x["SwingHigh"], "high"].tolist()
    lows = x.loc[x["SwingLow"], "low"].tolist()

    price = float(df["close"].iloc[-1])

    supports = sorted(
        [v for v in lows if v < price],
        reverse=True
    )

    resistances = sorted(
        [v for v in highs if v > price]
    )

    return (
        supports[0] if supports else np.nan,
        resistances[0] if resistances else np.nan,
        len(supports),
        len(resistances)
    )


def sweep_bos_fvg(df):
    x = swings(df, 2, 2)

    if len(x) < 10:
        return {
            "bull_sweep": False,
            "bear_sweep": False,
            "bull_bos": False,
            "bear_bos": False,
            "bull_choch": False,
            "bear_choch": False,
            "bull_fvg": False,
            "bear_fvg": False,
            "swing_high": float(x["high"].max()) if not x.empty else 0,
            "swing_low": float(x["low"].min()) if not x.empty else 0
        }

    last = x.iloc[-1]

    sh = x.loc[x["SwingHigh"], "high"]
    sl = x.loc[x["SwingLow"], "low"]

    ph = (
        float(sh.iloc[-1])
        if len(sh)
        else float(x["high"].iloc[-8:-1].max())
    )

    pl = (
        float(sl.iloc[-1])
        if len(sl)
        else float(x["low"].iloc[-8:-1].min())
    )

    bull_sweep = (
        float(last["low"]) < pl
        and float(last["close"]) > pl
    )

    bear_sweep = (
        float(last["high"]) > ph
        and float(last["close"]) < ph
    )

    prev_h = float(x["high"].iloc[-8:-1].max())
    prev_l = float(x["low"].iloc[-8:-1].min())

    bull_bos = float(last["close"]) > prev_h
    bear_bos = float(last["close"]) < prev_l

    recent = x.iloc[-20:]

    prior_high = float(
        recent["high"].iloc[:10].max()
    ) if len(recent) >= 10 else prev_h

    prior_low = float(
        recent["low"].iloc[:10].min()
    ) if len(recent) >= 10 else prev_l

    bull_choch = (
        float(last["close"]) > prior_high
        and not bull_bos
    )

    bear_choch = (
        float(last["close"]) < prior_low
        and not bear_bos
    )

    bull_fvg = (
        len(x) >= 3
        and float(x["low"].iloc[-1])
        > float(x["high"].iloc[-3])
    )

    bear_fvg = (
        len(x) >= 3
        and float(x["high"].iloc[-1])
        < float(x["low"].iloc[-3])
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
        "swing_high": ph,
        "swing_low": pl
    }


# ============================================================
# OI / VOLUME / ATR
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

    if "time" in df.columns:
        df["time"] = pd.to_numeric(
            df["time"],
            errors="coerce"
        )

    return (
        df.dropna(subset=["close"])
        .sort_values("time")
        .reset_index(drop=True)
    )


def oi_analysis(symbol):
    df = get_oi_history(symbol, 24)

    if len(df) < 7:
        return None, "⚪ UNKNOWN"

    cur = float(df["close"].iloc[-1])
    old = float(df["close"].iloc[-7])

    if old == 0:
        return None, "⚪ UNKNOWN"

    ch = (cur - old) / abs(old) * 100

    if ch >= 1:
        sig = "🔺 OI UP"
    elif ch <= -1:
        sig = "🔻 OI DOWN"
    else:
        sig = "⚪ OI FLAT"

    return ch, sig


def volume_analysis(df):
    if len(df) < 10 or "volume" not in df.columns:
        return 0.0

    avg = df["volume"].iloc[-7:-1].mean()

    if avg <= 0:
        return 0.0

    return float(df["volume"].iloc[-1] / avg)


def atr_analysis(df):
    x = add_atr(df)

    if len(x) < 25:
        return None, "⚪ UNKNOWN"

    if pd.isna(x["ATR"].iloc[-1]) or pd.isna(x["ATR"].iloc[-7]):
        return None, "⚪ UNKNOWN"

    a = float(x["ATR"].iloc[-1])
    old = float(x["ATR"].iloc[-7])

    d = a / old if old else 1

    if d >= 1.10:
        direction = "🔺 ATR EXPANDING"
    elif d <= 0.90:
        direction = "🔻 ATR CONTRACTING"
    else:
        direction = "⚪ ATR FLAT"

    return a, direction


def mtf_state(t5, t15, t1):
    bulls = sum(v == "🟢 BULL" for v in [t5, t15, t1])
    bears = sum(v == "🔴 BEAR" for v in [t5, t15, t1])

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

def deep_analysis(symbol, ticker):
    d5 = closed(get_candles(symbol, "5m", 36))
    d15 = closed(get_candles(symbol, "15m", 72))
    d1 = closed(get_candles(symbol, "1h", 120))

    if min(len(d5), len(d15), len(d1)) < 25:
        return None

    t5 = timeframe_trend(d5)
    t15 = timeframe_trend(d15)
    t1 = timeframe_trend(d1)

    mtf = mtf_state(t5, t15, t1)
    reg = regime_5d(symbol)

    sr_sup, sr_res, sup_count, res_count = sr_levels(d15)
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

    price = float(ticker["Price"])

    long_score = 0
    short_score = 0

    lr = []
    sr = []

    # MTF
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

    # 5D
    if "UPTREND" in reg["state"]:
        long_score += 2
        lr.append("5D uptrend")
    elif "DOWNTREND" in reg["state"]:
        short_score += 2
        sr.append("5D downtrend")
    elif "RANGE" in reg["state"]:
        long_score -= 1
        short_score -= 1

    # Structure
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

    # S/R
    if not pd.isna(sr_sup):
        dist = abs(price - sr_sup) / price
        if dist <= 0.01:
            long_score += 2
            lr.append("near support")

    if not pd.isna(sr_res):
        dist = abs(sr_res - price) / price
        if dist <= 0.01:
            short_score += 2
            sr.append("near resistance")

    # Volume
    if volx >= 2:
        long_score += 2
        short_score += 2
        lr.append("volume spike")
        sr.append("volume spike")
    elif volx >= 1.3:
        long_score += 1
        short_score += 1

    # OI
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
                lr.append("OI unwind after bull sweep")

            if struct["bear_sweep"]:
                short_score += 1
                sr.append("OI unwind after bear sweep")

    # Funding
    if fp is not None:
        if fp >= 0.05:
            short_score += 2
            sr.append("positive funding crowding")
            funding_signal = "🔴 Long crowded"
        elif fp <= -0.05:
            long_score += 2
            lr.append("negative funding crowding")
            funding_signal = "🟢 Short crowded"
        else:
            funding_signal = "⚪ Neutral"
    else:
        funding_signal = "⚪ Unavailable"

    # ATR
    if atr_dir == "🔺 ATR EXPANDING":
        long_score += 1
        short_score += 1
    elif atr_dir == "🔻 ATR CONTRACTING":
        long_score -= 1
        short_score -= 1

    # Order book
    if ob:
        imbalance = ob["imbalance"]

        if imbalance >= 25:
            long_score += 2
            lr.append("orderbook bid imbalance")
        elif imbalance <= -25:
            short_score += 2
            sr.append("orderbook ask imbalance")

    # Liquidation proxy
    if liq:
        if "BUY-SIDE" in liq["signal"]:
            long_score += 1
            lr.append("buy-side liquidation proxy")
        elif "SELL-SIDE" in liq["signal"]:
            short_score += 1
            sr.append("sell-side liquidation proxy")

    blocked = mtf == "⚪ MTF CONFLICT"

    if blocked:
        signal = "⛔ MTF CONFLICT"
    elif long_score > short_score and long_score >= 8:
        signal = "🟢 STRONG LONG"
    elif short_score > long_score and short_score >= 8:
        signal = "🔴 STRONG SHORT"
    elif long_score > short_score and long_score >= 5:
        signal = "🟡 LONG WATCH"
    elif short_score > long_score and short_score >= 5:
        signal = "🟠 SHORT WATCH"
    else:
        signal = "⚪ NO SIGNAL"

    score = max(long_score, short_score)

    leverage = ticker.get("Leverage")

    if leverage is None:
        leverage_display = "⚪ N/A"
        leverage_group = "UNKNOWN"
    elif leverage == 20:
        leverage_display = "20x"
        leverage_group = "20X + Vol/OI > 6"
    elif leverage > 20:
        leverage_display = f"{leverage:g}x"
        leverage_group = ">20X — NO Vol/OI FILTER"
    else:
        leverage_display = f"{leverage:g}x"
        leverage_group = "<20X / OTHER"

    return {
        "Coin": symbol,
        "Leverage": leverage_display,
        "Leverage Group": leverage_group,
        "Price": price,
        "24H Volume": ticker["24H Volume"],
        "OI": ticker["OI"],
        "Vol/OI": (
            round(float(ticker["Vol/OI"]), 2)
            if pd.notna(ticker["Vol/OI"])
            else None
        ),
        "5m": t5,
        "15m": t15,
        "1H": t1,
        "MTF": mtf,
        "5D Regime": reg["state"],
        "5D Range %": (
            round(reg["range_pct"], 2)
            if reg["range_pct"] is not None
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
        "S/R Count": f"{sup_count}/{res_count}",
        "Liquidity": (
            "🟢 BULL SWEEP"
            if struct["bull_sweep"]
            else "🔴 BEAR SWEEP"
            if struct["bear_sweep"]
            else "⚪ None"
        ),
        "BOS": (
            "🟢 BULL BOS"
            if struct["bull_bos"]
            else "🔴 BEAR BOS"
            if struct["bear_bos"]
            else "⚪ None"
        ),
        "CHOCH": (
            "🟢 BULL CHOCH"
            if struct["bull_choch"]
            else "🔴 BEAR CHOCH"
            if struct["bear_choch"]
            else "⚪ None"
        ),
        "FVG": (
            "🟢 BULL FVG"
            if struct["bull_fvg"]
            else "🔴 BEAR FVG"
            if struct["bear_fvg"]
            else "⚪ None"
        ),
        "ATR": round(atr, 8) if atr is not None else None,
        "ATR Direction": atr_dir,
        "Volume x": round(volx, 2),
        "OI Change %": (
            round(oi_ch, 2)
            if oi_ch is not None
            else None
        ),
        "OI Signal": oi_sig,
        "Funding %": (
            round(fp, 4)
            if fp is not None
            else None
        ),
        "Funding": funding_signal,
        "OB Imbalance %": (
            round(ob["imbalance"], 2)
            if ob
            else None
        ),
        "Bid Depth": (
            round(ob["bid_depth"], 2)
            if ob
            else None
        ),
        "Ask Depth": (
            round(ob["ask_depth"], 2)
            if ob
            else None
        ),
        "Liq Proxy": (
            liq["signal"]
            if liq
            else "⚪ Unknown"
        ),
        "Liq Proxy Score": (
            liq["score"]
            if liq
            else 0
        ),
        "Long Score": long_score,
        "Short Score": short_score,
        "Score": score,
        "Signal": signal,
        "Long Reason": " + ".join(lr) if lr else "None",
        "Short Reason": " + ".join(sr) if sr else "None"
    }


# ============================================================
# MARKET BUILD
# ============================================================

coins = get_all_perpetuals()
tickers = get_tickers()

if coins.empty or tickers.empty:
    st.error(
        "❌ Delta market data load nahi hua. "
        "API response/check internet karo."
    )
    st.stop()

market = (
    coins.merge(
        tickers,
        on="Coin",
        how="inner"
    )
    .dropna(subset=["Price"])
    .copy()
)

# ============================================================
# LEVERAGE CATEGORIES
# ============================================================

market_20x = market[
    (market["Leverage"].notna()) &
    (market["Leverage"] == 20) &
    (market["Vol/OI"].fillna(0) > VOL_OI_MIN)
].copy()

market_high = market[
    (market["Leverage"].notna()) &
    (market["Leverage"] > 20)
].copy()

market_other = market[
    (market["Leverage"].isna()) |
    (market["Leverage"] < 20)
].copy()

market_20x = market_20x.sort_values(
    "24H Volume",
    ascending=False
)

market_high = market_high.sort_values(
    "24H Volume",
    ascending=False
)

market_other = market_other.sort_values(
    "24H Volume",
    ascending=False
)

# ============================================================
# TOP METRICS
# ============================================================

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "All Perpetuals",
    len(market)
)

c2.metric(
    "20x + Vol/OI > 6",
    len(market_20x)
)

c3.metric(
    ">20x — No Vol/OI Filter",
    len(market_high)
)

c4.metric(
    "Leverage Unknown / <20x",
    len(market_other)
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
        "📋 Market Tables"
    ],
    horizontal=True
)

# ============================================================
# COMMON DEEP SCANNER
# ============================================================

def run_scanner(source_df, limit):
    candidates = source_df.head(limit)

    results = []

    if candidates.empty:
        return pd.DataFrame()

    bar = st.progress(0)

    total = len(candidates)

    for i, (_, row) in enumerate(candidates.iterrows()):
        r = deep_analysis(
            row["Coin"],
            row
        )

        if r:
            results.append(r)

        bar.progress(
            int((i + 1) / total * 100)
        )

    bar.empty()

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results)


# ============================================================
# LIVE SCANNER
# ============================================================

if mode == "🔥 Live Scanner":
    st.subheader("🔥 Leverage-Segmented Live Scanner")

    st.info(
        "20x contracts ke liye Vol/OI > 6 compulsory hai. "
        ">20x contracts par Vol/OI filter nahi lagaya gaya."
    )

    tab20, tabHigh = st.tabs([
        "⚡ 20X + Vol/OI > 6",
        "🚀 >20X — NO Vol/OI FILTER"
    ])

    with tab20:
        st.write(
            f"Deep scanning top {min(DEEP_SCAN_LIMIT, len(market_20x))} "
            "20x contracts..."
        )

        sig20 = run_scanner(
            market_20x,
            DEEP_SCAN_LIMIT
        )

        if sig20.empty:
            st.warning(
                "20x + Vol/OI > 6 category mein abhi data nahi mila."
            )
        else:
            st.dataframe(
                sig20.sort_values(
                    "Score",
                    ascending=False
                ),
                use_container_width=True,
                hide_index=True
            )

            st.subheader("🟢 20X LONG")
            st.dataframe(
                sig20.sort_values(
                    "Long Score",
                    ascending=False
                )[
                    [
                        "Coin", "Leverage", "Price",
                        "Vol/OI", "OI Change %",
                        "Funding %", "MTF",
                        "Liquidity", "BOS", "CHOCH",
                        "FVG", "OB Imbalance %",
                        "Liq Proxy", "Long Score",
                        "Long Reason"
                    ]
                ],
                use_container_width=True,
                hide_index=True
            )

            st.subheader("🔴 20X SHORT")
            st.dataframe(
                sig20.sort_values(
                    "Short Score",
                    ascending=False
                )[
                    [
                        "Coin", "Leverage", "Price",
                        "Vol/OI", "OI Change %",
                        "Funding %", "MTF",
                        "Liquidity", "BOS", "CHOCH",
                        "FVG", "OB Imbalance %",
                        "Liq Proxy", "Short Score",
                        "Short Reason"
                    ]
                ],
                use_container_width=True,
                hide_index=True
            )

    with tabHigh:
        st.write(
            f"Deep scanning top {min(DEEP_SCAN_LIMIT, len(market_high))} "
            ">20x contracts..."
        )

        sigHigh = run_scanner(
            market_high,
            DEEP_SCAN_LIMIT
        )

        if sigHigh.empty:
            st.warning(
                ">20x category mein abhi data nahi mila."
            )
        else:
            st.dataframe(
                sigHigh.sort_values(
                    "Score",
                    ascending=False
                ),
                use_container_width=True,
                hide_index=True
            )

            st.subheader("🟢 >20X LONG")
            st.dataframe(
                sigHigh.sort_values(
                    "Long Score",
                    ascending=False
                )[
                    [
                        "Coin", "Leverage", "Price",
                        "Vol/OI", "OI Change %",
                        "Funding %", "MTF",
                        "Liquidity", "BOS", "CHOCH",
                        "FVG", "OB Imbalance %",
                        "Liq Proxy", "Long Score",
                        "Long Reason"
                    ]
                ],
                use_container_width=True,
                hide_index=True
            )

            st.subheader("🔴 >20X SHORT")
            st.dataframe(
                sigHigh.sort_values(
                    "Short Score",
                    ascending=False
                )[
                    [
                        "Coin", "Leverage", "Price",
                        "Vol/OI", "OI Change %",
                        "Funding %", "MTF",
                        "Liquidity", "BOS", "CHOCH",
                        "FVG", "OB Imbalance %",
                        "Liq Proxy", "Short Score",
                        "Short Reason"
                    ]
                ],
                use_container_width=True,
                hide_index=True
            )

# ============================================================
# MARKET TABLES
# ============================================================

elif mode == "📋 Market Tables":
    st.subheader("📋 Raw Market Categories")

    tab1, tab2, tab3 = st.tabs([
        "20X + Vol/OI > 6",
        ">20X — No Vol/OI Filter",
        "Unknown / <20X"
    ])

    columns = [
        "Coin",
        "Leverage",
        "Price",
        "24H Volume",
        "OI",
        "Vol/OI",
        "Funding"
    ]

    with tab1:
        st.dataframe(
            market_20x[columns],
            use_container_width=True,
            hide_index=True
        )

    with tab2:
        st.dataframe(
            market_high[columns],
            use_container_width=True,
            hide_index=True
        )

    with tab3:
        st.dataframe(
            market_other[columns],
            use_container_width=True,
            hide_index=True
        )

# ============================================================
# ORDER BOOK
# ============================================================

elif mode == "📚 Order Book / Liquidity":
    st.subheader("📚 Delta L2 Order Book + Liquidity")

    available = (
        pd.concat(
            [
                market_high["Coin"],
                market_20x["Coin"],
                market_other["Coin"]
            ]
        )
        .drop_duplicates()
        .head(100)
        .tolist()
    )

    if not available:
        st.warning("Coins available nahi hain.")
    else:
        selected = st.selectbox(
            "Coin select karo",
            available
        )

        if st.button("🔍 Analyze Order Book"):
            ob = orderbook_analysis(
                selected,
                15
            )

            if not ob:
                st.error(
                    "Delta L2 order book data nahi mila."
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
                    f'{ob["bid_depth"]:,.2f}'
                )

                c4.metric(
                    "Ask Depth",
                    f'{ob["ask_depth"]:,.2f}'
                )

                imbalance = ob["imbalance"]

                if imbalance >= 25:
                    st.success(
                        f"🟢 Strong Bid Imbalance: {imbalance:.2f}%"
                    )
                elif imbalance <= -25:
                    st.error(
                        f"🔴 Strong Ask Imbalance: {imbalance:.2f}%"
                    )
                else:
                    st.info(
                        f"⚪ Balanced Order Book: {imbalance:.2f}%"
                    )

                left, right = st.columns(2)

                with left:
                    st.subheader("🟢 BIDS")
                    st.dataframe(
                        ob["bid_df"].sort_values(
                            "Price",
                            ascending=False
                        ),
                        use_container_width=True,
                        hide_index=True
                    )

                with right:
                    st.subheader("🔴 ASKS")
                    st.dataframe(
                        ob["ask_df"].sort_values(
                            "Price",
                            ascending=True
                        ),
                        use_container_width=True,
                        hide_index=True
                    )

                w1, w2 = st.columns(2)

                with w1:
                    st.metric(
                        "Largest Bid Wall",
                        f'{ob["largest_bid_price"]:.8f}'
                    )
                    st.write(
                        f'Size: {ob["largest_bid_size"]:,.2f}'
                    )

                with w2:
                    st.metric(
                        "Largest Ask Wall",
                        f'{ob["largest_ask_price"]:.8f}'
                    )
                    st.write(
                        f'Size: {ob["largest_ask_size"]:,.2f}'
                    )

                st.warning(
                    "⚠️ Visible order-book walls guaranteed support/"
                    "resistance nahi hain. Orders cancel ho sakte hain."
                )

# ============================================================
# LIQUIDATION PROXY
# ============================================================

else:
    st.subheader("💥 Liquidation Pressure Proxy")

    st.warning(
        "Ye actual liquidation feed nahi hai. "
        "Public trades + Delta L2 order book se liquidation-like "
        "pressure estimate kiya ja raha hai."
    )

    source = pd.concat(
        [
            market_high,
            market_20x
        ]
    ).drop_duplicates("Coin").head(30)

    rows = []

    bar = st.progress(0)

    total = len(source)

    for i, (_, row) in enumerate(source.iterrows()):
        symbol = row["Coin"]

        liq = liquidation_proxy(symbol)

        if liq:
            ob = orderbook_analysis(symbol, 15)

            rows.append({
                "Coin": symbol,
                "Leverage": (
                    f'{row["Leverage"]:g}x'
                    if pd.notna(row["Leverage"])
                    else "N/A"
                ),
                "Price": row["Price"],
                "Buy Volume": round(
                    liq["buy_volume"], 2
                ),
                "Sell Volume": round(
                    liq["sell_volume"], 2
                ),
                "Trade Delta %": round(
                    liq["delta_pct"], 2
                ),
                "OB Imbalance %": (
                    round(ob["imbalance"], 2)
                    if ob else None
                ),
                "Liquidation Proxy": liq["signal"],
                "Proxy Score": liq["score"]
            })

        if total:
            bar.progress(
                int((i + 1) / total * 100)
            )

    bar.empty()

    if rows:
        liq_df = pd.DataFrame(rows)

        st.dataframe(
            liq_df.sort_values(
                "Proxy Score",
                ascending=False
            ),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info(
            "Liquidation proxy data nahi mila."
        )

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
### 🔥 Scanner Logic

**20X CONTRACTS**
→ Vol/OI > 6  
→ MTF  
→ 5D Regime  
→ Support/Resistance  
→ Liquidity Sweep  
→ BOS / CHOCH  
→ FVG  
→ OI  
→ Funding  
→ Volume  
→ ATR  
→ Delta L2 Order Book  

**>20X CONTRACTS**
→ Vol/OI filter **OFF**  
→ MTF  
→ 5D Regime  
→ Support/Resistance  
→ Liquidity Sweep  
→ BOS / CHOCH  
→ FVG  
→ OI  
→ Funding  
→ Volume  
→ ATR  
→ Delta L2 Order Book
"""
)

st.caption(
    "⚠️ Leverage shown here is contract/product metadata when "
    "available, not your personal current position leverage. "
    "Liquidation Proxy is an estimate, not actual liquidation data."
)

if st.button("🔄 Refresh Scanner"):
    st.cache_data.clear()
    st.rerun()
