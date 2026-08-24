import streamlit as st
import requests
import pandas as pd
import numpy as np
import time

# ============================================================
# DELTA EXCHANGE INDIA - MTF REVERSAL / S-R SCANNER
# Clean rebuild
# ============================================================

BASE_URL = "https://api.india.delta.exchange"

HEADERS = {
    "Accept": "application/json",
    "User-Agent": "Delta-MTF-Scanner/2.0",
}

CACHE_TTL = 30
L2_CACHE_TTL = 10

st.set_page_config(
    page_title="Delta MTF Scanner",
    page_icon="🔥",
    layout="wide",
)

# ============================================================
# API
# ============================================================

def api_get(path, params=None, timeout=15):
    try:
        r = requests.get(
            BASE_URL + path,
            params=params or {},
            headers=HEADERS,
            timeout=timeout,
        )

        if r.status_code != 200:
            return None

        data = r.json()

        if not data.get("success", True):
            return None

        return data.get("result")

    except Exception:
        return None


# ============================================================
# PRODUCTS
# ============================================================

@st.cache_data(ttl=CACHE_TTL)
def get_products():
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

        rows.append(
            {
                "Coin": symbol,
                "ID": p.get("id"),
                "Product": p,
            }
        )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).drop_duplicates("Coin")


# ============================================================
# TICKERS
# ============================================================

@st.cache_data(ttl=CACHE_TTL)
def get_tickers():
    result = api_get("/v2/tickers")

    if not result:
        return pd.DataFrame()

    rows = []

    for x in result:
        symbol = x.get("symbol")

        if not symbol:
            continue

        try:
            price = float(
                x.get("close")
                or x.get("mark_price")
                or 0
            )

            volume = float(
                x.get("volume_24h")
                or x.get("volume")
                or 0
            )

            oi = float(
                x.get("open_interest")
                or x.get("oi")
                or 0
            )

        except Exception:
            continue

        if price <= 0:
            continue

        funding = np.nan

        try:
            if x.get("funding_rate") is not None:
                funding = float(x.get("funding_rate"))
        except Exception:
            pass

        rows.append(
            {
                "Coin": symbol,
                "Price": price,
                "24H Volume": volume,
                "OI": oi,
                "Funding": funding,
            }
        )

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    df["Vol/OI"] = (
        df["24H Volume"]
        / df["OI"].replace(0, np.nan)
    )

    return df


# ============================================================
# CANDLES
# ============================================================

@st.cache_data(ttl=CACHE_TTL)
def get_history(symbol, resolution, candles):
    now = int(time.time())

    seconds_map = {
        "1m": 60,
        "5m": 300,
        "15m": 900,
        "30m": 1800,
        "1h": 3600,
        "2h": 7200,
        "4h": 14400,
        "6h": 21600,
        "12h": 43200,
        "1d": 86400,
        "1w": 604800,
        "1M": 2592000,
    }

    step = seconds_map.get(resolution, 300)

    start = now - (step * (candles + 10))

    result = api_get(
        "/v2/history/candles",
        {
            "resolution": resolution,
            "symbol": symbol,
            "start": start,
            "end": now,
        },
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
        "volume",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(
                df[c],
                errors="coerce",
            )

    if "time" in df.columns:
        df["time"] = pd.to_numeric(
            df["time"],
            errors="coerce",
        )

        df = df.sort_values("time")

    required = [
        "open",
        "high",
        "low",
        "close",
    ]

    if not all(c in df.columns for c in required):
        return pd.DataFrame()

    df = df.dropna(subset=required)

    if "time" in df.columns:
        df = df.drop_duplicates("time")

    return df.tail(candles).reset_index(drop=True)


# ============================================================
# OI HISTORY
# ============================================================

@st.cache_data(ttl=CACHE_TTL)
def get_oi_history(symbol, candles=200):
    now = int(time.time())
    start = now - (900 * (candles + 10))

    result = api_get(
        "/v2/history/candles",
        {
            "resolution": "15m",
            "symbol": "OI:" + symbol,
            "start": start,
            "end": now,
        },
    )

    if not result:
        return pd.DataFrame()

    df = pd.DataFrame(result)

    if df.empty or "close" not in df.columns:
        return pd.DataFrame()

    df["close"] = pd.to_numeric(
        df["close"],
        errors="coerce",
    )

    return df.dropna(
        subset=["close"]
    ).reset_index(drop=True)


def calculate_oi_change(symbol):
    df = get_oi_history(symbol)

    if len(df) < 2:
        return np.nan

    old = float(df["close"].iloc[0])
    new = float(df["close"].iloc[-1])

    if old == 0:
        return np.nan

    return ((new - old) / abs(old)) * 100


# ============================================================
# L2
# ============================================================

@st.cache_data(ttl=L2_CACHE_TTL)
def get_orderbook(symbol, depth=15):
    result = api_get(
        "/v2/l2orderbook/" + symbol,
        {"depth": depth},
    )

    return result if isinstance(result, dict) else None


def orderbook_stats(symbol, depth=15):
    data = get_orderbook(symbol, depth)

    if not data:
        return None

    bids = data.get("buy") or []
    asks = data.get("sell") or []

    bid_rows = []
    ask_rows = []

    for x in bids:
        try:
            bid_rows.append(
                {
                    "Price": float(x["price"]),
                    "Size": float(x["size"]),
                }
            )
        except Exception:
            pass

    for x in asks:
        try:
            ask_rows.append(
                {
                    "Price": float(x["price"]),
                    "Size": float(x["size"]),
                }
            )
        except Exception:
            pass

    if not bid_rows or not ask_rows:
        return None

    bid = pd.DataFrame(bid_rows)
    ask = pd.DataFrame(ask_rows)

    bid = bid.sort_values(
        "Price",
        ascending=False,
    )

    ask = ask.sort_values(
        "Price",
        ascending=True,
    )

    bid_depth = bid["Size"].sum()
    ask_depth = ask["Size"].sum()

    total = bid_depth + ask_depth

    if total == 0:
        imbalance = 0
    else:
        imbalance = (
            (bid_depth - ask_depth)
            / total
            * 100
        )

    best_bid = bid["Price"].max()
    best_ask = ask["Price"].min()

    mid = (best_bid + best_ask) / 2

    return {
        "bid": bid,
        "ask": ask,
        "bid_depth": bid_depth,
        "ask_depth": ask_depth,
        "imbalance": imbalance,
        "best_bid": best_bid,
        "best_ask": best_ask,
        "mid": mid,
    }


# ============================================================
# PUBLIC TRADES
# ============================================================

@st.cache_data(ttl=L2_CACHE_TTL)
def get_trades(symbol):
    result = api_get(
        "/v2/trades/" + symbol
    )

    if not isinstance(result, dict):
        return pd.DataFrame()

    trades = result.get("trades") or []

    if not trades:
        return pd.DataFrame()

    df = pd.DataFrame(trades)

    for c in ["price", "size"]:
        if c in df.columns:
            df[c] = pd.to_numeric(
                df[c],
                errors="coerce",
            )

    return df


def trade_flow(symbol):
    df = get_trades(symbol)

    if df.empty or "side" not in df.columns:
        return None

    side = (
        df["side"]
        .astype(str)
        .str.lower()
    )

    buy = df.loc[
        side == "buy",
        "size",
    ].sum()

    sell = df.loc[
        side == "sell",
        "size",
    ].sum()

    total = buy + sell

    if total <= 0:
        return None

    delta = buy - sell

    return {
        "buy": float(buy),
        "sell": float(sell),
        "delta": float(delta),
        "delta_pct": float(
            delta / total * 100
        ),
    }


# ============================================================
# TREND
# ============================================================

def trend(df):
    if len(df) < 30:
        return "UNKNOWN"

    close = df["close"]

    ema9 = close.ewm(
        span=9,
        adjust=False,
    ).mean()

    ema21 = close.ewm(
        span=21,
        adjust=False,
    ).mean()

    ema50 = close.ewm(
        span=50,
        adjust=False,
    ).mean()

    last = close.iloc[-1]

    if (
        last > ema9.iloc[-1]
        and ema9.iloc[-1] > ema21.iloc[-1]
        and ema21.iloc[-1] > ema50.iloc[-1]
    ):
        return "BULL"

    if (
        last < ema9.iloc[-1]
        and ema9.iloc[-1] < ema21.iloc[-1]
        and ema21.iloc[-1] < ema50.iloc[-1]
    ):
        return "BEAR"

    return "MIXED"


# ============================================================
# SUPPORT / RESISTANCE
# ============================================================

def find_levels(df, lookback=3):
    if len(df) < (lookback * 2 + 5):
        return [], []

    highs = []
    lows = []

    for i in range(
        lookback,
        len(df) - lookback,
    ):
        high = float(df["high"].iloc[i])
        low = float(df["low"].iloc[i])

        left_high = df["high"].iloc[
            i - lookback:i
        ].max()

        right_high = df["high"].iloc[
            i + 1:i + lookback + 1
        ].max()

        left_low = df["low"].iloc[
            i - lookback:i
        ].min()

        right_low = df["low"].iloc[
            i + 1:i + lookback + 1
        ].min()

        if high >= left_high and high >= right_high:
            highs.append(
                {
                    "index": i,
                    "price": high,
                }
            )

        if low <= left_low and low <= right_low:
            lows.append(
                {
                    "index": i,
                    "price": low,
                }
            )

    return highs, lows


def cluster_levels(levels, tolerance=0.005):
    if not levels:
        return []

    clusters = []

    for level in levels:
        price = level["price"]
        placed = False

        for cluster in clusters:
            center = cluster["price"]

            if abs(price - center) / center <= tolerance:
                cluster["touches"] += 1
                cluster["prices"].append(price)
                cluster["indexes"].append(
                    level["index"]
                )
                cluster["price"] = float(
                    np.mean(cluster["prices"])
                )
                placed = True
                break

        if not placed:
            clusters.append(
                {
                    "price": price,
                    "touches": 1,
                    "prices": [price],
                    "indexes": [level["index"]],
                }
            )

    return sorted(
        clusters,
        key=lambda x: x["touches"],
        reverse=True,
    )


# ============================================================
# REPEATED STRUCTURE
# ============================================================

def repeated_structure(df):
    if len(df) < 15:
        return {
            "pattern": "NONE",
            "count": 0,
            "details": [],
        }

    highs, lows = find_levels(
        df,
        lookback=3,
    )

    resistance_clusters = cluster_levels(
        highs
    )

    support_clusters = cluster_levels(
        lows
    )

    details = []

    # Repeated resistance
    for c in resistance_clusters:
        if c["touches"] >= 2:
            details.append(
                {
                    "Type": "REPEATED RESISTANCE",
                    "Price": round(
                        c["price"],
                        8,
                    ),
                    "Touches": c["touches"],
                    "Candle Indexes": str(
                        c["indexes"]
                    ),
                }
            )

    # Repeated support
    for c in support_clusters:
        if c["touches"] >= 2:
            details.append(
                {
                    "Type": "REPEATED SUPPORT",
                    "Price": round(
                        c["price"],
                        8,
                    ),
                    "Touches": c["touches"],
                    "Candle Indexes": str(
                        c["indexes"]
                    ),
                }
            )

    # Higher highs
    if len(highs) >= 3:
        recent_highs = highs[-4:]

        values = [
            x["price"]
            for x in recent_highs
        ]

        if all(
            values[i] >= values[i - 1]
            for i in range(1, len(values))
        ):
            details.append(
                {
                    "Type": "REPEATED HIGHER HIGHS",
                    "Price": round(
                        values[-1],
                        8,
                    ),
                    "Touches": len(values),
                    "Candle Indexes": str(
                        [
                            x["index"]
                            for x in recent_highs
                        ]
                    ),
                }
            )

    # Lower lows
    if len(lows) >= 3:
        recent_lows = lows[-4:]

        values = [
            x["price"]
            for x in recent_lows
        ]

        if all(
            values[i] <= values[i - 1]
            for i in range(1, len(values))
        ):
            details.append(
                {
                    "Type": "REPEATED LOWER LOWS",
                    "Price": round(
                        values[-1],
                        8,
                    ),
                    "Touches": len(values),
                    "Candle Indexes": str(
                        [
                            x["index"]
                            for x in recent_lows
                        ]
                    ),
                }
            )

    if not details:
        return {
            "pattern": "NONE",
            "count": 0,
            "details": [],
        }

    pattern_names = list(
        dict.fromkeys(
            x["Type"]
            for x in details
        )
    )

    return {
        "pattern": " | ".join(pattern_names),
        "count": len(details),
        "details": details,
    }


# ============================================================
# STRUCTURE / BOS / SWEEP
# ============================================================

def structure(df):
    if len(df) < 20:
        return "NONE", "NONE"

    previous_high = df["high"].iloc[
        -11:-1
    ].max()

    previous_low = df["low"].iloc[
        -11:-1
    ].min()

    last = df.iloc[-1]

    if (
        last["low"] < previous_low
        and last["close"] > previous_low
    ):
        sweep = "BULL SWEEP"

    elif (
        last["high"] > previous_high
        and last["close"] < previous_high
    ):
        sweep = "BEAR SWEEP"

    else:
        sweep = "NONE"

    if last["close"] > previous_high:
        bos = "BULL BOS"

    elif last["close"] < previous_low:
        bos = "BEAR BOS"

    else:
        bos = "NONE"

    return sweep, bos


# ============================================================
# VOLUME
# ============================================================

def volume_multiple(df):
    if (
        len(df) < 8
        or "volume" not in df.columns
    ):
        return 0.0

    avg = df["volume"].iloc[
        -7:-1
    ].mean()

    if avg <= 0:
        return 0.0

    return float(
        df["volume"].iloc[-1] / avg
    )


# ============================================================
# TIMEFRAME CONFIG
# ============================================================

TIMEFRAMES = {
    "12H": ("12h", 60),
    "6H": ("6h", 120),
    "1D": ("1d", 365),
    "1W": ("1w", 52),
    "1M": ("1M", 12),
}


# ============================================================
# TIMEFRAME ANALYSIS
# ============================================================

def timeframe_analysis(symbol):
    output = {}

    for name, (resolution, candles) in TIMEFRAMES.items():
        df = get_history(
            symbol,
            resolution,
            candles,
        )

        if len(df) < 10:
            output[name] = {
                "trend": "NO DATA",
                "support": np.nan,
                "resistance": np.nan,
                "pattern": "NO DATA",
                "repeats": 0,
                "details": [],
            }
            continue

        highs, lows = find_levels(
            df,
            lookback=2,
        )

        supports = [
            x["price"]
            for x in lows
            if x["price"] < df["close"].iloc[-1]
        ]

        resistances = [
            x["price"]
            for x in highs
            if x["price"] > df["close"].iloc[-1]
        ]

        support = (
            max(supports)
            if supports
            else np.nan
        )

        resistance = (
            min(resistances)
            if resistances
            else np.nan
        )

        repeated = repeated_structure(df)

        output[name] = {
            "trend": trend(df),
            "support": support,
            "resistance": resistance,
            "pattern": repeated["pattern"],
            "repeats": repeated["count"],
            "details": repeated["details"],
            "candles": df,
        }

    return output


# ============================================================
# MAIN SCAN
# ============================================================

def scan_coin(symbol, ticker, depth):
    d5 = get_history(
        symbol,
        "5m",
        150,
    )

    d15 = get_history(
        symbol,
        "15m",
        150,
    )

    d1h = get_history(
        symbol,
        "1h",
        150,
    )

    if (
        len(d5) < 30
        or len(d15) < 30
        or len(d1h) < 30
    ):
        return None

    t5 = trend(d5)
    t15 = trend(d15)
    t1h = trend(d1h)

    trends = [t5, t15, t1h]

    bulls = trends.count("BULL")
    bears = trends.count("BEAR")

    if bulls == 3:
        mtf = "LONG ALIGNED"
    elif bears == 3:
        mtf = "SHORT ALIGNED"
    elif bulls >= 2:
        mtf = "LONG BIAS"
    elif bears >= 2:
        mtf = "SHORT BIAS"
    else:
        mtf = "CONFLICT"

    sweep, bos = structure(d5)

    vol_x = volume_multiple(d5)

    oi = calculate_oi_change(symbol)

    ob = orderbook_stats(
        symbol,
        depth,
    )

    flow = trade_flow(symbol)

    long_score = 0
    short_score = 0

    if mtf == "LONG ALIGNED":
        long_score += 4
    elif mtf == "SHORT ALIGNED":
        short_score += 4
    elif mtf == "LONG BIAS":
        long_score += 2
    elif mtf == "SHORT BIAS":
        short_score += 2

    if sweep == "BULL SWEEP":
        long_score += 2

    if sweep == "BEAR SWEEP":
        short_score += 2

    if bos == "BULL BOS":
        long_score += 3

    if bos == "BEAR BOS":
        short_score += 3

    if vol_x >= 2:
        long_score += 1
        short_score += 1

    if not np.isnan(oi):
        if oi > 1:
            if mtf.startswith("LONG"):
                long_score += 2
            elif mtf.startswith("SHORT"):
                short_score += 2

    if ob:
        if ob["imbalance"] >= 25:
            long_score += 2

        elif ob["imbalance"] <= -25:
            short_score += 2

    if flow:
        if flow["delta_pct"] >= 25:
            long_score += 1

        elif flow["delta_pct"] <= -25:
            short_score += 1

    if mtf == "CONFLICT":
        signal = "NO SIGNAL"

    elif max(
        long_score,
        short_score,
    ) < 5:
        signal = "NO SIGNAL"

    elif long_score > short_score:
        signal = (
            "STRONG LONG"
            if long_score >= 8
            else "LONG WATCH"
        )

    elif short_score > long_score:
        signal = (
            "STRONG SHORT"
            if short_score >= 8
            else "SHORT WATCH"
        )

    else:
        signal = "NO SIGNAL"

    return {
        "Coin": symbol,
        "Price": float(ticker["Price"]),
        "Vol/OI": float(
            ticker["Vol/OI"]
        )
        if pd.notna(ticker["Vol/OI"])
        else np.nan,
        "5m": t5,
        "15m": t15,
        "1H": t1h,
        "MTF": mtf,
        "Sweep": sweep,
        "BOS": bos,
        "Volume x": round(
            vol_x,
            2,
        ),
        "OI Change %": round(
            oi,
            2,
        )
        if not np.isnan(oi)
        else np.nan,
        "L2 Imbalance %": round(
            ob["imbalance"],
            2,
        )
        if ob
        else np.nan,
        "Long Score": long_score,
        "Short Score": short_score,
        "Score": max(
            long_score,
            short_score,
        ),
        "Signal": signal,
    }


# ============================================================
# LOAD MARKET
# ============================================================

products = get_products()
tickers = get_tickers()

if products.empty or tickers.empty:
    st.error(
        "Delta market data load nahi hua."
    )
    st.stop()

market = products.merge(
    tickers,
    on="Coin",
    how="inner",
)

market = market.dropna(
    subset=["Price"]
)

market = market.sort_values(
    "24H Volume",
    ascending=False,
).reset_index(drop=True)


# ============================================================
# HEADER
# ============================================================

st.title(
    "🔥 Delta MTF Reversal Scanner"
)

st.caption(
    "All perpetual coins → Volume/OI buckets → "
    "MTF → Support/Resistance → Repeated Structure → "
    "OI → L2 → Trade Flow"
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("Scanner Settings")

    depth = st.slider(
        "L2 Depth",
        5,
        50,
        15,
        5,
    )

    scan_limit = st.slider(
        "Deep Scan Coins",
        10,
        min(200, len(market)),
        min(50, len(market)),
        10,
    )

    if st.button(
        "🔄 Refresh Data"
    ):
        st.cache_data.clear()
        st.rerun()


# ============================================================
# MARKET BUCKETS
# ============================================================

all_coins = market.copy()

vol_oi_3 = all_coins[
    all_coins["Vol/OI"] > 3
].copy()

vol_oi_6 = all_coins[
    all_coins["Vol/OI"] > 6
].copy()

st.subheader(
    "📊 Market Overview"
)

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "All Perpetual Coins",
    len(all_coins),
)

c2.metric(
    "Vol/OI > 3",
    len(vol_oi_3),
)

c3.metric(
    "Vol/OI > 6",
    len(vol_oi_6),
)

c4.metric(
    "Highest Vol/OI",
    round(
        all_coins["Vol/OI"].max(),
        2,
    )
    if not all_coins.empty
    else 0,
)


# ============================================================
# BUCKET TABLES
# ============================================================

st.subheader(
    "📋 Volume / Open Interest Tables"
)

tab1, tab2, tab3 = st.tabs(
    [
        "All Coins",
        "Vol/OI > 3",
        "Vol/OI > 6",
    ]
)

columns = [
    "Coin",
    "Price",
    "24H Volume",
    "OI",
    "Vol/OI",
    "Funding",
]

with tab1:
    st.dataframe(
        all_coins[columns].head(200),
        use_container_width=True,
        hide_index=True,
    )

with tab2:
    st.dataframe(
        vol_oi_3[columns].head(200),
        use_container_width=True,
        hide_index=True,
    )

with tab3:
    st.dataframe(
        vol_oi_6[columns].head(200),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# NAVIGATION
# ============================================================

mode = st.radio(
    "Select Analysis",
    [
        "🔥 MTF Scanner",
        "📐 Support / Resistance",
        "🔁 Repeated Structure",
        "📚 L2 Order Book",
    ],
    horizontal=True,
)


# ============================================================
# MTF SCANNER
# ============================================================

if mode == "🔥 MTF Scanner":

    st.subheader(
        "🔥 Multi-Timeframe Scanner"
    )

    candidates = all_coins.head(
        scan_limit
    )

    results = []

    progress = st.progress(0)

    total = len(candidates)

    for i, (_, row) in enumerate(
        candidates.iterrows()
    ):

        try:
            result = scan_coin(
                row["Coin"],
                row,
                depth,
            )

            if result:
                results.append(result)

        except Exception:
            pass

        if total > 0:
            progress.progress(
                int(
                    ((i + 1) / total)
                    * 100
                )
            )

    progress.empty()

    if not results:
        st.warning(
            "Enough candle data available nahi hai."
        )
        st.stop()

    result_df = pd.DataFrame(results)

    st.dataframe(
        result_df.sort_values(
            "Score",
            ascending=False,
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader(
        "🟢 Long Candidates"
    )

    long_df = result_df.sort_values(
        "Long Score",
        ascending=False,
    )

    st.dataframe(
        long_df[
            [
                "Coin",
                "Price",
                "Vol/OI",
                "5m",
                "15m",
                "1H",
                "MTF",
                "Sweep",
                "BOS",
                "Volume x",
                "OI Change %",
                "L2 Imbalance %",
                "Long Score",
                "Signal",
            ]
        ].head(30),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader(
        "🔴 Short Candidates"
    )

    short_df = result_df.sort_values(
        "Short Score",
        ascending=False,
    )

    st.dataframe(
        short_df[
            [
                "Coin",
                "Price",
                "Vol/OI",
                "5m",
                "15m",
                "1H",
                "MTF",
                "Sweep",
                "BOS",
                "Volume x",
                "OI Change %",
                "L2 Imbalance %",
                "Short Score",
                "Signal",
            ]
        ].head(30),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# SUPPORT / RESISTANCE
# ============================================================

elif mode == "📐 Support / Resistance":

    st.subheader(
        "📐 Multi-Timeframe Support / Resistance"
    )

    symbol = st.selectbox(
        "Coin",
        all_coins["Coin"].tolist(),
    )

    if st.button(
        "Analyze S/R"
    ):

        analysis = timeframe_analysis(
            symbol
        )

        rows = []

        for tf, data in analysis.items():

            rows.append(
                {
                    "Timeframe": tf,
                    "Trend": data["trend"],
                    "Support": data["support"],
                    "Resistance": data[
                        "resistance"
                    ],
                    "Repeated Pattern": data[
                        "pattern"
                    ],
                    "Repeats": data[
                        "repeats"
                    ],
                }
            )

        sr_df = pd.DataFrame(rows)

        st.dataframe(
            sr_df,
            use_container_width=True,
            hide_index=True,
        )

        st.info(
            "1M = last 12 candles | "
            "1W = last 52 | "
            "1D = last 365 | "
            "12H = last 60 | "
            "6H = last 120"
        )


# ============================================================
# REPEATED STRUCTURE
# ============================================================

elif mode == "🔁 Repeated Structure":

    st.subheader(
        "🔁 Repeated Resistance / Support Detector"
    )

    symbol = st.selectbox(
        "Coin",
        all_coins["Coin"].tolist(),
        key="structure_coin",
    )

    if st.button(
        "Find Repeated Structure"
    ):

        analysis = timeframe_analysis(
            symbol
        )

        for tf, data in analysis.items():

            st.markdown(
                "### " + tf
            )

            st.write(
                "Trend:",
                data["trend"],
            )

            st.write(
                "Pattern:",
                data["pattern"],
            )

            st.write(
                "Repeat Count:",
                data["repeats"],
            )

            if data["details"]:

                details = pd.DataFrame(
                    data["details"]
                )

                st.dataframe(
                    details,
                    use_container_width=True,
                    hide_index=True,
                )

            else:
                st.caption(
                    "Is timeframe par "
                    "repeated structure nahi mila."
                )

            st.divider()


# ============================================================
# L2
# ============================================================

else:

    st.subheader(
        "📚 Live L2 Order Book"
    )

    symbol = st.selectbox(
        "Coin",
        all_coins["Coin"].tolist(),
        key="l2_coin",
    )

    if st.button(
        "Load Order Book"
    ):

        ob = orderbook_stats(
            symbol,
            depth,
        )

        if not ob:
            st.error(
                "L2 data available nahi hai."
            )
            st.stop()

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Best Bid",
            f'{ob["best_bid"]:.8g}',
        )

        c2.metric(
            "Best Ask",
            f'{ob["best_ask"]:.8g}',
        )

        c3.metric(
            "Bid Depth",
            f'{ob["bid_depth"]:,.2f}',
        )

        c4.metric(
            "Ask Depth",
            f'{ob["ask_depth"]:,.2f}',
        )

        imbalance = ob["imbalance"]

        if imbalance >= 25:

            st.success(
                "🟢 Bid Dominant: "
                + str(round(imbalance, 2))
                + "%"
            )

        elif imbalance <= -25:

            st.error(
                "🔴 Ask Dominant: "
                + str(round(imbalance, 2))
                + "%"
            )

        else:

            st.info(
                "⚪ Balanced: "
                + str(round(imbalance, 2))
                + "%"
            )

        left, right = st.columns(2)

        with left:

            st.write(
                "🟢 BID"
            )

            bid = ob["bid"].copy()

            bid["Notional"] = (
                bid["Price"]
                * bid["Size"]
            )

            st.dataframe(
                bid,
                use_container_width=True,
                hide_index=True,
            )

        with right:

            st.write(
                "🔴 ASK"
            )

            ask = ob["ask"].copy()

            ask["Notional"] = (
                ask["Price"]
                * ask["Size"]
            )

            st.dataframe(
                ask,
                use_container_width=True,
                hide_index=True,
            )

        st.warning(
            "L2 wall cancel ho sakti hai. "
            "Visible liquidity ko guaranteed support "
            "ya resistance na samjhein."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Delta Exchange India public API based analytical scanner."
)

st.caption(
    "⚠️ Vol/OI leverage nahi hai. "
    "Ye Volume ÷ Open Interest ratio hai."
)

st.caption(
    "⚠️ Actual liquidation feed aur liquidation-proxy "
    "alag concepts hain. Is scanner mein public data "
    "se analytical signals generate kiye ja rahe hain."
)
