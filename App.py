import streamlit as st
import requests
import pandas as pd
import numpy as np
import time

# ============================================================
# DELTA EXCHANGE INDIA - AUTO MTF / S-R / LEVERAGE SCANNER
# Version 3.0
#
# Main features:
# - ALL live perpetual contracts
# - Vol/OI > configurable threshold (default 3)
# - Leverage buckets (20x+ are NOT removed)
# - Automatic MTF Support / Resistance
# - 6H=120, 12H=60, 1D=365, 1W=52, 1M=12
# - Repeated support/resistance
# - Higher-high / lower-low sequences
# - Breakout / retest / failed-breakout detection
# - Candle details for detected levels
# - Automatic ranking; no manual S/R hunting
#
# IMPORTANT:
# Vol/OI = 24H Volume / Open Interest. It is NOT leverage.
# Actual maximum leverage depends on the product metadata/API.
# ============================================================

BASE_URL = "https://api.india.delta.exchange"
HEADERS = {
    "Accept": "application/json",
    "User-Agent": "Delta-Auto-MTF-Scanner/3.0",
}

CACHE_TTL = 30
L2_CACHE_TTL = 10

st.set_page_config(
    page_title="Delta Auto MTF Scanner",
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

def first_number(d, keys):
    for k in keys:
        v = d.get(k)
        if v is not None and v != "":
            try:
                return float(v)
            except Exception:
                pass
    return np.nan


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

        # Delta product schemas can change; read common leverage names
        # without assuming that one field is always present.
        max_lev = first_number(
            p,
            [
                "max_leverage",
                "maximum_leverage",
                "leverage",
                "default_leverage",
            ],
        )
        default_lev = first_number(
            p,
            ["default_leverage"],
        )

        rows.append(
            {
                "Coin": symbol,
                "ID": p.get("id"),
                "Max Leverage": max_lev,
                "Default Leverage": default_lev,
                "Product": p,
            }
        )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).drop_duplicates("Coin").reset_index(drop=True)


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
        df["24H Volume"] /
        df["OI"].replace(0, np.nan)
    )
    return df


# ============================================================
# CANDLES
# ============================================================

SECONDS = {
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
    # Calendar months are not exactly 30 days, but this is
    # only used to build the API start window.
    "1M": 2592000,
}


@st.cache_data(ttl=CACHE_TTL)
def get_history(symbol, resolution, candles):
    now = int(time.time())
    step = SECONDS.get(resolution, 300)

    start = now - step * (int(candles) + 15)

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

    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "time" in df.columns:
        df["time"] = pd.to_numeric(df["time"], errors="coerce")
        df = df.sort_values("time")
        df = df.drop_duplicates("time")

    required = ["open", "high", "low", "close"]
    if not all(c in df.columns for c in required):
        return pd.DataFrame()

    df = df.dropna(subset=required)
    return df.tail(int(candles)).reset_index(drop=True)


# ============================================================
# BASIC INDICATORS
# ============================================================

def trend(df):
    if len(df) < 30:
        return "UNKNOWN"

    close = df["close"]
    e9 = close.ewm(span=9, adjust=False).mean()
    e21 = close.ewm(span=21, adjust=False).mean()
    e50 = close.ewm(span=50, adjust=False).mean()
    last = close.iloc[-1]

    if last > e9.iloc[-1] > e21.iloc[-1] > e50.iloc[-1]:
        return "BULL"
    if last < e9.iloc[-1] < e21.iloc[-1] < e50.iloc[-1]:
        return "BEAR"
    return "MIXED"


def volume_multiple(df):
    if len(df) < 8 or "volume" not in df.columns:
        return 0.0

    avg = df["volume"].iloc[-7:-1].mean()
    if avg <= 0:
        return 0.0
    return float(df["volume"].iloc[-1] / avg)


# ============================================================
# PIVOTS / S-R
# ============================================================

def find_levels(df, lookback=2):
    if len(df) < lookback * 2 + 5:
        return [], []

    highs = []
    lows = []

    for i in range(lookback, len(df) - lookback):
        h = float(df["high"].iloc[i])
        l = float(df["low"].iloc[i])

        left_h = df["high"].iloc[i-lookback:i].max()
        right_h = df["high"].iloc[i+1:i+lookback+1].max()

        left_l = df["low"].iloc[i-lookback:i].min()
        right_l = df["low"].iloc[i+1:i+lookback+1].min()

        if h >= left_h and h >= right_h:
            highs.append({"index": i, "price": h})

        if l <= left_l and l <= right_l:
            lows.append({"index": i, "price": l})

    return highs, lows


def cluster_levels(levels, tolerance=0.005):
    if not levels:
        return []

    clusters = []

    for item in levels:
        price = float(item["price"])
        found = None

        for c in clusters:
            center = c["price"]
            if center != 0 and abs(price-center)/abs(center) <= tolerance:
                found = c
                break

        if found is None:
            clusters.append({
                "price": price,
                "touches": 1,
                "prices": [price],
                "indexes": [item["index"]],
            })
        else:
            found["touches"] += 1
            found["prices"].append(price)
            found["indexes"].append(item["index"])
            found["price"] = float(np.mean(found["prices"]))

    return sorted(
        clusters,
        key=lambda x: (x["touches"], x["price"]),
        reverse=True,
    )


def candle_time(df, index):
    if "time" not in df.columns:
        return str(index)

    try:
        return time.strftime(
            "%Y-%m-%d %H:%M",
            time.gmtime(int(df["time"].iloc[index])),
        )
    except Exception:
        return str(index)


def candle_detail(df, index, level_type, level_price):
    row = df.iloc[index]

    return {
        "Candle Index": int(index),
        "Candle Time UTC": candle_time(df, index),
        "Type": level_type,
        "Level": float(level_price),
        "Open": float(row["open"]),
        "High": float(row["high"]),
        "Low": float(row["low"]),
        "Close": float(row["close"]),
        "Volume": (
            float(row["volume"])
            if "volume" in df.columns
            and pd.notna(row.get("volume", np.nan))
            else np.nan
        ),
    }


def nearest_sr(df, tolerance=0.006):
    if df.empty:
        return {
            "support": np.nan,
            "resistance": np.nan,
            "major_support": np.nan,
            "major_resistance": np.nan,
            "support_touches": 0,
            "resistance_touches": 0,
            "support_details": [],
            "resistance_details": [],
        }

    current = float(df["close"].iloc[-1])
    highs, lows = find_levels(df, lookback=2)

    hc = cluster_levels(highs, tolerance)
    lc = cluster_levels(lows, tolerance)

    supports = [x for x in lc if x["price"] < current]
    resistances = [x for x in hc if x["price"] > current]

    support = max(
        [x["price"] for x in supports],
        default=np.nan,
    )
    resistance = min(
        [x["price"] for x in resistances],
        default=np.nan,
    )

    # Major level = highest-touch level below/above current.
    major_support_item = (
        max(supports, key=lambda x: x["touches"])
        if supports else None
    )
    major_res_item = (
        max(resistances, key=lambda x: x["touches"])
        if resistances else None
    )

    support_details = []
    for c in sorted(
        supports,
        key=lambda x: abs(current-x["price"])
    )[:5]:
        for idx in c["indexes"][-5:]:
            support_details.append(
                candle_detail(
                    df, idx, "SUPPORT", c["price"]
                )
            )

    resistance_details = []
    for c in sorted(
        resistances,
        key=lambda x: abs(current-x["price"])
    )[:5]:
        for idx in c["indexes"][-5:]:
            resistance_details.append(
                candle_detail(
                    df, idx, "RESISTANCE", c["price"]
                )
            )

    return {
        "support": support,
        "resistance": resistance,
        "major_support": (
            major_support_item["price"]
            if major_support_item else np.nan
        ),
        "major_resistance": (
            major_res_item["price"]
            if major_res_item else np.nan
        ),
        "support_touches": (
            major_support_item["touches"]
            if major_support_item else 0
        ),
        "resistance_touches": (
            major_res_item["touches"]
            if major_res_item else 0
        ),
        "support_details": support_details,
        "resistance_details": resistance_details,
    }


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

    highs, lows = find_levels(df, lookback=2)
    hc = cluster_levels(highs)
    lc = cluster_levels(lows)

    details = []

    for c in hc:
        if c["touches"] >= 2:
            for idx in c["indexes"][-5:]:
                details.append(
                    candle_detail(
                        df,
                        idx,
                        "REPEATED RESISTANCE",
                        c["price"],
                    )
                )

    for c in lc:
        if c["touches"] >= 2:
            for idx in c["indexes"][-5:]:
                details.append(
                    candle_detail(
                        df,
                        idx,
                        "REPEATED SUPPORT",
                        c["price"],
                    )
                )

    if len(highs) >= 3:
        rh = highs[-4:]
        vals = [x["price"] for x in rh]
        if all(vals[i] > vals[i-1] for i in range(1, len(vals))):
            details.append({
                "Candle Index": rh[-1]["index"],
                "Candle Time UTC": candle_time(df, rh[-1]["index"]),
                "Type": "HIGHER-HIGH SEQUENCE",
                "Level": vals[-1],
                "Open": np.nan,
                "High": vals[-1],
                "Low": np.nan,
                "Close": np.nan,
                "Volume": np.nan,
            })

    if len(lows) >= 3:
        rl = lows[-4:]
        vals = [x["price"] for x in rl]
        if all(vals[i] < vals[i-1] for i in range(1, len(vals))):
            details.append({
                "Candle Index": rl[-1]["index"],
                "Candle Time UTC": candle_time(df, rl[-1]["index"]),
                "Type": "LOWER-LOW SEQUENCE",
                "Level": vals[-1],
                "Open": np.nan,
                "High": np.nan,
                "Low": vals[-1],
                "Close": np.nan,
                "Volume": np.nan,
            })

    if not details:
        return {
            "pattern": "NONE",
            "count": 0,
            "details": [],
        }

    names = list(dict.fromkeys(x["Type"] for x in details))
    return {
        "pattern": " | ".join(names),
        "count": len(details),
        "details": details,
    }


# ============================================================
# BREAKOUT / RETEST / FAILED BREAKOUT
# ============================================================

def breakout_status(df):
    if len(df) < 8:
        return "NONE", np.nan

    current = float(df["close"].iloc[-1])

    highs, lows = find_levels(df.iloc[:-1], lookback=2)
    resistance = max(
        [x["price"] for x in highs],
        default=np.nan,
    )
    support = min(
        [x["price"] for x in lows],
        default=np.nan,
    )

    recent = df.tail(5)

    if pd.notna(resistance):
        if current > resistance:
            return "BULL BREAKOUT", resistance

        # Failed breakout: high crossed resistance but closed below.
        if (
            recent["high"].max() > resistance
            and recent["close"].iloc[-1] < resistance
        ):
            return "FAILED BULL BREAKOUT", resistance

        # Retest: previous candle above, current close back above.
        if len(recent) >= 2:
            prev = recent.iloc[-2]
            last = recent.iloc[-1]
            if (
                prev["close"] > resistance
                and last["low"] <= resistance
                and last["close"] > resistance
            ):
                return "BULL RETEST HOLD", resistance

    if pd.notna(support):
        if current < support:
            return "BEAR BREAKDOWN", support

        if (
            recent["low"].min() < support
            and recent["close"].iloc[-1] > support
        ):
            return "FAILED BEAR BREAKDOWN", support

        if len(recent) >= 2:
            prev = recent.iloc[-2]
            last = recent.iloc[-1]
            if (
                prev["close"] < support
                and last["high"] >= support
                and last["close"] < support
            ):
                return "BEAR RETEST FAIL", support

    return "NONE", np.nan


# ============================================================
# TIMEFRAME CONFIG
# ============================================================

TIMEFRAMES = {
    "6H": ("6h", 120),
    "12H": ("12h", 60),
    "1D": ("1d", 365),
    "1W": ("1w", 52),
    "1M": ("1M", 12),
}


def fmt(v):
    if v is None or pd.isna(v):
        return "N/A"
    return f"{float(v):.8g}"


# ============================================================
# AUTO MTF ANALYSIS
# ============================================================

def timeframe_analysis(symbol):
    output = {}

    for name, (resolution, candles) in TIMEFRAMES.items():
        df = get_history(symbol, resolution, candles)

        if len(df) < 10:
            output[name] = {
                "trend": "NO DATA",
                "support": np.nan,
                "resistance": np.nan,
                "major_support": np.nan,
                "major_resistance": np.nan,
                "support_touches": 0,
                "resistance_touches": 0,
                "pattern": "NO DATA",
                "repeats": 0,
                "breakout": "NO DATA",
                "break_level": np.nan,
                "details": [],
                "candles": df,
            }
            continue

        sr = nearest_sr(df)
        rep = repeated_structure(df)
        bo, bo_level = breakout_status(df)

        output[name] = {
            "trend": trend(df),
            "support": sr["support"],
            "resistance": sr["resistance"],
            "major_support": sr["major_support"],
            "major_resistance": sr["major_resistance"],
            "support_touches": sr["support_touches"],
            "resistance_touches": sr["resistance_touches"],
            "pattern": rep["pattern"],
            "repeats": rep["count"],
            "breakout": bo,
            "break_level": bo_level,
            "details": (
                sr["support_details"]
                + sr["resistance_details"]
                + rep["details"]
            ),
            "candles": df,
        }

    return output


# ============================================================
# MTF SCORE
# ============================================================

def mtf_score(analysis, price):
    long_score = 0
    short_score = 0
    bullish_tfs = 0
    bearish_tfs = 0
    sr_rows = []

    for tf, d in analysis.items():
        tr = d["trend"]

        if tr == "BULL":
            bullish_tfs += 1
            long_score += 2
        elif tr == "BEAR":
            bearish_tfs += 1
            short_score += 2

        if pd.notna(d["support"]):
            distance = abs(price - d["support"]) / price * 100
            if distance <= 3:
                long_score += 1

        if pd.notna(d["resistance"]):
            distance = abs(d["resistance"] - price) / price * 100
            if distance <= 3:
                short_score += 1

        if d["support_touches"] >= 3:
            long_score += 1

        if d["resistance_touches"] >= 3:
            short_score += 1

        if "BULL" in d["breakout"]:
            long_score += 3
        if "BEAR" in d["breakout"]:
            short_score += 3

        sr_rows.append({
            "Timeframe": tf,
            "Trend": tr,
            "Support": d["support"],
            "Resistance": d["resistance"],
            "Major Support": d["major_support"],
            "Major Resistance": d["major_resistance"],
            "S Touches": d["support_touches"],
            "R Touches": d["resistance_touches"],
            "Structure": d["pattern"],
            "Breakout/Retest": d["breakout"],
        })

    if bullish_tfs >= 4:
        bias = "STRONG LONG"
    elif bearish_tfs >= 4:
        bias = "STRONG SHORT"
    elif bullish_tfs >= 3:
        bias = "LONG BIAS"
    elif bearish_tfs >= 3:
        bias = "SHORT BIAS"
    else:
        bias = "MIXED"

    return {
        "long_score": long_score,
        "short_score": short_score,
        "score": max(long_score, short_score),
        "bias": bias,
        "sr_rows": sr_rows,
    }


# ============================================================
# LEVERAGE BUCKET
# ============================================================

def leverage_bucket(max_lev):
    if pd.isna(max_lev):
        return "Leverage N/A"
    if max_lev <= 10:
        return "≤10x"
    if max_lev <= 20:
        return ">10x–20x"
    if max_lev <= 50:
        return ">20x–50x"
    return ">50x"


# ============================================================
# MARKET-WIDE AUTO SCAN
# ============================================================

def auto_scan_row(row):
    symbol = row["Coin"]
    price = float(row["Price"])

    analysis = timeframe_analysis(symbol)
    scores = mtf_score(analysis, price)

    # A compact structure summary for the master table.
    patterns = []
    breakouts = []
    for tf, d in analysis.items():
        if d["pattern"] not in ("NONE", "NO DATA"):
            patterns.append(tf + ": " + d["pattern"])
        if d["breakout"] not in ("NONE", "NO DATA"):
            breakouts.append(tf + ": " + d["breakout"])

    # Nearest S/R across all timeframes.
    supports = []
    resistances = []
    for tf, d in analysis.items():
        if pd.notna(d["support"]):
            supports.append((tf, d["support"]))
        if pd.notna(d["resistance"]):
            resistances.append((tf, d["resistance"]))

    nearest_support = np.nan
    nearest_support_tf = ""
    if supports:
        x = min(
            supports,
            key=lambda z: abs(price-z[1])
        )
        nearest_support_tf, nearest_support = x

    nearest_resistance = np.nan
    nearest_resistance_tf = ""
    if resistances:
        x = min(
            resistances,
            key=lambda z: abs(price-z[1])
        )
        nearest_resistance_tf, nearest_resistance = x

    signal = "NO SIGNAL"
    if scores["bias"] == "STRONG LONG" and scores["long_score"] >= 7:
        signal = "STRONG LONG"
    elif scores["bias"] in ("LONG BIAS",) and scores["long_score"] >= 5:
        signal = "LONG WATCH"
    elif scores["bias"] == "STRONG SHORT" and scores["short_score"] >= 7:
        signal = "STRONG SHORT"
    elif scores["bias"] in ("SHORT BIAS",) and scores["short_score"] >= 5:
        signal = "SHORT WATCH"

    return {
        "Coin": symbol,
        "Price": price,
        "Vol/OI": float(row["Vol/OI"]) if pd.notna(row["Vol/OI"]) else np.nan,
        "Max Leverage": row["Max Leverage"],
        "Default Leverage": row["Default Leverage"],
        "Leverage Bucket": leverage_bucket(row["Max Leverage"]),
        "MTF Bias": scores["bias"],
        "MTF Score": scores["score"],
        "Long Score": scores["long_score"],
        "Short Score": scores["short_score"],
        "Nearest Support": nearest_support,
        "Support TF": nearest_support_tf,
        "Nearest Resistance": nearest_resistance,
        "Resistance TF": nearest_resistance_tf,
        "Repeated Structure": " || ".join(patterns) if patterns else "NONE",
        "Breakout / Retest": " || ".join(breakouts) if breakouts else "NONE",
        "Signal": signal,
        "_analysis": analysis,
    }


# ============================================================
# OPTIONAL L2
# ============================================================

@st.cache_data(ttl=L2_CACHE_TTL)
def get_orderbook(symbol, depth=15):
    result = api_get(
        "/v2/l2orderbook/" + symbol,
        {"depth": int(depth)},
    )
    return result if isinstance(result, dict) else None


def orderbook_stats(symbol, depth=15):
    data = get_orderbook(symbol, depth)
    if not data:
        return None

    bids = data.get("buy") or []
    asks = data.get("sell") or []

    bid = []
    ask = []

    for x in bids:
        try:
            bid.append({
                "Price": float(x["price"]),
                "Size": float(x["size"]),
            })
        except Exception:
            pass

    for x in asks:
        try:
            ask.append({
                "Price": float(x["price"]),
                "Size": float(x["size"]),
            })
        except Exception:
            pass

    if not bid or not ask:
        return None

    bid = pd.DataFrame(bid).sort_values("Price", ascending=False)
    ask = pd.DataFrame(ask).sort_values("Price", ascending=True)

    bd = float(bid["Size"].sum())
    ad = float(ask["Size"].sum())
    total = bd + ad

    imbalance = (bd-ad)/total*100 if total else 0.0

    return {
        "bid": bid,
        "ask": ask,
        "bid_depth": bd,
        "ask_depth": ad,
        "imbalance": imbalance,
        "best_bid": float(bid["Price"].max()),
        "best_ask": float(ask["Price"].min()),
    }


# ============================================================
# LOAD MARKET
# ============================================================

products = get_products()
tickers = get_tickers()

if products.empty or tickers.empty:
    st.error("Delta market data load nahi hua. Refresh karke dobara try karein.")
    st.stop()

market = products.merge(tickers, on="Coin", how="inner")
market = market.dropna(subset=["Price"]).copy()
market = market.sort_values("24H Volume", ascending=False).reset_index(drop=True)

# ============================================================
# HEADER
# ============================================================

st.title("🔥 Delta Auto MTF / S-R Scanner")
st.caption(
    "All live perpetuals → Vol/OI filter → leverage buckets → "
    "automatic 6H/12H/1D/1W/1M S/R → repeated structure → breakout/retest"
)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("⚙️ Scanner Settings")

    min_vol_oi = st.number_input(
        "Minimum Vol/OI",
        min_value=0.0,
        value=3.0,
        step=0.5,
    )

    scan_count = st.slider(
        "Automatic deep S/R scan",
        min_value=5,
        max_value=min(150, max(5, len(market))),
        value=min(40, max(5, len(market))),
        step=5,
        help="Market table mein saare eligible coins rahenge. Deep MTF analysis API load ko control karne ke liye top N eligible coins par chalega.",
    )

    depth = st.slider(
        "L2 depth",
        min_value=5,
        max_value=50,
        value=15,
        step=5,
    )

    if st.button("🔄 Refresh All"):
        st.cache_data.clear()
        st.rerun()

eligible = market[
    market["Vol/OI"].fillna(0) > min_vol_oi
].copy()

# ============================================================
# MARKET OVERVIEW
# ============================================================

st.subheader("📊 Market Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric("All Perpetuals", len(market))
c2.metric(f"Vol/OI > {min_vol_oi:g}", len(eligible))

known_lev = market["Max Leverage"].dropna()
c3.metric(
    "Known Leverage",
    len(known_lev),
)

c4.metric(
    "Highest Vol/OI",
    round(float(market["Vol/OI"].max()), 2)
    if not market.empty else 0,
)

# ============================================================
# ALL ELIGIBLE MARKET TABLE
# ============================================================

st.subheader("📋 All Eligible Coins")

market_cols = [
    "Coin",
    "Price",
    "24H Volume",
    "OI",
    "Vol/OI",
    "Max Leverage",
    "Default Leverage",
    "Leverage Bucket",
    "Funding",
]

market_view = eligible[market_cols].copy()
st.dataframe(
    market_view.head(300),
    use_container_width=True,
    hide_index=True,
)

# ============================================================
# LEVERAGE TABLES
# ============================================================

st.subheader("⚡ Leverage-wise Tables")

lev_tabs = st.tabs([
    "≤10x",
    ">10x–20x",
    ">20x–50x",
    ">50x",
    "Leverage N/A",
])

for tab, bucket in zip(
    lev_tabs,
    ["≤10x", ">10x–20x", ">20x–50x", ">50x", "Leverage N/A"],
):
    with tab:
        x = eligible[
            eligible["Leverage Bucket"] == bucket
        ][market_cols].copy()

        st.write(f"{bucket}: {len(x)} coins")
        st.dataframe(
            x.head(300),
            use_container_width=True,
            hide_index=True,
        )

# ============================================================
# NAVIGATION
# ============================================================

mode = st.radio(
    "Analysis",
    [
        "🔥 AUTO MTF SCANNER",
        "📐 AUTO S/R DETAILS",
        "🔁 REPEATED STRUCTURE",
        "📚 L2 ORDER BOOK",
    ],
    horizontal=True,
)

# ============================================================
# AUTO SCANNER
# ============================================================

if mode == "🔥 AUTO MTF SCANNER":
    st.subheader("🔥 Automatic Market-wide MTF Scanner")

    if eligible.empty:
        st.warning("Vol/OI filter ke baad koi coin nahi mila.")
        st.stop()

    candidates = eligible.sort_values(
        ["Vol/OI", "24H Volume"],
        ascending=False,
    ).head(scan_count)

    st.info(
        f"{len(candidates)} eligible coins ka automatic "
        f"6H/12H/1D/1W/1M S/R scan chalega. "
        "20x+ leverage coins filter se remove nahi kiye gaye hain."
    )

    results = []
    progress = st.progress(0)
    total = len(candidates)

    for i, (_, row) in enumerate(candidates.iterrows()):
        try:
            results.append(auto_scan_row(row))
        except Exception:
            pass

        progress.progress(int((i + 1) / total * 100))

    progress.empty()

    if not results:
        st.warning("Automatic scan ke liye enough candle data nahi mila.")
        st.stop()

    scan_df = pd.DataFrame(results)

    display_cols = [
        "Coin",
        "Price",
        "Vol/OI",
        "Max Leverage",
        "Leverage Bucket",
        "MTF Bias",
        "MTF Score",
        "Nearest Support",
        "Support TF",
        "Nearest Resistance",
        "Resistance TF",
        "Repeated Structure",
        "Breakout / Retest",
        "Signal",
    ]

    st.dataframe(
        scan_df.sort_values(
            ["MTF Score", "Vol/OI"],
            ascending=False,
        )[display_cols],
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("🟢 Long Watch")
    long_df = scan_df[
        scan_df["Long Score"] > scan_df["Short Score"]
    ].sort_values(
        ["Long Score", "Vol/OI"],
        ascending=False,
    )
    st.dataframe(
        long_df[display_cols].head(30),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("🔴 Short Watch")
    short_df = scan_df[
        scan_df["Short Score"] > scan_df["Long Score"]
    ].sort_values(
        ["Short Score", "Vol/OI"],
        ascending=False,
    )
    st.dataframe(
        short_df[display_cols].head(30),
        use_container_width=True,
        hide_index=True,
    )

    # Save results in session state for detail pages.
    st.session_state["auto_scan_df"] = scan_df

# ============================================================
# AUTO S/R DETAILS
# ============================================================

elif mode == "📐 AUTO S/R DETAILS":
    st.subheader("📐 Automatic Multi-Timeframe Support / Resistance")

    if eligible.empty:
        st.warning("Eligible coins nahi mile.")
        st.stop()

    # Use automatic ranking; user does NOT need to manually hunt S/R.
    ranked = eligible.sort_values(
        ["Vol/OI", "24H Volume"],
        ascending=False,
    )

    symbol = st.selectbox(
        "Coin",
        ranked["Coin"].head(scan_count).tolist(),
        help="List automatically Vol/OI ke basis par ban rahi hai.",
    )

    if st.button("🔎 Show Automatic S/R"):
        analysis = timeframe_analysis(symbol)
        price = float(
            ranked.loc[
                ranked["Coin"] == symbol,
                "Price"
            ].iloc[0]
        )

        st.metric("Current Price", fmt(price))

        rows = []
        for tf, d in analysis.items():
            rows.append({
                "Timeframe": tf,
                "Trend": d["trend"],
                "Support": d["support"],
                "Resistance": d["resistance"],
                "Major Support": d["major_support"],
                "Major Resistance": d["major_resistance"],
                "S Touches": d["support_touches"],
                "R Touches": d["resistance_touches"],
                "Repeated Structure": d["pattern"],
                "Breakout / Retest": d["breakout"],
            })

        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("### 🧱 Detected S/R Candle Details")

        for tf, d in analysis.items():
            st.markdown(f"#### {tf}")

            details = d["details"]
            if not details:
                st.caption("Is timeframe par strong level detail nahi mila.")
                continue

            detail_df = pd.DataFrame(details).drop_duplicates(
                subset=["Candle Index", "Type", "Level"]
            )

            st.dataframe(
                detail_df,
                use_container_width=True,
                hide_index=True,
            )

# ============================================================
# REPEATED STRUCTURE
# ============================================================

elif mode == "🔁 REPEATED STRUCTURE":
    st.subheader("🔁 Automatic Repeated Resistance / Support")

    if eligible.empty:
        st.warning("Eligible coins nahi mile.")
        st.stop()

    candidates = eligible.sort_values(
        ["Vol/OI", "24H Volume"],
        ascending=False,
    ).head(scan_count)

    rows = []
    details = []

    progress = st.progress(0)
    total = len(candidates)

    for i, (_, row) in enumerate(candidates.iterrows()):
        symbol = row["Coin"]

        try:
            analysis = timeframe_analysis(symbol)

            for tf, d in analysis.items():
                if d["pattern"] not in ("NONE", "NO DATA"):
                    rows.append({
                        "Coin": symbol,
                        "Vol/OI": row["Vol/OI"],
                        "Max Leverage": row["Max Leverage"],
                        "Timeframe": tf,
                        "Trend": d["trend"],
                        "Pattern": d["pattern"],
                        "Repeat Count": d["repeats"],
                        "Support": d["support"],
                        "Resistance": d["resistance"],
                    })

                    for item in d["details"]:
                        z = dict(item)
                        z["Coin"] = symbol
                        z["Timeframe"] = tf
                        z["Vol/OI"] = row["Vol/OI"]
                        z["Max Leverage"] = row["Max Leverage"]
                        details.append(z)
        except Exception:
            pass

        progress.progress(int((i + 1) / total * 100))

    progress.empty()

    if rows:
        st.dataframe(
            pd.DataFrame(rows).sort_values(
                ["Repeat Count", "Vol/OI"],
                ascending=False,
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Repeated structure nahi mila.")

    if details:
        st.markdown("### 🕯️ Exact Candle Details")
        detail_df = pd.DataFrame(details).drop_duplicates(
            subset=["Coin", "Timeframe", "Candle Index", "Type", "Level"]
        )
        st.dataframe(
            detail_df.sort_values(
                ["Coin", "Timeframe", "Candle Index"]
            ),
            use_container_width=True,
            hide_index=True,
        )

# ============================================================
# L2
# ============================================================

else:
    st.subheader("📚 Live L2 Order Book")

    symbol = st.selectbox(
        "Coin",
        eligible["Coin"].tolist() if not eligible.empty
        else market["Coin"].tolist(),
        key="l2_coin",
    )

    if st.button("Load L2"):
        ob = orderbook_stats(symbol, depth)

        if not ob:
            st.error("L2 data available nahi hai.")
            st.stop()

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Best Bid", fmt(ob["best_bid"]))
        c2.metric("Best Ask", fmt(ob["best_ask"]))
        c3.metric("Bid Depth", f'{ob["bid_depth"]:,.2f}')
        c4.metric("Ask Depth", f'{ob["ask_depth"]:,.2f}')

        imb = ob["imbalance"]

        if imb >= 25:
            st.success(f"🟢 Bid Dominant: {imb:.2f}%")
        elif imb <= -25:
            st.error(f"🔴 Ask Dominant: {imb:.2f}%")
        else:
            st.info(f"⚪ Balanced: {imb:.2f}%")

        left, right = st.columns(2)

        with left:
            bid = ob["bid"].copy()
            bid["Notional"] = bid["Price"] * bid["Size"]
            st.write("🟢 BID")
            st.dataframe(
                bid,
                use_container_width=True,
                hide_index=True,
            )

        with right:
            ask = ob["ask"].copy()
            ask["Notional"] = ask["Price"] * ask["Size"]
            st.write("🔴 ASK")
            st.dataframe(
                ask,
                use_container_width=True,
                hide_index=True,
            )

        st.warning(
            "Visible L2 walls cancel ho sakti hain; "
            "inhe guaranteed support/resistance na samjhein."
        )

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Delta Exchange India public API based analytical scanner."
)

st.caption(
    "Vol/OI = 24H Volume ÷ Open Interest. "
    "Ye leverage nahi hai."
)

st.caption(
    "20x+ coins ko deliberately filter nahi kiya gaya. "
    "Leverage metadata API mein available na ho to N/A bucket use hota hai."
)

st.caption(
    "S/R pivot-based analytical levels hain; visible orderbook "
    "liquidity aur historical S/R guaranteed future levels nahi hain."
)
