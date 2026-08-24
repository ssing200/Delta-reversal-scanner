import streamlit as st
import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime, timezone

# ============================================================
# DELTA MTF STRUCTURE + REVERSAL SCANNER
# ============================================================
# Public Delta Exchange India API
#
# Main logic:
# 1. ALL live perpetual contracts
# 2. Vol/OI > 3 filter
# 3. Leverage is DISPLAY ONLY, NOT a filter
# 4. MTF trend
# 5. MTF Support / Resistance
# 6. Repeated resistance / support detection
# 7. BOS + liquidity sweep
# 8. Price + OI relationship
# 9. Volume expansion
# 10. Funding
# 11. Public trade delta
# 12. L2 order-book imbalance
# 13. Confluence score
# 14. A+ / Strong / Watch / No Trade
# ============================================================

BASE_URL = "https://api.india.delta.exchange"

HEADERS = {
    "Accept": "application/json",
    "User-Agent": "Delta-MTF-Structure-Scanner/2.0",
}

st.set_page_config(
    page_title="Delta MTF Structure Scanner",
    page_icon="🔥",
    layout="wide",
)

# ============================================================
# SETTINGS
# ============================================================

CACHE_TTL = 30
L2_CACHE_TTL = 10

MIN_VOL_OI = 3.0
DEFAULT_DEPTH = 15
DEFAULT_SCAN_LIMIT = 50

# Timeframes requested by you
TIMEFRAMES = {
    "1M": {
        "resolution": "1M",
        "candles": 12,
        "hours": 24 * 31 * 12,
    },
    "1W": {
        "resolution": "1W",
        "candles": 52,
        "hours": 24 * 7 * 52,
    },
    "1D": {
        "resolution": "1D",
        "candles": 365,
        "hours": 24 * 365,
    },
    "12H": {
        "resolution": "12h",
        "candles": 60,
        "hours": 12 * 60,
    },
    "6H": {
        "resolution": "6h",
        "candles": 120,
        "hours": 6 * 120,
    },
    "1H": {
        "resolution": "1h",
        "candles": 120,
        "hours": 120,
    },
    "15m": {
        "resolution": "15m",
        "candles": 200,
        "hours": 50,
    },
    "5m": {
        "resolution": "5m",
        "candles": 250,
        "hours": 21,
    },
}

# ============================================================
# API
# ============================================================

def api_get(path, params=None, timeout=15):
    try:
        response = requests.get(
            BASE_URL + path,
            params=params or {},
            headers=HEADERS,
            timeout=timeout,
        )

        if response.status_code != 200:
            return None

        payload = response.json()

        if isinstance(payload, dict):
            if payload.get("success", True) is False:
                return None

            return payload.get("result")

        return None

    except (
        requests.RequestException,
        ValueError,
        TypeError,
    ):
        return None


# ============================================================
# PRODUCTS
# ============================================================

@st.cache_data(ttl=CACHE_TTL)
def get_perpetual_products():

    result = api_get("/v2/products")

    if not result:
        return pd.DataFrame()

    rows = []

    for item in result:

        if item.get("contract_type") != "perpetual_futures":
            continue

        if item.get("state") != "live":
            continue

        if item.get("trading_status") != "operational":
            continue

        symbol = item.get("symbol")

        if not symbol:
            continue

        # Leverage is DISPLAY ONLY.
        leverage = (
            item.get("max_leverage")
            or item.get("default_leverage")
            or item.get("leverage")
            or item.get("maximum_leverage")
        )

        try:
            leverage = float(leverage)
        except (TypeError, ValueError):
            leverage = np.nan

        rows.append(
            {
                "Coin": symbol,
                "ID": item.get("id"),
                "Leverage": leverage,
                "Description": item.get("description", ""),
            }
        )

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

@st.cache_data(ttl=CACHE_TTL)
def get_tickers():

    result = api_get("/v2/tickers")

    if not result:
        return pd.DataFrame()

    rows = []

    for item in result:

        symbol = item.get("symbol")

        if not symbol:
            continue

        try:
            price = float(
                item.get("close")
                or item.get("mark_price")
                or 0
            )

            volume = float(
                item.get("volume_24h")
                or item.get("volume")
                or 0
            )

            oi = float(
                item.get("open_interest")
                or item.get("oi")
                or 0
            )

        except (TypeError, ValueError):
            continue

        if price <= 0:
            continue

        funding_raw = item.get("funding_rate")

        try:
            funding = (
                float(funding_raw)
                if funding_raw is not None
                else np.nan
            )
        except (TypeError, ValueError):
            funding = np.nan

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
# HISTORY
# ============================================================

@st.cache_data(ttl=CACHE_TTL)
def get_history(
    symbol,
    resolution,
    candles,
):

    # Extra candles for indicator calculations
    multiplier = 1.25

    hours = TIMEFRAMES.get(
        resolution,
        {}
    ).get(
        "hours",
        48,
    )

    end = int(time.time())

    start = end - int(
        hours * 3600 * multiplier
    )

    result = api_get(
        "/v2/history/candles",
        {
            "resolution": resolution,
            "symbol": symbol,
            "start": start,
            "end": end,
        },
    )

    if not result:
        return pd.DataFrame()

    df = pd.DataFrame(result)

    if df.empty:
        return df

    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    for col in numeric_cols:

        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
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

    if not all(
        col in df.columns
        for col in required
    ):
        return pd.DataFrame()

    df = df.dropna(
        subset=required
    )

    if "time" in df.columns:

        df = df.drop_duplicates(
            "time"
        )

    # Keep requested amount + small buffer
    return df.tail(
        candles
    ).reset_index(drop=True)


# ============================================================
# ATR
# ============================================================

def add_atr(
    df,
    period=14,
):

    x = df.copy()

    previous_close = (
        x["close"].shift(1)
    )

    tr = pd.concat(
        [
            x["high"] - x["low"],
            (
                x["high"]
                - previous_close
            ).abs(),
            (
                x["low"]
                - previous_close
            ).abs(),
        ],
        axis=1,
    ).max(axis=1)

    x["TR"] = tr

    x["ATR"] = (
        tr.rolling(period)
        .mean()
    )

    return x


# ============================================================
# TREND
# ============================================================

def trend_label(df):

    if len(df) < 30:
        return "⚪ UNKNOWN"

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

    last = float(
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
        last > e9
        and e9 > e21
        and e21 > e50
    ):
        return "🟢 BULL"

    if (
        last < e9
        and e9 < e21
        and e21 < e50
    ):
        return "🔴 BEAR"

    return "🟡 MIXED"


# ============================================================
# SWING POINTS
# ============================================================

def find_swing_points(
    df,
    left=2,
    right=2,
):

    highs = []
    lows = []

    if len(df) < (
        left + right + 5
    ):
        return highs, lows

    for i in range(
        left,
        len(df) - right,
    ):

        high = float(
            df["high"].iloc[i]
        )

        low = float(
            df["low"].iloc[i]
        )

        left_highs = df[
            "high"
        ].iloc[
            i - left:i
        ]

        right_highs = df[
            "high"
        ].iloc[
            i + 1:i + right + 1
        ]

        left_lows = df[
            "low"
        ].iloc[
            i - left:i
        ]

        right_lows = df[
            "low"
        ].iloc[
            i + 1:i + right + 1
        ]

        if (
            high >= left_highs.max()
            and high >= right_highs.max()
        ):
            highs.append(i)

        if (
            low <= left_lows.min()
            and low <= right_lows.min()
        ):
            lows.append(i)

    return highs, lows


# ============================================================
# SUPPORT / RESISTANCE ZONES
# ============================================================

def build_sr_zones(
    df,
    max_zones=5,
):

    if len(df) < 15:
        return {
            "supports": [],
            "resistances": [],
        }

    x = add_atr(df)

    atr = x["ATR"].iloc[-1]

    if pd.isna(atr) or atr <= 0:
        atr = (
            float(
                x["close"].iloc[-1]
            )
            * 0.01
        )

    # Dynamic zone width
    zone_width = max(
        atr * 0.55,
        float(
            x["close"].iloc[-1]
        ) * 0.0025,
    )

    highs, lows = find_swing_points(
        x,
        left=2,
        right=2,
    )

    current_price = float(
        x["close"].iloc[-1]
    )

    resistance_points = []

    for idx in highs:

        level = float(
            x["high"].iloc[idx]
        )

        if level <= 0:
            continue

        resistance_points.append(
            {
                "level": level,
                "index": idx,
            }
        )

    support_points = []

    for idx in lows:

        level = float(
            x["low"].iloc[idx]
        )

        if level <= 0:
            continue

        support_points.append(
            {
                "level": level,
                "index": idx,
            }
        )

    def cluster(
        points,
        kind,
    ):

        if not points:
            return []

        # Recent first
        points = sorted(
            points,
            key=lambda z: z["index"],
            reverse=True,
        )

        clusters = []

        for p in points:

            placed = False

            for zone in clusters:

                if abs(
                    p["level"]
                    - zone["center"]
                ) <= zone_width:

                    zone["points"].append(p)

                    levels = [
                        q["level"]
                        for q in zone["points"]
                    ]

                    zone["center"] = float(
                        np.mean(levels)
                    )

                    zone["low"] = float(
                        min(levels)
                    )

                    zone["high"] = float(
                        max(levels)
                    )

                    placed = True
                    break

            if not placed:

                clusters.append(
                    {
                        "center": p["level"],
                        "low": p["level"],
                        "high": p["level"],
                        "points": [p],
                        "kind": kind,
                    }
                )

        result = []

        for zone in clusters:

            touches = len(
                zone["points"]
            )

            # Last occurrence
            last_point = max(
                zone["points"],
                key=lambda z: z["index"],
            )

            first_point = min(
                zone["points"],
                key=lambda z: z["index"],
            )

            result.append(
                {
                    "center": zone["center"],
                    "low": zone["low"],
                    "high": zone["high"],
                    "touches": touches,
                    "first_index": first_point["index"],
                    "last_index": last_point["index"],
                    "points": zone["points"],
                }
            )

        # Nearest to price first
        if kind == "RESISTANCE":

            result = [
                z
                for z in result
                if z["center"] >= current_price
            ]

            result.sort(
                key=lambda z:
                z["center"]
                - current_price
            )

        else:

            result = [
                z
                for z in result
                if z["center"] <= current_price
            ]

            result.sort(
                key=lambda z:
                current_price
                - z["center"]
            )

        return result[:max_zones]

    return {
        "supports": cluster(
            support_points,
            "SUPPORT",
        ),
        "resistances": cluster(
            resistance_points,
            "RESISTANCE",
        ),
    }


# ============================================================
# REPEATED STRUCTURE
# ============================================================

def repeated_structure(
    df,
    timeframe,
):

    if len(df) < 15:
        return {
            "pattern": "NONE",
            "direction": "NONE",
            "repetitions": 0,
            "details": [],
            "text": "No data",
        }

    x = add_atr(df)

    atr = x["ATR"].iloc[-1]

    if pd.isna(atr) or atr <= 0:
        atr = float(
            x["close"].iloc[-1]
        ) * 0.01

    highs, lows = find_swing_points(
        x,
        left=2,
        right=2,
    )

    # --------------------------------------------------------
    # Extract recent swing highs/lows
    # --------------------------------------------------------

    high_points = []

    for idx in highs:

        high_points.append(
            {
                "index": idx,
                "price": float(
                    x["high"].iloc[idx]
                ),
            }
        )

    low_points = []

    for idx in lows:

        low_points.append(
            {
                "index": idx,
                "price": float(
                    x["low"].iloc[idx]
                ),
            }
        )

    # Need at least 3 swings
    if len(high_points) < 3 and len(low_points) < 3:

        return {
            "pattern": "NONE",
            "direction": "NONE",
            "repetitions": 0,
            "details": [],
            "text": "Insufficient swings",
        }

    details = []

    # --------------------------------------------------------
    # Higher High sequence
    # --------------------------------------------------------

    recent_highs = high_points[-6:]

    hh_count = 0

    for i in range(
        1,
        len(recent_highs),
    ):

        if (
            recent_highs[i]["price"]
            > recent_highs[i - 1]["price"]
            * 1.002
        ):
            hh_count += 1

    # --------------------------------------------------------
    # Lower High sequence
    # --------------------------------------------------------

    lh_count = 0

    for i in range(
        1,
        len(recent_highs),
    ):

        if (
            recent_highs[i]["price"]
            < recent_highs[i - 1]["price"]
            * 0.998
        ):
            lh_count += 1

    # --------------------------------------------------------
    # Higher Low
    # --------------------------------------------------------

    recent_lows = low_points[-6:]

    hl_count = 0

    for i in range(
        1,
        len(recent_lows),
    ):

        if (
            recent_lows[i]["price"]
            > recent_lows[i - 1]["price"]
            * 1.002
        ):
            hl_count += 1

    # --------------------------------------------------------
    # Lower Low
    # --------------------------------------------------------

    ll_count = 0

    for i in range(
        1,
        len(recent_lows),
    ):

        if (
            recent_lows[i]["price"]
            < recent_lows[i - 1]["price"]
            * 0.998
        ):
            ll_count += 1

    # --------------------------------------------------------
    # Detailed candle records
    # --------------------------------------------------------

    def candle_detail(
        point,
        label,
    ):

        idx = point["index"]

        row = x.iloc[idx]

        timestamp = ""

        if "time" in x.columns:

            try:
                timestamp = datetime.fromtimestamp(
                    float(row["time"]),
                    tz=timezone.utc,
                ).strftime(
                    "%Y-%m-%d %H:%M UTC"
                )
            except Exception:
                timestamp = str(
                    row["time"]
                )

        return {
            "Timeframe": timeframe,
            "Candle": idx,
            "Time": timestamp,
            "Open": float(row["open"]),
            "High": float(row["high"]),
            "Low": float(row["low"]),
            "Close": float(row["close"]),
            "Volume": float(
                row.get(
                    "volume",
                    np.nan,
                )
            ),
            "Pattern": label,
            "Level": point["price"],
        }

    # Keep recent structure candles
    for p in recent_highs[-5:]:
        details.append(
            candle_detail(
                p,
                "SWING HIGH",
            )
        )

    for p in recent_lows[-5:]:
        details.append(
            candle_detail(
                p,
                "SWING LOW",
            )
        )

    # --------------------------------------------------------
    # Determine dominant repeated pattern
    # --------------------------------------------------------

    if hh_count >= 2 and hl_count >= 2:

        return {
            "pattern": "HIGHER HIGH + HIGHER LOW",
            "direction": "BULL",
            "repetitions": max(
                hh_count,
                hl_count,
            ),
            "details": details,
            "text": (
                f"{timeframe}: "
                f"HH {hh_count} / "
                f"HL {hl_count}"
            ),
        }

    if ll_count >= 2 and lh_count >= 2:

        return {
            "pattern": "LOWER LOW + LOWER HIGH",
            "direction": "BEAR",
            "repetitions": max(
                ll_count,
                lh_count,
            ),
            "details": details,
            "text": (
                f"{timeframe}: "
                f"LL {ll_count} / "
                f"LH {lh_count}"
            ),
        }

    if hh_count >= 2:

        return {
            "pattern": "REPEATED NEW RESISTANCE / HIGHER HIGH",
            "direction": "BULL_STRUCTURE",
            "repetitions": hh_count,
            "details": details,
            "text": (
                f"{timeframe}: "
                f"new resistance/high "
                f"{hh_count} times"
            ),
        }

    if ll_count >= 2:

        return {
            "pattern": "REPEATED NEW SUPPORT / LOWER LOW",
            "direction": "BEAR_STRUCTURE",
            "repetitions": ll_count,
            "details": details,
            "text": (
                f"{timeframe}: "
                f"new low/support break "
                f"{ll_count} times"
            ),
        }

    if lh_count >= 2:

        return {
            "pattern": "LOWER HIGH / RESISTANCE REJECTION",
            "direction": "BEAR",
            "repetitions": lh_count,
            "details": details,
            "text": (
                f"{timeframe}: "
                f"lower high {lh_count} times"
            ),
        }

    if hl_count >= 2:

        return {
            "pattern": "HIGHER LOW / SUPPORT HOLD",
            "direction": "BULL",
            "repetitions": hl_count,
            "details": details,
            "text": (
                f"{timeframe}: "
                f"higher low {hl_count} times"
            ),
        }

    return {
        "pattern": "MIXED",
        "direction": "MIXED",
        "repetitions": 0,
        "details": details,
        "text": f"{timeframe}: Mixed",
    }


# ============================================================
# SUPPORT / RESISTANCE SUMMARY
# ============================================================

def sr_summary(
    df,
    timeframe,
):

    sr = build_sr_zones(
        df,
        max_zones=5,
    )

    price = float(
        df["close"].iloc[-1]
    )

    supports = sr["supports"]
    resistances = sr["resistances"]

    nearest_support = (
        supports[0]
        if supports
        else None
    )

    nearest_resistance = (
        resistances[0]
        if resistances
        else None
    )

    return {
        "timeframe": timeframe,
        "price": price,
        "supports": supports,
        "resistances": resistances,
        "nearest_support": nearest_support,
        "nearest_resistance": nearest_resistance,
    }


# ============================================================
# BOS + SWEEP
# ============================================================

def structure_signal(
    df,
):

    if len(df) < 20:

        return {
            "sweep": "⚪ None",
            "bos": "⚪ None",
        }

    last = df.iloc[-1]

    previous_high = float(
        df["high"].iloc[-10:-1].max()
    )

    previous_low = float(
        df["low"].iloc[-10:-1].min()
    )

    bull_sweep = (
        float(last["low"])
        < previous_low
        and float(last["close"])
        > previous_low
    )

    bear_sweep = (
        float(last["high"])
        > previous_high
        and float(last["close"])
        < previous_high
    )

    bull_bos = (
        float(last["close"])
        > previous_high
    )

    bear_bos = (
        float(last["close"])
        < previous_low
    )

    if bull_sweep:
        sweep = "🟢 BULL SWEEP"
    elif bear_sweep:
        sweep = "🔴 BEAR SWEEP"
    else:
        sweep = "⚪ None"

    if bull_bos:
        bos = "🟢 BULL BOS"
    elif bear_bos:
        bos = "🔴 BEAR BOS"
    else:
        bos = "⚪ None"

    return {
        "sweep": sweep,
        "bos": bos,
    }


# ============================================================
# VOLUME
# ============================================================

def volume_analysis(
    df,
):

    if (
        len(df) < 8
        or "volume" not in df.columns
    ):
        return 0.0

    avg_volume = float(
        df["volume"]
        .iloc[-7:-1]
        .mean()
    )

    if avg_volume <= 0:
        return 0.0

    return float(
        df["volume"].iloc[-1]
        / avg_volume
    )


# ============================================================
# OI HISTORY
# ============================================================

@st.cache_data(ttl=CACHE_TTL)
def get_oi_history(
    symbol,
    hours=48,
):

    end = int(time.time())

    start = end - int(
        hours * 3600
    )

    result = api_get(
        "/v2/history/candles",
        {
            "resolution": "15m",
            "symbol": f"OI:{symbol}",
            "start": start,
            "end": end,
        },
    )

    if not result:
        return pd.DataFrame()

    df = pd.DataFrame(result)

    if (
        df.empty
        or "close" not in df.columns
    ):
        return pd.DataFrame()

    df["close"] = pd.to_numeric(
        df["close"],
        errors="coerce",
    )

    if "time" in df.columns:

        df["time"] = pd.to_numeric(
            df["time"],
            errors="coerce",
        )

        df = df.sort_values(
            "time"
        )

    return df.dropna(
        subset=["close"]
    ).reset_index(drop=True)


def oi_change(
    symbol,
):

    df = get_oi_history(
        symbol,
        48,
    )

    if len(df) < 2:
        return None

    old = float(
        df["close"].iloc[0]
    )

    current = float(
        df["close"].iloc[-1]
    )

    if old == 0:
        return None

    return (
        (current - old)
        / abs(old)
        * 100
    )


# ============================================================
# PRICE + OI INTERPRETATION
# ============================================================

def price_oi_relationship(
    df,
    oi_change_value,
):

    if (
        oi_change_value is None
        or len(df) < 2
    ):
        return "⚪ UNKNOWN"

    price_change = (
        float(df["close"].iloc[-1])
        - float(df["close"].iloc[-2])
    )

    if (
        price_change > 0
        and oi_change_value > 0
    ):
        return "🟢 PRICE↑ OI↑ — POSITION BUILD"

    if (
        price_change < 0
        and oi_change_value > 0
    ):
        return "🔴 PRICE↓ OI↑ — SHORT BUILD"

    if (
        price_change > 0
        and oi_change_value < 0
    ):
        return "🟡 PRICE↑ OI↓ — SHORT COVER"

    if (
        price_change < 0
        and oi_change_value < 0
    ):
        return "🟠 PRICE↓ OI↓ — LONG EXIT"

    return "⚪ BALANCED"


# ============================================================
# ORDER BOOK
# ============================================================

@st.cache_data(ttl=L2_CACHE_TTL)
def get_orderbook(
    symbol,
    depth=15,
):

    result = api_get(
        f"/v2/l2orderbook/{symbol}",
        {
            "depth": int(depth),
        },
    )

    return (
        result
        if isinstance(result, dict)
        else None
    )


def parse_orderbook(
    symbol,
    depth=15,
):

    data = get_orderbook(
        symbol,
        depth,
    )

    if not data:
        return None

    bids = (
        data.get("buy")
        or []
    )

    asks = (
        data.get("sell")
        or []
    )

    bid_rows = []
    ask_rows = []

    for row in bids:

        try:
            bid_rows.append(
                {
                    "Price": float(
                        row["price"]
                    ),
                    "Size": float(
                        row["size"]
                    ),
                }
            )
        except (
            KeyError,
            TypeError,
            ValueError,
        ):
            pass

    for row in asks:

        try:
            ask_rows.append(
                {
                    "Price": float(
                        row["price"]
                    ),
                    "Size": float(
                        row["size"]
                    ),
                }
            )
        except (
            KeyError,
            TypeError,
            ValueError,
        ):
            pass

    if (
        not bid_rows
        or not ask_rows
    ):
        return None

    bid_df = (
        pd.DataFrame(bid_rows)
        .sort_values(
            "Price",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    ask_df = (
        pd.DataFrame(ask_rows)
        .sort_values(
            "Price",
            ascending=True,
        )
        .reset_index(drop=True)
    )

    bid_depth = float(
        bid_df["Size"].sum()
    )

    ask_depth = float(
        ask_df["Size"].sum()
    )

    total_depth = (
        bid_depth + ask_depth
    )

    imbalance = (
        (
            bid_depth
            - ask_depth
        )
        / total_depth
        * 100
        if total_depth > 0
        else 0
    )

    best_bid = float(
        bid_df["Price"].max()
    )

    best_ask = float(
        ask_df["Price"].min()
    )

    mid = (
        best_bid
        + best_ask
    ) / 2

    spread = (
        best_ask
        - best_bid
    )

    return {
        "symbol": symbol,
        "bid_df": bid_df,
        "ask_df": ask_df,
        "bid_depth": bid_depth,
        "ask_depth": ask_depth,
        "imbalance": imbalance,
        "best_bid": best_bid,
        "best_ask": best_ask,
        "mid": mid,
        "spread": spread,
    }


# ============================================================
# PUBLIC TRADES
# ============================================================

@st.cache_data(ttl=L2_CACHE_TTL)
def get_recent_trades(
    symbol,
):

    result = api_get(
        f"/v2/trades/{symbol}"
    )

    if not isinstance(
        result,
        dict,
    ):
        return pd.DataFrame()

    trades = (
        result.get("trades")
        or []
    )

    if not trades:
        return pd.DataFrame()

    df = pd.DataFrame(
        trades
    )

    if "price" in df.columns:

        df["price"] = pd.to_numeric(
            df["price"],
            errors="coerce",
        )

    if "size" in df.columns:

        df["size"] = pd.to_numeric(
            df["size"],
            errors="coerce",
        )

    return df


def trade_flow(
    symbol,
):

    df = get_recent_trades(
        symbol
    )

    if (
        df.empty
        or "side" not in df.columns
        or "size" not in df.columns
    ):
        return None

    side = (
        df["side"]
        .astype(str)
        .str.lower()
    )

    buy = float(
        df.loc[
            side == "buy",
            "size",
        ].sum()
    )

    sell = float(
        df.loc[
            side == "sell",
            "size",
        ].sum()
    )

    total = (
        buy + sell
    )

    if total <= 0:
        return None

    delta = (
        buy - sell
    )

    delta_pct = (
        delta
        / total
        * 100
    )

    return {
        "buy": buy,
        "sell": sell,
        "delta": delta,
        "delta_pct": delta_pct,
        "trades": len(df),
    }


# ============================================================
# FUNDING
# ============================================================

def funding_interpretation(
    funding,
):

    if pd.isna(funding):
        return "⚪ UNKNOWN"

    # Thresholds intentionally conservative
    if funding >= 0.0005:
        return "🔴 HIGH POSITIVE"

    if funding <= -0.0005:
        return "🟢 HIGH NEGATIVE"

    return "⚪ NORMAL"


# ============================================================
# NEAR S/R
# ============================================================

def sr_position(
    price,
    sr_data,
):

    support = (
        sr_data.get(
            "nearest_support"
        )
    )

    resistance = (
        sr_data.get(
            "nearest_resistance"
        )
    )

    support_distance = np.nan
    resistance_distance = np.nan

    if support:

        support_distance = (
            price
            - support["center"]
        ) / price * 100

    if resistance:

        resistance_distance = (
            resistance["center"]
            - price
        ) / price * 100

    if (
        pd.notna(
            support_distance
        )
        and support_distance <= 1
    ):
        location = (
            "🟢 NEAR SUPPORT"
        )

    elif (
        pd.notna(
            resistance_distance
        )
        and resistance_distance <= 1
    ):
        location = (
            "🔴 NEAR RESISTANCE"
        )

    else:
        location = "⚪ MID RANGE"

    return {
        "location": location,
        "support_distance": support_distance,
        "resistance_distance": resistance_distance,
    }


# ============================================================
# CONFLUENCE SCORING
# ============================================================

def calculate_score(
    mtf_bias,
    structure,
    volume_x,
    oi_rel,
    oi_ch,
    funding,
    flow,
    ob,
    repeated_patterns,
    sr_1d,
):

    long_score = 0
    short_score = 0

    # --------------------------------------------------------
    # MTF = 20
    # --------------------------------------------------------

    if mtf_bias == "🟢 LONG ALIGNED":
        long_score += 20

    elif mtf_bias == "🔴 SHORT ALIGNED":
        short_score += 20

    elif mtf_bias == "🟢 LONG BIAS":
        long_score += 12

    elif mtf_bias == "🔴 SHORT BIAS":
        short_score += 12

    # --------------------------------------------------------
    # Structure = 15
    # --------------------------------------------------------

    if "BULL" in structure["sweep"]:
        long_score += 4

    if "BEAR" in structure["sweep"]:
        short_score += 4

    if "BULL" in structure["bos"]:
        long_score += 7

    if "BEAR" in structure["bos"]:
        short_score += 7

    # --------------------------------------------------------
    # Volume = 10
    # --------------------------------------------------------

    if volume_x >= 2:
        long_score += 5
        short_score += 5

    elif volume_x >= 1.5:
        long_score += 3
        short_score += 3

    # --------------------------------------------------------
    # OI + Price = 15
    # --------------------------------------------------------

    if "PRICE↑ OI↑" in oi_rel:
        long_score += 10

    elif "PRICE↓ OI↑" in oi_rel:
        short_score += 10

    elif "SHORT COVER" in oi_rel:
        long_score += 5

    elif "LONG EXIT" in oi_rel:
        short_score += 5

    if (
        oi_ch is not None
        and abs(oi_ch) >= 2
    ):

        if oi_ch > 0:
            long_score += 3
            short_score += 3

    # --------------------------------------------------------
    # Funding = 5
    # --------------------------------------------------------

    if not pd.isna(funding):

        if (
            funding >= 0.0005
            and mtf_bias.startswith("🔴")
        ):
            short_score += 5

        elif (
            funding <= -0.0005
            and mtf_bias.startswith("🟢")
        ):
            long_score += 5

    # --------------------------------------------------------
    # Trade Flow = 10
    # --------------------------------------------------------

    if flow:

        delta = flow["delta_pct"]

        if delta >= 25:
            long_score += 7

        elif delta >= 15:
            long_score += 4

        elif delta <= -25:
            short_score += 7

        elif delta <= -15:
            short_score += 4

    # --------------------------------------------------------
    # L2 = 5
    # --------------------------------------------------------

    if ob:

        imbalance = ob["imbalance"]

        if imbalance >= 25:
            long_score += 5

        elif imbalance <= -25:
            short_score += 5

    # --------------------------------------------------------
    # Repeated structure = 5
    # --------------------------------------------------------

    bull_repeat = 0
    bear_repeat = 0

    for p in repeated_patterns:

        if p["direction"] in [
            "BULL",
            "BULL_STRUCTURE",
        ]:
            bull_repeat += p[
                "repetitions"
            ]

        if p["direction"] == "BEAR":
            bear_repeat += p[
                "repetitions"
            ]

    if bull_repeat >= 3:
        long_score += 5

    elif bull_repeat >= 2:
        long_score += 3

    if bear_repeat >= 3:
        short_score += 5

    elif bear_repeat >= 2:
        short_score += 3

    # --------------------------------------------------------
    # S/R penalty / confirmation
    # --------------------------------------------------------

    if sr_1d:

        if (
            sr_1d["location"]
            == "🟢 NEAR SUPPORT"
        ):
            long_score += 3

        elif (
            sr_1d["location"]
            == "🔴 NEAR RESISTANCE"
        ):
            short_score += 3

    return (
        min(long_score, 100),
        min(short_score, 100),
    )


# ============================================================
# SIGNAL
# ============================================================

def signal_from_scores(
    long_score,
    short_score,
):

    highest = max(
        long_score,
        short_score,
    )

    # Avoid trades when scores are close
    difference = abs(
        long_score
        - short_score
    )

    if (
        highest < 50
        or difference < 8
    ):
        return (
            "⚪ NO TRADE",
            "NO TRADE",
        )

    if (
        long_score >= 80
        and long_score > short_score
    ):
        return (
            "🔥 A+ LONG",
            "A+",
        )

    if (
        short_score >= 80
        and short_score > long_score
    ):
        return (
            "🔥 A+ SHORT",
            "A+",
        )

    if (
        long_score >= 70
        and long_score > short_score
    ):
        return (
            "🟢 STRONG LONG",
            "STRONG",
        )

    if (
        short_score >= 70
        and short_score > long_score
    ):
        return (
            "🔴 STRONG SHORT",
            "STRONG",
        )

    if (
        long_score > short_score
    ):
        return (
            "🟡 LONG WATCH",
            "WATCH",
        )

    return (
        "🟠 SHORT WATCH",
        "WATCH",
    )


# ============================================================
# SINGLE COIN SCAN
# ============================================================

def scan_coin(
    symbol,
    ticker,
    depth,
):

    histories = {}
    sr_all = {}
    repeated_all = []

    # --------------------------------------------------------
    # MTF LOAD
    # --------------------------------------------------------

    for tf, cfg in TIMEFRAMES.items():

        df = get_history(
            symbol,
            cfg["resolution"],
            cfg["candles"],
        )

        if not df.empty:

            histories[tf] = df

            # Only major timeframes for S/R
            if tf in [
                "1M",
                "1W",
                "1D",
                "12H",
                "6H",
            ]:

                sr_all[tf] = sr_summary(
                    df,
                    tf,
                )

                repeat = repeated_structure(
                    df,
                    tf,
                )

                repeated_all.append(
                    repeat
                )

    # Minimum data
    required_tf = [
        "5m",
        "15m",
        "1H",
    ]

    if not all(
        tf in histories
        and len(histories[tf]) >= 20
        for tf in required_tf
    ):
        return None

    # --------------------------------------------------------
    # Trend
    # --------------------------------------------------------

    t5 = trend_label(
        histories["5m"]
    )

    t15 = trend_label(
        histories["15m"]
    )

    t1h = trend_label(
        histories["1H"]
    )

    trend_list = [
        t5,
        t15,
        t1h,
    ]

    bulls = sum(
        x == "🟢 BULL"
        for x in trend_list
    )

    bears = sum(
        x == "🔴 BEAR"
        for x in trend_list
    )

    if bulls == 3:
        mtf_bias = (
            "🟢 LONG ALIGNED"
        )

    elif bears == 3:
        mtf_bias = (
            "🔴 SHORT ALIGNED"
        )

    elif bulls >= 2:
        mtf_bias = (
            "🟢 LONG BIAS"
        )

    elif bears >= 2:
        mtf_bias = (
            "🔴 SHORT BIAS"
        )

    else:
        mtf_bias = "⚪ CONFLICT"

    # --------------------------------------------------------
    # 5m structure
    # --------------------------------------------------------

    structure = structure_signal(
        histories["5m"]
    )

    # --------------------------------------------------------
    # Volume
    # --------------------------------------------------------

    volume_x = volume_analysis(
        histories["5m"]
    )

    # --------------------------------------------------------
    # OI
    # --------------------------------------------------------

    oi_ch = oi_change(
        symbol
    )

    oi_rel = price_oi_relationship(
        histories["15m"],
        oi_ch,
    )

    # --------------------------------------------------------
    # L2
    # --------------------------------------------------------

    ob = parse_orderbook(
        symbol,
        depth,
    )

    # --------------------------------------------------------
    # Trade Flow
    # --------------------------------------------------------

    flow = trade_flow(
        symbol
    )

    # --------------------------------------------------------
    # Funding
    # --------------------------------------------------------

    try:
        funding = float(
            ticker["Funding"]
        )
    except (
        TypeError,
        ValueError,
    ):
        funding = np.nan

    # --------------------------------------------------------
    # 1D S/R
    # --------------------------------------------------------

    sr_1d = sr_all.get(
        "1D"
    )

    price = float(
        ticker["Price"]
    )

    sr_loc = (
        sr_position(
            price,
            sr_1d,
        )
        if sr_1d
        else {
            "location": "⚪ UNKNOWN",
            "support_distance": np.nan,
            "resistance_distance": np.nan,
        }
    )

    # --------------------------------------------------------
    # Score
    # --------------------------------------------------------

    long_score, short_score = (
        calculate_score(
            mtf_bias,
            structure,
            volume_x,
            oi_rel,
            oi_ch,
            funding,
            flow,
            ob,
            repeated_all,
            sr_loc,
        )
    )

    signal, grade = (
        signal_from_scores(
            long_score,
            short_score,
        )
    )

    # --------------------------------------------------------
    # Repeated pattern summary
    # --------------------------------------------------------

    repeated_texts = []

    for p in repeated_all:

        if (
            p["repetitions"] >= 2
            and p["pattern"] != "MIXED"
        ):
            repeated_texts.append(
                p["text"]
            )

    repeated_summary = (
        " | ".join(
            repeated_texts
        )
        if repeated_texts
        else "None"
    )

    # --------------------------------------------------------
    # Major timeframe S/R summary
    # --------------------------------------------------------

    sr_values = {}

    for tf in [
        "1M",
        "1W",
        "1D",
        "12H",
        "6H",
    ]:

        item = sr_all.get(tf)

        if not item:
            sr_values[
                f"{tf} Support"
            ] = np.nan

            sr_values[
                f"{tf} Resistance"
            ] = np.nan

            continue

        support = (
            item["nearest_support"]
        )

        resistance = (
            item[
                "nearest_resistance"
            ]
        )

        sr_values[
            f"{tf} Support"
        ] = (
            support["center"]
            if support
            else np.nan
        )

        sr_values[
            f"{tf} Resistance"
        ] = (
            resistance["center"]
            if resistance
            else np.nan
        )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    return {
        "Coin": symbol,
        "Price": price,
        "Leverage": ticker.get(
            "Leverage
