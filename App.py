import streamlit as st
import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime, timezone

# ============================================================
# DELTA REVERSAL / MARKET STRUCTURE SCANNER
# CLEAN SINGLE FILE VERSION
# ============================================================

BASE_URL = "https://api.india.delta.exchange"

HEADERS = {
    "Accept": "application/json",
    "User-Agent": "Delta-Market-Structure-Scanner/2.0",
}

CACHE_TTL = 20
DEFAULT_DEPTH = 15

# Vol/OI threshold requested by user
MIN_VOL_OI = 3.0

# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Delta Advanced Scanner",
    page_icon="🔥",
    layout="wide",
)

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

        data = response.json()

        if isinstance(data, dict):
            if data.get("success", True) is False:
                return None

            return data.get("result")

        return data

    except Exception:
        return None


# ============================================================
# PRODUCTS
# ============================================================


@st.cache_data(ttl=CACHE_TTL)
def get_products():
    result = api_get("/v2/products")

    if not isinstance(result, list):
        return pd.DataFrame()

    rows = []

    for item in result:
        if not isinstance(item, dict):
            continue

        contract_type = str(
            item.get("contract_type", "")
        ).lower()

        state = str(
            item.get("state", "")
        ).lower()

        trading_status = str(
            item.get("trading_status", "")
        ).lower()

        if contract_type != "perpetual_futures":
            continue

        if state != "live":
            continue

        if trading_status not in (
            "",
            "operational",
        ):
            continue

        symbol = item.get("symbol")

        if not symbol:
            continue

        # Try multiple possible leverage fields.
        leverage = None

        for key in [
            "max_leverage",
            "maximum_leverage",
            "leverage",
            "default_leverage",
        ]:
            value = item.get(key)

            if value is not None:
                try:
                    leverage = float(value)
                    break
                except Exception:
                    pass

        rows.append(
            {
                "Coin": str(symbol),
                "Product ID": item.get("id"),
                "Max Leverage": leverage,
                "Raw Product": item,
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

    if not isinstance(result, list):
        return pd.DataFrame()

    rows = []

    for item in result:
        if not isinstance(item, dict):
            continue

        symbol = item.get("symbol")

        if not symbol:
            continue

        try:
            price = float(
                item.get("close")
                or item.get("mark_price")
                or 0
            )
        except Exception:
            price = 0

        try:
            volume = float(
                item.get("volume_24h")
                or item.get("volume")
                or 0
            )
        except Exception:
            volume = 0

        try:
            oi = float(
                item.get("open_interest")
                or item.get("oi")
                or 0
            )
        except Exception:
            oi = 0

        if price <= 0:
            continue

        funding = np.nan

        try:
            raw_funding = item.get("funding_rate")

            if raw_funding is not None:
                funding = float(raw_funding)
        except Exception:
            pass

        rows.append(
            {
                "Coin": str(symbol),
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

    df["Vol/OI"] = df["Vol/OI"].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return df


# ============================================================
# CANDLE HISTORY
# ============================================================


@st.cache_data(ttl=CACHE_TTL)
def get_history(
    symbol,
    resolution="5m",
    hours=48,
):
    end = int(time.time())
    start = end - int(hours * 3600)

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

    if not isinstance(result, list):
        return pd.DataFrame()

    df = pd.DataFrame(result)

    if df.empty:
        return df

    for col in [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]:
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
        df = df.sort_values("time")

    return df.reset_index(drop=True)


# ============================================================
# TIMEFRAME CONFIG
# ============================================================

TIMEFRAMES = {
    "5m": {
        "resolution": "5m",
        "candles": 432,
        "hours": 36,
    },
    "15m": {
        "resolution": "15m",
        "candles": 288,
        "hours": 72,
    },
    "1H": {
        "resolution": "1h",
        "candles": 240,
        "hours": 240,
    },
    "6H": {
        "resolution": "6h",
        "candles": 120,
        "hours": 720,
    },
    "12H": {
        "resolution": "12h",
        "candles": 60,
        "hours": 720,
    },
    "1D": {
        "resolution": "1d",
        "candles": 365,
        "hours": 365 * 24,
    },
    "1W": {
        "resolution": "1w",
        "candles": 52,
        "hours": 52 * 7 * 24,
    },
    "1M": {
        "resolution": "1M",
        "candles": 12,
        "hours": 365 * 24,
    },
}


# ============================================================
# OI HISTORY
# ============================================================


@st.cache_data(ttl=CACHE_TTL)
def get_oi_history(symbol, hours=48):
    end = int(time.time())
    start = end - int(hours * 3600)

    result = api_get(
        "/v2/history/candles",
        {
            "resolution": "15m",
            "symbol": f"OI:{symbol}",
            "start": start,
            "end": end,
        },
    )

    if not isinstance(result, list):
        return pd.DataFrame()

    df = pd.DataFrame(result)

    if df.empty or "close" not in df.columns:
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
        df = df.sort_values("time")

    return df.dropna(
        subset=["close"]
    ).reset_index(drop=True)


def oi_change(symbol):
    df = get_oi_history(symbol, 48)

    if len(df) < 2:
        return np.nan

    old = float(df["close"].iloc[0])
    current = float(df["close"].iloc[-1])

    if old == 0:
        return np.nan

    return (
        (current - old)
        / abs(old)
        * 100
    )


# ============================================================
# ATR
# ============================================================


def add_atr(df, period=14):
    x = df.copy()

    previous_close = x["close"].shift(1)

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

    x["ATR"] = tr.rolling(
        period
    ).mean()

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

    if (
        last > ema9.iloc[-1]
        and ema9.iloc[-1]
        > ema21.iloc[-1]
        and ema21.iloc[-1]
        > ema50.iloc[-1]
    ):
        return "🟢 BULL"

    if (
        last < ema9.iloc[-1]
        and ema9.iloc[-1]
        < ema21.iloc[-1]
        and ema21.iloc[-1]
        < ema50.iloc[-1]
    ):
        return "🔴 BEAR"

    return "🟡 MIXED"


# ============================================================
# ORDER BOOK
# ============================================================


@st.cache_data(ttl=10)
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

    if isinstance(result, dict):
        return result

    return None


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
        except Exception:
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
        except Exception:
            pass

    if not bid_rows or not ask_rows:
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

    total = (
        bid_depth
        + ask_depth
    )

    imbalance = (
        (bid_depth - ask_depth)
        / total
        * 100
        if total > 0
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

    return {
        "bid_df": bid_df,
        "ask_df": ask_df,
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


@st.cache_data(ttl=10)
def get_recent_trades(symbol):
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

    df = pd.DataFrame(trades)

    for col in [
        "price",
        "size",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce",
            )

    return df


def trade_flow(symbol):
    df = get_recent_trades(symbol)

    if df.empty:
        return None

    if "side" not in df.columns:
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

    total = buy + sell

    if total <= 0:
        return None

    delta = buy - sell

    return {
        "buy": buy,
        "sell": sell,
        "delta": delta,
        "delta_pct": (
            delta / total * 100
        ),
    }


# ============================================================
# PIVOT SUPPORT / RESISTANCE
# ============================================================


def find_pivots(
    df,
    left=3,
    right=3,
):
    if len(df) < (
        left + right + 5
    ):
        return [], []

    highs = []
    lows = []

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
            i + 1:i + 1 + right
        ]

        left_lows = df[
            "low"
        ].iloc[
            i - left:i
        ]

        right_lows = df[
            "low"
        ].iloc[
            i + 1:i + 1 + right
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
# LEVEL CLUSTERING
# ============================================================


def cluster_levels(
    df,
    indices,
    level_type,
    tolerance_pct=0.35,
):
    if not indices:
        return []

    levels = []

    for idx in indices:
        if level_type == "RESISTANCE":
            price = float(
                df["high"].iloc[idx]
            )
        else:
            price = float(
                df["low"].iloc[idx]
            )

        if price <= 0:
            continue

        matched = None

        for level in levels:
            distance = (
                abs(price - level["price"])
                / level["price"]
                * 100
            )

            if distance <= tolerance_pct:
                matched = level
                break

        candle_time = None

        if "time" in df.columns:
            try:
                candle_time = datetime.fromtimestamp(
                    float(
                        df["time"].iloc[idx]
                    ),
                    tz=timezone.utc,
                ).strftime(
                    "%Y-%m-%d %H:%M"
                )
            except Exception:
                candle_time = None

        candle = {
            "index": idx,
            "price": price,
            "time": candle_time,
            "open": float(
                df["open"].iloc[idx]
            ),
            "high": float(
                df["high"].iloc[idx]
            ),
            "low": float(
                df["low"].iloc[idx]
            ),
            "close": float(
                df["close"].iloc[idx]
            ),
            "volume": (
                float(
                    df["volume"].iloc[idx]
                )
                if "volume" in df.columns
                else np.nan
            ),
        }

        if matched is None:
            levels.append(
                {
                    "price": price,
                    "touches": 1,
                    "candles": [candle],
                }
            )
        else:
            old_price = matched["price"]

            count = matched["touches"]

            matched["price"] = (
                (
                    old_price * count
                )
                + price
            ) / (count + 1)

            matched["touches"] += 1
            matched["candles"].append(
                candle
            )

    return levels


# ============================================================
# REPEATED SUPPORT / RESISTANCE
# ============================================================


def structure_analysis(
    df,
    timeframe,
):
    if len(df) < 20:
        return {
            "support": [],
            "resistance": [],
            "events": [],
        }

    atr_df = add_atr(df)

    atr_value = (
        float(
            atr_df["ATR"].iloc[-1]
        )
        if pd.notna(
            atr_df["ATR"].iloc[-1]
        )
        else None
    )

    tolerance = 0.35

    if atr_value is not None:
        price = float(
            df["close"].iloc[-1]
        )

        if price > 0:
            atr_pct = (
                atr_value
                / price
                * 100
            )

            tolerance = max(
                0.20,
                min(
                    0.80,
                    atr_pct * 0.60,
                ),
            )

    highs, lows = find_pivots(
        df,
        left=3,
        right=3,
    )

    resistances = cluster_levels(
        df,
        highs,
        "RESISTANCE",
        tolerance,
    )

    supports = cluster_levels(
        df,
        lows,
        "SUPPORT",
        tolerance,
    )

    events = []

    # --------------------------------------------------------
    # Repeated resistance
    # --------------------------------------------------------

    for level in resistances:
        if level["touches"] >= 2:
            candles = level[
                "candles"
            ]

            events.append(
                {
                    "Timeframe": timeframe,
                    "Type": "REPEATED RESISTANCE",
                    "Level": level["price"],
                    "Touches": level["touches"],
                    "First Candle": candles[0],
                    "Last Candle": candles[-1],
                }
            )

    # --------------------------------------------------------
    # Repeated support
    # --------------------------------------------------------

    for level in supports:
        if level["touches"] >= 2:
            candles = level[
                "candles"
            ]

            events.append(
                {
                    "Timeframe": timeframe,
                    "Type": "REPEATED SUPPORT",
                    "Level": level["price"],
                    "Touches": level["touches"],
                    "First Candle": candles[0],
                    "Last Candle": candles[-1],
                }
            )

    # --------------------------------------------------------
    # Resistance expansion
    #
    # New highs are repeatedly appearing while
    # important previous lows/supports are not broken.
    # --------------------------------------------------------

    if len(highs) >= 3:
        recent_high_indices = highs[-5:]

        values = [
            float(
                df["high"].iloc[i]
            )
            for i in recent_high_indices
        ]

        rising_count = 0

        for i in range(
            1,
            len(values),
        ):
            if values[i
