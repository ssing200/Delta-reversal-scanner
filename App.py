import streamlit as st
import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime, timezone

# ============================================================
# DELTA REVERSAL / STRUCTURE SCANNER
# Clean rebuild
#
# Logic:
#   1. ALL live perpetual contracts
#   2. Vol/OI > 3 candidate filter
#   3. Leverage-wise classification (NO leverage filter)
#   4. MTF trend
#   5. Support / Resistance
#   6. Repeated resistance / support detection
#   7. BOS / Sweep
#   8. Volume / OI
#   9. L2 imbalance
#  10. Public trade-flow pressure
#
# IMPORTANT:
# Delta public REST API does NOT provide a universal
# all-trader liquidation feed. "Liquidation Pressure"
# below is only a public-data pressure PROXY.
# ============================================================

BASE_URL = "https://api.india.delta.exchange"

HEADERS = {
    "Accept": "application/json",
    "User-Agent": "Delta-Structure-Scanner/2.0",
}

CACHE_TTL = 20
L2_CACHE_TTL = 10

DEFAULT_DEPTH = 15
DEFAULT_VOL_OI = 3.0
DEFAULT_SCAN_LIMIT = 50


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Delta MTF Structure Scanner",
    page_icon="🔥",
    layout="wide",
)


# ============================================================
# API
# ============================================================

def api_get(path, params=None, timeout=15):
    """Safe public GET request."""

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

        if not isinstance(data, dict):
            return None

        if data.get("success") is False:
            return None

        return data.get("result")

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
def get_products():

    result = api_get("/v2/products")

    if not result:
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

        if state not in ("live", "listed", ""):
            continue

        if trading_status not in (
            "operational",
            "",
        ):
            continue

        symbol = item.get("symbol")

        if not symbol:
            continue

        # Delta product metadata can vary.
        # Try multiple common leverage fields.
        leverage = None

        for key in [
            "max_leverage",
            "maximum_leverage",
            "leverage",
        ]:
            value = item.get(key)

            if value is not None:
                try:
                    leverage = float(value)
                    break
                except (
                    TypeError,
                    ValueError,
                ):
                    pass

        # Some API versions expose leverage in specs.
        specs = item.get("specs")

        if leverage is None and isinstance(
            specs, dict
        ):
            for key in [
                "max_leverage",
                "maximum_leverage",
                "leverage",
            ]:
                value = specs.get(key)

                if value is not None:
                    try:
                        leverage = float(value)
                        break
                    except (
                        TypeError,
                        ValueError,
                    ):
                        pass

        rows.append(
            {
                "Coin": symbol,
                "Product ID": item.get("id"),
                "Max Leverage": leverage,
                "Description": item.get(
                    "description",
                    "",
                ),
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

        if not isinstance(item, dict):
            continue

        symbol = item.get("symbol")

        if not symbol:
            continue

        def number(*keys):

            for key in keys:

                value = item.get(key)

                if value is None:
                    continue

                try:
                    return float(value)
                except (
                    TypeError,
                    ValueError,
                ):
                    pass

            return 0.0

        price = number(
            "close",
            "mark_price",
            "spot_price",
        )

        volume = number(
            "volume_24h",
            "volume",
        )

        oi = number(
            "open_interest",
            "oi",
        )

        funding_raw = item.get(
            "funding_rate"
        )

        try:
            funding = (
                float(funding_raw)
                if funding_raw is not None
                else np.nan
            )
        except (
            TypeError,
            ValueError,
        ):
            funding = np.nan

        if price <= 0:
            continue

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
# MERGED MARKET
# ============================================================

@st.cache_data(ttl=CACHE_TTL)
def get_market():

    products = get_products()
    tickers = get_tickers()

    if products.empty or tickers.empty:
        return pd.DataFrame()

    df = products.merge(
        tickers,
        on="Coin",
        how="left",
    )

    df = df.dropna(
        subset=["Price"]
    )

    df["Vol/OI"] = pd.to_numeric(
        df["Vol/OI"],
        errors="coerce",
    )

    df["Max Leverage"] = pd.to_numeric(
        df["Max Leverage"],
        errors="coerce",
    )

    return (
        df.sort_values(
            "24H Volume",
            ascending=False,
        )
        .reset_index(drop=True)
    )


# ============================================================
# HISTORY
# ============================================================

RESOLUTION_MAP = {
    "5m": "5m",
    "15m": "15m",
    "1H": "1h",
    "4H": "4h",
    "6H": "6h",
    "12H": "12h",
    "1D": "1d",
    "1W": "1w",
    "1M": "1M",
}


@st.cache_data(ttl=CACHE_TTL)
def get_history(
    symbol,
    resolution,
    candles=200,
):

    resolution = RESOLUTION_MAP.get(
        resolution,
        resolution,
    )

    # Approximate lookback based on requested candle count.
    seconds_map = {
        "5m": 300,
        "15m": 900,
        "1h": 3600,
        "4h": 14400,
        "6h": 21600,
        "12h": 43200,
        "1d": 86400,
        "1w": 604800,
        "1M": 2592000,
    }

    candle_seconds = seconds_map.get(
        resolution,
        3600,
    )

    end = int(time.time())

    start = end - (
        candles
        * candle_seconds
        * 2
    )

    result = api_get(
        "/v2/history/candles",
        {
            "resolution": resolution,
            "symbol": symbol,
            "start": start,
            "end": end,
        },
        timeout=15,
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

        df = df.sort_values(
            "time"
        )

    required = [
        "open",
        "high",
        "low",
        "close",
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

        df = (
            df.drop_duplicates("time")
            .reset_index(drop=True)
        )

    else:

        df = df.reset_index(
            drop=True
        )

    return df.tail(candles).reset_index(
        drop=True
    )


# ============================================================
# OI HISTORY
# ============================================================

@st.cache_data(ttl=CACHE_TTL)
def get_oi_history(
    symbol,
    hours=72,
):

    end = int(time.time())
    start = end - (
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

    return (
        df.dropna(
            subset=["close"]
        )
        .reset_index(drop=True)
    )


def oi_change(
    symbol,
    hours=48,
):

    df = get_oi_history(
        symbol,
        hours,
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
# ATR
# ============================================================

def add_atr(
    df,
    period=14,
):

    x = df.copy()

    previous = x[
        "close"
    ].shift(1)

    tr = pd.concat(
        [
            x["high"] - x["low"],
            (
                x["high"]
                - previous
            ).abs(),
            (
                x["low"]
                - previous
            ).abs(),
        ],
        axis=1,
    ).max(axis=1)

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
        return "UNKNOWN"

    close = df[
        "close"
    ]

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
        return "BULL"

    if (
        last < ema9.iloc[-1]
        and ema9.iloc[-1]
        < ema21.iloc[-1]
        and ema21.iloc[-1]
        < ema50.iloc[-1]
    ):
        return "BEAR"

    return "MIXED"


# ============================================================
# CANDLE TIME
# ============================================================

def candle_time(
    timestamp,
):

    try:

        ts = float(
            timestamp
        )

        if ts > 10_000_000_000:
            ts /= 1000

        return datetime.fromtimestamp(
            ts,
            tz=timezone.utc,
        ).strftime(
            "%Y-%m-%d %H:%M UTC"
        )

    except Exception:

        return ""


# ============================================================
# SUPPORT / RESISTANCE
# ============================================================

def detect_pivots(
    df,
    left=3,
    right=3,
):

    if len(df) < (
        left + right + 5
    ):
        return (
            pd.DataFrame(),
            pd.DataFrame(),
        )

    supports = []
    resistances = []

    for i in range(
        left,
        len(df) - right,
    ):

        low = float(
            df["low"].iloc[i]
        )

        high = float(
            df["high"].iloc[i]
        )

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

        if (
            low <= float(
                left_lows.min()
            )
            and low <= float(
                right_lows.min()
            )
        ):

            supports.append(
                {
                    "index": i,
                    "price": low,
                    "time": (
                        candle_time(
                            df[
                                "time"
                            ].iloc[i]
                        )
                        if "time"
                        in df.columns
                        else ""
                    ),
                }
            )

        if (
            high >= float(
                left_highs.max()
            )
            and high >= float(
                right_highs.max()
            )
        ):

            resistances.append(
                {
                    "index": i,
                    "price": high,
                    "time": (
                        candle_time(
                            df[
                                "time"
                            ].iloc[i]
                        )
                        if "time"
                        in df.columns
                        else ""
                    ),
                }
            )

    return (
        pd.DataFrame(supports),
        pd.DataFrame(resistances),
    )


# ============================================================
# REPEATED LEVELS
# ============================================================

def cluster_levels(
    levels,
    tolerance_pct=0.35,
):

    if levels.empty:
        return []

    sorted_levels = levels.sort_values(
        "price"
    ).reset_index(drop=True)

    clusters = []

    for _, row in sorted_levels.iterrows():

        price = float(
            row["price"]
        )

        placed = False

        for cluster in clusters:

            center = cluster[
                "center"
            ]

            distance = (
                abs(price - center)
                / center
                * 100
            )

            if distance <= tolerance_pct:

                cluster["rows"].append(
                    row.to_dict()
                )

                prices = [
                    float(
                        r["price"]
                    )
                    for r in cluster[
                        "rows"
                    ]
                ]

                cluster[
                    "center"
                ] = float(
                    np.mean(prices)
                )

                placed = True
                break

        if not placed:

            clusters.append(
                {
                    "center": price,
                    "rows": [
                        row.to_dict()
                    ],
                }
            )

    return clusters


def repeated_level_analysis(
    df,
    timeframe,
    level_type,
):

    supports, resistances = (
        detect_pivots(df)
    )

    levels = (
        supports
        if level_type == "SUPPORT"
        else resistances
    )

    if levels.empty:
        return []

    clusters = cluster_levels(
        levels
    )

    output = []

    for cluster in clusters:

        rows = cluster[
            "rows"
        ]

        count = len(rows)

        if count < 2:
            continue

        prices = [
            float(
                r["price"]
            )
            for r in rows
        ]

        times = [
            r.get("time", "")
            for r in rows
        ]

        output.append(
            {
                "Timeframe": timeframe,
                "Type": level_type,
                "Level": round(
                    float(
                        np.mean(
                            prices
                        )
                    ),
                    8,
                ),
                "Touches": count,
                "First Candle": times[0],
                "Last Candle": times[-1],
                "Candle Details": (
                    " | ".join(
                        f"{t} @ {p:.8g}"
                        for t, p in zip(
                            times,
                            prices,
                        )
                    )
                ),
            }
        )

    return output


# ============================================================
# BREAK / REJECTION PATTERN
# ============================================================

def progressive_structure(
    df,
    timeframe,
):

    if len(df) < 30:
        return []

    supports, resistances = (
        detect_pivots(
            df,
            left=3,
            right=3,
        )
    )

    events = []

    # --------------------------------------------------------
    # Progressive higher resistance
    # --------------------------------------------------------

    if len(resistances) >= 3:

        r = resistances.sort_values(
            "index"
        ).reset_index(drop=True)

        for i in range(
            2,
            len(r),
        ):

            a = float(
                r["price"].iloc[
                    i - 2
                ]
            )

            b = float(
                r["price"].iloc[
                    i - 1
                ]
            )

            c = float(
                r["price"].iloc[
                    i
                ]
            )

            if (
                b > a
                and c > b
            ):

                events.append(
                    {
                        "Timeframe": timeframe,
                        "Pattern": (
                            "PROGRESSIVE "
                            "HIGHER RESISTANCE"
                        ),
                        "Level": c,
                        "First": a,
                        "Second": b,
                        "Latest": c,
                        "Details": (
                            "Resistance "
                            "higher-high sequence"
                        ),
                    }
                )

    # --------------------------------------------------------
    # Progressive lower support
    # --------------------------------------------------------

    if len(supports) >= 3:

        s = supports.sort_values(
            "index"
        ).reset_index(drop=True)

        for i in range(
            2,
            len(s),
        ):

            a = float(
                s["price"].iloc[
                    i - 2
                ]
            )

            b = float(
                s["price"].iloc[
                    i - 1
                ]
            )

            c = float(
                s["price"].iloc[
                    i
                ]
            )

            if (
                b < a
                and c < b
            ):

                events.append(
                    {
                        "Timeframe": timeframe,
                        "Pattern": (
                            "PROGRESSIVE "
                            "LOWER SUPPORT"
                        ),
                        "Level":
