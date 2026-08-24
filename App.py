import streamlit as st
import requests
import pandas as pd
import numpy as np
import time


# ============================================================
# DELTA EXCHANGE INDIA
# AUTO MTF / S-R / LEVERAGE / STRUCTURE SCANNER
# CLEAN RESET VERSION
# ============================================================

BASE_URL = "https://api.india.delta.exchange"

HEADERS = {
    "Accept": "application/json",
    "User-Agent": "Delta-Auto-MTF-Scanner/4.0",
}

CACHE_TTL = 30
L2_CACHE_TTL = 10


# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="Delta Auto MTF Scanner",
    page_icon="🔥",
    layout="wide",
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_float(value, default=np.nan):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def first_number(data, keys):
    if not isinstance(data, dict):
        return np.nan

    for key in keys:
        if key in data:
            value = safe_float(data.get(key))
            if pd.notna(value):
                return value

    return np.nan


def fmt(value):
    if value is None:
        return "N/A"

    try:
        if pd.isna(value):
            return "N/A"
        return f"{float(value):.8g}"
    except Exception:
        return "N/A"


def api_get(path, params=None, timeout=15):
    """
    Safe GET request.
    Returns API result or None.
    """

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

    if isinstance(result, dict):
        result = result.get("data") or result.get("products") or []

    if not isinstance(result, list):
        return pd.DataFrame()

    rows = []

    for product in result:

        if not isinstance(product, dict):
            continue

        contract_type = str(
            product.get("contract_type", "")
        ).lower()

        state = str(
            product.get("state", "")
        ).lower()

        trading_status = str(
            product.get("trading_status", "")
        ).lower()

        if contract_type != "perpetual_futures":
            continue

        if state and state != "live":
            continue

        if trading_status and trading_status != "operational":
            continue

        symbol = product.get("symbol")

        if not symbol:
            continue

        max_leverage = first_number(
            product,
            [
                "max_leverage",
                "maximum_leverage",
                "maxLeverage",
                "leverage",
            ],
        )

        default_leverage = first_number(
            product,
            [
                "default_leverage",
                "defaultLeverage",
            ],
        )

        rows.append(
            {
                "Coin": str(symbol),
                "ID": product.get("id"),
                "Max Leverage": max_leverage,
                "Default Leverage": default_leverage,
            }
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    df = df.drop_duplicates(
        subset=["Coin"]
    ).reset_index(drop=True)

    return df


# ============================================================
# LEVERAGE BUCKET
# ============================================================

def leverage_bucket(max_leverage):

    if max_leverage is None:
        return "Leverage N/A"

    try:
        if pd.isna(max_leverage):
            return "Leverage N/A"
    except Exception:
        return "Leverage N/A"

    if max_leverage <= 10:
        return "≤10x"

    if max_leverage <= 20:
        return ">10x–20x"

    if max_leverage <= 50:
        return ">20x–50x"

    return ">50x"


# ============================================================
# TICKERS
# ============================================================

@st.cache_data(ttl=CACHE_TTL)
def get_tickers():

    result = api_get("/v2/tickers")

    if not result:
        return pd.DataFrame()

    if isinstance(result, dict):
        result = result.get("data") or result.get("tickers") or []

    if not isinstance(result, list):
        return pd.DataFrame()

    rows = []

    for ticker in result:

        if not isinstance(ticker, dict):
            continue

        symbol = ticker.get("symbol")

        if not symbol:
            continue

        price = first_number(
            ticker,
            [
                "close",
                "mark_price",
                "last_price",
                "price",
            ],
        )

        volume = first_number(
            ticker,
            [
                "volume_24h",
                "volume",
                "turnover_24h",
            ],
            )

        oi = first_number(
            ticker,
            [
                "open_interest",
                "oi",
            ],
        )

        if pd.isna(price) or price <= 0:
            continue

        if pd.isna(volume):
            volume = 0.0

        if pd.isna(oi):
            oi = 0.0

        funding = first_number(
            ticker,
            [
                "funding_rate",
                "funding",
            ],
        )

        rows.append(
            {
                "Coin": str(symbol),
                "Price": price,
                "24H Volume": volume,
                "OI": oi,
                "Funding": funding,
            }
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    df["Vol/OI"] = (
        df["24H Volume"]
        / df["OI"].replace(0, np.nan)
    )

    return df


# ============================================================
# CANDLE SETTINGS
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
    "1M": 2592000,
}


TIMEFRAMES = {
    "6H": ("6h", 120),
    "12H": ("12h", 60),
    "1D": ("1d", 365),
    "1W": ("1w", 52),
    "1M": ("1M", 12),
}


# ============================================================
# HISTORY / CANDLES
# ============================================================

@st.cache_data(ttl=CACHE_TTL)
def get_history(symbol, resolution, candles):

    try:
        candles = int(candles)
    except Exception:
        candles = 100

    now = int(time.time())

    step = SECONDS.get(
        resolution,
        300,
    )

    start = now - (
        step * (candles + 20)
    )

    result = api_get(
        "/v2/history/candles",
        {
            "resolution": resolution,
            "symbol": symbol,
            "start": start,
            "end": now,
        },
        timeout=20,
    )

    if not result:
        return pd.DataFrame()

    if isinstance(result, dict):
        result = (
            result.get("data")
            or result.get("candles")
            or []
        )

    if not isinstance(result, list):
        return pd.DataFrame()

    try:
        df = pd.DataFrame(result)
    except Exception:
        return pd.DataFrame()

    if df.empty:
        return df

    for column in [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    if "time" in df.columns:

        df["time"] = pd.to_numeric(
            df["time"],
            errors="coerce",
        )

        df = df.dropna(
            subset=["time"]
        )

        df = df.sort_values("time")

        df = df.drop_duplicates(
            "time"
        )

    required = [
        "open",
        "high",
        "low",
        "close",
    ]

    if not all(
        column in df.columns
        for column in required
    ):
        return pd.DataFrame()

    df = df.dropna(
        subset=required
    )

    return df.tail(
        candles
    ).reset_index(drop=True)


# ============================================================
# TREND
# ============================================================

def trend(df):

    if df is None or len(df) < 30:
        return "UNKNOWN"

    try:
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

    except Exception:
        return "UNKNOWN"


# ============================================================
# VOLUME MULTIPLE
# ============================================================

def volume_multiple(df):

    if (
        df is None
        or len(df) < 8
        or "volume" not in df.columns
    ):
        return 0.0

    try:
        average_volume = df[
            "volume"
        ].iloc[-7:-1].mean()

        if average_volume <= 0:
            return 0.0

        return float(
            df["volume"].iloc[-1]
            / average_volume
        )

    except Exception:
        return 0.0


# ============================================================
# PIVOT LEVELS
# ============================================================

def find_levels(
    df,
    lookback=2,
):

    if (
        df is None
        or len(df) < (
            lookback * 2 + 5
        )
    ):
        return [], []

    highs = []
    lows = []

    for i in range(
        lookback,
        len(df) - lookback,
    ):

        try:
            high_value = float(
                df["high"].iloc[i]
            )

            low_value = float(
                df["low"].iloc[i]
            )

            left_high = df[
                "high"
            ].iloc[
                i - lookback:i
            ].max()

            right_high = df[
                "high"
            ].iloc[
                i + 1:i + lookback + 1
            ].max()

            left_low = df[
                "low"
            ].iloc[
                i - lookback:i
            ].min()

            right_low = df[
                "low"
            ].iloc[
                i + 1:i + lookback + 1
            ].min()

            if (
                high_value >= left_high
                and high_value >= right_high
            ):
                highs.append(
                    {
                        "index": i,
                        "price": high_value,
                    }
                )

            if (
                low_value <= left_low
                and low_value <= right_low
            ):
                lows.append(
                    {
                        "index": i,
                        "price": low_value,
                    }
                )

        except Exception:
            continue

    return highs, lows


# ============================================================
# CLUSTER S/R
# ============================================================

def cluster_levels(
    levels,
    tolerance=0.005,
):

    if not levels:
        return []

    clusters = []

    for item in levels:

        price = safe_float(
            item.get("price")
        )

        if pd.isna(price):
            continue

        found = None

        for cluster in clusters:

            center = cluster["price"]

            if center == 0:
                continue

            difference = (
                abs(price - center)
                / abs(center)
            )

            if difference <= tolerance:
                found = cluster
                break

        if found is None:

            clusters.append(
                {
                    "price": price,
                    "touches": 1,
                    "prices": [price],
                    "indexes": [
                        item["index"]
                    ],
                }
            )

        else:

            found["touches"] += 1

            found["prices"].append(
                price
            )

            found["indexes"].append(
                item["index"]
            )

            found["price"] = float(
                np.mean(
                    found["prices"]
                )
            )

    return sorted(
        clusters,
        key=lambda x: (
            x["touches"],
            x["price"],
        ),
        reverse=True,
    )


# ============================================================
# CANDLE TIME
# ============================================================

def candle_time(
    df,
    index,
):

    if (
        df is None
        or "time" not in df.columns
    ):
        return str(index)

    try:

        timestamp = int(
            df["time"].iloc[index]
        )

        return time.strftime(
            "%Y-%m-%d %H:%M",
            time.gmtime(timestamp),
        )

    except Exception:
        return str(index)


# ============================================================
# CANDLE DETAIL
# ============================================================

def candle_detail(
    df,
    index,
    level_type,
    level_price,
):

    try:
        row = df.iloc[index]
    except Exception:
        return {}

    volume = np.nan

    if (
        "volume" in df.columns
        and pd.notna(
            row.get(
                "volume",
                np.nan,
            )
        )
    ):
        volume = safe_float(
            row["volume"]
        )

    return {
        "Candle Index": int(index),
        "Candle Time UTC": candle_time(
            df,
            index,
        ),
        "Type": level_type,
        "Level": safe_float(
            level_price
        ),
        "Open": safe_float(
            row["open"]
        ),
        "High": safe_float(
            row["high"]
        ),
        "Low": safe_float(
            row["low"]
        ),
        "Close": safe_float(
            row["close"]
        ),
        "Volume": volume,
    }


# ============================================================
# NEAREST SUPPORT / RESISTANCE
# ============================================================

def nearest_sr(
    df,
    tolerance=0.006,
):

    empty_result = {
        "support": np.nan,
        "resistance": np.nan,
        "major_support": np.nan,
        "major_resistance": np.nan,
        "support_touches": 0,
        "resistance_touches": 0,
        "support_details": [],
        "resistance_details": [],
    }

    if df is None or df.empty:
        return empty_result

    try:
        current = float(
            df["close"].iloc[-1]
        )
    except Exception:
        return empty_result

    highs, lows = find_levels(
        df,
        lookback=2,
    )

    high_clusters = cluster_levels(
        highs,
        tolerance,
    )

    low_clusters = cluster_levels(
        lows,
        tolerance,
    )

    supports = [
        x
        for x in low_clusters
        if x["price"] < current
    ]

    resistances = [
        x
        for x in high_clusters
        if x["price"] > current
    ]

    support = max(
        [
            x["price"]
            for x in supports
        ],
        default=np.nan,
    )

    resistance = min(
        [
            x["price"]
            for x in resistances
        ],
        default=np.nan,
    )

    major_support_item = (
        max(
            supports,
            key=lambda x: x["touches"],
        )
        if supports
        else None
    )

    major_resistance_item = (
        max(
            resistances,
            key=lambda x: x["touches"],
        )
        if resistances
        else None
    )

    support_details = []

    for cluster in sorted(
        supports,
        key=lambda x: abs(
            current - x["price"]
        ),
    )[:5]:

        for index in cluster[
            "indexes"
        ][-5:]:

            detail = candle_detail(
                df,
                index,
                "SUPPORT",
                cluster["price"],
            )

            if detail:
                support_details.append(
                    detail
                )

    resistance_details = []

    for cluster in sorted(
        resistances,
        key=lambda x: abs(
            current - x["price"]
        ),
    )[:5]:

        for index in cluster[
            "indexes"
        ][-5:]:

            detail = candle_detail(
                df,
                index,
                "RESISTANCE",
                cluster["price"],
            )

            if detail:
                resistance_details.append(
                    detail
                )

    return {
        "support": support,
        "resistance": resistance,
        "major_support": (
            major_support_item["price"]
            if major_support_item
            else np.nan
        ),
        "major_resistance": (
            major_resistance_item["price"]
            if major_resistance_item
            else np.nan
        ),
        "support_touches": (
            major_support_item["touches"]
            if major_support_item
            else 0
        ),
        "resistance_touches": (
            major_resistance_item["touches"]
            if major_resistance_item
            else 0
        ),
        "support_details": support_details,
        "resistance_details": resistance_details,
    }


# ============================================================
# REPEATED STRUCTURE
# ============================================================

def repeated_structure(df):

    if df is None or len(df) < 15:

        return {
            "pattern": "NONE",
            "count": 0,
            "details": [],
        }

    highs, lows = find_levels(
        df,
        lookback=2,
    )

    high_clusters = cluster_levels(
        highs
    )

    low_clusters = cluster_levels(
        lows
    )

    details = []

    for cluster in high_clusters:

        if cluster["touches"] >= 2:

            for index in cluster[
                "indexes"
            ][-5:]:

                detail = candle_detail(
                    df,
                    index,
                    "REPEATED RESISTANCE",
                    cluster["price"],
                )

                if detail:
                    details.append(
                        detail
                    )

    for cluster in low_clusters:

        if cluster["touches"] >= 2:

            for index in cluster[
                "indexes"
            ][-5:]:

                detail = candle_detail(
                    df,
                    index,
                    "REPEATED SUPPORT",
                    cluster["price"],
                )

                if detail:
                    details.append(
                        detail
                    )

    if len(highs) >= 3:

        recent_highs = highs[-4:]

        values = [
            x["price"]
            for x in recent_highs
        ]

        if all(
            values[i] > values[i - 1]
            for i in range(
                1,
                len(values),
            )
        ):

            last_item = recent_highs[-1]

            details.append(
                {
                    "Candle Index": last_item[
                        "index"
                    ],
                    "Candle Time UTC": candle_time(
                        df,
                        last_item["index"],
                    ),
                    "Type": "HIGHER-HIGH SEQUENCE",
                    "Level": values[-1],
                    "Open": np.nan,
                    "High": values[-1],
                    "Low": np.nan,
                    "Close": np.nan,
                    "Volume": np.nan,
                }
            )

    if len(lows) >= 3:

        recent_lows = lows[-4:]

        values = [
            x["price"]
            for x in recent_lows
        ]

        if all(
            values[i] < values[i - 1]
            for i in range(
                1,
                len(values),
            )
        ):

            last_item = recent_lows[-1]

            details.append(
                {
                    "Candle Index": last_item[
                        "index"
                    ],
                    "Candle Time UTC": candle_time(
                        df,
                        last_item["index"],
                    ),
                    "Type": "LOWER-LOW SEQUENCE",
                    "Level": values[-1],
                    "Open": np.nan,
                    "High": np.nan,
                    "Low": values[-1],
                    "Close": np.nan,
                    "Volume": np.nan,
                }
            )

    if not details:

        return {
            "pattern": "NONE",
            "count": 0,
            "details": [],
        }

    names = list(
        dict.fromkeys(
            item["Type"]
            for item in details
        )
    )

    return {
        "pattern": " | ".join(names),
        "count": len(details),
        "details": details,
    }


# ============================================================
# BREAKOUT / RETEST / FAILED BREAKOUT
# ============================================================

def breakout_status(df):

    if df is None or len(df) < 8:
        return "NONE", np.nan

    try:
        current = float(
            df["close"].iloc[-1]
        )
    except Exception:
        return "NONE", np.nan

    previous_df = df.iloc[:-1]

    highs, lows = find_levels(
        previous_df,
        lookback=2,
    )

    resistance = max(
        [
            x["price"]
            for x in highs
        ],
        default=np.nan,
    )

    support = min(
        [
            x["price"]
            for x in lows
        ],
        default=np.nan,
    )

    recent = df.tail(5)

    # ----------------------------
    # RESISTANCE
    # ----------------------------

    if pd.notna(resistance):

        if current > resistance:
            return (
                "BULL BREAKOUT",
                resistance,
            )

        if (
            recent["high"].max()
            > resistance
            and recent["close"].iloc[-1]
            < resistance
        ):
            return (
                "FAILED BULL BREAKOUT",
                resistance,
            )

        if len(recent) >= 2:

            previous = recent.iloc[-2]
            last = recent.iloc[-1]

            if (
                previous["close"]
                > resistance
                and last["low"]
                <= resistance
                and last["close"]
                > resistance
            ):
                return (
                    "BULL RETEST HOLD",
                    resistance,
                )

    # ----------------------------
    # SUPPORT
    # ----------------------------

    if pd.notna(support):

        if current < support:
            return (
                "BEAR BREAKDOWN",
                support,
            )

        if (
            recent["low"].min()
            < support
            and recent["close"].iloc[-1]
            > support
        ):
            return (
                "FAILED BEAR BREAKDOWN",
                support,
            )

        if len(recent) >= 2:

            previous = recent.iloc[-2]
            last = recent.iloc[-1]

            if (
                previous["close"]
                < support
                and last["high"]
                >= support
                and last["close"]
                < support
            ):
                return (
                    "BEAR RETEST FAIL",
                    support,
                )

    return "NONE", np.nan


# ============================================================
# TIMEFRAME ANALYSIS
# ============================================================

def timeframe_analysis(symbol):

    output = {}

    for timeframe, config in TIMEFRAMES.items():

        resolution, candle_count = config

        try:
            df = get_history(
                symbol,
                resolution,
                candle_count,
            )
        except Exception:
            df = pd.DataFrame()

        if df is None or len(df) < 10:

            output[timeframe] = {
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

        repeated = repeated_structure(
            df
        )

        breakout, breakout_level = (
            breakout_status(df)
        )

        details = (
            sr["support_details"]
            + sr["resistance_details"]
            + repeated["details"]
        )

        output[timeframe] = {
            "trend": trend(df),
            "support": sr["support"],
            "resistance": sr["resistance"],
            "major_support": sr[
                "major_support"
            ],
            "major_resistance": sr[
                "major_resistance"
            ],
            "support_touches": sr[
                "support_touches"
            ],
            "resistance_touches": sr[
                "resistance_touches"
            ],
            "pattern": repeated[
                "pattern"
            ],
            "repeats": repeated[
                "count"
            ],
            "breakout": breakout,
            "break_level": breakout_level,
            "details": details,
            "candles": df,
        }

    return output


# ============================================================
# MTF SCORE
# ============================================================

def mtf_score(
    analysis,
    price,
):

    long_score = 0
    short_score = 0

    bullish_tfs = 0
    bearish_tfs = 0

    sr_rows = []

    for timeframe, data in analysis.items():

        current_trend = data[
            "trend"
        ]

        if current_trend == "BULL":

            bullish_tfs += 1
            long_score += 2

        elif current_trend == "BEAR":

            bearish_tfs += 1
            short_score += 2

        support = data[
            "support"
        ]

        resistance = data[
            "resistance"
        ]

        if pd.notna(support):

            distance = (
                abs(price - support)
                / price
                * 100
            )

            if distance <= 3:
                long_score += 1

        if pd.notna(resistance):

            distance = (
                abs(
                    resistance - price
                )
                / price
                * 100
            )

            if distance <= 3:
                short_score += 1

        if data[
            "support_touches"
        ] >= 3:

            long_score += 1

        if data[
            "resistance_touches"
        ] >= 3:

            short_score += 1

        breakout = data[
            "breakout"
        ]

        if (
            "BULL" in breakout
        ):
            long_score += 3

        if (
            "BEAR" in breakout
        ):
            short_score += 3

        sr_rows.append(
            {
                "Timeframe": timeframe,
                "Trend": current_trend,
                "Support": support,
                "Resistance": resistance,
                "Major Support": data[
                    "major_support"
                ],
                "Major Resistance": data[
                    "major_resistance"
                ],
                "S Touches": data[
                    "support_touches"
                ],
                "R Touches": data[
                    "resistance_touches"
                ],
                "Structure": data[
                    "pattern"
                ],
                "Breakout/Retest": breakout,
            }
        )

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
        "score": max(
            long_score,
            short_score,
        ),
        "bias": bias,
        "sr_rows": sr_rows,
    }


# ============================================================
# AUTO SCAN ROW
# ============================================================

def auto_scan_row(row):

    symbol = row["Coin"]

    price = safe_float(
        row["Price"],
        0,
    )

    if price <= 0:
        raise ValueError(
            "Invalid price"
        )

    analysis = timeframe_analysis(
        symbol
    )

    scores = mtf_score(
        analysis,
        price,
    )

    patterns = []
    breakouts = []

    supports = []
    resistances = []

    for timeframe, data in analysis.items():

        pattern = data[
            "pattern"
        ]

        breakout = data[
            "breakout"
        ]

        if pattern not in (
            "NONE",
            "NO DATA",
        ):
            patterns.append(
                f"{timeframe}: {pattern}"
            )

        if breakout not in (
            "NONE",
            "NO DATA",
        ):
            breakouts.append(
                f"{timeframe}: {breakout}"
            )

        if pd.notna(
            data["support"]
        ):
            supports.append(
                (
                    timeframe,
                    data["support"],
                )
            )

        if pd.notna(
            data["resistance"]
        ):
            resistances.append(
                (
                    timeframe,
                    data["resistance"],
                )
            )

    nearest_support = np.nan
    nearest_support_tf = ""

    if supports:

        nearest_support_tf, nearest_support = min(
            supports,
            key=lambda item: abs(
                price - item[1]
            ),
        )

    nearest_resistance = np.nan
    nearest_resistance_tf = ""

    if resistances:

        nearest_resistance_tf, nearest_resistance = min(
            resistances,
            key=lambda item: abs(
                price - item[1]
            ),
        )

    signal = "NO SIGNAL"

    if (
        scores["bias"]
        == "STRONG LONG"
        and scores["long_score"] >= 7
    ):
        signal = "STRONG LONG"

    elif (
        scores["bias"]
        == "LONG BIAS"
        and scores["long_score"] >= 5
    ):
        signal = "LONG WATCH"

    elif (
        scores["bias"]
        == "STRONG SHORT"
        and scores["short_score"] >= 7
    ):
        signal = "STRONG SHORT"

    elif (
        scores["bias"]
        == "SHORT BIAS"
        and scores["short_score"] >= 5
    ):
        signal = "SHORT WATCH"

    return {
        "Coin": symbol,
        "Price": price,
        "Vol/OI": safe_float(
            row["Vol/OI"]
        ),
        "Max Leverage": row[
            "Max Leverage"
        ],
        "Default Leverage": row[
            "Default Leverage"
        ],
        "Leverage Bucket": leverage_bucket(
            row["Max Leverage"]
        ),
        "MTF Bias": scores[
            "bias"
        ],
        "MTF Score": scores[
            "score"
        ],
        "Long Score": scores[
            "long_score"
        ],
        "Short Score": scores[
            "short_score"
        ],
        "Nearest Support": nearest_support,
        "Support TF": nearest_support_tf,
        "Nearest Resistance": nearest_resistance,
        "Resistance TF": nearest_resistance_tf,
        "Repeated Structure": (
            " || ".join(patterns)
            if patterns
            else "NONE"
        ),
        "Breakout / Retest": (
            " || ".join(breakouts)
            if breakouts
            else "NONE"
        ),
        "Signal": signal,
        "_analysis": analysis,
    }


# ============================================================
# ORDER BOOK
# ============================================================

@st.cache_data(ttl=L2_CACHE_TTL)
def get_orderbook(
    symbol,
    depth=15,
):

    result = api_get(
        "/v2/l2orderbook/" + str(symbol),
        {
            "depth": int(depth),
        },
    )

    if isinstance(result, dict):
        return result

    return None


def orderbook_stats(
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
        or data.get("bids")
        or []
    )

    asks = (
        data.get("sell")
        or data.get("asks")
        or []
    )

    bid_rows = []
    ask_rows = []

    for item in bids:

        if not isinstance(
            item,
            dict,
        ):
            continue

        price = safe_float(
            item.get("price")
        )

        size = safe_float(
            item.get("size")
        )

        if (
            pd.notna(price)
            and pd.notna(size)
        ):
            bid_rows.append(
                {
                    "Price": price,
                    "Size": size,
                }
            )

    for item in asks:

        if not isinstance(
            item,
            dict,
        ):
            continue

        price = safe_float(
            item.get("price")
        )

        size = safe_float(
            item.get("size")
        )

        if (
            pd.notna(price)
            and pd.notna(size)
        ):
            ask_rows.append(
                {
                    "Price": price,
                    "Size": size,
                }
            )

    if not bid_rows or not ask_rows:
        return None

    bid = pd.DataFrame(
        bid_rows
    ).sort_values(
        "Price",
        ascending=False,
    )

    ask = pd.DataFrame(
        ask_rows
    ).sort_values(
        "Price",
        ascending=True,
    )

    bid_depth = float(
        bid["Size"].sum()
    )

    ask_depth = float(
        ask["Size"].sum()
    )

    total_depth = (
        bid_depth + ask_depth
    )

    imbalance = (
        (bid_depth - ask_depth)
        / total_depth
        * 100
        if total_depth
        else 0.0
    )

    return {
        "bid": bid,
        "ask": ask,
        "bid_depth": bid_depth,
        "ask_depth": ask_depth,
        "imbalance": imbalance,
        "best_bid": float(
            bid["Price"].max()
        ),
        "best_ask": float(
            ask["Price"].min()
        ),
    }


# ============================================================
# LOAD MARKET
# ============================================================

products = get_products()

tickers = get_tickers()


if products.empty:

    st.error(
        "❌ Delta products API se market data nahi mila."
    )

    st.info(
        "Refresh karke dobara try karein."
    )

    st.stop()


if tickers.empty:

    st.error(
        "❌ Delta ticker API se market data nahi mila."
    )

    st.info(
        "Refresh karke dobara try karein."
    )

    st.stop()


# ============================================================
# MERGE
# ============================================================

market = products.merge(
    tickers,
    on="Coin",
    how="inner",
)


if market.empty:

    st.error(
        "❌ Products aur tickers merge hone ke baad market empty hai."
    )

    st.stop()


market = market.dropna(
    subset=["Price"]
).copy()


# ============================================================
# IMPORTANT FIX:
# CREATE LEVERAGE BUCKET BEFORE market_view
# ============================================================

market["Leverage Bucket"] = (
    market["Max Leverage"].apply(
        leverage_bucket
    )
)


# Make sure expected columns exist.
# This prevents pandas KeyError if API schema changes.

required_market_columns = [
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


for column in required_market_columns:

    if column not in market.columns:

        market[column] = np.nan


market = market.sort_values(
    "24H Volume",
    ascending=False,
).reset_index(
    drop=True
)


# ============================================================
# HEADER
# ============================================================

st.title(
    "🔥 Delta Auto MTF / S-R Scanner"
)

st.caption(
    "All live perpetuals → Vol/OI filter → "
    "leverage buckets → automatic 6H/12H/1D/1W/1M "
    "S/R → repeated structure → breakout/retest"
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "⚙️ Scanner Settings"
    )

    min_vol_oi = st.number_input(
        "Minimum Vol/OI",
        min_value=0.0,
        value=3.0,
        step=0.5,
    )

    max_scan = min(
        150,
        max(
            5,
            len(market),
        ),
    )

    default_scan = min(
        40,
        max_scan,
    )

    scan_count = st.slider(
        "Automatic deep S/R scan",
        min_value=5,
        max_value=max_scan,
        value=default_scan,
        step=5,
        help=(
            "All eligible coins market table mein rahenge. "
            "Deep MTF analysis top N eligible coins par chalega."
        ),
    )

    depth = st.slider(
        "L2 depth",
        min_value=5,
        max_value=50,
        value=15,
        step=5,
    )

    if st.button(
        "🔄 Refresh All"
    ):

        st.cache_data.clear()

        st.rerun()


# ============================================================
# ELIGIBLE
# ============================================================

eligible = market[
    market["Vol/OI"].fillna(0)
    > min_vol_oi
].copy()


# ============================================================
# MARKET OVERVIEW
# ============================================================

st.subheader(
    "📊 Market Overview"
)

c1, c2, c3, c4 = st.columns(4)


c1.metric(
    "All Perpetuals",
    len(market),
)


c2.metric(
    f"Vol/OI > {min_vol_oi:g}",
    len(eligible),
)


known_leverage = (
    market["Max Leverage"]
    .dropna()
)


c3.metric(
    "Known Leverage",
    len(known_leverage),
)


highest_vol_oi = (
    market["Vol/OI"]
    .replace(
        [np.inf, -np.inf],
        np.nan,
    )
    .max()
)


if pd.isna(highest_vol_oi):
    highest_vol_oi = 0


c4.metric(
    "Highest Vol/OI",
    round(
        float(highest_vol_oi),
        2,
    ),
)


# ============================================================
# ALL ELIGIBLE MARKET TABLE
# ============================================================

st.subheader(
    "📋 All Eligible Coins"
)


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


market_view = eligible[
    market_cols
].copy()


st.dataframe(
    market_view.head(300),
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# LEVERAGE TABLES
# ============================================================

st.subheader(
    "⚡ Leverage-wise Tables"
)


leverage_tabs = st.tabs(
    [
        "≤10x",
        ">10x–20x",
        ">20x–50x",
        ">50x",
        "Leverage N/A",
    ]
)


leverage_buckets = [
    "≤10x",
    ">10x–20x",
    ">20x–50x",
    ">50x",
    "Leverage N/A",
]


for tab, bucket in zip(
    leverage_tabs,
    leverage_buckets,
):

    with tab:

        bucket_df = eligible[
            eligible["Leverage Bucket"]
            == bucket
        ][market_cols].copy()

        st.write(
            f"{bucket}: {len(bucket_df)} coins"
        )

        st.dataframe(
            bucket_df.head(300),
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
# AUTO MTF SCANNER
# ============================================================

if mode == "🔥 AUTO MTF SCANNER":

    st.subheader(
        "🔥 Automatic Market-wide MTF Scanner"
    )

    if eligible.empty:

        st.warning(
            "Vol/OI filter ke baad koi coin nahi mila."
        )

        st.stop()

    candidates = (
        eligible
        .sort_values(
            [
                "Vol/OI",
                "24H Volume",
            ],
            ascending=False,
        )
        .head(scan_count)
    )

    st.info(
        f"{len(candidates)} eligible coins ka "
        "automatic 6H/12H/1D/1W/1M S/R scan chalega. "
        "20x+ leverage coins filter se remove nahi kiye gaye hain."
    )

    results = []

    progress = st.progress(0)

    total = len(candidates)

    for position, (_, row) in enumerate(
        candidates.iterrows()
    ):

        try:

            result = auto_scan_row(
                row
            )

            results.append(
                result
            )

        except Exception:
            pass

        progress.progress(
            int(
                (position + 1)
                / total
                * 100
            )
        )

    progress.empty()

    if not results:

        st.warning(
            "Automatic scan ke liye candle data nahi mila."
        )

        st.stop()

    scan_df = pd.DataFrame(
        results
    )

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

    scan_display = (
        scan_df
        .sort_values(
            [
                "MTF Score",
                "Vol/OI",
            ],
            ascending=False,
        )[display_cols]
    )

    st.dataframe(
        scan_display,
        use_container_width=True,
        hide_index=True,
    )


    # ========================================================
    # LONG WATCH
    # ========================================================

    st.subheader(
        "🟢 Long Watch"
    )

    long_df = scan_df[
        scan_df["Long Score"]
        > scan_df["Short Score"]
    ].sort_values(
        [
            "Long Score",
            "Vol/OI",
        ],
        ascending=False,
    )

    st.dataframe(
        long_df[display_cols].head(30),
        use_container_width=True,
        hide_index=True,
    )


    # ========================================================
    # SHORT WATCH
    # ========================================================

    st.subheader(
        "🔴 Short Watch"
    )

    short_df = scan_df[
        scan_df["Short Score"]
        > scan_df["Long Score"]
    ].sort_values(
        [
            "Short Score",
            "Vol/OI",
        ],
        ascending=False,
    )

    st.dataframe(
        short_df[display_cols].head(30),
        use_container_width=True,
        hide_index=True,
    )


    st.session_state[
        "auto_scan_df"
    ] = scan_df


# ============================================================
# AUTO S/R DETAILS
# ============================================================

elif mode == "📐 AUTO S/R DETAILS":

    st.subheader(
        "📐 Automatic Multi-Timeframe Support / Resistance"
    )

    if eligible.empty:

        st.warning(
            "Eligible coins nahi mile."
        )

        st.stop()

    ranked = (
        eligible
        .sort_values(
            [
                "Vol/OI",
                "24H Volume",
            ],
            ascending=False,
        )
    )

    symbol_list = (
        ranked["Coin"]
        .head(scan_count)
        .tolist()
    )

    if not symbol_list:

        st.warning(
            "Coin list empty hai."
        )

        st.stop()

    symbol = st.selectbox(
        "Coin",
        symbol_list,
        help=(
            "Coin list automatically Vol/OI "
            "ke basis par ranked hai."
        ),
    )

    if st.button(
        "🔎 Show Automatic S/R"
    ):

        analysis = timeframe_analysis(
            symbol
        )

        price_series = ranked.loc[
            ranked["Coin"] == symbol,
            "Price",
        ]

        if price_series.empty:

            st.error(
                "Price data available nahi hai."
            )

            st.stop()

        price = float(
            price_series.iloc[0]
        )

        st.metric(
            "Current Price",
            fmt(price),
        )

        rows = []

        for timeframe, data in analysis.items():

            rows.append(
                {
                    "Timeframe": timeframe,
                    "Trend": data[
                        "trend"
                    ],
                    "Support": data[
                        "support"
                    ],
                    "Resistance": data[
                        "resistance"
                    ],
                    "Major Support": data[
                        "major_support"
                    ],
                    "Major Resistance": data[
                        "major_resistance"
                    ],
                    "S Touches": data[
                        "support_touches"
                    ],
                    "R Touches": data[
                        "resistance_touches"
                    ],
                    "Repeated Structure": data[
                        "pattern"
                    ],
                    "Breakout / Retest": data[
                        "breakout"
                    ],
                }
            )

        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown(
            "### 🧱 Detected S/R Candle Details"
        )

        for timeframe, data in analysis.items():

            st.markdown(
                f"#### {timeframe}"
            )

            details = data[
                "details"
            ]

            if not details:

                st.caption(
                    "Is timeframe par strong level detail nahi mila."
                )

                continue

            try:

                detail_df = (
                    pd.DataFrame(
                        details
                    )
                    .drop_duplicates(
                        subset=[
                            "Candle Index",
                            "Type",
                            "Level",
                        ]
                    )
                )

                st.dataframe(
                    detail_df,
                    use_container_width=True,
                    hide_index=True,
                )

            except Exception:

                st.caption(
                    "Candle details display nahi ho paaye."
                )


# ============================================================
# REPEATED STRUCTURE
# ============================================================

elif mode == "🔁 REPEATED STRUCTURE":

    st.subheader(
        "🔁 Automatic Repeated Resistance / Support"
    )

    if eligible.empty:

        st.warning(
            "Eligible coins nahi mile."
        )

        st.stop()

    candidates = (
        eligible
        .sort_values(
            [
                "Vol/OI",
                "24H Volume",
            ],
            ascending=False,
        )
        .head(scan_count)
    )

    structure_rows = []

    structure_details = []

    progress = st.progress(0)

    total = len(candidates)

    for position, (_, row) in enumerate(
        candidates.iterrows()
    ):

        symbol = row[
            "Coin"
        ]

        try:

            analysis = timeframe_analysis(
                symbol
            )

            for timeframe, data in analysis.items():

                pattern = data[
                    "pattern"
                ]

                if pattern in (
                    "NONE",
                    "NO DATA",
                ):
                    continue

                structure_rows.append(
                    {
                        "Coin": symbol,
                        "Vol/OI": row[
                            "Vol/OI"
                        ],
                        "Max Leverage": row[
                            "Max Leverage"
                        ],
                        "Timeframe": timeframe,
                        "Trend": data[
                            "trend"
                        ],
                        "Pattern": pattern,
                        "Repeat Count": data[
                            "repeats"
                        ],
                        "Support": data[
                            "support"
                        ],
                        "Resistance": data[
                            "resistance"
                        ],
                    }
                )

                for item in data[
                    "details"
                ]:

                    detail = dict(
                        item
                    )

                    detail[
                        "Coin"
                    ] = symbol

                    detail[
                        "Timeframe"
                    ] = timeframe

                    detail[
                        "Vol/OI"
                    ] = row[
                        "Vol/OI"
                    ]

                    detail[
                        "Max Leverage"
                    ] = row[
                        "Max Leverage"
                    ]

                    structure_details.append(
                        detail
                    )

        except Exception:
            pass

        progress.progress(
            int(
                (position + 1)
                / total
                * 100
            )
        )

    progress.empty()

    if structure_rows:

        structure_df = (
            pd.DataFrame(
                structure_rows
            )
            .sort_values(
                [
                    "Repeat Count",
                    "Vol/OI",
                ],
                ascending=False,
            )
        )

        st.dataframe(
            structure_df,
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.info(
            "Repeated structure nahi mila."
        )

    if structure_details:

        st.markdown(
            "### 🕯️ Exact Candle Details"
        )

        detail_df = (
            pd.DataFrame(
                structure_details
            )
            .drop_duplicates(
                subset=[
                    "Coin",
                    "Timeframe",
                    "Candle Index",
                    "Type",
                    "Level",
                ]
            )
            .sort_values(
                [
                    "Coin",
                    "Timeframe",
                    "Candle Index",
                ]
            )
        )

        st.dataframe(
            detail_df,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# L2 ORDER BOOK
# ============================================================

else:

    st.subheader(
        "📚 Live L2 Order Book"
    )

    if eligible.empty:

        l2_symbols = market[
            "Coin"
        ].tolist()

    else:

        l2_symbols = eligible[
            "Coin"
        ].tolist()

    if not l2_symbols:

        st.warning(
            "Koi coin available nahi hai."
        )

        st.stop()

    symbol = st.selectbox(
        "Coin",
        l2_symbols,
        key="l2_coin",
    )

    if st.button(
        "Load L2"
    ):

        orderbook = orderbook_stats(
            symbol,
            depth,
        )

        if not orderbook:

            st.error(
                "L2 data available nahi hai."
            )

            st.stop()

        c1, c2, c3, c4 = st.columns(
            4
        )

        c1.metric(
            "Best Bid",
            fmt(
                orderbook[
                    "best_bid"
                ]
            ),
        )

        c2.metric(
            "Best Ask",
            fmt(
                orderbook[
                    "best_ask"
                ]
            ),
        )

        c3.metric(
            "Bid Depth",
            f'{orderbook["bid_depth"]:,.2f}',
        )

        c4.metric(
            "Ask Depth",
            f'{orderbook["ask_depth"]:,.2f}',
        )

        imbalance = orderbook[
            "imbalance"
        ]

        if imbalance >= 25:

            st.success(
                f"🟢 Bid Dominant: {imbalance:.2f}%"
            )

        elif imbalance <= -25:

            st.error(
                f"🔴 Ask Dominant: {imbalance:.2f}%"
            )

        else:

            st.info(
                f"⚪ Balanced: {imbalance:.2f}%"
            )

        left, right = st.columns(
            2
        )

        with left:

            bid = orderbook[
                "bid"
            ].copy()

            bid[
                "Notional"
            ] = (
                bid["Price"]
                * bid["Size"]
            )

            st.write(
                "🟢 BID"
            )

            st.dataframe(
                bid,
                use_container_width=True,
                hide_index=True,
            )

        with right:

            ask = orderbook[
                "ask"
            ].copy()

            ask[
                "Notional"
            ] = (
                ask["Price"]
                * ask["Size"]
            )

            st.write(
                "🔴 ASK"
            )

            st.dataframe(
                ask,
                use_container_width=True,
                hide_index=True,
            )

        st.warning(
            "Visible L2 walls cancel ho sakti hain. "
            "Order-book liquidity ko guaranteed support/resistance na samjhein."
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
    "20x+ coins ko deliberately filter nahi kiya gaya."
)

st.caption(
    "Leverage metadata API mein available na ho "
    "to Leverage N/A bucket use hota hai."
)

st.caption(
    "S/R pivot-based analytical levels hain. "
    "Ye guaranteed future support/resistance nahi hain."
)

st.caption(
    "Scanner educational/analytical use ke liye hai; "
    "trade execution ya profit guarantee nahi deta."
)
