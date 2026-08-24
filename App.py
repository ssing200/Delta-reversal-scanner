import streamlit as st
import requests
import pandas as pd
import numpy as np
import time

# ============================================================
# DELTA REVERSAL SCANNER
# 20X+ MAX LEVERAGE FILTER
# Delta Exchange India Public API
# ============================================================

BASE_URL = "https://api.india.delta.exchange"

HEADERS = {
    "Accept": "application/json",
    "User-Agent": "Delta-Reversal-Scanner-20X/2.0",
}

st.set_page_config(
    page_title="Delta Reversal Scanner 20X+",
    page_icon="🔥",
    layout="wide",
)

# ============================================================
# SETTINGS
# ============================================================

CACHE_TTL = 20
TOP_COINS = 25
DEFAULT_DEPTH = 15
DEFAULT_MIN_LEVERAGE = 20.0


# ============================================================
# API HELPER
# ============================================================

def api_get(path, params=None, timeout=12):
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

        if not payload.get("success", True):
            return None

        return payload.get("result")

    except (
        requests.RequestException,
        ValueError,
        TypeError,
    ):
        return None


# ============================================================
# PRODUCTS + MAX LEVERAGE
# ============================================================

@st.cache_data(ttl=CACHE_TTL)
def get_perpetual_products():

    result = api_get("/v2/products")

    if not result:
        return pd.DataFrame(
            columns=[
                "Coin",
                "ID",
                "Max Leverage",
            ]
        )

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

        # ----------------------------------------------------
        # Try common leverage fields
        # ----------------------------------------------------

        max_leverage = None

        possible_fields = [
            "max_leverage",
            "default_leverage",
            "leverage",
        ]

        for field in possible_fields:

            value = item.get(field)

            if value is not None:

                try:
                    max_leverage = float(value)
                    break

                except (
                    TypeError,
                    ValueError,
                ):
                    pass

        rows.append(
            {
                "Coin": symbol,
                "ID": item.get("id"),
                "Max Leverage": max_leverage,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "Coin",
                "ID",
                "Max Leverage",
            ]
        )

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

        except (
            TypeError,
            ValueError,
        ):
            continue

        if price <= 0:
            continue

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
        /
        df["OI"].replace(
            0,
            np.nan,
        )
    )

    return df


# ============================================================
# CANDLES
# ============================================================

@st.cache_data(ttl=CACHE_TTL)
def get_history(
    symbol,
    resolution="5m",
    hours=48,
):

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
            "end": end,
        },
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

    return (
        df.dropna(
            subset=["close"]
        )
        .reset_index(drop=True)
    )


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
            "depth": int(depth)
        },
    )

    return (
        result
        if isinstance(
            result,
            dict,
        )
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

    bids = data.get("buy") or []
    asks = data.get("sell") or []

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
        bid_depth
        + ask_depth
    )

    imbalance = (
        (
            bid_depth
            - ask_depth
        )
        / total_depth
        * 100
        if total_depth > 0
        else 0.0
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

@st.cache_data(ttl=10)
def get_recent_trades(
    symbol
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

    if "side" not in df.columns:
        return df

    return (
        df.dropna(
            subset=[
                "price",
                "size",
            ]
        )
        .reset_index(drop=True)
    )


def trade_flow(
    symbol
):

    df = get_recent_trades(
        symbol
    )

    if (
        df.empty
        or "side" not in df.columns
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

    total = buy + sell

    if total <= 0:
        return None

    delta = buy - sell

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
# OI CHANGE
# ============================================================

def oi_change(
    symbol
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
        (
            current - old
        )
        /
        abs(old)
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

    x["ATR"] = (
        tr.rolling(
            period
        ).mean()
    )

    return x


# ============================================================
# TREND
# ============================================================

def trend_label(
    df
):

    if len(df) < 30:
        return "⚪ UNKNOWN"

    close = df["close"]

    ema9 = (
        close.ewm(
            span=9,
            adjust=False,
        )
        .mean()
    )

    ema21 = (
        close.ewm(
            span=21,
            adjust=False,
        )
        .mean()
    )

    ema50 = (
        close.ewm(
            span=50,
            adjust=False,
        )
        .mean()
    )

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
# STRUCTURE
# ============================================================

def structure_signal(
    df
):

    if len(df) < 20:

        return {
            "sweep": "⚪ None",
            "bos": "⚪ None",
        }

    x = df.copy()

    last = x.iloc[-1]

    previous_high = float(
        x["high"]
        .iloc[-10:-1]
        .max()
    )

    previous_low = float(
        x["low"]
        .iloc[-10:-1]
        .min()
    )

    bull_sweep = (
        float(last["low"])
        < previous_low
        and
        float(last["close"])
        > previous_low
    )

    bear_sweep = (
        float(last["high"])
        > previous_high
        and
        float(last["close"])
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
# LIQUIDATION-LIKE PRESSURE
# ============================================================

def liquidation_proxy(
    symbol
):

    flow = trade_flow(
        symbol
    )

    ob = parse_orderbook(
        symbol,
        DEFAULT_DEPTH,
    )

    if (
        flow is None
        and ob is None
    ):
        return None

    oi_ch = oi_change(
        symbol
    )

    score = 0

    reasons = []

    if flow is not None:

        d = flow[
            "delta_pct"
        ]

        if abs(d) >= 40:

            score += 2

            reasons.append(
                "strong trade delta"
            )

        elif abs(d) >= 25:

            score += 1

            reasons.append(
                "trade delta"
            )

    if ob is not None:

        imbalance = ob[
            "imbalance"
        ]

        if abs(imbalance) >= 35:

            score += 2

            reasons.append(
                "strong L2 imbalance"
            )

        elif abs(imbalance) >= 20:

            score += 1

            reasons.append(
                "L2 imbalance"
            )

    if (
        oi_ch is not None
        and abs(oi_ch) >= 2
    ):

        score += 1

        reasons.append(
            "OI displacement"
        )

    if flow is not None:

        if (
            flow["delta_pct"]
            > 0
        ):

            side = (
                "BUY-SIDE PRESSURE"
            )

        elif (
            flow["delta_pct"]
            < 0
        ):

            side = (
                "SELL-SIDE PRESSURE"
            )

        else:

            side = "BALANCED"

    else:

        side = "UNKNOWN"

    if score >= 4:

        level = "🔥 HIGH"

    elif score >= 2:

        level = "🟡 MEDIUM"

    else:

        level = "⚪ LOW"

    return {
        "level": level,
        "side": side,
        "score": score,
        "delta_pct": (
            flow["delta_pct"]
            if flow is not None
            else np.nan
        ),
        "oi_change": oi_ch,
        "ob_imbalance": (
            ob["imbalance"]
            if ob is not None
            else np.nan
        ),
        "reason": (
            ", ".join(reasons)
            if reasons
            else "None"
        ),
    }


# ============================================================
# SCAN COIN
# ============================================================

def scan_coin(
    symbol,
    ticker,
):

    d5 = get_history(
        symbol,
        "5m",
        36,
    )

    d15 = get_history(
        symbol,
        "15m",
        72,
    )

    d1h = get_history(
        symbol,
        "1h",
        120,
    )

    if min(
        len(d5),
        len(d15),
        len(d1h),
    ) < 25:

        return None

    t5 = trend_label(d5)

    t15 = trend_label(d15)

    t1h = trend_label(d1h)

    bulls = sum(
        x == "🟢 BULL"
        for x in [
            t5,
            t15,
            t1h,
        ]
    )

    bears = sum(
        x == "🔴 BEAR"
        for x in [
            t5,
            t15,
            t1h,
        ]
    )

    if bulls == 3:

        mtf = "🟢 LONG ALIGNED"

    elif bears == 3:

        mtf = "🔴 SHORT ALIGNED"

    elif bulls >= 2:

        mtf = "🟢 LONG BIAS"

    elif bears >= 2:

        mtf = "🔴 SHORT BIAS"

    else:

        mtf = "⚪ CONFLICT"

    structure = structure_signal(
        d5
    )

    atr_df = add_atr(
        d5
    )

    atr = (
        float(
            atr_df["ATR"].iloc[-1]
        )
        if not pd.isna(
            atr_df["ATR"].iloc[-1]
        )
        else np.nan
    )

    volume_x = 0.0

    if (
        len(d5) >= 8
        and "volume" in d5.columns
    ):

        avg_volume = float(
            d5["volume"]
            .iloc[-7:-1]
            .mean()
        )

        if avg_volume > 0:

            volume_x = float(
                d5["volume"].iloc[-1]
                / avg_volume
            )

    oi_ch = oi_change(
        symbol
    )

    ob = parse_orderbook(
        symbol,
        DEFAULT_DEPTH,
    )

    liq = liquidation_proxy(
        symbol
    )

    long_score = 0
    short_score = 0

    if (
        mtf
        == "🟢 LONG ALIGNED"
    ):

        long_score += 4

    elif (
        mtf
        == "🔴 SHORT ALIGNED"
    ):

        short_score += 4

    elif (
        mtf
        == "🟢 LONG BIAS"
    ):

        long_score += 2

    elif (
        mtf
        == "🔴 SHORT BIAS"
    ):

        short_score += 2

    if (
        "BULL"
        in structure["sweep"]
    ):

        long_score += 2

    if (
        "BEAR"
        in structure["sweep"]
    ):

        short_score += 2

    if (
        "BULL"
        in structure["bos"]
    ):

        long_score += 3

    if (
        "BEAR"
        in structure["bos"]
    ):

        short_score += 3

    if volume_x >= 2:

        long_score += 1
        short_score += 1

    if oi_ch is not None:

        if oi_ch >= 1:

            if mtf.startswith(
                "🟢"
            ):

                long_score += 2

            elif mtf.startswith(
                "🔴"
            ):

                short_score += 2

    if ob is not None:

        if (
            ob["imbalance"]
            >= 25
        ):

            long_score += 2

        elif (
            ob["imbalance"]
            <= -25
        ):

            short_score += 2

    if liq is not None:

        if (
            liq["side"]
            == "BUY-SIDE PRESSURE"
        ):

            long_score += 1

        elif (
            liq["side"]
            == "SELL-SIDE PRESSURE"
        ):

            short_score += 1

    if (
        mtf == "⚪ CONFLICT"
        or
        max(
            long_score,
            short_score,
        ) < 5
    ):

        signal = (
            "⚪ NO SIGNAL"
        )

    elif (
        long_score
        > short_score
    ):

        signal = (
            "🟢 STRONG LONG"
            if long_score >= 8
            else "🟡 LONG WATCH"
        )

    elif (
        short_score
        > long_score
    ):

        signal = (
            "🔴 STRONG SHORT"
            if short_score >= 8
            else "🟠 SHORT WATCH"
        )

    else:

        signal = (
            "⚪ NO SIGNAL"
        )

    return {

        "Coin": symbol,

        "Price": float(
            ticker["Price"]
        ),

        "Max Leverage": (
            float(
                ticker[
                    "Max Leverage"
                ]
            )
            if pd.notna(
                ticker[
                    "Max Leverage"
                ]
            )
            else np.nan
        ),

        "Vol/OI": round(
            float(
                ticker["Vol/OI"]
            )
            if pd.notna(
                ticker["Vol/OI"]
            )
            else 0,
            2,
        ),

        "5m": t5,

        "15m": t15,

        "1H": t1h,

        "MTF": mtf,

        "Sweep": structure[
            "sweep"
        ],

        "BOS": structure[
            "bos"
        ],

        "Volume x": round(
            volume_x,
            2,
        ),

        "OI Change %": (
            round(
                oi_ch,
                2,
            )
            if oi_ch is not None
            else np.nan
        ),

        "OB Imbalance %": (
            round(
                ob["imbalance"],
                2,
            )
            if ob is not None
            else np.nan
        ),

        "Liq Pressure": (
            liq["level"]
            if liq is not None
            else "⚪ UNKNOWN"
        ),

        "Long Score":
            long_score,

        "Short Score":
            short_score,

        "Score":
            max(
                long_score,
                short_score,
            ),

        "Signal":
            signal,
    }


# ============================================================
# MARKET LOAD
# ============================================================

products = (
    get_perpetual_products()
)

tickers = get_tickers()

if (
    products.empty
    or tickers.empty
):

    st.error(
        "❌ Delta market data load nahi hua. "
        "Thodi der baad Refresh dabao."
    )

    st.stop()


market = products.merge(
    tickers,
    on="Coin",
    how="left",
)


# ============================================================
# LEVERAGE CLEANUP
# ============================================================

market[
    "Max Leverage"
] = pd.to_numeric(
    market[
        "Max Leverage"
    ],
    errors="coerce",
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "⚙️ Scanner Settings"
    )

    min_leverage = st.number_input(
        "Minimum Max Leverage",
        min_value=1.0,
        max_value=200.0,
        value=20.0,
        step=1.0,
        help=(
            "Sirf un perpetual contracts ko "
            "scan karega jinka maximum leverage "
            "is value se zyada hai."
        ),
    )

    min_vol_oi = st.number_input(
        "Minimum Vol/OI",
        min_value=0.0,
        value=6.0,
        step=1.0,
    )

    scan_limit = st.slider(
        "Deep scan coins",
        min_value=5,
        max_value=50,
        value=TOP_COINS,
        step=5,
    )

    depth = st.slider(
        "L2 depth",
        min_value=5,
        max_value=50,
        value=DEFAULT_DEPTH,
        step=5,
    )

    if st.button(
        "🔄 Refresh All Data"
    ):

        st.cache_data.clear()

        st.rerun()


# ============================================================
# LEVERAGE FILTER
# ============================================================

market_leverage = market[
    market[
        "Max Leverage"
    ].notna()
    &
    (
        market[
            "Max Leverage"
        ]
        > min_leverage
    )
].copy()


# ============================================================
# VOLUME/OI FILTER
# ============================================================

market_filtered = market_leverage[
    market_leverage[
        "Vol/OI"
    ].fillna(0)
    > min_vol_oi
].copy()


market_filtered = (
    market_filtered
    .sort_values(
        "24H Volume",
        ascending=False,
    )
    .reset_index(drop=True)
)


# ============================================================
# HEADER
# ============================================================

st.title(
    "🔥 Delta Reversal Scanner"
)

st.caption(
    "20X+ Max Leverage → Volume/OI → "
    "MTF → Structure → OI → L2 → "
    "Trade Flow → Liquidation-like Pressure"
)


# ============================================================
# MARKET METRICS
# ============================================================

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Total Perpetuals",
    len(market),
)

c2.metric(
    f"Max Leverage > {min_leverage:g}x",
    len(market_leverage),
)

c3.metric(
    f"Leverage + Vol/OI > {min_vol_oi:g}",
    len(market_filtered),
)

c4.metric(
    "Highest Max Leverage",
    (
        f'{market["Max Leverage"].max():.0f}x'
        if market[
            "Max Leverage"
        ].notna().any()
        else "N/A"
    ),
)


# ============================================================
# LEVERAGE UNIVERSE
# ============================================================

with st.expander(
    "📊 View 20X+ Leverage Coins"
):

    leverage_view = (
        market_leverage[
            [
                "Coin",
                "Price",
                "Max Leverage",
                "24H Volume",
                "OI",
                "Vol/OI",
                "Funding",
            ]
        ]
        .sort_values(
            "Max Leverage",
            ascending=False,
        )
    )

    st.dataframe(
        leverage_view,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# PAGE
# ============================================================

mode = st.radio(
    "Page",
    [
        "🔥 Live Scanner",
        "📚 L2 Order Book",
        "💥 Liquidation Pressure",
        "🌡️ Liquidity Heatmap",
    ],
    horizontal=True,
)


# ============================================================
# LIVE SCANNER
# ============================================================

if mode == "🔥 Live Scanner":

    candidates = (
        market_filtered
        .head(scan_limit)
    )

    if candidates.empty:

        st.warning(
            "20x+ leverage aur Vol/OI "
            "filters ke baad koi coin nahi mila."
        )

        st.stop()

    st.info(
        f"{len(candidates)} coins deep scan honge."
    )

    results = []

    progress = st.progress(0)

    for i, (
        _,
        row,
    ) in enumerate(
        candidates.iterrows()
    ):

        try:

            result = scan_coin(
                row["Coin"],
                row,
            )

            if result is not None:

                results.append(
                    result
                )

        except Exception:
            pass

        progress.progress(
            int(
                (
                    i + 1
                )
                /
                len(candidates)
                * 100
            )
        )

    progress.empty()

    if not results:

        st.warning(
            "Scanner ko enough market "
            "data nahi mila."
        )

        st.stop()

    df = pd.DataFrame(
        results
    )

    st.subheader(
        "🎯 Complete Scanner"
    )

    st.dataframe(
        df.sort_values(
            "Score",
            ascending=False,
        ),
        use_container_width=True,
        hide_index=True,
    )


    # --------------------------------------------------------
    # LONG
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "🟢 Long Watch"
        )

        long_df = (
            df.sort_values(
                "Long Score",
                ascending=False,
            )
        )

        st.dataframe(
            long_df[
                [
                    "Coin",
                    "Price",
                    "Max Leverage",
                    "MTF",
                    "Sweep",
                    "BOS",
                    "Volume x",
                    "OI Change %",
                    "OB Imbalance %",
                    "Liq Pressure",
                    "Long Score",
                    "Signal",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )


    # --------------------------------------------------------
    # SHORT
    # --------------------------------------------------------

    with col2:

        st.subheader(
            "🔴 Short Watch"
        )

        short_df = (
            df.sort_values(
                "Short Score",
                ascending=False,
            )
        )

        st.dataframe(
            short_df[
                [
                    "Coin",
                    "Price",
                    "Max Leverage",
                    "MTF",
                    "Sweep",
                    "BOS",
                    "Volume x",
                    "OI Change %",
                    "OB Imbalance %",
                    "Liq Pressure",
                    "Short Score",
                    "Signal",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# L2 ORDER BOOK
# ============================================================

elif mode == "📚 L2 Order Book":

    st.subheader(
        "📚 Delta L2 Order Book"
    )

    symbols = (
        market_leverage[
            "Coin"
        ]
        .head(100)
        .tolist()
    )

    if not symbols:

        st.warning(
            "20x+ leverage coins available nahi hain."
        )

        st.stop()

    selected = st.selectbox(
        "Coin",
        symbols,
    )

    selected_lev = market_leverage.loc[
        market_leverage["Coin"]
        == selected,
        "Max Leverage",
    ]

    if not selected_lev.empty:

        st.info(
            f"⚡ {selected} maximum leverage: "
            f"{selected_lev.iloc[0]:.0f}x"
        )

    if st.button(
        "🔍 Load L2"
    ):

        ob = parse_orderbook(
            selected,
            depth,
        )

        if ob is None:

            st.error(
                "Delta L2 orderbook data nahi mila."
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
            f'{ob["bid_depth"]:,.0f}',
        )

        c4.metric(
            "Ask Depth",
            f'{ob["ask_depth"]:,.0f}',
        )

        imbalance = ob[
            "imbalance"
        ]

        if imbalance >= 25:

            st.success(
                f"🟢 Bid imbalance: "
                f"{imbalance:.2f}%"
            )

        elif imbalance <= -25:

            st.error(
                f"🔴 Ask imbalance: "
                f"{imbalance:.2f}%"
            )

        else:

            st.info(
                f"⚪ Balanced: "
                f"{imbalance:.2f}%"
            )

        left, right = st.columns(2)

        with left:

            st.subheader(
                "🟢 BIDS"
            )

            bid = ob[
                "bid_df"
            ].copy()

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

            st.subheader(
                "🔴 ASKS"
            )

            ask = ob[
                "ask_df"
            ].copy()

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
            "⚠️ Visible L2 walls cancel ho sakte hain."
        )


# ============================================================
# LIQUIDATION PRESSURE
# ============================================================

elif mode == "💥 Liquidation Pressure":

    st.subheader(
        "💥 Liquidation-like Pressure Scanner"
    )

    st.warning(
        "Actual liquidation volume nahi hai. "
        "Ye public trades + L2 + OI based pressure proxy hai."
    )

    candidates = (
        market_filtered
        .head(scan_limit)
    )

    if candidates.empty:

        st.warning(
            "20x+ leverage + Vol/OI "
            "filter ke baad koi coin nahi mila."
        )

        st.stop()

    rows = []

    progress = st.progress(0)

    for i, (
        _,
        row,
    ) in enumerate(
        candidates.iterrows()
    ):

        symbol = row[
            "Coin"
        ]

        try:

            liq = liquidation_proxy(
                symbol
            )

            if liq is not None:

                rows.append(
                    {
                        "Coin": symbol,

                        "Price": row[
                            "Price"
                        ],

                        "Max Leverage": row[
                            "Max Leverage"
                        ],

                        "Pressure": liq[
                            "level"
                        ],

                        "Side": liq[
                            "side"
                        ],

                        "Proxy Score": liq[
                            "score"
                        ],

                        "Trade Delta %": (
                            round(
                                liq[
                                    "delta_pct"
                                ],
                                2,
                            )
                            if pd.notna(
                                liq[
                                    "delta_pct"
                                ]
                            )
                            else np.nan
                        ),

                        "OI Change %": (
                            round(
                                liq[
                                    "oi_change"
                                ],
                                2,
                            )
                            if liq[
                                "oi_change"
                            ] is not None
                            else np.nan
                        ),

                        "L2 Imbalance %": (
                            round(
                                liq[
                                    "ob_imbalance"
                                ],
                                2,
                            )
                            if pd.notna(
                                liq[
                                    "ob_imbalance"
                                ]
                            )
                            else np.nan
                        ),

                        "Reason": liq[
                            "reason"
                        ],
                    }
                )

        except Exception:
            pass

        progress.progress(
            int(
                (
                    i + 1
                )
                /
                len(candidates)
                * 100
            )
        )

    progress.empty()

    if rows:

        liq_df = pd.DataFrame(
            rows
        )

        st.dataframe(
            liq_df.sort_values(
                "Proxy Score",
                ascending=False,
            ),
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.info(
            "Liquidation-pressure data nahi mila."
        )


# ============================================================
# LIQUIDITY HEATMAP
# ============================================================

else:

    st.subheader(
        "🌡️ Delta L2 Liquidity Heatmap"
    )

    st.info(
        "Current L2 snapshot. "
        "Historical heatmap ke liye snapshots store karne honge."
    )

    symbols = (
        market_leverage[
            "Coin"
        ]
        .head(100)
        .tolist()
    )

    if not symbols:

        st.warning(
            "20x+ leverage coins available nahi hain."
        )

        st.stop()

    selected = st.selectbox(
        "Coin",
        symbols,
        key="heatmap_coin",
    )

    if st.button(
        "🌡️ Build Heatmap"
    ):

        parsed = liquidity_levels(
            selected,
            depth,
        )

        if parsed is None:

            st.error(
                "L2 data nahi mila."
            )

            st.stop()

        ob, levels = parsed

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Mid",
            f'{ob["mid"]:.8g}',
        )

        c2.metric(
            "Bid / Ask",
            f'{ob["bid_depth"]:,.0f} / '
            f'{ob["ask_depth"]:,.0f}',
        )

        c3.metric(
            "Imbalance",
            f'{ob["imbalance"]:.2f}%',
        )

        display = levels[
            [
                "Side",
                "Price",
                "Distance %",
                "Size",
                "Notional",
            ]
        ].copy()

        styled = (
            display.style
            .background_gradient(
                subset=["Size"],
                cmap="RdYlGn",
            )
            .format(
                {
                    "Price": "{:.8g}",
                    "Distance %": "{:.4f}",
                    "Size": "{:.4f}",
                    "Notional": "{:.2f}",
                }
            )
        )

        st.dataframe(
            styled,
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "Heat intensity = current visible L2 size."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Data source: Delta Exchange India public API."
)

st.caption(
    "⚠️ 20x+ Max Leverage ka matlab ye nahi hai "
    "ki har trade par 20x+ leverage automatically available "
    "ya suitable hai. Actual leverage limits account, "
    "product aur current exchange rules par depend kar sakti hain."
)

st.caption(
    "⚠️ Liquidation Pressure actual liquidation feed nahi "
    "balki public trade + L2 + OI based proxy hai."
)
