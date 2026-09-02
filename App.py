import streamlit as st
import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta, timezone

# ============================================================
# DELTA REVERSAL SCANNER - CLEAN REBUILD
# Public Delta Exchange India API only
# ============================================================

BASE_URL = "https://api.india.delta.exchange"
HEADERS = {
    "Accept": "application/json",
    "User-Agent": "Delta-Reversal-Scanner-Clean/1.0",
}

st.set_page_config(
    page_title="Delta Reversal Scanner",
    page_icon="🔥",
    layout="wide",
)

# ----------------------------
# Settings
# ----------------------------
CACHE_TTL = 20
TOP_COINS = 25
DEFAULT_DEPTH = 15


# ============================================================
# 6-MONTH CANDLE RESEARCH ENGINE
# ============================================================

RESEARCH_RESOLUTION = "4h"
RESEARCH_DAYS = 183
RESEARCH_TRAIN_FRACTION = 0.60
RESEARCH_MIN_TRADES = 20
RESEARCH_ATR_PERIOD = 14
RESEARCH_LOOKBACK = 30
RESEARCH_HORIZON = 12
RESEARCH_STOP_R = 1.0
RESEARCH_TARGET_R = 1.5


def _to_utc_ts(x):
    ts = pd.Timestamp(x)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def get_research_candles(symbol, days=RESEARCH_DAYS, resolution=RESEARCH_RESOLUTION):
    """
    Download 6-month 4H candles in safe <=2000-candle chunks.
    A 6-month 4H series is ~1100 candles, so normally one request is enough.
    """
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)

    # Delta accepts unix timestamps for history endpoint.
    result = api_get(
        "/v2/history/candles",
        params={
            "resolution": resolution,
            "symbol": symbol,
            "start": int(start.timestamp()),
            "end": int(end.timestamp()),
        },
        timeout=20,
    )

    if not result:
        return pd.DataFrame()

    rows = []
    for x in result:
        if isinstance(x, dict):
            rows.append(x)
        elif isinstance(x, (list, tuple)) and len(x) >= 6:
            rows.append({
                "time": x[0],
                "open": x[1],
                "high": x[2],
                "low": x[3],
                "close": x[4],
                "volume": x[5],
            })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    rename = {}
    for c in df.columns:
        lc = str(c).lower()
        if lc in ("time", "timestamp", "start"):
            rename[c] = "time"
        elif lc in ("open", "o"):
            rename[c] = "open"
        elif lc in ("high", "h"):
            rename[c] = "high"
        elif lc in ("low", "l"):
            rename[c] = "low"
        elif lc in ("close", "c"):
            rename[c] = "close"
        elif lc in ("volume", "v"):
            rename[c] = "volume"
    df = df.rename(columns=rename)

    required = ["time", "open", "high", "low", "close"]
    if any(c not in df.columns for c in required):
        return pd.DataFrame()

    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Delta may return seconds or milliseconds.
    t = pd.to_numeric(df["time"], errors="coerce")
    if t.dropna().median() > 1e12:
        df["time"] = pd.to_datetime(t, unit="ms", utc=True)
    else:
        df["time"] = pd.to_datetime(t, unit="s", utc=True)

    df = (
        df.dropna(subset=["time", "open", "high", "low", "close"])
          .sort_values("time")
          .drop_duplicates("time")
          .reset_index(drop=True)
    )
    return df


def research_features(df):
    """Create only information available at or before each candle."""
    d = df.copy()

    prev_close = d["close"].shift(1)
    tr = pd.concat([
        d["high"] - d["low"],
        (d["high"] - prev_close).abs(),
        (d["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)

    d["atr"] = tr.rolling(RESEARCH_ATR_PERIOD).mean()
    d["atr_pct"] = d["atr"] / d["close"].replace(0, np.nan)

    d["ema20"] = d["close"].ewm(span=20, adjust=False).mean()
    d["ema50"] = d["close"].ewm(span=50, adjust=False).mean()
    d["ema20_slope"] = d["ema20"].pct_change(5)

    d["prev_high20"] = d["high"].shift(1).rolling(20).max()
    d["prev_low20"] = d["low"].shift(1).rolling(20).min()
    d["prev_high30"] = d["high"].shift(1).rolling(30).max()
    d["prev_low30"] = d["low"].shift(1).rolling(30).min()

    d["range30"] = d["prev_high30"] - d["prev_low30"]
    d["range_width"] = d["range30"] / d["close"].replace(0, np.nan)

    d["range_pos"] = (
        (d["close"] - d["prev_low30"]) /
        d["range30"].replace(0, np.nan)
    )

    d["body"] = (d["close"] - d["open"]).abs()
    d["body_pct"] = d["body"] / d["close"].replace(0, np.nan)

    d["upper_wick"] = d["high"] - d[["open", "close"]].max(axis=1)
    d["lower_wick"] = d[["open", "close"]].min(axis=1) - d["low"]

    if "volume" in d.columns:
        d["vol_ma20"] = d["volume"].shift(1).rolling(20).mean()
        d["vol_mult"] = d["volume"] / d["vol_ma20"].replace(0, np.nan)
    else:
        d["vol_mult"] = np.nan

    # Efficiency ratio: net movement / total movement.
    net = (d["close"] - d["close"].shift(10)).abs()
    movement = d["close"].diff().abs().rolling(10).sum()
    d["efficiency"] = net / movement.replace(0, np.nan)

    d["bull"] = d["close"] > d["open"]
    d["bear"] = d["close"] < d["open"]

    return d


def classify_behavior(row):
    """
    Behaviour-first classification.
    RANGE: low directional efficiency + relatively contained range.
    TREND_UP/DOWN: directional efficiency + EMA alignment/slope.
    TRANSITION: everything else.
    """
    eff = row.get("efficiency", np.nan)
    slope = row.get("ema20_slope", np.nan)
    ema20 = row.get("ema20", np.nan)
    ema50 = row.get("ema50", np.nan)
    width = row.get("range_width", np.nan)

    if pd.isna(eff) or pd.isna(slope) or pd.isna(ema20) or pd.isna(ema50):
        return "TRANSITION"

    if eff < 0.25 and (pd.isna(width) or width < 0.18):
        return "RANGE"

    if eff >= 0.35 and ema20 > ema50 and slope > 0:
        return "TREND_UP"

    if eff >= 0.35 and ema20 < ema50 and slope < 0:
        return "TREND_DOWN"

    return "TRANSITION"


def detect_research_setups(d, i):
    """
    Returns setup candidates using only candles <= i.
    The actual trade starts at candle i+1 open, avoiding look-ahead.
    """
    r = d.iloc[i]
    out = []

    if pd.isna(r["atr"]) or r["atr"] <= 0:
        return out

    behavior = classify_behavior(r)
    pos = r["range_pos"]

    # ---------------- RANGE ----------------
    if behavior == "RANGE" and not pd.isna(pos):
        # Lower-edge sweep + reclaim.
        if pos <= 0.30 and r["low"] < r["prev_low20"] and r["close"] > r["prev_low20"]:
            out.append(("RANGE_SWEEP_LONG", "LONG", behavior))

        # Lower-edge bullish rejection without requiring a sweep.
        elif pos <= 0.25 and r["bull"] and r["lower_wick"] >= r["body"] * 0.8:
            out.append(("RANGE_REJECTION_LONG", "LONG", behavior))

        # Upper-edge sweep + rejection.
        if pos >= 0.70 and r["high"] > r["prev_high20"] and r["close"] < r["prev_high20"]:
            out.append(("RANGE_SWEEP_SHORT", "SHORT", behavior))

        elif pos >= 0.75 and r["bear"] and r["upper_wick"] >= r["body"] * 0.8:
            out.append(("RANGE_REJECTION_SHORT", "SHORT", behavior))

    # ---------------- TREND ----------------
    if behavior == "TREND_UP":
        # Pullback/reclaim of EMA20.
        if r["low"] <= r["ema20"] and r["close"] > r["ema20"] and r["bull"]:
            out.append(("TREND_PULLBACK_LONG", "LONG", behavior))

        # Fresh breakout continuation.
        if r["close"] > r["prev_high20"] and (pd.isna(r["vol_mult"]) or r["vol_mult"] >= 1.2):
            out.append(("BREAKOUT_LONG", "LONG", behavior))

    if behavior == "TREND_DOWN":
        if r["high"] >= r["ema20"] and r["close"] < r["ema20"] and r["bear"]:
            out.append(("TREND_PULLBACK_SHORT", "SHORT", behavior))

        if r["close"] < r["prev_low20"] and (pd.isna(r["vol_mult"]) or r["vol_mult"] >= 1.2):
            out.append(("BREAKOUT_SHORT", "SHORT", behavior))

    # ---------------- TRANSITION / COMPRESSION BREAK ----------------
    if behavior in ("RANGE", "TRANSITION"):
        if r["close"] > r["prev_high20"] and (pd.isna(r["vol_mult"]) or r["vol_mult"] >= 1.2):
            out.append(("COMPRESSION_BREAKOUT_LONG", "LONG", behavior))

        if r["close"] < r["prev_low20"] and (pd.isna(r["vol_mult"]) or r["vol_mult"] >= 1.2):
            out.append(("COMPRESSION_BREAKOUT_SHORT", "SHORT", behavior))

    return out


def simulate_trade(d, entry_idx, side):
    """
    Entry at next candle OPEN.
    Stop = 1 ATR, target = 1.5 ATR.
    If SL and TP occur in the same candle, conservatively count SL first.
    """
    if entry_idx >= len(d):
        return None

    entry = float(d.iloc[entry_idx]["open"])
    atr = float(d.iloc[entry_idx - 1]["atr"])

    if not np.isfinite(entry) or not np.isfinite(atr) or atr <= 0:
        return None

    if side == "LONG":
        sl = entry - RESEARCH_STOP_R * atr
        tp = entry + RESEARCH_TARGET_R * atr
    else:
        sl = entry + RESEARCH_STOP_R * atr
        tp = entry - RESEARCH_TARGET_R * atr

    last = min(len(d) - 1, entry_idx + RESEARCH_HORIZON)

    for j in range(entry_idx, last + 1):
        h = float(d.iloc[j]["high"])
        l = float(d.iloc[j]["low"])

        if side == "LONG":
            hit_sl = l <= sl
            hit_tp = h >= tp
        else:
            hit_sl = h >= sl
            hit_tp = l <= tp

        if hit_sl and hit_tp:
            return {
                "outcome": "LOSS",
                "r": -RESEARCH_STOP_R,
                "bars": j - entry_idx + 1,
                "entry": entry,
                "sl": sl,
                "tp": tp,
            }

        if hit_tp:
            return {
                "outcome": "WIN",
                "r": RESEARCH_TARGET_R,
                "bars": j - entry_idx + 1,
                "entry": entry,
                "sl": sl,
                "tp": tp,
            }

        if hit_sl:
            return {
                "outcome": "LOSS",
                "r": -RESEARCH_STOP_R,
                "bars": j - entry_idx + 1,
                "entry": entry,
                "sl": sl,
                "tp": tp,
            }

    # Time exit: mark-to-close R, not artificially a win/loss.
    close = float(d.iloc[last]["close"])
    if side == "LONG":
        r_value = (close - entry) / atr
    else:
        r_value = (entry - close) / atr

    return {
        "outcome": "TIME_EXIT",
        "r": float(r_value),
        "bars": last - entry_idx + 1,
        "entry": entry,
        "sl": sl,
        "tp": tp,
    }


def wilson_lower_bound(wins, n, z=1.96):
    if n <= 0:
        return 0.0
    p = wins / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    spread = z * np.sqrt((p * (1 - p) / n) + z * z / (4 * n * n))
    return (centre - spread) / denom


def run_coin_research(symbol):
    d = get_research_candles(symbol)
    if d.empty or len(d) < 150:
        return pd.DataFrame(), pd.DataFrame()

    d = research_features(d)

    trades = []
    for i in range(60, len(d) - 1):
        candidates = detect_research_setups(d, i)

        for setup, side, behavior in candidates:
            result = simulate_trade(d, i + 1, side)
            if not result:
                continue

            trades.append({
                "Coin": symbol,
                "Time": d.iloc[i]["time"],
                "Setup": setup,
                "Side": side,
                "Behavior": behavior,
                "Outcome": result["outcome"],
                "R": result["r"],
                "Bars": result["bars"],
                "Entry": result["entry"],
                "SL": result["sl"],
                "TP": result["tp"],
                "VolumeMultiple": d.iloc[i].get("vol_mult", np.nan),
                "Efficiency": d.iloc[i].get("efficiency", np.nan),
                "RangePosition": d.iloc[i].get("range_pos", np.nan),
            })

    if not trades:
        return pd.DataFrame(), d

    return pd.DataFrame(trades), d


def aggregate_research(trades):
    if trades.empty:
        return pd.DataFrame()

    rows = []
    for setup, g in trades.groupby("Setup"):
        n = len(g)
        wins = int((g["Outcome"] == "WIN").sum())
        losses = int((g["Outcome"] == "LOSS").sum())
        time_exits = int((g["Outcome"] == "TIME_EXIT").sum())

        rows.append({
            "Setup": setup,
            "Trades": n,
            "Wins": wins,
            "Losses": losses,
            "Time exits": time_exits,
            "Win %": round(100 * wins / n, 2),
            "Wilson LB %": round(100 * wilson_lower_bound(wins, n), 2),
            "Avg R": round(g["R"].mean(), 3),
            "Median R": round(g["R"].median(), 3),
            "Avg bars": round(g["Bars"].mean(), 2),
            "Coins": g["Coin"].nunique(),
        })

    return pd.DataFrame(rows).sort_values(
        ["Wilson LB %", "Avg R", "Trades"],
        ascending=False
    ).reset_index(drop=True)


def validation_research(trades):
    if trades.empty:
        return pd.DataFrame()

    t = trades.sort_values(["Coin", "Time"]).copy()
    rows = []

    # Per-coin time split prevents a coin's early observations from contaminating
    # its own validation statistics.
    for coin, g in t.groupby("Coin"):
        cut = int(len(g) * RESEARCH_TRAIN_FRACTION)
        if cut <= 0 or cut >= len(g):
            continue

        v = g.iloc[cut:]
        for setup, sg in v.groupby("Setup"):
            n = len(sg)
            wins = int((sg["Outcome"] == "WIN").sum())
            rows.append({
                "Coin": coin,
                "Setup": setup,
                "Trades": n,
                "Win %": 100 * wins / n if n else 0,
                "Wilson LB %": 100 * wilson_lower_bound(wins, n),
                "Avg R": sg["R"].mean(),
                "Median R": sg["R"].median(),
            })

    return pd.DataFrame(rows)


def common_validation_summary(validation):
    if validation.empty:
        return pd.DataFrame()

    rows = []
    for setup, g in validation.groupby("Setup"):
        n = int(g["Trades"].sum())
        wins_est = (g["Win %"] / 100 * g["Trades"]).sum()
        win_pct = 100 * wins_est / n if n else 0

        rows.append({
            "Setup": setup,
            "Validation trades": n,
            "Validation Win %": round(win_pct, 2),
            "Min Wilson LB %": round(g["Wilson LB %"].min(), 2),
            "Avg R": round((g["Avg R"] * g["Trades"]).sum() / n, 3) if n else 0,
            "Coins": g["Coin"].nunique(),
            "Usable?": (
                "YES"
                if n >= RESEARCH_MIN_TRADES and g["Wilson LB %"].min() >= 50
                else "NO"
            ),
        })

    return pd.DataFrame(rows).sort_values(
        ["Usable?", "Validation Win %", "Avg R"],
        ascending=[False, False, False]
    ).reset_index(drop=True)


def research_universe():
    products = get_perpetual_products()
    if products.empty:
        return []

    # All live perpetuals. Research does NOT apply the 20x Vol/OI filter;
    # that filter belongs to the live trading-universe rule.
    return products["Coin"].dropna().astype(str).unique().tolist()


def render_research_page():
    st.header("🧪 6-Month Candle Research")

    st.info(
        "Research-first mode: 6-month 4H candles are scanned candle-by-candle. "
        "Signals enter on the next candle open, so the backtest does not use "
        "the future candle to create the signal."
    )

    universe = research_universe()
    if not universe:
        st.error("Delta perpetual products could not be loaded.")
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        max_coins = st.number_input(
            "Coins to research",
            min_value=1,
            max_value=max(1, len(universe)),
            value=min(50, len(universe)),
            step=1,
        )
    with c2:
        st.write("Resolution")
        st.code("4h")
    with c3:
        st.write("History")
        st.code("~6 months")

    selected = universe[:int(max_coins)]

    if st.button("🚀 RUN 6-MONTH RESEARCH", type="primary"):
        all_trades = []
        progress = st.progress(0.0)
        status = st.empty()

        for n, coin in enumerate(selected, 1):
            status.write(f"Scanning {coin} ({n}/{len(selected)})...")
            try:
                trades, _ = run_coin_research(coin)
                if not trades.empty:
                    all_trades.append(trades)
            except Exception as exc:
                st.warning(f"{coin}: {exc}")

            progress.progress(n / len(selected))

        if not all_trades:
            st.error("No research trades were generated.")
            return

        trades = pd.concat(all_trades, ignore_index=True)
        summary = aggregate_research(trades)

        st.subheader("1. Common patterns")
        st.dataframe(summary, use_container_width=True)

        st.subheader("2. Validation — later part of each coin")
        validation = validation_research(trades)
        vsummary = common_validation_summary(validation)
        st.dataframe(vsummary, use_container_width=True)

        st.subheader("3. Behaviour breakdown")
        behaviour = (
            trades.groupby(["Behavior", "Setup"])
            .agg(
                Trades=("R", "size"),
                WinPct=("Outcome", lambda x: 100 * (x == "WIN").mean()),
                AvgR=("R", "mean"),
                AvgBars=("Bars", "mean"),
                Coins=("Coin", "nunique"),
            )
            .reset_index()
        )
        behaviour["WinPct"] = behaviour["WinPct"].round(2)
        behaviour["AvgR"] = behaviour["AvgR"].round(3)
        behaviour["AvgBars"] = behaviour["AvgBars"].round(2)
        st.dataframe(behaviour, use_container_width=True)

        st.subheader("4. Per-coin evidence")
        per_coin = (
            trades.groupby(["Coin", "Setup"])
            .agg(
                Trades=("R", "size"),
                WinPct=("Outcome", lambda x: 100 * (x == "WIN").mean()),
                AvgR=("R", "mean"),
                MedianR=("R", "median"),
                AvgBars=("Bars", "mean"),
            )
            .reset_index()
        )
        st.dataframe(per_coin, use_container_width=True)

        st.subheader("5. Raw candle-level trades")
        st.dataframe(trades.tail(500), use_container_width=True)

        csv = trades.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download research trades CSV",
            data=csv,
            file_name="delta_6m_research_trades.csv",
            mime="text/csv",
        )

        st.session_state["research_trades"] = trades
        st.session_state["research_summary"] = summary
        st.session_state["research_validation"] = vsummary



# ============================================================
# API HELPER
# ============================================================

def api_get(path, params=None, timeout=12):
    """Safe public GET request to Delta India API."""
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

    except (requests.RequestException, ValueError, TypeError):
        return None


# ============================================================
# PRODUCTS
# ============================================================

@st.cache_data(ttl=CACHE_TTL)
def get_perpetual_products():
    result = api_get("/v2/products")

    if not result:
        return pd.DataFrame(columns=["Coin", "ID"])

    rows = []

    for item in result:
        if item.get("contract_type") != "perpetual_futures":
            continue

        if item.get("state") != "live":
            continue

        if item.get("trading_status") != "operational":
            continue

        symbol = item.get("symbol")
        if symbol:
            rows.append(
                {
                    "Coin": symbol,
                    "ID": item.get("id"),
                }
            )

    if not rows:
        return pd.DataFrame(columns=["Coin", "ID"])

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
# CANDLES / OI HISTORY
# ============================================================

@st.cache_data(ttl=CACHE_TTL)
def get_history(symbol, resolution="5m", hours=48):
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

    df = pd.DataFrame(result)

    if df.empty:
        return df

    for col in ["open", "high", "low", "close", "volume"]:
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

    required = ["open", "high", "low", "close"]

    if not all(c in df.columns for c in required):
        return pd.DataFrame()

    df = df.dropna(subset=required)

    if "time" in df.columns:
        df = df.sort_values("time")

    return (
        df.drop_duplicates("time")
        .reset_index(drop=True)
        if "time" in df.columns
        else df.reset_index(drop=True)
    )


@st.cache_data(ttl=CACHE_TTL)
def get_oi_history(symbol, hours=48):
    end = int(time.time())
    start = end - int(hours * 3600)

    # Delta documents OI history as OI:<symbol>
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

    return (
        df.dropna(subset=["close"])
        .reset_index(drop=True)
    )


# ============================================================
# ORDER BOOK
# ============================================================

@st.cache_data(ttl=10)
def get_orderbook(symbol, depth=15):
    result = api_get(
        f"/v2/l2orderbook/{symbol}",
        {"depth": int(depth)},
    )

    return result if isinstance(result, dict) else None


def parse_orderbook(symbol, depth=15):
    data = get_orderbook(symbol, depth)

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
                    "Price": float(row["price"]),
                    "Size": float(row["size"]),
                }
            )
        except (KeyError, TypeError, ValueError):
            pass

    for row in asks:
        try:
            ask_rows.append(
                {
                    "Price": float(row["price"]),
                    "Size": float(row["size"]),
                }
            )
        except (KeyError, TypeError, ValueError):
            pass

    if not bid_rows or not ask_rows:
        return None

    bid_df = (
        pd.DataFrame(bid_rows)
        .sort_values("Price", ascending=False)
        .reset_index(drop=True)
    )

    ask_df = (
        pd.DataFrame(ask_rows)
        .sort_values("Price", ascending=True)
        .reset_index(drop=True)
    )

    bid_depth = float(bid_df["Size"].sum())
    ask_depth = float(ask_df["Size"].sum())
    total_depth = bid_depth + ask_depth

    imbalance = (
        (bid_depth - ask_depth)
        / total_depth
        * 100
        if total_depth > 0
        else 0.0
    )

    best_bid = float(bid_df["Price"].max())
    best_ask = float(ask_df["Price"].min())
    mid = (best_bid + best_ask) / 2
    spread = best_ask - best_bid

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
def get_recent_trades(symbol):
    result = api_get(f"/v2/trades/{symbol}")

    if not isinstance(result, dict):
        return pd.DataFrame()

    trades = result.get("trades") or []

    if not trades:
        return pd.DataFrame()

    df = pd.DataFrame(trades)

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
        # Newer feeds can use buyer-role fields in websocket,
        # but REST public trades normally exposes side.
        return df

    return df.dropna(
        subset=["price", "size"]
    ).reset_index(drop=True)


def trade_flow(symbol):
    df = get_recent_trades(symbol)

    if df.empty or "side" not in df.columns:
        return None

    side = df["side"].astype(str).str.lower()

    buy = float(
        df.loc[side == "buy", "size"].sum()
    )
    sell = float(
        df.loc[side == "sell", "size"].sum()
    )

    total = buy + sell

    if total <= 0:
        return None

    delta = buy - sell
    delta_pct = delta / total * 100

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

def oi_change(symbol):
    df = get_oi_history(symbol, 48)

    if len(df) < 2:
        return None

    old = float(df["close"].iloc[0])
    current = float(df["close"].iloc[-1])

    if old == 0:
        return None

    return (current - old) / abs(old) * 100


# ============================================================
# ATR / TREND / STRUCTURE
# ============================================================

def add_atr(df, period=14):
    x = df.copy()

    previous_close = x["close"].shift(1)

    tr = pd.concat(
        [
            x["high"] - x["low"],
            (x["high"] - previous_close).abs(),
            (x["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    x["ATR"] = tr.rolling(period).mean()

    return x


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

    last = float(close.iloc[-1])

    if (
        last > ema9.iloc[-1]
        and ema9.iloc[-1] > ema21.iloc[-1]
        and ema21.iloc[-1] > ema50.iloc[-1]
    ):
        return "🟢 BULL"

    if (
        last < ema9.iloc[-1]
        and ema9.iloc[-1] < ema21.iloc[-1]
        and ema21.iloc[-1] < ema50.iloc[-1]
    ):
        return "🔴 BEAR"

    return "🟡 MIXED"


def structure_signal(df):
    if len(df) < 20:
        return {
            "sweep": "⚪ None",
            "bos": "⚪ None",
        }

    x = df.copy()
    last = x.iloc[-1]

    previous_high = float(
        x["high"].iloc[-10:-1].max()
    )
    previous_low = float(
        x["low"].iloc[-10:-1].min()
    )

    bull_sweep = (
        float(last["low"]) < previous_low
        and float(last["close"]) > previous_low
    )

    bear_sweep = (
        float(last["high"]) > previous_high
        and float(last["close"]) < previous_high
    )

    bull_bos = float(last["close"]) > previous_high
    bear_bos = float(last["close"]) < previous_low

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
# LIQUIDATION-LIKE PRESSURE PROXY
# ============================================================

def liquidation_proxy(symbol):
    """
    IMPORTANT:
    Delta's public REST API does NOT provide a public,
    all-trader liquidation feed.

    This function therefore estimates liquidation-like
    pressure from public Delta data:
      - aggressive/public trade imbalance
      - OI change
      - price movement
      - L2 order-book imbalance

    It is NOT actual liquidation volume.
    """

    flow = trade_flow(symbol)
    ob = parse_orderbook(symbol, DEFAULT_DEPTH)

    if flow is None and ob is None:
        return None

    oi_ch = oi_change(symbol)

    score = 0
    reasons = []

    if flow is not None:
        d = flow["delta_pct"]

        if abs(d) >= 40:
            score += 2
            reasons.append("strong trade delta")
        elif abs(d) >= 25:
            score += 1
            reasons.append("trade delta")

    if ob is not None:
        imbalance = ob["imbalance"]

        if abs(imbalance) >= 35:
            score += 2
            reasons.append("strong L2 imbalance")
        elif abs(imbalance) >= 20:
            score += 1
            reasons.append("L2 imbalance")

    if oi_ch is not None and abs(oi_ch) >= 2:
        score += 1
        reasons.append("OI displacement")

    if flow is not None:
        if flow["delta_pct"] > 0:
            side = "BUY-SIDE PRESSURE"
        elif flow["delta_pct"] < 0:
            side = "SELL-SIDE PRESSURE"
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
        "reason": ", ".join(reasons) if reasons else "None",
    }


# ============================================================
# CURRENT LIQUIDITY SNAPSHOT / HEATMAP TABLE
# ============================================================

def liquidity_levels(symbol, depth=15):
    ob = parse_orderbook(symbol, depth)

    if ob is None:
        return None

    bid = ob["bid_df"].copy()
    ask = ob["ask_df"].copy()

    mid = ob["mid"]

    bid["Side"] = "BID"
    ask["Side"] = "ASK"

    bid["Distance %"] = (
        (mid - bid["Price"])
        / mid
        * 100
    )

    ask["Distance %"] = (
        (ask["Price"] - mid)
        / mid
        * 100
    )

    levels = pd.concat(
        [bid, ask],
        ignore_index=True,
    )

    levels["Notional"] = (
        levels["Price"]
        * levels["Size"]
    )

    levels["Distance %"] = levels[
        "Distance %"
    ].round(4)

    levels["Size"] = levels[
        "Size"
    ].round(4)

    levels["Notional"] = levels[
        "Notional"
    ].round(2)

    return ob, levels.sort_values(
        ["Side", "Distance %"]
    )


# ============================================================
# SCANNER ROW
# ============================================================

def scan_coin(symbol, ticker):
    d5 = get_history(symbol, "5m", 36)
    d15 = get_history(symbol, "15m", 72)
    d1h = get_history(symbol, "1h", 120)

    if min(len(d5), len(d15), len(d1h)) < 25:
        return None

    t5 = trend_label(d5)
    t15 = trend_label(d15)
    t1h = trend_label(d1h)

    bulls = sum(
        x == "🟢 BULL"
        for x in [t5, t15, t1h]
    )
    bears = sum(
        x == "🔴 BEAR"
        for x in [t5, t15, t1h]
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

    structure = structure_signal(d5)

    atr_df = add_atr(d5)
    atr = (
        float(atr_df["ATR"].iloc[-1])
        if not pd.isna(atr_df["ATR"].iloc[-1])
        else np.nan
    )

    volume_x = 0.0
    if len(d5) >= 8 and "volume" in d5.columns:
        avg_volume = float(
            d5["volume"].iloc[-7:-1].mean()
        )
        if avg_volume > 0:
            volume_x = float(
                d5["volume"].iloc[-1]
                / avg_volume
            )

    oi_ch = oi_change(symbol)
    ob = parse_orderbook(symbol, DEFAULT_DEPTH)
    liq = liquidation_proxy(symbol)

    long_score = 0
    short_score = 0

    if mtf == "🟢 LONG ALIGNED":
        long_score += 4
    elif mtf == "🔴 SHORT ALIGNED":
        short_score += 4
    elif mtf == "🟢 LONG BIAS":
        long_score += 2
    elif mtf == "🔴 SHORT BIAS":
        short_score += 2

    if "BULL" in structure["sweep"]:
        long_score += 2

    if "BEAR" in structure["sweep"]:
        short_score += 2

    if "BULL" in structure["bos"]:
        long_score += 3

    if "BEAR" in structure["bos"]:
        short_score += 3

    if volume_x >= 2:
        long_score += 1
        short_score += 1

    if oi_ch is not None:
        if oi_ch >= 1:
            if mtf.startswith("🟢"):
                long_score += 2
            elif mtf.startswith("🔴"):
                short_score += 2

    if ob is not None:
        if ob["imbalance"] >= 25:
            long_score += 2
        elif ob["imbalance"] <= -25:
            short_score += 2

    if liq is not None:
        if liq["side"] == "BUY-SIDE PRESSURE":
            long_score += 1
        elif liq["side"] == "SELL-SIDE PRESSURE":
            short_score += 1

    if (
        mtf == "⚪ CONFLICT"
        or max(long_score, short_score) < 5
    ):
        signal = "⚪ NO SIGNAL"
    elif long_score > short_score:
        signal = (
            "🟢 STRONG LONG"
            if long_score >= 8
            else "🟡 LONG WATCH"
        )
    elif short_score > long_score:
        signal = (
            "🔴 STRONG SHORT"
            if short_score >= 8
            else "🟠 SHORT WATCH"
        )
    else:
        signal = "⚪ NO SIGNAL"

    return {
        "Coin": symbol,
        "Price": float(ticker["Price"]),
        "Vol/OI": round(
            float(ticker["Vol/OI"])
            if pd.notna(ticker["Vol/OI"])
            else 0,
            2,
        ),
        "5m": t5,
        "15m": t15,
        "1H": t1h,
        "MTF": mtf,
        "Sweep": structure["sweep"],
        "BOS": structure["bos"],
        "Volume x": round(volume_x, 2),
        "OI Change %": (
            round(oi_ch, 2)
            if oi_ch is not None
            else np.nan
        ),
        "OB Imbalance %": (
            round(ob["imbalance"], 2)
            if ob is not None
            else np.nan
        ),
        "Liq Pressure": (
            liq["level"]
            if liq is not None
            else "⚪ UNKNOWN"
        ),
        "Long Score": long_score,
        "Short Score": short_score,
        "Score": max(long_score, short_score),
        "Signal": signal,
    }


# ============================================================
# MARKET LOAD
# ============================================================

products = get_perpetual_products()
tickers = get_tickers()

if products.empty or tickers.empty:
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

market = market.dropna(subset=["Price"])

market = market.sort_values(
    "24H Volume",
    ascending=False,
).reset_index(drop=True)

st.title("🔥 Delta Reversal Scanner")
st.caption(
    "Delta India public data → MTF → Structure → OI → "
    "Funding → L2 Order Book → Trade Flow → "
    "Liquidation-like pressure"
)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("⚙️ Scanner Settings")

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

    if st.button("🔄 Refresh All Data"):
        st.cache_data.clear()
        st.rerun()

# Live universe rule:
# exact 20x => Vol/OI > threshold
# >20x => NO Vol/OI filter
lev_num = pd.to_numeric(market["Leverage"], errors="coerce")
exact_20x = market[
    (lev_num == 20) &
    (market["Vol/OI"].fillna(0) > min_vol_oi)
].copy()
gt_20x = market[lev_num > 20].copy()
market_filtered = (
    pd.concat([exact_20x, gt_20x], ignore_index=True)
    .drop_duplicates(subset=["Coin"])
)

st.metric(
    "Perpetual contracts",
    len(market),
)

st.metric(
    f"Coins with Vol/OI > {min_vol_oi:g}",
    len(market_filtered),
)

mode = st.radio(
    "Page",
    [
        "🧪 6M Research", "🔥 Live Scanner",
        "📚 L2 Order Book",
        "💥 Liquidation Pressure",
        "🌡️ Liquidity Heatmap",
    ],
    horizontal=True,
)


# ============================================================
# 6-MONTH RESEARCH
# ============================================================

if mode == "🧪 6M Research":
    render_research_page()
    st.stop()


# ============================================================
# LIVE SCANNER
# ============================================================

if mode == "🔥 Live Scanner":
    candidates = market_filtered.head(scan_limit)

    if candidates.empty:
        st.warning(
            "Vol/OI filter ke baad koi coin nahi mila."
        )
        st.stop()

    st.info(
        f"{len(candidates)} coins scan honge. "
        "Deep scan API calls zyada ho sakti hain."
    )

    results = []
    progress = st.progress(0)

    for i, (_, row) in enumerate(
        candidates.iterrows()
    ):
        try:
            result = scan_coin(
                row["Coin"],
                row,
            )
            if result is not None:
                results.append(result)
        except Exception:
            # One bad symbol must not crash the whole app.
            pass

        progress.progress(
            int(
                (i + 1)
                / len(candidates)
                * 100
            )
        )

    progress.empty()

    if not results:
        st.warning(
            "Scanner ko enough candle/orderbook data nahi mila."
        )
        st.stop()

    df = pd.DataFrame(results)

    st.subheader("🎯 Complete Scanner")

    st.dataframe(
        df.sort_values(
            "Score",
            ascending=False,
        ),
        use_container_width=True,
        hide_index=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🟢 Long Watch")

        long_df = df.sort_values(
            "Long Score",
            ascending=False,
        )

        st.dataframe(
            long_df[
                [
                    "Coin",
                    "Price",
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

    with col2:
        st.subheader("🔴 Short Watch")

        short_df = df.sort_values(
            "Short Score",
            ascending=False,
        )

        st.dataframe(
            short_df[
                [
                    "Coin",
                    "Price",
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
    st.subheader("📚 Delta L2 Order Book")

    symbols = market["Coin"].head(100).tolist()

    if not symbols:
        st.warning("Coins available nahi hain.")
        st.stop()

    selected = st.selectbox(
        "Coin",
        symbols,
    )

    if st.button("🔍 Load L2"):
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

        imbalance = ob["imbalance"]

        if imbalance >= 25:
            st.success(
                f"🟢 Bid imbalance: {imbalance:.2f}%"
            )
        elif imbalance <= -25:
            st.error(
                f"🔴 Ask imbalance: {imbalance:.2f}%"
            )
        else:
            st.info(
                f"⚪ Balanced: {imbalance:.2f}%"
            )

        left, right = st.columns(2)

        with left:
            st.subheader("🟢 BIDS")

            bid = ob["bid_df"].copy()
            bid["Notional"] = (
                bid["Price"] * bid["Size"]
            )

            st.dataframe(
                bid,
                use_container_width=True,
                hide_index=True,
            )

        with right:
            st.subheader("🔴 ASKS")

            ask = ob["ask_df"].copy()
            ask["Notional"] = (
                ask["Price"] * ask["Size"]
            )

            st.dataframe(
                ask,
                use_container_width=True,
                hide_index=True,
            )

        st.warning(
            "⚠️ Visible L2 walls can be cancelled. "
            "Wall ko guaranteed support/resistance mat samjho."
        )


# ============================================================
# LIQUIDATION PRESSURE
# ============================================================

elif mode == "💥 Liquidation Pressure":
    st.subheader(
        "💥 Liquidation-like Pressure Scanner"
    )

    st.warning(
        "Important: Delta public API par sab traders ki "
        "actual liquidation feed public REST endpoint ke "
        "roop me available nahi hai. Is page ka signal "
        "public trades + L2 + OI se pressure proxy hai, "
        "actual liquidation volume nahi."
    )

    candidates = market_filtered.head(
        scan_limit
    )

    if candidates.empty:
        st.warning(
            "Vol/OI filter ke baad koi coin nahi mila."
        )
        st.stop()

    rows = []
    progress = st.progress(0)

    for i, (_, row) in enumerate(
        candidates.iterrows()
    ):
        symbol = row["Coin"]

        try:
            liq = liquidation_proxy(symbol)

            if liq is not None:
                rows.append(
                    {
                        "Coin": symbol,
                        "Price": row["Price"],
                        "Pressure": liq["level"],
                        "Side": liq["side"],
                        "Proxy Score": liq["score"],
                        "Trade Delta %": round(
                            liq["delta_pct"], 2
                        )
                        if pd.notna(
                            liq["delta_pct"]
                        )
                        else np.nan,
                        "OI Change %": round(
                            liq["oi_change"], 2
                        )
                        if liq["oi_change"]
                        is not None
                        else np.nan,
                        "L2 Imbalance %": round(
                            liq["ob_imbalance"], 2
                        )
                        if pd.notna(
                            liq["ob_imbalance"]
                        )
                        else np.nan,
                        "Reason": liq["reason"],
                    }
                )
        except Exception:
            pass

        progress.progress(
            int(
                (i + 1)
                / len(candidates)
                * 100
            )
        )

    progress.empty()

    if rows:
        liq_df = pd.DataFrame(rows)

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
        "Ye current L2 snapshot ka heatmap hai. "
        "Historical heatmap ke liye order-book snapshots "
        "continuously store karne honge."
    )

    symbols = market["Coin"].head(100).tolist()

    selected = st.selectbox(
        "Coin",
        symbols,
        key="heatmap_coin",
    )

    if st.button("🌡️ Build Heatmap"):
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

        # Size based heat styling.
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
            "Green/yellow intensity = larger visible "
            "order size in the current L2 snapshot."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Data source: Delta Exchange India public API. "
    "L2 orderbook: /v2/l2orderbook/{symbol}; "
    "public trades: /v2/trades/{symbol}; "
    "OHLC/OI history: /v2/history/candles."
)

st.caption(
    "⚠️ Scanner analytical tool hai. Orderbook walls "
    "cancel ho sakte hain aur liquidation-pressure "
    "proxy actual liquidation feed nahi hai."
)
