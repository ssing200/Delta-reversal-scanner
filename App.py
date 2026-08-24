import streamlit as st
import requests
import pandas as pd
import numpy as np
import time

# ============================================================
# DELTA REVERSAL SCANNER - FINAL FIXED ( >20x FIX )
# ============================================================

BASE_URL = "https://api.india.delta.exchange"
HEADERS = {
    "Accept": "application/json",
    "User-Agent": "Delta-Reversal-Scanner-Final/2.2",
}

st.set_page_config(
    page_title="Delta Reversal Scanner",
    page_icon="🔥",
    layout="wide",
)

CACHE_TTL = 20
DEFAULT_DEPTH = 15


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
    except (requests.RequestException, ValueError, TypeError):
        return None


@st.cache_data(ttl=CACHE_TTL)
def get_perpetual_products():
    result = api_get("/v2/products", {"contract_types": "perpetual_futures"})
    if not result:
        return pd.DataFrame(columns=["Coin", "ID", "Max Leverage", "Default Leverage"])

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

        # Robust leverage calculation
        default_lev = 20.0
        try:
            default_lev = float(item.get("default_leverage") or 20)
        except:
            pass

        max_lev = default_lev
        try:
            im = float(item.get("initial_margin") or 0)
            if im > 0:
                # initial_margin is in percent (0.5 = 0.5%, 5 = 5%)
                calculated = round(100 / im)
                max_lev = max(calculated, default_lev)
        except:
            max_lev = default_lev

        rows.append({
            "Coin": symbol,
            "ID": item.get("id"),
            "Max Leverage": int(max_lev),
            "Default Leverage": default_lev,
        })

    if not rows:
        return pd.DataFrame(columns=["Coin", "ID", "Max Leverage", "Default Leverage"])

    return pd.DataFrame(rows).drop_duplicates("Coin").reset_index(drop=True)


@st.cache_data(ttl=CACHE_TTL)
def get_tickers():
    result = api_get("/v2/tickers", {"contract_types": "perpetual_futures"})
    if not result:
        return pd.DataFrame()

    rows = []
    for item in result:
        symbol = item.get("symbol")
        if not symbol:
            continue
        try:
            price = float(item.get("close") or item.get("mark_price") or 0)
            volume = float(item.get("volume_24h") or item.get("volume") or 0)
            oi = float(item.get("open_interest") or item.get("oi") or 0)
        except:
            continue
        if price <= 0:
            continue

        funding = np.nan
        try:
            fr = item.get("funding_rate")
            if fr is not None:
                funding = float(fr)
        except:
            pass

        rows.append({
            "Coin": symbol,
            "Price": price,
            "24H Volume": volume,
            "OI": oi,
            "Funding": funding,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["Vol/OI"] = df["24H Volume"] / df["OI"].replace(0, np.nan)
    return df


@st.cache_data(ttl=CACHE_TTL)
def get_history(symbol, resolution="5m", hours=48):
    end = int(time.time())
    start = end - int(hours * 3600)
    result = api_get("/v2/history/candles", {
        "resolution": resolution,
        "symbol": symbol,
        "start": start,
        "end": end,
    })
    if not result:
        return pd.DataFrame()
    df = pd.DataFrame(result)
    if df.empty:
        return df
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "time" in df.columns:
        df["time"] = pd.to_numeric(df["time"], errors="coerce")
        df = df.sort_values("time")
    required = ["open", "high", "low", "close"]
    if not all(c in df.columns for c in required):
        return pd.DataFrame()
    df = df.dropna(subset=required)
    return df.drop_duplicates("time").reset_index(drop=True) if "time" in df.columns else df.reset_index(drop=True)


@st.cache_data(ttl=CACHE_TTL)
def get_oi_history(symbol, hours=48):
    end = int(time.time())
    start = end - int(hours * 3600)
    result = api_get("/v2/history/candles", {
        "resolution": "15m",
        "symbol": f"OI:{symbol}",
        "start": start,
        "end": end,
    })
    if not result:
        return pd.DataFrame()
    df = pd.DataFrame(result)
    if df.empty or "close" not in df.columns:
        return pd.DataFrame()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    if "time" in df.columns:
        df["time"] = pd.to_numeric(df["time"], errors="coerce")
        df = df.sort_values("time")
    return df.dropna(subset=["close"]).reset_index(drop=True)


@st.cache_data(ttl=10)
def get_orderbook(symbol, depth=15):
    result = api_get(f"/v2/l2orderbook/{symbol}", {"depth": int(depth)})
    return result if isinstance(result, dict) else None


def parse_orderbook(symbol, depth=15):
    data = get_orderbook(symbol, depth)
    if not data:
        return None
    bids = data.get("buy") or []
    asks = data.get("sell") or []
    bid_rows, ask_rows = [], []
    for row in bids:
        try:
            bid_rows.append({"Price": float(row["price"]), "Size": float(row["size"])})
        except:
            pass
    for row in asks:
        try:
            ask_rows.append({"Price": float(row["price"]), "Size": float(row["size"])})
        except:
            pass
    if not bid_rows or not ask_rows:
        return None
    bid_df = pd.DataFrame(bid_rows).sort_values("Price", ascending=False).reset_index(drop=True)
    ask_df = pd.DataFrame(ask_rows).sort_values("Price", ascending=True).reset_index(drop=True)
    bid_depth = float(bid_df["Size"].sum())
    ask_depth = float(ask_df["Size"].sum())
    total = bid_depth + ask_depth
    imbalance = (bid_depth - ask_depth) / total * 100 if total > 0 else 0.0
    best_bid = float(bid_df["Price"].max())
    best_ask = float(ask_df["Price"].min())
    mid = (best_bid + best_ask) / 2
    return {
        "symbol": symbol, "bid_df": bid_df, "ask_df": ask_df,
        "bid_depth": bid_depth, "ask_depth": ask_depth,
        "imbalance": imbalance, "best_bid": best_bid,
        "best_ask": best_ask, "mid": mid, "spread": best_ask - best_bid,
    }


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
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
    if "size" in df.columns:
        df["size"] = pd.to_numeric(df["size"], errors="coerce")
    if "side" not in df.columns:
        return df
    return df.dropna(subset=["price", "size"]).reset_index(drop=True)


def trade_flow(symbol):
    df = get_recent_trades(symbol)
    if df.empty or "side" not in df.columns:
        return None
    side = df["side"].astype(str).str.lower()
    buy = float(df.loc[side == "buy", "size"].sum())
    sell = float(df.loc[side == "sell", "size"].sum())
    total = buy + sell
    if total <= 0:
        return None
    delta = buy - sell
    return {"buy": buy, "sell": sell, "delta": delta, "delta_pct": delta / total * 100, "trades": len(df)}


def oi_change(symbol):
    df = get_oi_history(symbol, 48)
    if len(df) < 2:
        return None
    old = float(df["close"].iloc[0])
    current = float(df["close"].iloc[-1])
    if old == 0:
        return None
    return (current - old) / abs(old) * 100


def add_atr(df, period=14):
    x = df.copy()
    prev = x["close"].shift(1)
    tr = pd.concat([
        x["high"] - x["low"],
        (x["high"] - prev).abs(),
        (x["low"] - prev).abs(),
    ], axis=1).max(axis=1)
    x["ATR"] = tr.rolling(period).mean()
    return x


def trend_label(df):
    if len(df) < 30:
        return "⚪ UNKNOWN"
    close = df["close"]
    ema9 = close.ewm(span=9, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    last = float(close.iloc[-1])
    if last > ema9.iloc[-1] and ema9.iloc[-1] > ema21.iloc[-1] and ema21.iloc[-1] > ema50.iloc[-1]:
        return "🟢 BULL"
    if last < ema9.iloc[-1] and ema9.iloc[-1] < ema21.iloc[-1] and ema21.iloc[-1] < ema50.iloc[-1]:
        return "🔴 BEAR"
    return "🟡 MIXED"


def structure_signal(df):
    if len(df) < 20:
        return {"sweep": "⚪ None", "bos": "⚪ None"}
    x = df.copy()
    last = x.iloc[-1]
    prev_high = float(x["high"].iloc[-10:-1].max())
    prev_low = float(x["low"].iloc[-10:-1].min())
    bull_sweep = float(last["low"]) < prev_low and float(last["close"]) > prev_low
    bear_sweep = float(last["high"]) > prev_high and float(last["close"]) < prev_high
    bull_bos = float(last["close"]) > prev_high
    bear_bos = float(last["close"]) < prev_low
    sweep = "🟢 BULL SWEEP" if bull_sweep else "🔴 BEAR SWEEP" if bear_sweep else "⚪ None"
    bos = "🟢 BULL BOS" if bull_bos else "🔴 BEAR BOS" if bear_bos else "⚪ None"
    return {"sweep": sweep, "bos": bos}


def liquidation_proxy(symbol):
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
        imb = ob["imbalance"]
        if abs(imb) >= 35:
            score += 2
            reasons.append("strong L2 imbalance")
        elif abs(imb) >= 20:
            score += 1
            reasons.append("L2 imbalance")
    if oi_ch is not None and abs(oi_ch) >= 2:
        score += 1
        reasons.append("OI displacement")
    side = "BUY-SIDE PRESSURE" if (flow and flow["delta_pct"] > 0) else "SELL-SIDE PRESSURE" if (flow and flow["delta_pct"] < 0) else "UNKNOWN"
    level = "🔥 HIGH" if score >= 4 else "🟡 MEDIUM" if score >= 2 else "⚪ LOW"
    return {
        "level": level, "side": side, "score": score,
        "delta_pct": flow["delta_pct"] if flow else np.nan,
        "oi_change": oi_ch,
        "ob_imbalance": ob["imbalance"] if ob else np.nan,
        "reason": ", ".join(reasons) if reasons else "None",
    }


def liquidity_levels(symbol, depth=15):
    ob = parse_orderbook(symbol, depth)
    if ob is None:
        return None
    bid = ob["bid_df"].copy()
    ask = ob["ask_df"].copy()
    mid = ob["mid"]
    bid["Side"] = "BID"
    ask["Side"] = "ASK"
    bid["Distance %"] = (mid - bid["Price"]) / mid * 100
    ask["Distance %"] = (ask["Price"] - mid) / mid * 100
    levels = pd.concat([bid, ask], ignore_index=True)
    levels["Notional"] = levels["Price"] * levels["Size"]
    levels["Distance %"] = levels["Distance %"].round(4)
    levels["Size"] = levels["Size"].round(4)
    levels["Notional"] = levels["Notional"].round(2)
    return ob, levels.sort_values(["Side", "Distance %"])


def scan_coin(symbol, ticker, max_leverage=20):
    d5 = get_history(symbol, "5m", 36)
    d15 = get_history(symbol, "15m", 72)
    d1h = get_history(symbol, "1h", 120)
    if min(len(d5), len(d15), len(d1h)) < 25:
        return None

    t5 = trend_label(d5)
    t15 = trend_label(d15)
    t1h = trend_label(d1h)
    bulls = sum(x == "🟢 BULL" for x in [t5, t15, t1h])
    bears = sum(x == "🔴 BEAR" for x in [t5, t15, t1h])

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
    atr = float(atr_df["ATR"].iloc[-1]) if not pd.isna(atr_df["ATR"].iloc[-1]) else np.nan
    if pd.notna(atr) and atr / ticker["Price"] < 0.0012:
        return None

    volume_x = 0.0
    if len(d5) >= 8 and "volume" in d5.columns:
        avg = float(d5["volume"].iloc[-7:-1].mean())
        if avg > 0:
            volume_x = float(d5["volume"].iloc[-1] / avg)

    oi_ch = oi_change(symbol)
    ob = parse_orderbook(symbol, DEFAULT_DEPTH)
    liq = liquidation_proxy(symbol)

    long_score = short_score = 0

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

    if volume_x >= 2.2:
        long_score += 2
        short_score += 2
    elif volume_x >= 1.6:
        long_score += 1
        short_score += 1

    if oi_ch is not None:
        if oi_ch >= 1.5 and mtf.startswith("🟢"):
            long_score += 2
        elif oi_ch >= 1.5 and mtf.startswith("🔴"):
            short_score += 2
        elif oi_ch <= -1.5:
            long_score = max(0, long_score - 1)
            short_score = max(0, short_score - 1)

    if ob is not None:
        if ob["imbalance"] >= 28:
            long_score += 2
        elif ob["imbalance"] <= -28:
            short_score += 2

    if liq is not None:
        if liq["side"] == "BUY-SIDE PRESSURE":
            long_score += 1
            if ob is not None and ob["imbalance"] > 15:
                long_score += 1
        elif liq["side"] == "SELL-SIDE PRESSURE":
            short_score += 1
            if ob is not None and ob["imbalance"] < -15:
                short_score += 1

    funding = ticker.get("Funding", np.nan)
    if pd.notna(funding):
        if funding > 0.0004:
            short_score += 1
        elif funding < -0.0004:
            long_score += 1

    if mtf == "⚪ CONFLICT" or max(long_score, short_score) < 6:
        signal = "⚪ NO SIGNAL"
    elif long_score > short_score:
        if long_score >= 10 and mtf.startswith("🟢") and ("BULL" in structure["sweep"] or "BULL" in structure["bos"]):
            signal = "🟢 STRONG LONG"
        elif long_score >= 7:
            signal = "🟡 LONG WATCH"
        else:
            signal = "⚪ NO SIGNAL"
    elif short_score > long_score:
        if short_score >= 10 and mtf.startswith("🔴") and ("BEAR" in structure["sweep"] or "BEAR" in structure["bos"]):
            signal = "🔴 STRONG SHORT"
        elif short_score >= 7:
            signal = "🟠 SHORT WATCH"
        else:
            signal = "⚪ NO SIGNAL"
    else:
        signal = "⚪ NO SIGNAL"

    return {
        "Coin": symbol,
        "Max Leverage": max_leverage,
        "Price": float(ticker["Price"]),
        "Vol/OI": round(float(ticker["Vol/OI"]) if pd.notna(ticker["Vol/OI"]) else 0, 2),
        "Funding": round(float(funding), 6) if pd.notna(funding) else np.nan,
        "5m": t5, "15m": t15, "1H": t1h, "MTF": mtf,
        "Sweep": structure["sweep"], "BOS": structure["bos"],
        "Volume x": round(volume_x, 2),
        "OI Change %": round(oi_ch, 2) if oi_ch is not None else np.nan,
        "OB Imbalance %": round(ob["imbalance"], 2) if ob is not None else np.nan,
        "Liq Pressure": liq["level"] if liq is not None else "⚪ UNKNOWN",
        "Long Score": long_score, "Short Score": short_score,
        "Score": max(long_score, short_score), "Signal": signal,
    }


# ============================================================
# LOAD DATA
# ============================================================

products = get_perpetual_products()
tickers = get_tickers()

if products.empty or tickers.empty:
    st.error("❌ Data load nahi hua. Refresh dabao.")
    st.stop()

market = products.merge(tickers, on="Coin", how="left")
market = market.dropna(subset=["Price"])
market = market.sort_values("24H Volume", ascending=False).reset_index(drop=True)

st.title("🔥 Delta Reversal Scanner — Fixed >20x")
st.caption("Leverage split + Sharp scoring | >20x threshold soft rakha gaya hai")

with st.sidebar:
    st.header("⚙️ Settings")
    min_vol_oi_high = st.number_input("Min Vol/OI for > 20x", min_value=0.0, value=1.5, step=0.5)
    min_vol_oi_low  = st.number_input("Min Vol/OI for ≤ 20x", min_value=0.0, value=1.5, step=0.5)
    scan_limit = st.slider("Deep scan coins", 5, 40, 30, 5)
    depth = st.slider("L2 depth", 5, 50, DEFAULT_DEPTH, 5)
    if st.button("🔄 Refresh All Data"):
        st.cache_data.clear()
        st.rerun()

# Filters
high_mask = (market["Max Leverage"] > 20) & (market["Vol/OI"].fillna(0) >= min_vol_oi_high)
low_mask  = (market["Max Leverage"] <= 20) & (market["Vol/OI"].fillna(0) >= min_vol_oi_low)
market_filtered = market[high_mask | low_mask].copy()
market_filtered = market_filtered.sort_values("24H Volume", ascending=False).reset_index(drop=True)

# Strong debug
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Perps", len(market))
c2.metric("Total >20x", len(market[market["Max Leverage"] > 20]))
c3.metric("Total ≤20x", len(market[market["Max Leverage"] <= 20]))
c4.metric(f">20x after filter", len(market[high_mask]))
c5.metric(f"≤20x after filter", len(market[low_mask]))

# Show top high leverage coins for debug
with st.expander("🔍 Debug: Top >20x coins (before deep scan)"):
    high_debug = market[market["Max Leverage"] > 20][["Coin", "Max Leverage", "Price", "24H Volume", "Vol/OI"]].head(15)
    st.dataframe(high_debug, use_container_width=True, hide_index=True)

mode = st.radio("Page", ["🔥 Live Scanner", "📚 L2 Order Book", "💥 Liquidation Pressure", "🌡️ Liquidity Heatmap"], horizontal=True)


if mode == "🔥 Live Scanner":
    candidates = market_filtered.head(scan_limit)
    if candidates.empty:
        st.warning("Filter ke baad koi coin nahi mila.")
        st.stop()

    st.info(f"Scanning {len(candidates)} coins...")
    results = []
    progress = st.progress(0)
    for i, (_, row) in enumerate(candidates.iterrows()):
        try:
            res = scan_coin(row["Coin"], row, max_leverage=int(row.get("Max Leverage", 20)))
            if res:
                results.append(res)
        except:
            pass
        progress.progress(int((i + 1) / len(candidates) * 100))
    progress.empty()

    if not results:
        st.warning("Enough data nahi mila.")
        st.stop()

    df = pd.DataFrame(results)

    st.subheader("🎯 Complete Scanner")
    st.dataframe(df.sort_values("Score", ascending=False), use_container_width=True, hide_index=True)

    low_df = df[df["Max Leverage"] <= 20].sort_values("Score", ascending=False)
    high_df = df[df["Max Leverage"] > 20].sort_values("Score", ascending=False)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📉 ≤ 20x Leverage")
        if low_df.empty:
            st.info("Koi coin nahi mila")
        else:
            st.dataframe(low_df, use_container_width=True, hide_index=True)
    with col2:
        st.subheader("🚀 > 20x Leverage")
        if high_df.empty:
            st.info("Koi coin nahi mila")
        else:
            st.dataframe(high_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🟢 Long Watch")
        long_df = df[df["Long Score"] >= 6].sort_values("Long Score", ascending=False)
        st.dataframe(long_df[["Coin", "Max Leverage", "Price", "MTF", "Sweep", "BOS", "Volume x", "OI Change %", "OB Imbalance %", "Funding", "Liq Pressure", "Long Score", "Signal"]], use_container_width=True, hide_index=True)
    with c2:
        st.subheader("🔴 Short Watch")
        short_df = df[df["Short Score"] >= 6].sort_values("Short Score", ascending=False)
        st.dataframe(short_df[["Coin", "Max Leverage", "Price", "MTF", "Sweep", "BOS", "Volume x", "OI Change %", "OB Imbalance %", "Funding", "Liq Pressure", "Short Score", "Signal"]], use_container_width=True, hide_index=True)


elif mode == "📚 L2 Order Book":
    st.subheader("📚 L2 Order Book")
    selected = st.selectbox("Coin", market["Coin"].head(100).tolist())
    if st.button("🔍 Load L2"):
        ob = parse_orderbook(selected, depth)
        if not ob:
            st.error("Data nahi mila")
            st.stop()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Best Bid", f'{ob["best_bid"]:.8g}')
        c2.metric("Best Ask", f'{ob["best_ask"]:.8g}')
        c3.metric("Bid Depth", f'{ob["bid_depth"]:,.0f}')
        c4.metric("Ask Depth", f'{ob["ask_depth"]:,.0f}')
        imb = ob["imbalance"]
        if imb >= 25:
            st.success(f"🟢 Bid imbalance: {imb:.2f}%")
        elif imb <= -25:
            st.error(f"🔴 Ask imbalance: {imb:.2f}%")
        else:
            st.info(f"⚪ Balanced: {imb:.2f}%")
        left, right = st.columns(2)
        with left:
            st.subheader("🟢 BIDS")
            b = ob["bid_df"].copy()
            b["Notional"] = b["Price"] * b["Size"]
            st.dataframe(b, use_container_width=True, hide_index=True)
        with right:
            st.subheader("🔴 ASKS")
            a = ob["ask_df"].copy()
            a["Notional"] = a["Price"] * a["Size"]
            st.dataframe(a, use_container_width=True, hide_index=True)


elif mode == "💥 Liquidation Pressure":
    st.subheader("💥 Liquidation-like Pressure")
    st.warning("Proxy only (trades + L2 + OI). Actual liquidation feed nahi hai.")
    candidates = market_filtered.head(scan_limit)
    rows = []
    progress = st.progress(0)
    for i, (_, row) in enumerate(candidates.iterrows()):
        try:
            liq = liquidation_proxy(row["Coin"])
            if liq:
                rows.append({
                    "Coin": row["Coin"], "Max Leverage": row.get("Max Leverage", 20),
                    "Price": row["Price"], "Pressure": liq["level"], "Side": liq["side"],
                    "Proxy Score": liq["score"],
                    "Trade Delta %": round(liq["delta_pct"], 2) if pd.notna(liq["delta_pct"]) else np.nan,
                    "OI Change %": round(liq["oi_change"], 2) if liq["oi_change"] is not None else np.nan,
                    "L2 Imbalance %": round(liq["ob_imbalance"], 2) if pd.notna(liq["ob_imbalance"]) else np.nan,
                    "Reason": liq["reason"],
                })
        except:
            pass
        progress.progress(int((i + 1) / len(candidates) * 100))
    progress.empty()
    if rows:
        st.dataframe(pd.DataFrame(rows).sort_values("Proxy Score", ascending=False), use_container_width=True, hide_index=True)
    else:
        st.info("Data nahi mila")


else:
    st.subheader("🌡️ L2 Liquidity Heatmap")
    selected = st.selectbox("Coin", market["Coin"].head(100).tolist(), key="hm")
    if st.button("🌡️ Build Heatmap"):
        parsed = liquidity_levels(selected, depth)
        if not parsed:
            st.error("Data nahi mila")
            st.stop()
        ob, levels = parsed
        c1, c2, c3 = st.columns(3)
        c1.metric("Mid", f'{ob["mid"]:.8g}')
        c2.metric("Bid / Ask", f'{ob["bid_depth"]:,.0f} / {ob["ask_depth"]:,.0f}')
        c3.metric("Imbalance", f'{ob["imbalance"]:.2f}%')
        display = levels[["Side", "Price", "Distance %", "Size", "Notional"]].copy()
        styled = display.style.background_gradient(subset=["Size"], cmap="RdYlGn").format({
            "Price": "{:.8g}", "Distance %": "{:.4f}", "Size": "{:.4f}", "Notional": "{:.2f}"
        })
        st.dataframe(styled, use_container_width=True, hide_index=True)

st.divider()
st.caption("Data: Delta Exchange India public API | Analytical tool only.")
