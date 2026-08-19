import streamlit as st
import requests
import pandas as pd
import numpy as np
import time

BASE_URL = "https://api.india.delta.exchange"
HEADERS = {"Accept": "application/json", "User-Agent": "Delta-Reversal-Scanner/9.0"}

CACHE_SECONDS = 120
DEEP_SCAN_LIMIT = 30
VOL_OI_MIN = 6.0
RR_DEFAULT = 2.0

st.set_page_config(page_title="Delta Reversal Scanner PRO 9", layout="wide")
st.title("🔥 Delta Reversal Scanner PRO 9")
st.caption("MTF → 5D Regime → S/R → Sweep → BOS/CHOCH → FVG → OI → Funding → Volume → ATR")

# ---------------- API ----------------
def api_get(path, params=None):
    try:
        r = requests.get(BASE_URL + path, params=params, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            return None
        data = r.json()
        if data.get("success") is False:
            return None
        return data.get("result", [])
    except Exception:
        return None

@st.cache_data(ttl=CACHE_SECONDS)
def get_all_perpetuals():
    result = api_get("/v2/products")
    if not result:
        return pd.DataFrame()
    rows = []
    for p in result:
        if p.get("contract_type") != "perpetual_futures":
            continue
        if p.get("state") != "live" or p.get("trading_status") != "operational":
            continue
        s = p.get("symbol")
        if s:
            rows.append({"Coin": s, "ID": p.get("id")})
    return pd.DataFrame(rows).drop_duplicates("Coin")

@st.cache_data(ttl=CACHE_SECONDS)
def get_tickers():
    result = api_get("/v2/tickers")
    if not result:
        return pd.DataFrame()
    rows = []
    for p in result:
        s = p.get("symbol")
        if not s:
            continue
        try:
            price = float(p.get("close", p.get("mark_price", 0)) or 0)
            vol = float(p.get("volume_24h", p.get("volume", 0)) or 0)
            oi = float(p.get("open_interest", p.get("oi", 0)) or 0)
        except Exception:
            continue
        if price <= 0:
            continue
        raw = p.get("funding_rate", p.get("funding"))
        try:
            funding = float(raw) if raw is not None else None
        except Exception:
            funding = None
        rows.append({"Coin": s, "Price": price, "24H Volume": vol, "OI": oi, "Funding": funding})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["Vol/OI"] = df["24H Volume"] / df["OI"].replace(0, np.nan)
    return df

@st.cache_data(ttl=CACHE_SECONDS)
def get_candles(symbol, resolution, hours):
    end = int(time.time())
    start = end - int(hours * 3600)
    result = api_get("/v2/history/candles", {
        "resolution": resolution, "symbol": symbol, "start": start, "end": end
    })
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
    df = df.dropna(subset=["open", "high", "low", "close"])
    return df.sort_values("time").drop_duplicates("time").reset_index(drop=True)

@st.cache_data(ttl=CACHE_SECONDS)
def get_oi_history(symbol, hours=24):
    end = int(time.time())
    start = end - hours * 3600
    result = api_get("/v2/history/candles", {
        "resolution": "15m", "symbol": "OI:" + symbol, "start": start, "end": end
    })
    if not result:
        return pd.DataFrame()
    df = pd.DataFrame(result)
    if df.empty or "close" not in df.columns:
        return pd.DataFrame()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    return df.dropna(subset=["close"]).sort_values("time").reset_index(drop=True)

# ---------------- indicators ----------------
def add_atr(df, period=14):
    x = df.copy()
    pc = x["close"].shift(1)
    tr = pd.concat([
        x["high"] - x["low"],
        (x["high"] - pc).abs(),
        (x["low"] - pc).abs()
    ], axis=1).max(axis=1)
    x["ATR"] = tr.rolling(period).mean()
    x["ATRpct"] = x["ATR"] / x["close"] * 100
    return x

def swings(df, left=2, right=2):
    x = df.copy()
    x["SwingHigh"] = False
    x["SwingLow"] = False
    for i in range(left, len(x)-right):
        if x["high"].iloc[i] > x["high"].iloc[i-left:i].max() and x["high"].iloc[i] > x["high"].iloc[i+1:i+right+1].max():
            x.loc[x.index[i], "SwingHigh"] = True
        if x["low"].iloc[i] < x["low"].iloc[i-left:i].min() and x["low"].iloc[i] < x["low"].iloc[i+1:i+right+1].min():
            x.loc[x.index[i], "SwingLow"] = True
    return x

def closed(df):
    # Drop the currently forming candle when time is available.
    if df.empty or "time" not in df.columns:
        return df
    now = int(time.time())
    if len(df) >= 2:
        last_t = int(df["time"].iloc[-1])
        # If the final candle started less than one resolution period ago,
        # it is likely still forming. A conservative check is used.
        # Caller supplies enough history; remove only an obviously current candle.
        if now - last_t < 60:
            return df.iloc[:-1].copy()
    return df

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
        return {"state": "⚪ UNKNOWN", "range_pct": None}
    x = df.iloc[-120:] if len(df) > 120 else df
    first = x["close"].iloc[0]
    last = x["close"].iloc[-1]
    hi = x["high"].max()
    lo = x["low"].min()
    rng = (hi-lo)/lo*100 if lo else 0
    move = (last-first)/first*100 if first else 0
    if move > 4:
        state = "🟢 5D UPTREND"
    elif move < -4:
        state = "🔴 5D DOWNTREND"
    elif rng > 12 and abs(move) < 3:
        state = "🟡 5D RANGE"
    else:
        state = "⚪ 5D UNCERTAINTY"
    return {"state": state, "range_pct": rng}

def sr_levels(df, lookback=80, tol=0.006):
    x = swings(df.iloc[-lookback:].copy(), 2, 2)
    highs = x.loc[x["SwingHigh"], "high"].tolist()
    lows = x.loc[x["SwingLow"], "low"].tolist()
    price = float(df["close"].iloc[-1])
    supports = sorted([v for v in lows if v < price], reverse=True)
    resistances = sorted([v for v in highs if v > price])
    support = supports[0] if supports else np.nan
    resistance = resistances[0] if resistances else np.nan
    return support, resistance, len(supports), len(resistances)

def sweep_bos_fvg(df):
    x = swings(df, 2, 2)
    last = x.iloc[-1]
    sh = x.loc[x["SwingHigh"], "high"]
    sl = x.loc[x["SwingLow"], "low"]
    ph = sh.iloc[-1] if len(sh) else x["high"].iloc[-8:-1].max()
    pl = sl.iloc[-1] if len(sl) else x["low"].iloc[-8:-1].min()
    bull_sweep = last["low"] < pl and last["close"] > pl
    bear_sweep = last["high"] > ph and last["close"] < ph
    prev_h = x["high"].iloc[-8:-1].max()
    prev_l = x["low"].iloc[-8:-1].min()
    bull_bos = last["close"] > prev_h
    bear_bos = last["close"] < prev_l
    # CHOCH approximation: break opposite to recent directional structure.
    recent = x.iloc[-20:]
    if len(recent) >= 10:
        prior_high = recent["high"].iloc[:10].max()
        prior_low = recent["low"].iloc[:10].min()
    else:
        prior_high, prior_low = prev_h, prev_l
    bull_choch = last["close"] > prior_high and not bull_bos
    bear_choch = last["close"] < prior_low and not bear_bos
    bull_fvg = len(x) >= 3 and x["low"].iloc[-1] > x["high"].iloc[-3]
    bear_fvg = len(x) >= 3 and x["high"].iloc[-1] < x["low"].iloc[-3]
    return {
        "bull_sweep": bull_sweep, "bear_sweep": bear_sweep,
        "bull_bos": bull_bos, "bear_bos": bear_bos,
        "bull_choch": bull_choch, "bear_choch": bear_choch,
        "bull_fvg": bull_fvg, "bear_fvg": bear_fvg,
        "swing_high": float(ph), "swing_low": float(pl)
    }

def oi_analysis(symbol):
    df = get_oi_history(symbol, 24)
    if len(df) < 7:
        return None, "⚪ UNKNOWN"
    cur = float(df["close"].iloc[-1])
    old = float(df["close"].iloc[-7])
    if old == 0:
        return None, "⚪ UNKNOWN"
    ch = (cur-old)/abs(old)*100
    return ch, ("🔺 OI UP" if ch >= 1 else "🔻 OI DOWN" if ch <= -1 else "⚪ OI FLAT")

def volume_analysis(df):
    if len(df) < 10:
        return 0
    avg = df["volume"].iloc[-7:-1].mean()
    return float(df["volume"].iloc[-1]/avg) if avg > 0 else 0

def atr_analysis(df):
    x = add_atr(df)
    if len(x) < 25 or pd.isna(x["ATR"].iloc[-1]) or pd.isna(x["ATR"].iloc[-7]):
        return None, "⚪ UNKNOWN"
    a = float(x["ATR"].iloc[-1])
    old = float(x["ATR"].iloc[-7])
    d = a/old if old else 1
    direction = "🔺 ATR EXPANDING" if d >= 1.10 else "🔻 ATR CONTRACTING" if d <= .90 else "⚪ ATR FLAT"
    return a, direction

def mtf_state(t5, t15, t1):
    bulls = sum(v == "🟢 BULL" for v in [t5,t15,t1])
    bears = sum(v == "🔴 BEAR" for v in [t5,t15,t1])
    if bulls == 3: return "🟢 MTF ALIGNED LONG"
    if bears == 3: return "🔴 MTF ALIGNED SHORT"
    if bulls >= 2 and bears == 0: return "🟢 MTF LONG BIAS"
    if bears >= 2 and bulls == 0: return "🔴 MTF SHORT BIAS"
    if bulls == 0 and bears == 0: return "🟡 MTF RANGE/MIXED"
    return "⚪ MTF CONFLICT"

# ---------------- live ----------------
def deep_analysis(symbol, ticker):
    d5 = closed(get_candles(symbol, "5m", 36))
    d15 = closed(get_candles(symbol, "15m", 72))
    d1 = closed(get_candles(symbol, "1h", 120))
    if min(len(d5),len(d15),len(d1)) < 25:
        return None

    t5, t15, t1 = timeframe_trend(d5), timeframe_trend(d15), timeframe_trend(d1)
    mtf = mtf_state(t5,t15,t1)
    reg = regime_5d(symbol)
    sr_sup, sr_res, sup_count, res_count = sr_levels(d15)
    struct = sweep_bos_fvg(d5)
    atr, atr_dir = atr_analysis(d5)
    volx = volume_analysis(d5)
    oi_ch, oi_sig = oi_analysis(symbol)

    funding = ticker.get("Funding")
    fp = float(funding)*100 if funding is not None else None
    price = float(ticker["Price"])

    long_score = short_score = 0
    lr, sr = [], []

    # MTF: strongest condition
    if mtf == "🟢 MTF ALIGNED LONG": long_score += 4; lr.append("5m+15m+1H aligned")
    elif mtf == "🔴 MTF ALIGNED SHORT": short_score += 4; sr.append("5m+15m+1H aligned")
    elif mtf == "🟢 MTF LONG BIAS": long_score += 2; lr.append("MTF long bias")
    elif mtf == "🔴 MTF SHORT BIAS": short_score += 2; sr.append("MTF short bias")
    elif mtf == "⚪ MTF CONFLICT":
        long_score -= 2; short_score -= 2

    # 5D regime
    if "UPTREND" in reg["state"]: long_score += 2; lr.append("5D uptrend")
    elif "DOWNTREND" in reg["state"]: short_score += 2; sr.append("5D downtrend")
    elif "RANGE" in reg["state"]: long_score -= 1; short_score -= 1

    # Structure
    if struct["bull_sweep"]: long_score += 2; lr.append("liquidity sweep")
    if struct["bear_sweep"]: short_score += 2; sr.append("liquidity sweep")
    if struct["bull_bos"]: long_score += 3; lr.append("BOS")
    if struct["bear_bos"]: short_score += 3; sr.append("BOS")
    if struct["bull_choch"]: long_score += 2; lr.append("CHOCH")
    if struct["bear_choch"]: short_score += 2; sr.append("CHOCH")
    if struct["bull_fvg"]: long_score += 2; lr.append("bull FVG")
    if struct["bear_fvg"]: short_score += 2; sr.append("bear FVG")

    # S/R confluence
    if not pd.isna(sr_sup):
        dist = abs(price-sr_sup)/price
        if dist <= .01: long_score += 2; lr.append("near support")
    if not pd.isna(sr_res):
        dist = abs(sr_res-price)/price
        if dist <= .01: short_score += 2; sr.append("near resistance")

    # Volume
    if volx >= 2: long_score += 2; short_score += 2; lr.append("volume spike"); sr.append("volume spike")
    elif volx >= 1.3: long_score += 1; short_score += 1

    # OI displacement
    if oi_ch is not None:
        if oi_ch >= 1:
            if mtf.startswith("🟢"): long_score += 2; lr.append("OI expansion")
            if mtf.startswith("🔴"): short_score += 2; sr.append("OI expansion")
        elif oi_ch <= -1:
            if struct["bull_sweep"]: long_score += 1; lr.append("OI unwind after sweep")
            if struct["bear_sweep"]: short_score += 1; sr.append("OI unwind after sweep")

    # Funding
    if fp is not None:
        if fp >= .05: short_score += 2; sr.append("positive funding crowding"); funding_signal="🔴 Long crowded"
        elif fp <= -.05: long_score += 2; lr.append("negative funding crowding"); funding_signal="🟢 Short crowded"
        else: funding_signal="⚪ Neutral"
    else:
        funding_signal="⚪ Unavailable"

    # ATR
    if atr_dir == "🔺 ATR EXPANDING": long_score += 1; short_score += 1
    elif atr_dir == "🔻 ATR CONTRACTING": long_score -= 1; short_score -= 1

    # Hard block on MTF conflict
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
    return {
        "Coin": symbol, "Price": price, "24H Volume": ticker["24H Volume"],
        "OI": ticker["OI"], "Vol/OI": round(float(ticker["Vol/OI"]),2),
        "5m":t5, "15m":t15, "1H":t1, "MTF":mtf,
        "5D Regime":reg["state"], "5D Range %":round(reg["range_pct"],2) if reg["range_pct"] else None,
        "Support":round(sr_sup,8) if not pd.isna(sr_sup) else None,
        "Resistance":round(sr_res,8) if not pd.isna(sr_res) else None,
        "S/R Count":f"{sup_count}/{res_count}",
        "Liquidity":("🟢 BULL SWEEP" if struct["bull_sweep"] else "🔴 BEAR SWEEP" if struct["bear_sweep"] else "⚪ None"),
        "BOS":("🟢 BULL BOS" if struct["bull_bos"] else "🔴 BEAR BOS" if struct["bear_bos"] else "⚪ None"),
        "CHOCH":("🟢 BULL CHOCH" if struct["bull_choch"] else "🔴 BEAR CHOCH" if struct["bear_choch"] else "⚪ None"),
        "FVG":("🟢 BULL FVG" if struct["bull_fvg"] else "🔴 BEAR FVG" if struct["bear_fvg"] else "⚪ None"),
        "ATR":round(atr,8) if atr else None, "ATR Direction":atr_dir,
        "Volume x":round(volx,2), "OI Change %":round(oi_ch,2) if oi_ch is not None else None,
        "OI Signal":oi_sig, "Funding %":round(fp,4) if fp is not None else None,
        "Funding":funding_signal, "Long Score":long_score, "Short Score":short_score,
        "Score":score, "Signal":signal,
        "Long Reason":" + ".join(lr) if lr else "None",
        "Short Reason":" + ".join(sr) if sr else "None"
    }

# ---------------- backtest ----------------
def backtest_symbol(symbol, days=7, rr=2.0, threshold=8):
    # Conservative historical test using only data available before each candle.
    df = get_candles(symbol, "5m", days*24)
    if len(df) < 180:
        return []
    df = add_atr(swings(df))
    trades=[]
    for i in range(80, len(df)-50):
        hist=df.iloc[:i].copy()
        cur=df.iloc[i]
        if len(hist)<60: continue

        # Regime and MTF proxies from historical data.
        def tr(x):
            if len(x)<30:return "MIXED"
            c=x["close"]; e9=c.ewm(span=9,adjust=False).mean(); e21=c.ewm(span=21,adjust=False).mean()
            return "BULL" if c.iloc[-1]>e9.iloc[-1]>e21.iloc[-1] else "BEAR" if c.iloc[-1]<e9.iloc[-1]<e21.iloc[-1] else "MIXED"
        t5=tr(hist.iloc[-60:])
        t15=tr(hist.iloc[::3].iloc[-60:])
        t1=tr(hist.iloc[::12].iloc[-60:])
        bulls=[t5,t15,t1].count("BULL"); bears=[t5,t15,t1].count("BEAR")
        if bulls==3: mtf_long=True; mtf_short=False
        elif bears==3: mtf_short=True; mtf_long=False
        else: mtf_long=mtf_short=False

        sw=hist
        sh=sw.loc[sw["SwingHigh"],"high"]; sl=sw.loc[sw["SwingLow"],"low"]
        if sh.empty or sl.empty: continue
        ph,pl=sh.iloc[-1],sl.iloc[-1]
        bull_sweep=cur["low"]<pl and cur["close"]>pl
        bear_sweep=cur["high"]>ph and cur["close"]<ph
        ph8=hist["high"].iloc[-8:].max(); pl8=hist["low"].iloc[-8:].min()
        bull_bos=cur["close"]>ph8; bear_bos=cur["close"]<pl8
        bull_fvg=cur["low"]>hist["high"].iloc[-3]
        bear_fvg=cur["high"]<hist["low"].iloc[-3]
        av=hist["volume"].iloc[-7:].mean()
        vx=cur["volume"]/av if av>0 else 0
        atr_rising=(not pd.isna(cur["ATR"]) and not pd.isna(hist["ATR"].iloc[-7]) and cur["ATR"]>hist["ATR"].iloc[-7]*1.1)

        ls=ss=0
        if mtf_long: ls+=4
        if mtf_short: ss+=4
        if bull_sweep: ls+=2
        if bear_sweep: ss+=2
        if bull_bos: ls+=3
        if bear_bos: ss+=3
        if bull_fvg: ls+=2
        if bear_fvg: ss+=2
        if vx>=2: ls+=2; ss+=2
        elif vx>=1.3: ls+=1; ss+=1
        if atr_rising: ls+=1; ss+=1

        def simulate(side, score):
            if score<threshold:return None
            entry=float(cur["close"])
            if side=="LONG":
                stop=min(float(cur["low"]),float(pl)); risk=entry-stop
                if risk<=0:return None
                target=entry+risk*rr
            else:
                stop=max(float(cur["high"]),float(ph)); risk=stop-entry
                if risk<=0:return None
                target=entry-risk*rr
            for j in range(i+1,min(i+50,len(df))):
                f=df.iloc[j]
                if side=="LONG":
                    # If both occur in same candle, pessimistically count SL first.
                    if f["low"]<=stop:return {"Side":side,"Score":score,"Result":"LOSS","R":-1,"Time":cur["time"],"Entry":entry,"SL":stop,"TP":target}
                    if f["high"]>=target:return {"Side":side,"Score":score,"Result":"WIN","R":rr,"Time":cur["time"],"Entry":entry,"SL":stop,"TP":target}
                else:
                    if f["high"]>=stop:return {"Side":side,"Score":score,"Result":"LOSS","R":-1,"Time":cur["time"],"Entry":entry,"SL":stop,"TP":target}
                    if f["low"]<=target:return {"Side":side,"Score":score,"Result":"WIN","R":rr,"Time":cur["time"],"Entry":entry,"SL":stop,"TP":target}
            return None
        a=simulate("LONG",ls); b=simulate("SHORT",ss)
        if a and b: trades.append(a if a["Score"]>=b["Score"] else b)
        elif a: trades.append(a)
        elif b: trades.append(b)
    return trades

# ---------------- UI ----------------
coins=get_all_perpetuals(); tickers=get_tickers()
if coins.empty or tickers.empty:
    st.error("❌ Market data load nahi hua.")
    st.stop()
market=coins.merge(tickers,on="Coin",how="left").dropna(subset=["Price"])
market=market[market["Vol/OI"]>VOL_OI_MIN].sort_values("24H Volume",ascending=False)

st.metric("Coins after Vol/OI > 6",len(market))
mode=st.radio("Mode",["🔥 Live Scanner","📊 Backtest"],horizontal=True)

if mode=="🔥 Live Scanner":
    candidates=market.head(DEEP_SCAN_LIMIT)
    st.info(f"Sirf Vol/OI > 6 wale top {len(candidates)} active coins deep scan honge.")
    results=[]; bar=st.progress(0)
    for i,(_,row) in enumerate(candidates.iterrows()):
        r=deep_analysis(row["Coin"],row)
        if r: results.append(r)
        bar.progress(int((i+1)/len(candidates)*100))
    bar.empty()
    sig=pd.DataFrame(results)
    if sig.empty:
        st.warning("Signal data nahi mila.")
    else:
        st.subheader("🎯 Complete Scanner")
        st.dataframe(sig.sort_values("Score",ascending=False),use_container_width=True,hide_index=True)
        st.subheader("🟢 LONG")
        st.dataframe(sig[["Coin","Price","MTF","5D Regime","Support","Resistance","Liquidity","BOS","CHOCH","FVG","ATR Direction","Volume x","OI Change %","Funding %","Long Score","Long Reason"]].sort_values("Long Score",ascending=False),use_container_width=True,hide_index=True)
        st.subheader("🔴 SHORT")
        st.dataframe(sig[["Coin","Price","MTF","5D Regime","Support","Resistance","Liquidity","BOS","CHOCH","FVG","ATR Direction","Volume x","OI Change %","Funding %","Short Score","Short Reason"]].sort_values("Short Score",ascending=False),use_container_width=True,hide_index=True)
        strong=sig[(sig["Score"]>=8)&(~sig["Signal"].str.contains("CONFLICT",na=False))]
        st.subheader("🔥 STRONG 8+")
        st.dataframe(strong,use_container_width=True,hide_index=True) if not strong.empty else st.info("Abhi 8+ aligned setup nahi mila.")

else:
    st.subheader("📊 Historical Backtest")
    c1,c2,c3=st.columns(3)
    with c1: days=st.slider("Days",2,30,7)
    with c2: rr=st.selectbox("Risk : Reward",[1.0,1.5,2.0,2.5,3.0],index=2)
    with c3: threshold=st.selectbox("Minimum Score",[6,7,8,9,10],index=2)
    limit=st.slider("Coins",1,min(15,len(market)) if len(market) else 1,min(10,len(market)) if len(market) else 1)
    if st.button("▶️ Run Backtest"):
        trades=[]
        bar=st.progress(0)
        for i,(_,row) in enumerate(market.head(limit).iterrows()):
            trades.extend(backtest_symbol(row["Coin"],days,rr,threshold))
            bar.progress(int((i+1)/limit*100))
        bar.empty()
        if not trades:
            st.warning("Historical trades nahi mile.")
        else:
            bt=pd.DataFrame(trades)
            wins=(bt["Result"]=="WIN").sum(); total=len(bt)
            wr=wins/total*100
            total_r=bt["R"].sum()
            gp=bt.loc[bt["R"]>0,"R"].sum(); gl=abs(bt.loc[bt["R"]<0,"R"].sum())
            pf=gp/gl if gl else np.inf
            eq=bt["R"].cumsum(); dd=eq-eq.cummax()
            m1,m2,m3,m4,m5=st.columns(5)
            m1.metric("Trades",total); m2.metric("Win Rate",f"{wr:.2f}%"); m3.metric("Total R",f"{total_r:.2f}")
            m4.metric("Profit Factor","∞" if np.isinf(pf) else f"{pf:.2f}"); m5.metric("Max DD",f"{dd.min():.2f} R")
            st.subheader("Score Performance")
            ss=bt.groupby("Score").agg(Trades=("Result","count"),Wins=("Result",lambda x:(x=="WIN").sum()),Total_R=("R","sum"),Avg_R=("R","mean")).reset_index()
            ss["Win %"]=ss["Wins"]/ss["Trades"]*100
            st.dataframe(ss,use_container_width=True,hide_index=True)
            st.subheader("Side Performance")
            side=bt.groupby("Side").agg(Trades=("Result","count"),Wins=("Result",lambda x:(x=="WIN").sum()),Total_R=("R","sum"),Avg_R=("R","mean")).reset_index()
            side["Win %"]=side["Wins"]/side["Trades"]*100
            st.dataframe(side,use_container_width=True,hide_index=True)
            st.subheader("📈 Equity Curve")
            st.line_chart(pd.DataFrame({"R":eq.values}))
            st.subheader("📋 Trade Log")
            st.dataframe(bt.sort_values("Time",ascending=False),use_container_width=True,hide_index=True)

st.divider()
st.write("""
**Current logic:** Vol/OI > 6 → 5m/15m/1H alignment → 5-day regime → support/resistance → liquidity sweep → BOS/CHOCH → FVG → OI displacement → funding → volume → ATR.

**Important:** 8+ score is a screening threshold, not a guaranteed probability. Backtest should be run across multiple coins and periods; avoid judging the strategy from one short sample.
""")

if st.button("🔄 Refresh Scanner"):
    st.cache_data.clear()
    st.rerun()
