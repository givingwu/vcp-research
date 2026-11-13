from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Literal
import json
import numpy as np
import pandas as pd

STRATEGIES = ("A_trend", "B_vcp_early", "C_breakout", "D_pullback")

@dataclass
class Config:
    starting_equity: float = 100_000.0
    risk_fraction: float = 0.0035
    max_position_fraction: float = 0.20
    max_gross_fraction: float = 1.0
    max_positions: int = 10
    commission_bps: float = 2.0
    slippage_bps: float = 5.0
    stop_pct: float = 0.07
    max_hold_bars: int = 80
    min_dollar_volume: float = 2_000_000.0
    train_end: str = "2019-12-31"
    validation_end: str = "2022-12-31"
    test_start: str = "2023-01-01"
    test_end: str = "2026-11-01"


def load_prices(folder: Path) -> dict[str, pd.DataFrame]:
    result = {}
    for path in sorted(folder.glob("*.csv")):
        d = pd.read_csv(path)
        d.columns = [c.lower().strip() for c in d.columns]
        required = {"date", "open", "high", "low", "close", "volume"}
        if not required <= set(d.columns):
            raise ValueError(f"{path}: missing {required - set(d.columns)}")
        d["date"] = pd.to_datetime(d["date"])
        d = d.sort_values("date").drop_duplicates("date").set_index("date")
        for col in ["open", "high", "low", "close", "volume"]:
            d[col] = pd.to_numeric(d[col], errors="coerce")
        d = d.dropna(subset=list(required - {"date"}))
        d = d[(d[["open", "high", "low", "close"]] > 0).all(axis=1) & (d.volume >= 0)]
        if (d.high < d[["open", "close", "low"]].max(axis=1)).any() or (d.low > d[["open", "close"]].min(axis=1)).any():
            raise ValueError(f"{path}: inconsistent OHLC")
        if len(d) >= 250:
            result[path.stem.upper()] = d
    if not result:
        raise ValueError(f"No valid >=250-bar CSV files found in {folder}")
    return result


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    c = d.close
    d["sma50"] = c.rolling(50).mean()
    d["sma200"] = c.rolling(200).mean()
    d["mom126"] = c / c.shift(126) - 1
    d["volume50"] = d.volume.rolling(50).mean()
    d["dollarvol50"] = (c * d.volume).rolling(50).mean()
    d["trend"] = (
        (c > d.sma50) & (d.sma50 > d.sma200)
        & (d.sma50 > d.sma50.shift(20))
        & (d.sma200 > d.sma200.shift(20))
    )
    return d


def vcp_at(d: pd.DataFrame, i: int) -> tuple[bool, float, float]:
    """Only evaluate bars <=i; return (valid, pivot, structural stop).

    Deliberately objective approximation of discretionary VCP, not J Law's exact method.
    Confirm extrema using 3 bars to right, then pair peaks with subsequent troughs.
    """
    if i < 235 or not bool(d.iloc[i].trend):
        return False, np.nan, np.nan
    start = max(0, i-94)
    sub = d.iloc[start:i+1]
    high, low = sub.high.to_numpy(), sub.low.to_numpy()
    peaks, troughs = [], []
    for j in range(3, len(sub)-3):
        if high[j] == np.max(high[j-3:j+4]) and high[j] > np.max(high[j+1:j+4]):
            peaks.append(j)
        if low[j] == np.min(low[j-3:j+4]) and low[j] < np.min(low[j+1:j+4]):
            troughs.append(j)
    swings = []
    for pk in peaks:
        ts = [t for t in troughs if pk < t <= pk+24]
        if ts:
            tr = ts[0]
            swings.append((pk, tr, (high[pk]-low[tr])/high[pk]))
    # Require strictly ordered three contractions; each trough/peak must follow previous trough.
    if len(swings) < 3:
        return False, np.nan, np.nan
    for a in range(len(swings)-3, -1, -1):
        seq = swings[a:a+3]
        if seq[-1][1] < len(sub)-27 and a != len(swings)-3:
            continue
        if not (seq[0][1] < seq[1][0] and seq[1][1] < seq[2][0]):
            continue
        dep = [z[2] for z in seq]
        if not (0.08 <= dep[0] <= 0.40 and 0.025 <= dep[2] < 0.12 and dep[0] > dep[1] > dep[2]):
            continue
        if len(sub)-1-seq[-1][1] > 15:
            continue
        last_tr = seq[-1][1]
        vol_baseline = float(d.volume.iloc[max(0, i-50):i+1].median())
        # Smaller volume at most recent trough compared with base's median.
        if vol_baseline <= 0 or np.median(sub.volume.iloc[max(0,last_tr-3):last_tr+1]) >= vol_baseline:
            continue
        # Pivot is *previously confirmed* resistance; require closing price below pivot
        # to prevent look-ahead and avoid post-breakout shape reuse.
        pivot = max(high[seq[-2][0]], high[seq[-1][0]])
        if d.close.iloc[i] >= pivot:
            continue
        stop = float(low[last_tr]) * 0.995
        return True, float(pivot), stop
    return False, np.nan, np.nan


def build_signals(raw: dict[str, pd.DataFrame], min_dollar_volume: float = 2_000_000.0):
    data = {sym:add_features(d) for sym,d in raw.items()}
    signals = {s:{} for s in STRATEGIES}
    for sym,d in data.items():
        # rank of momentum vs peers will be added daily below.
        signals["A_trend"][sym] = pd.DataFrame(index=d.index, data={"entry":False, "stop":np.nan, "exit":False})
        for s in STRATEGIES[1:]:
            signals[s][sym] = pd.DataFrame(index=d.index, data={"entry":False, "stop":np.nan, "exit":False})
    dates = sorted(set().union(*(set(d.index) for d in data.values())))
    for date in dates:
        eligible = []
        for sym,d in data.items():
            if date in d.index:
                r = d.loc[date]
                if r.trend and pd.notna(r.mom126) and r.dollarvol50 >= min_dollar_volume:
                    eligible.append((sym, float(r.mom126)))
        eligible.sort(key=lambda x:x[1], reverse=True)
        # Cross-sectional momentum top 30%, at least one symbol.
        leaders = {sym for sym,_ in eligible[:max(1, int(np.ceil(len(eligible)*0.3)))]}
        for sym,d in data.items():
            if date in d.index:
                ix = d.index.get_loc(date)
                row = d.iloc[ix]
                for s in STRATEGIES:
                    signals[s][sym].at[date,"exit"] = bool(
                        (not row.trend) or (ix >= 20 and row.close < d.low.iloc[ix-20:ix].min())
                    )
                if sym in leaders and ix >= 220:
                    # A: buy first qualification, with 20-day closing high confirmation.
                    signals["A_trend"][sym].at[date,"entry"] = bool(
                        row.close > d.high.iloc[ix-20:ix].max()
                    )
                    signals["A_trend"][sym].at[date,"stop"] = row.close * 0.93
    # One-pass finite state machine, using only completed bars to decide orders.
    for sym,d in data.items():
        pending = None
        last_active = False
        for i in range(235,len(d)):
            date = d.index[i]
            # No cross sectional lookahead; true leadership recomputed by date across all available symbols.
            candidates = sorted([(s, float(x.loc[date,"mom126"])) for s,x in data.items()
                                 if date in x.index and bool(x.loc[date,"trend"])
                                 and x.loc[date,"dollarvol50"] >= min_dollar_volume],key=lambda x:x[1],reverse=True)
            leader_set = {s for s,_ in candidates[:max(1,int(np.ceil(len(candidates)*0.3)))]}
            leader_ok = sym in leader_set
            if not leader_ok:
                pending = None
                last_active = False
                continue
            # Pattern recognition at yesterday's close; current bar can trigger breakout.
            found,pivot,struct_stop = vcp_at(d,i-1)
            if found and (not last_active or pending is None):
                pending = {"pivot":pivot,"stop":struct_stop,"born":i-1,"breakout":None,
                           "breakout_vol":None}
            last_active = found
            if pending is None:
                continue
            if i-pending["born"] > 22:
                pending = None
                continue
            r=d.iloc[i]
            if pending["breakout"] is None:
                if r.close > pending["pivot"] and r.volume >= d.volume.iloc[i-50:i].mean()*1.3:
                    if i-pending["born"] <= 12:
                        signals["C_breakout"][sym].at[date,"entry"] = True
                        signals["C_breakout"][sym].at[date,"stop"] = pending["stop"]
                    pending["breakout"] = i
                    pending["breakout_vol"] = float(r.volume)
                elif r.low < pending["stop"]:
                    pending = None
                elif i-pending["born"] <= 10 and not signals["B_vcp_early"][sym].entry.iloc[max(0,i-10):i].any():
                    signals["B_vcp_early"][sym].at[date,"entry"] = True
                    signals["B_vcp_early"][sym].at[date,"stop"] = pending["stop"]
            else:
                # First close near pivot; volume below breakout; no invalidation of structural stop.
                if (i-pending["breakout"] <= 12 and i>pending["breakout"]
                    and abs(r.close/pending["pivot"]-1) <= 0.025
                    and r.low > pending["stop"]
                    and r.volume < pending["breakout_vol"]*0.8):
                    signals["D_pullback"][sym].at[date,"entry"] = True
                    signals["D_pullback"][sym].at[date,"stop"] = pending["stop"]
                    pending = None
                elif r.low < pending["stop"]:
                    pending = None
    return data, signals


def backtest(data:dict[str,pd.DataFrame], signals:dict[str,pd.DataFrame], cfg:Config, strategy:str, start:str,end:str):
    """Daily next-open signals, gap-aware stops, cash-based long-only portfolio.
    Mark from first day starting equity, fills on next open after signal close.
    """
    dates=sorted({date for d in data.values() for date in d.index if pd.Timestamp(start) <= date <= pd.Timestamp(end)})
    cash=cfg.starting_equity
    holdings={}
    pending={}
    trades=[]; curve=[]
    last_close={}
    for date in dates:
        for sym in data:
            if date not in data[sym].index: continue
            row=data[sym].loc[date]
            # Exit at next open for yesterday's trend-exit, before intraday stop checks.
            if sym in holdings and pending.get(sym,{}).get("exit"):
                h=holdings.pop(sym)
                px=float(row.open)*(1-cfg.slippage_bps/10000)
                cash += h["qty"]*px*(1-cfg.commission_bps/10000)
                trades.append({"symbol":sym,"entry_date":h["date"],"exit_date":date,"entry_price":h["entry"],"exit_price":px,
                               "qty":h["qty"],"pnl":h["qty"]*(px-h["entry"])-h["fees"]-h["qty"]*px*cfg.commission_bps/10000,"reason":"signal"})
            if sym in holdings and row.low <= holdings[sym]["stop"]:
                h=holdings.pop(sym)
                px=min(float(row.open),h["stop"])*(1-cfg.slippage_bps/10000)
                cash += h["qty"]*px*(1-cfg.commission_bps/10000)
                trades.append({"symbol":sym,"entry_date":h["date"],"exit_date":date,"entry_price":h["entry"],"exit_price":px,
                               "qty":h["qty"],"pnl":h["qty"]*(px-h["entry"])-h["fees"]-h["qty"]*px*cfg.commission_bps/10000,"reason":"stop"})
            pending.pop(sym,None) if pending.get(sym,{}).get("exit") else None
            last_close[sym] = float(row.close)
        # All entry orders generated yesterday are eligible at today's open.
        requests=[]
        for sym,order in list(pending.items()):
            if "entry" in order and sym not in holdings and date in data[sym].index:
                requests.append((sym,order))
        # Deterministic alphabetical order. Portfolio cap, risk cap and cash cap enforced.
        equity_open = cash + sum(h["qty"]*float(data[s].loc[date].open if date in data[s].index else last_close.get(s,h["entry"])) for s,h in holdings.items())
        for sym,order in sorted(requests):
            if len(holdings)>=cfg.max_positions:break
            row=data[sym].loc[date]
            entry=float(row.open)*(1+cfg.slippage_bps/10000)
            stop=min(float(order["entry"]),entry*0.99) # ensure stop below entry
            # Gap up can increase stop distance, never move stop above fill.
            risk_per_share=entry-stop
            gross=sum(h["qty"]*float(data[s].loc[date].open if date in data[s].index else last_close.get(s,h["entry"])) for s,h in holdings.items())
            budget=min(cash/(1+cfg.commission_bps/10000),
                       equity_open*cfg.max_position_fraction,
                       max(0,equity_open*cfg.max_gross_fraction-gross))
            qty=int(min(budget/entry, equity_open*cfg.risk_fraction/risk_per_share)) if risk_per_share>0 else 0
            if qty>0:
                fees=qty*entry*cfg.commission_bps/10000
                cash-=qty*entry+fees
                holdings[sym]={"qty":qty,"entry":entry,"stop":stop,"date":date,"fees":fees}
            pending.pop(sym,None)
        # Fill an entry-day stop conservatively if daily low crosses the level.
        for sym in list(holdings):
            if holdings[sym]["date"] != date or date not in data[sym].index:
                continue
            h=holdings[sym]; bar=data[sym].loc[date]
            if bar.low <= h["stop"]:
                px=h["stop"]*(1-cfg.slippage_bps/10000)
                cash += h["qty"]*px*(1-cfg.commission_bps/10000)
                trades.append({"symbol":sym,"entry_date":h["date"],"exit_date":date,"entry_price":h["entry"],"exit_price":px,
                               "qty":h["qty"],"pnl":h["qty"]*(px-h["entry"])-h["fees"]-h["qty"]*px*cfg.commission_bps/10000,"reason":"entry_day_stop"})
                holdings.pop(sym)
        total=cash+sum(h["qty"]*last_close.get(s,h["entry"]) for s,h in holdings.items())
        curve.append({"date":date,"equity":total,"cash":cash,"positions":len(holdings)})
        for sym,d in data.items():
            if date not in d.index:continue
            sig=signals[sym].loc[date]
            if sym in holdings:
                h=holdings[sym]
                if bool(sig["exit"]) or d.index.get_loc(date)-d.index.get_loc(h["date"])>=cfg.max_hold_bars:
                    pending[sym]={"exit":True}
            elif bool(sig["entry"]) and sym not in pending:
                pending[sym]={"entry":float(sig["stop"])}
    # Mark-to-market positions remain open; do not pretend they realized PnL.
    c=pd.DataFrame(curve).set_index("date") if curve else pd.DataFrame(columns=["equity","cash","positions"])
    t=pd.DataFrame(trades)
    return c,t


def metrics(curve:pd.DataFrame,trades:pd.DataFrame, starting_equity:float)->dict:
    if curve.empty:return {}
    eq=curve.equity
    rets=eq.pct_change().fillna(eq.iloc[0]/starting_equity-1)
    years=max(len(eq)/252,1/252)
    peak=eq.cummax().clip(lower=starting_equity)
    dd=eq/peak-1
    stdev=rets.std(ddof=0)
    return {"start_equity":round(float(starting_equity),2),"end_equity":round(float(eq.iloc[-1]),2),
            "total_return":round(float(eq.iloc[-1]/starting_equity-1),4),
            "cagr":round(float((eq.iloc[-1]/starting_equity)**(1/years)-1),4),
            "max_drawdown":round(float(dd.min()),4),
            "sharpe_rf0":round(float(rets.mean()/stdev*np.sqrt(252)),3) if stdev>0 else None,
            "trades_closed":int(len(trades)),
            "win_rate":round(float((trades.pnl>0).mean()),4) if len(trades) else None,
            "avg_pnl_closed":round(float(trades.pnl.mean()),2) if len(trades) else None,
            "avg_positions":round(float(curve.positions.mean()),2)}


def run(folder:Path, out:Path,cfg:Config)->dict:
    raw=load_prices(folder)
    data,signals=build_signals(raw,cfg.min_dollar_volume)
    out.mkdir(parents=True,exist_ok=True)
    result={"metadata":{"symbols":list(data),"assumptions":asdict(cfg),
                        "data_warning":"CSV bars are not independently PIT verified; no delisted-universe or dividend/borrow model.",
                        "status":"RESEARCH_ONLY_NOT_VALIDATED"}}
    periods={"train":("2000-01-01",cfg.train_end),
             "validation":(str(pd.Timestamp(cfg.train_end)+pd.Timedelta(days=1)),cfg.validation_end),
             "test":(cfg.test_start,cfg.test_end)}
    for p,(start,end) in periods.items():
        result[p]={}
        for strat in STRATEGIES:
            c,t=backtest(data,signals[strat],cfg,strat,start,end)
            result[p][strat]=metrics(c,t,cfg.starting_equity)
            if not c.empty:c.to_csv(out/f"{p}_{strat}_equity.csv")
            t.to_csv(out/f"{p}_{strat}_trades.csv",index=False)
    (out/"summary.json").write_text(json.dumps(result,indent=2,ensure_ascii=False,default=str))
    return result