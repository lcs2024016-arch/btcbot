"""9/15 EMA Scalper v3 (uzairblaoch Pine) + ML filter: signals, candidates, trade simulation, features.

Shared by train.py (history), bot.py (live) and the parity tests, so live and backtest use one code path.

Rules (Pine defaults; see pine/ema915_v3_london_ny.pine):
  trend      EMA9 vs EMA15 + EMA9 slope over 3 bars + gap (point thresholds, as in the Pine)
  normal     EMA9 inside the candle body + strong / pin / engulfing candle, not a displacement bar
  FVG        displacement candle (body > 2x avg, touching an EMA, in trend) -> next bar retraces
             30-70% of it without breaking its extreme ("Conservative" mode)
  bias       1h EMA9/15 + slope + close (visible once the 1h bar has closed)
  sessions   London 08:00-12:00 UTC (13:30-17:30 IST), New York 13:00-17:00 UTC (18:30-22:30 IST),
             Monday-Friday only
  orders     market entry at the next 5m open; SL = signal candle extreme (+/-1 pt), TP = 2R (-/+1 pt)
  exit       forced market exit at the open 25 min after entry, so every trade is < 30 min (fee-free)
  ML         LightGBM P(win) on OHLCV features; trade only if p >= cut (top 35% of recent signals)
"""
import numpy as np
import pandas as pd
from numba import njit

P = dict(fast=9, slow=15, rr=2.0, body_min=0.6, opp_wick=0.3, angle_lb=3, min_slope=0.3, min_gap=0.5, buf=1.0,
         big_body=2.0, rmin=0.3, rmax=0.7, htf_fast=9, htf_slow=15, htf_slope_len=5, htf_min_slope=0.5)
CAP = 4                     # exit at the open of entry bar + 5 (25 min)
TF = "5min"
SESSIONS = {"London": (8, 12), "New York": (13, 17)}     # UTC hours of the signal bar
WEEKDAYS = (0, 1, 2, 3, 4)                                # Monday-Friday only (weekend volume is thin)
IST = "Asia/Kolkata"
# Delta Exchange India BTCUSD perpetual (api /v2/products/BTCUSD) and the paper account
CONTRACT_BTC = 0.001        # 1 contract = 0.001 BTC, minimum order 1 contract
TICK = 0.5                  # price step
USDINR = 85.0               # Delta India's fixed conversion rate
CAPITAL_INR = 1500.0
RISK_PCT = 1.0              # target risk per trade
MAX_RISK_PCT = 2.0          # if 1 contract risks more than this, skip the trade
# Margin (from leverage_report.html): leverage does not change P&L at fixed 1% risk, only the margin locked.
# 50x cross is the lowest setting that never caps the lot count; cross margin backs the position with the
# whole balance, so liquidation stays far beyond the stop. Isolated 200x liquidated 69 of 379 trades.
LEVERAGE = 50
MARGIN_MODE = "cross"
MAINT_MARGIN = 0.0025       # Delta BTCUSD maintenance margin (0.25% of notional)
LIQ_SAFETY = 2.0            # liquidation must be >= 2x farther than the stop, else fewer lots
LGB = dict(objective="binary", learning_rate=0.03, n_estimators=300, num_leaves=15, min_child_samples=200,
           subsample=0.7, subsample_freq=1, colsample_bytree=0.7, reg_lambda=10.0, verbose=-1)
KEEP = 0.35                 # trade the model's top 35% of signals
PRIO = {0: "normal long", 1: "normal short", 2: "FVG long", 3: "FVG short"}
HOW = {0: "stop", 1: "target", 2: "stop (gap)", 3: "target (gap)", 4: "time exit"}


# ---------------------------------------------------------------- indicators

def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi(c, n=14):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def atr(df, n=14):
    pc = df.close.shift(1)
    tr = pd.concat([df.high - df.low, (df.high - pc).abs(), (df.low - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def adx(df, n=14):
    h, l = df.high, df.low
    up, dn = h.diff(), -l.diff()
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    ndm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    a = atr(df, n)
    pdi = 100 * pdm.ewm(alpha=1 / n, adjust=False).mean() / a
    ndi = 100 * ndm.ewm(alpha=1 / n, adjust=False).mean() / a
    dx = 100 * (pdi - ndi).abs() / (pdi + ndi)
    return dx.ewm(alpha=1 / n, adjust=False).mean()


def align(s, src_tf, dst_index, dst_tf=TF):
    """Higher-TF series (indexed by bar open) onto dst bars; visible once the src bar has closed."""
    s = s.copy()
    s.index = s.index + pd.Timedelta(src_tf)
    return s.reindex(dst_index + pd.Timedelta(dst_tf), method="ffill").to_numpy()


def resample(df, tf):
    return df.resample(tf, label="left", closed="left").agg(
        dict(open="first", high="max", low="min", close="last", volume="sum")).dropna()


# ---------------------------------------------------------------- signals

def signals(df, p=P):
    """Raw Pine conditions (no session / bias filter) + the 1h bias for each direction."""
    o, h, l, c = df.open, df.high, df.low, df.close
    e9, e15 = ema(c, p["fast"]), ema(c, p["slow"])
    body, rng = (c - o).abs(), h - l
    avg_body = body.rolling(20).mean()
    cmin, cmax = np.minimum(o, c), np.maximum(o, c)
    strong_bull = (rng > 0) & (c > o) & (body / rng >= p["body_min"]) & ((h - cmax) <= rng * p["opp_wick"])
    strong_bear = (rng > 0) & (c < o) & (body / rng >= p["body_min"]) & ((cmin - l) <= rng * p["opp_wick"])
    slope = e9 - e9.shift(p["angle_lb"])
    gap = (e9 - e15).abs()
    up = (e9 > e15) & (slope > p["min_slope"]) & (gap > p["min_gap"])
    dn = (e9 < e15) & (slope < -p["min_slope"]) & (gap > p["min_gap"])
    touch_any = ((cmin <= e9) & (cmax >= e9)) | ((cmin <= e15) & (cmax >= e15))
    bull_eng = (c > o) & (c > h.shift(1)) & (o < l.shift(1)) & touch_any
    bear_eng = (c < o) & (c < l.shift(1)) & (o > h.shift(1)) & touch_any
    bull_pin = (c > o) & (rng > 0) & ((o - l) > (h - c) * 2) & ((o - l) / rng >= 0.6) & (l <= e9) & (e9 <= cmax)
    bear_pin = (c < o) & (rng > 0) & ((h - c) > (o - l) * 2) & ((h - c) / rng >= 0.6) & (cmin <= e9) & (e9 <= h)
    disp_bull = (body > avg_body * p["big_body"]) & strong_bull & touch_any & up
    disp_bear = (body > avg_body * p["big_body"]) & strong_bear & touch_any & dn
    ema_in_body = (cmin <= e9) & (cmax >= e9)
    normal_buy = up & ema_in_body & (strong_bull | bull_pin | bull_eng) & ~disp_bull
    normal_sell = dn & ema_in_body & (strong_bear | bear_pin | bear_eng) & ~disp_bear

    hc = c.resample("60min", label="left", closed="left").last().dropna()
    hf, hs = ema(hc, p["htf_fast"]), ema(hc, p["htf_slow"])
    hslope = hf - ema(hc.shift(p["htf_slope_len"]), p["htf_fast"])
    hb = ((hf > hs) & (hslope > p["htf_min_slope"]) & (hc > hf)).astype(float)
    hr = ((hf < hs) & (hslope < -p["htf_min_slope"]) & (hc < hf)).astype(float)
    b = lambda s: s.fillna(False).to_numpy()
    return dict(nbuy=b(normal_buy), nsell=b(normal_sell), dbull=b(disp_bull), dbear=b(disp_bear),
                htf_bull=np.nan_to_num(align(hb, "60min", df.index)) > 0,
                htf_bear=np.nan_to_num(align(hr, "60min", df.index)) > 0)


def session_of(hours):
    out = np.full(len(hours), "", dtype=object)
    for name, (lo, hi) in SESSIONS.items():
        out[(hours >= lo) & (hours < hi)] = name
    return out


def candidates(df, p=P):
    """Every signal the rules can fire assuming flat (engine priority order within a bar)."""
    s = signals(df, p)
    O, H, L, C = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    rows = []
    for prio, (mask, d) in enumerate(((s["nbuy"], 1), (s["nsell"], -1))):
        for i in np.flatnonzero(mask):
            rows.append((i, prio, d, 0, C[i], L[i] if d == 1 else H[i]))
    dbull, dbear = s["dbull"], s["dbear"]
    for i in np.flatnonzero(dbull[:-1] & ~dbull[1:]) + 1:          # displacement at i-1, not overwritten at i
        h0, l0 = H[i - 1], L[i - 1]
        r = h0 - l0
        zl, zh = h0 - r * p["rmax"], h0 - r * p["rmin"]
        if L[i] <= zh and H[i] >= zl and L[i] > l0:
            rows.append((i, 2, 1, 1, max(zl, L[i]), l0))
    for i in np.flatnonzero(dbear[:-1] & ~dbear[1:]) + 1:
        h0, l0 = H[i - 1], L[i - 1]
        r = h0 - l0
        zl, zh = l0 + r * p["rmin"], l0 + r * p["rmax"]
        if H[i] >= zl and L[i] <= zh and H[i] < h0:
            rows.append((i, 3, -1, 1, min(zh, H[i]), h0))
    t = pd.DataFrame(rows, columns=["bar", "prio", "dir", "fvg", "e", "sl"])
    t = t[t.e != t.sl].sort_values(["bar", "prio"]).reset_index(drop=True)
    b, d = t.bar.to_numpy(), t.dir.to_numpy()
    r = d * (t.e - t.sl)
    t["stop"] = t.sl + d * p["buf"]
    t["tgt"] = t.e + d * r * p["rr"] - d * p["buf"]
    t["time"] = df.index[b]
    t["session"] = session_of(df.index.hour[b])
    hb, hr = s["htf_bull"][b], s["htf_bear"][b]
    t["htf_ok"] = np.where(d == 1, hb, hr)
    t["htf_s"] = np.where(d == 1, hb.astype(int) - hr.astype(int), hr.astype(int) - hb.astype(int))
    t["strategy"] = (t.session != "") & t.htf_ok & np.isin(df.index.dayofweek[b], WEEKDAYS)
    return t


# ---------------------------------------------------------------- trade simulation (candle model)

@njit(cache=True)
def _sim(O, H, L, bars, dirs, risks, stops, tgts, cap, slip):
    k, n = len(bars), len(O)
    out = np.full((k, 4), np.nan)          # exit_bar, R, how, exit_px
    for j in range(k):
        i0 = bars[j] + 1
        d, stop, tgt = dirs[j], stops[j], tgts[j]
        if i0 + cap + 1 >= n:
            continue
        fill = O[i0] * (1 + d * slip)
        px, how, xb = -1.0, -1, -1
        for i in range(i0, i0 + cap + 1):
            if d == 1:
                if O[i] <= stop:
                    px, how = O[i] * (1 - slip), 2
                elif O[i] >= tgt:
                    px, how = O[i], 3
                else:
                    hs, ht = L[i] <= stop, H[i] >= tgt
                    if hs and ht:                               # TradingView: nearer extreme first
                        if (H[i] - O[i]) < (O[i] - L[i]):
                            px, how = tgt, 1
                        else:
                            px, how = stop * (1 - slip), 0
                    elif hs:
                        px, how = stop * (1 - slip), 0
                    elif ht:
                        px, how = tgt, 1
            else:
                if O[i] >= stop:
                    px, how = O[i] * (1 + slip), 2
                elif O[i] <= tgt:
                    px, how = O[i], 3
                else:
                    hs, ht = H[i] >= stop, L[i] <= tgt
                    if hs and ht:
                        if (O[i] - L[i]) < (H[i] - O[i]):
                            px, how = tgt, 1
                        else:
                            px, how = stop * (1 + slip), 0
                    elif hs:
                        px, how = stop * (1 + slip), 0
                    elif ht:
                        px, how = tgt, 1
            if px > 0:
                xb = i
                break
        if px < 0:
            xb = i0 + cap + 1
            px, how = O[xb] * (1 - d * slip), 4
        out[j, 0], out[j, 1], out[j, 2], out[j, 3] = xb, d * (px - fill) / risks[j], how, px
    return out


def label(df, t, slip=0.0):
    """Outcome of each candidate traded on its own. R is per sizing distance |e - sl| (1% risk)."""
    O, H, L = (df[k].to_numpy() for k in ("open", "high", "low"))
    out = _sim(O, H, L, t.bar.to_numpy().astype(np.int64), t.dir.to_numpy().astype(np.int64),
               (t.e - t.sl).abs().to_numpy(), t.stop.to_numpy(), t.tgt.to_numpy(), CAP, slip)
    ok = ~np.isnan(out[:, 0])
    t = t[ok].copy()
    xb = out[ok, 0].astype(int)
    t["R"], t["exit_bar"], t["how"], t["exit_px"] = out[ok, 1], xb, out[ok, 2].astype(int), out[ok, 3]
    t["entry_px"] = O[t.bar.to_numpy() + 1] * (1 + t.dir.to_numpy() * slip)
    t["label_end"] = df.index[xb] + pd.Timedelta(TF)
    t["y"] = (t.R > 0).astype(int)
    return t


def execute(t, take):
    """One position at a time; flat again at the close of the exit bar."""
    tt = t[take]
    keep, busy = [], -1
    for k, (b, xb) in enumerate(zip(tt.bar.to_numpy(), tt.exit_bar.to_numpy())):
        if b >= busy:
            keep.append(k)
            busy = xb
    return tt.iloc[keep]


# ---------------------------------------------------------------- features (OHLCV only)

def market_features(df):
    """Direction-dependent columns end in _s and are multiplied by the trade direction later.
    Zero-volume bars (exchange maintenance) are skipped, then the result is mapped back onto df."""
    full = df.index
    df = df[df.volume > 0]
    c, h, l, v = df.close, df.high, df.low, df.volume
    A = atr(df, 14)
    r1 = np.log(c).diff()
    e20, e50, e200 = ema(c, 20), ema(c, 50), ema(c, 200)
    bb_mid, bb_sd = c.rolling(20).mean(), c.rolling(20).std()
    vwap = (c * v).rolling(48).sum() / v.rolling(48).sum()
    up, dn = h.diff(), -l.diff()
    pdi = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index).ewm(alpha=1 / 14, adjust=False).mean() / A
    ndi = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index).ewm(alpha=1 / 14, adjust=False).mean() / A
    rng = (h - l).replace(0, np.nan)
    X = pd.DataFrame(index=df.index)
    for k in (1, 3, 6, 12, 36, 144):
        X[f"ret{k}_s"] = np.log(c / c.shift(k)) / (A / c)
    X["dist_e20_s"], X["dist_e50_s"], X["dist_e200_s"] = (c - e20) / A, (c - e50) / A, (c - e200) / A
    X["e50_e200_s"] = (e50 - e200) / A
    X["bbpos_s"] = (c - bb_mid) / (2 * bb_sd)
    X["vwap_dev_s"] = (c - vwap) / A
    X["di_diff_s"] = 100 * (pdi - ndi)
    X["rsi_c_s"] = rsi(c, 14) - 50
    X["close_loc_s"] = (c - l) / rng - 0.5
    X["body_s"] = (c - df.open) / rng
    for k in (12, 36, 144):
        X[f"rvol{k}"] = r1.rolling(k).std() * 1e4
    X["atr_pct"] = A / c * 1e4
    X["atr_ratio"] = A / atr(df, 100)
    X["adx"] = adx(df, 14)
    X["bb_width"] = 4 * bb_sd / c * 1e4
    X["vol_z20"] = (v - v.rolling(20).mean()) / v.rolling(20).std()
    X["vol_z100"] = (v - v.rolling(100).mean()) / v.rolling(100).std()
    X["range_atr"] = rng / A
    X["hour"] = df.index.hour
    X["dow"] = df.index.dayofweek
    nxt = (df.index + pd.Timedelta(TF)).hour
    X["sess_13_21"] = (nxt >= 13) & (nxt < 21)
    for tf in ("1h", "4h"):
        g = resample(df, tf)
        X[f"trend_{tf}_s"] = align((g.close - ema(g.close, 50)) / atr(g, 14), tf, df.index)
        X[f"ret_{tf}x6_s"] = align(np.log(g.close / g.close.shift(6)) / (atr(g, 14) / g.close), tf, df.index)
    X["rule_adx22"] = X.adx > 22
    X["rule_vol12"] = v > 1.2 * v.shift(1).rolling(20).mean()
    X["rule_range2atr"] = rng <= 2 * A
    return X.reindex(full), A.reindex(full).to_numpy()


def features(df, t, X=None, A=None):
    if X is None:
        X, A = market_features(df)
    b = t.bar.to_numpy()
    F = X.iloc[b].reset_index(drop=True).astype(float)
    d = t.dir.to_numpy()
    signed = [c for c in F.columns if c.endswith("_s")]
    F[signed] = F[signed].mul(d, axis=0)
    c, o, h, l = (df[k].to_numpy() for k in ("close", "open", "high", "low"))
    e9, e15 = ema(df.close, 9).to_numpy(), ema(df.close, 15).to_numpy()
    body = np.abs(c - o)
    avg_body = pd.Series(body).rolling(20).mean().to_numpy()
    rng = np.where(h - l > 0, h - l, np.nan)
    F["dir"] = d
    F["fvg"] = t.fvg.to_numpy()
    F["htf_s"] = t.htf_s.to_numpy()
    F["ln_ny"] = (t.session != "").to_numpy().astype(float)
    F["stop_atr"] = (t.e - t.sl).abs().to_numpy() / A[b]
    F["stop_bp"] = (t.e - t.sl).abs().to_numpy() / t.e.to_numpy() * 1e4
    F["ema_gap_s"] = d * (e9[b] - e15[b]) / A[b]
    F["ema_slope_s"] = d * (e9[b] - e9[b - 3]) / A[b]
    F["body_avg"] = body[b] / avg_body[b]
    F["body_rng"] = body[b] / rng[b]
    return F.replace([np.inf, -np.inf], np.nan)


# ---------------------------------------------------------------- Delta India sizing (INR paper account)

def round_tick(px):
    return np.round(np.asarray(px, float) / TICK) * TICK


def liq_distance(n, price, equity_inr):
    """Adverse move (fraction of price) that liquidates n lots at `price` (Delta: margin - maintenance)."""
    notional = n * CONTRACT_BTC * price
    margin = equity_inr / USDINR if MARGIN_MODE == "cross" else notional / LEVERAGE
    return (margin - MAINT_MARGIN * notional) / notional


def contracts(equity_inr, e, sl):
    """Whole contracts for RISK_PCT of equity; 1 contract if that rounds to 0 but stays <= MAX_RISK_PCT.
    Capped by the margin available at LEVERAGE and so that liquidation sits >= LIQ_SAFETY x the stop distance."""
    risk_1 = CONTRACT_BTC * abs(e - sl) * USDINR                   # INR lost per contract at the stop
    if risk_1 <= 0 or equity_inr <= 0:
        return 0
    n = int(equity_inr * RISK_PCT / 100 // risk_1)
    if n == 0 and risk_1 <= equity_inr * MAX_RISK_PCT / 100:
        n = 1
    n = min(n, int(equity_inr / USDINR * LEVERAGE // (CONTRACT_BTC * e)))
    stop_dist = abs(e - sl) / e
    while n > 0 and liq_distance(n, e, equity_inr) < LIQ_SAFETY * stop_dist:
        n -= 1
    return n


def pnl_inr(n, d, entry, exit_):
    return n * CONTRACT_BTC * d * (exit_ - entry) * USDINR


def inr_curve(tr, capital=CAPITAL_INR):
    """Replay trades (needs dir, e, sl, entry_px, exit_px) on the INR account with whole contracts."""
    eq, rows = capital, []
    for r in tr.itertuples():
        n = contracts(eq, r.e, r.sl)
        pl = pnl_inr(n, r.dir, r.entry_px, r.exit_px) if n else 0.0
        eq += pl
        rows.append((n, pl, eq, n * CONTRACT_BTC * abs(r.e - r.sl) * USDINR / (eq - pl) * 100 if n else 0.0))
    out = tr.copy()
    out["contracts"], out["pnl_inr"], out["equity_inr"], out["risk_pct"] = zip(*rows) if rows else ([],) * 4
    return out


# ---------------------------------------------------------------- stats

def stats(r, times=None, risk_pct=1.0):
    r = np.asarray(r, float)
    if len(r) == 0:
        return dict(n=0, win=np.nan, avgR=np.nan, PF=np.nan, totR=0.0, ret=0.0, maxDD=0.0, per_mo=np.nan)
    eq = np.concatenate([[1.0], np.cumprod(np.maximum(1 + risk_pct / 100 * r, 0))])
    pos, neg = r[r > 0].sum(), -r[r < 0].sum()
    per_mo = np.nan
    if times is not None and len(times) > 1:
        per_mo = len(r) / max((pd.Timestamp(max(times)) - pd.Timestamp(min(times))).days / 30.4, 1)
    return dict(n=len(r), win=(r > 0).mean() * 100, avgR=r.mean(), PF=pos / neg if neg else np.inf,
                totR=r.sum(), ret=(eq[-1] - 1) * 100, maxDD=(1 - eq / np.maximum.accumulate(eq)).max() * 100,
                per_mo=per_mo)
