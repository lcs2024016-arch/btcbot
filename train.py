"""Update history, run the walk-forward backtest, train the live model.

    python train.py                  # update data + train the live model (monthly GitHub job)
    python train.py --walkforward    # also re-run the monthly walk-forward backtest (~5 min)
    python train.py --no-update      # skip the data download

History: Binance USD-M BTCUSDT 5m (data/5m/YYYY-MM.parquet), topped up from data.binance.vision.
Live model: LightGBM on every candidate whose trade ended before today; the cutoff is the
65th percentile of its scores on the most recent 20% of London/NY strategy signals (= keep top 35%),
the same rule the walk-forward uses each month.
"""
import argparse
import io
import json
import os
import zipfile

import lightgbm as lgb
import numpy as np
import pandas as pd
import requests
from sklearn.metrics import roc_auc_score

import strategy as S

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "5m")          # one parquet per month: YYYY-MM.parquet
MODEL = os.path.join(HERE, "model")
BV = "https://data.binance.vision/data/futures/um/{kind}/klines/BTCUSDT/5m/BTCUSDT-5m-{tag}.zip"
WF_START = pd.Timestamp("2021-01-01", tz="UTC")


def _zip_frame(url):
    r = requests.get(url, timeout=60)
    if r.status_code != 200:
        return None
    z = zipfile.ZipFile(io.BytesIO(r.content))
    raw = z.read(z.namelist()[0]).decode()
    has_header = raw.startswith("open_time")
    d = pd.read_csv(io.StringIO(raw), header=0 if has_header else None).iloc[:, :6]
    d.columns = ["time", "open", "high", "low", "close", "volume"]
    unit = "us" if d.time.iloc[0] > 1e14 else "ms"
    d.index = pd.to_datetime(d.pop("time"), unit=unit, utc=True)
    return d.astype(float)


def load_history():
    files = sorted(f for f in os.listdir(DATA) if f.endswith(".parquet"))
    return pd.concat([pd.read_parquet(os.path.join(DATA, f)) for f in files])


def save_months(df, months):
    for y, m in months:
        g = df[(df.index.year == y) & (df.index.month == m)]
        g.to_parquet(os.path.join(DATA, f"{y}-{m:02d}.parquet"), compression="zstd")


def update_data():
    df = load_history()
    last = df.index[-1]
    today = pd.Timestamp.now(tz="UTC").normalize()
    parts, day = [], (last + pd.Timedelta("5min")).normalize()
    while day < today:
        month_end = day + pd.offsets.MonthBegin(1)
        if day.day == 1 and month_end <= today:                      # whole month available
            f = _zip_frame(BV.format(kind="monthly", tag=day.strftime("%Y-%m")))
            if f is not None:
                parts.append(f)
                day = month_end
                continue
        f = _zip_frame(BV.format(kind="daily", tag=day.strftime("%Y-%m-%d")))
        if f is None:
            break                                                    # not published yet
        parts.append(f)
        day += pd.Timedelta("1D")
    if parts:
        new = pd.concat([df] + parts)
        new = new[~new.index.duplicated(keep="last")].sort_index()
        added = new.index.difference(df.index)
        save_months(new, sorted(set(zip(added.year, added.month))))
        print(f"data: {len(new) - len(df)} new bars, now {new.index[0]} -> {new.index[-1]}")
        return new
    print(f"data: up to date ({last})")
    return df


def dataset(df):
    t = S.label(df, S.candidates(df))
    F = S.features(df, t)
    return t.reset_index(drop=True), F


def fit_with_cut(t, F, tr):
    """Fit on rows `tr`; cutoff = (1-KEEP) quantile of scores on the last 20% of strategy signals."""
    split = t.time[tr].quantile(0.8)
    iva = tr & (t.time >= split).to_numpy() & t.strategy.to_numpy()
    mdl = lgb.LGBMClassifier(**S.LGB).fit(F[tr], t.y[tr])
    cut = float(np.quantile(mdl.predict_proba(F[iva])[:, 1], 1 - S.KEEP))
    return mdl, cut


def walk_forward(t, F):
    t = t.copy()
    t["p"], t["cut"] = np.nan, np.nan
    emb = pd.Timedelta("1D")
    for m0 in pd.date_range(WF_START, t.time.max(), freq="MS"):
        te = ((t.time >= m0) & (t.time < m0 + pd.offsets.MonthBegin(1))).to_numpy()
        tr = (t.label_end < m0 - emb).to_numpy()
        if te.sum() == 0:
            continue
        mdl, cut = fit_with_cut(t, F, tr)
        t.loc[te, "p"] = mdl.predict_proba(F[te])[:, 1]
        t.loc[te, "cut"] = cut
        print(f"  {m0:%Y-%m}: train {tr.sum():6d}  cut {cut:.3f}", end="\r")
    print()
    return t[t.p.notna()].copy()


def summarize(wf):
    strat = wf.strategy.to_numpy()
    ml = S.execute(wf, strat & (wf.p >= wf.cut).to_numpy())
    base = S.execute(wf, strat)
    s = wf[strat]
    recent = pd.Timestamp("2024-01-01", tz="UTC")
    out = dict(auc=roc_auc_score(s.y, s.p), auc_2024=roc_auc_score(s[s.time >= recent].y, s[s.time >= recent].p),
               start=str(wf.time.min())[:10], end=str(wf.time.max())[:10], periods={}, years={})
    for name, x in (("ml", ml), ("base", base)):
        for per, g in (("all", x), ("2024+", x[x.time >= recent])):
            out["periods"][f"{name}_{per}"] = {k: float(v) for k, v in S.stats(g.R, g.time).items()}
        for y, g in x.groupby(x.time.dt.year):
            out["years"].setdefault(str(y), {})[name] = {k: float(v) for k, v in S.stats(g.R, g.time).items()}
    for name, x in (("ml", ml), ("base", base)):
        for per, g in (("all", x), ("2024+", x[x.time >= recent])):
            c = S.inr_curve(g)
            traded = c[c.contracts > 0]
            out["periods"][f"{name}_{per}"].update(
                inr_end=float(c.equity_inr.iloc[-1]), inr_skipped=int((c.contracts == 0).sum()),
                inr_maxDD=float((1 - c.equity_inr / c.equity_inr.cummax().clip(lower=S.CAPITAL_INR)).max() * 100),
                inr_med_risk=float(traded.risk_pct.median()), inr_med_contracts=float(traded.contracts.median()))
    cols = ["time", "session", "prio", "dir", "fvg", "e", "sl", "entry_px", "stop", "tgt", "exit_px", "how", "R",
            "p", "cut"]
    ml.assign(account="ml")[cols + ["account"]].to_csv(os.path.join(MODEL, "backtest_ml_trades.csv"), index=False)
    base.assign(account="base")[cols + ["account"]].to_csv(os.path.join(MODEL, "backtest_base_trades.csv"),
                                                           index=False)
    with open(os.path.join(MODEL, "backtest.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    for k, v in out["periods"].items():
        print(f"{k:12s} n {v['n']:5.0f}  win {v['win']:5.1f}%  avgR {v['avgR']:+.3f}  PF {v['PF']:.2f}  "
              f"maxDD {v['maxDD']:.1f}%  | Rs 1500 -> Rs {v['inr_end']:,.0f} (DD {v['inr_maxDD']:.0f}%, "
              f"{v['inr_skipped']} skipped, median {v['inr_med_contracts']:.0f} contract(s) = "
              f"{v['inr_med_risk']:.2f}% risk)")
    print(f"AUC {out['auc']:.3f} (2024+ {out['auc_2024']:.3f})")


def train_live(df, t, F):
    tr = (t.label_end < pd.Timestamp.now(tz="UTC").normalize()).to_numpy()
    mdl, cut = fit_with_cut(t, F, tr)
    os.makedirs(MODEL, exist_ok=True)
    mdl.booster_.save_model(os.path.join(MODEL, "model.txt"))
    meta = dict(cut=cut, keep=S.KEEP, trained_at=str(pd.Timestamp.now(tz="UTC"))[:19],
                data_end=str(df.index[-1]), n_train=int(tr.sum()), features=list(F.columns))
    with open(os.path.join(MODEL, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    print(f"model: {tr.sum()} training signals, cut {cut:.4f}, data to {df.index[-1]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--walkforward", action="store_true")
    ap.add_argument("--no-update", action="store_true")
    a = ap.parse_args()
    df = load_history() if a.no_update else update_data()
    t, F = dataset(df)
    if a.walkforward:
        summarize(walk_forward(t, F))
    train_live(df, t, F)


if __name__ == "__main__":
    main()
