"""HTML backtest of the forward-test bot's exact settings: 9/15 EMA + ML, 1% risk, 50x cross margin,
Delta India lots (0.001 BTC), Rs 1,500 start, Mon-Fri London + New York (IST).

Trades: model/backtest_*_trades.csv (monthly walk-forward, zero slippage, fee-free).
Sizing:  strategy.py contracts() - the same function the bot uses.
The page recomputes the monthly SIP in JavaScript; tests/test_sip.js checks it against sip_sim() here.

    python research/backtest50_report.py   # -> docs/backtest_50x.html
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import strategy as S  # noqa: E402

PERIODS = {"2021": "2021-01-01", "2024": "2024-01-01"}


def account(tr):
    """Replay trades on the INR account with the bot's sizing; add margin and liquidation distance."""
    c = S.inr_curve(tr)
    eq_before = c.equity_inr - c.pnl_inr
    c["margin_inr"] = c.contracts * S.CONTRACT_BTC * c.e / S.LEVERAGE * S.USDINR
    c["liq_pct"] = [S.liq_distance(n, e, q) * 100 if n else np.nan for n, e, q in zip(c.contracts, c.e, eq_before)]
    c["stop_pct"] = (c.e - c.sl).abs() / c.e * 100
    return c[c.contracts > 0].copy(), int((c.contracts == 0).sum())


def summary(c, skipped):
    eq = np.concatenate([[S.CAPITAL_INR], c.equity_inr])
    dd = 1 - eq / np.maximum.accumulate(eq)
    months = max((c.time.max() - c.time.min()).days / 30.44, 1)
    wins, losses = c.pnl_inr[c.pnl_inr > 0], c.pnl_inr[c.pnl_inr <= 0]
    streak = run = 0
    for v in c.pnl_inr:
        run = run + 1 if v <= 0 else 0
        streak = max(streak, run)
    return dict(n=len(c), per_mo=len(c) / months, win=float((c.pnl_inr > 0).mean() * 100),
                avgR=float(c.R.mean()), PF=float(wins.sum() / -losses.sum()) if len(losses) else None,
                final=float(eq[-1]), ret=float(eq[-1] / S.CAPITAL_INR * 100 - 100),
                cagr=float(((eq[-1] / S.CAPITAL_INR) ** (12 / months) - 1) * 100), maxDD=float(dd.max() * 100),
                avg_win=float(wins.mean()) if len(wins) else 0, avg_loss=float(losses.mean()) if len(losses) else 0,
                streak=streak, skipped=skipped, med_lots=float(c.contracts.median()), max_lots=int(c.contracts.max()),
                med_margin=float(c.margin_inr.median()), max_margin=float(c.margin_inr.max()),
                med_liq=float(c.liq_pct.median()), min_liq=float(c.liq_pct.min()), med_stop=float(c.stop_pct.median()),
                med_risk=float(c.risk_pct.median()), best=float(c.pnl_inr.max()), worst=float(c.pnl_inr.min()))


def breakdown(c, key):
    out = []
    for k, g in c.groupby(key, sort=True):
        start = g.equity_inr.iloc[0] - g.pnl_inr.iloc[0]
        eq = np.concatenate([[start], g.equity_inr])
        out.append(dict(k=str(k), n=len(g), win=float((g.pnl_inr > 0).mean() * 100), avgR=float(g.R.mean()),
                        pnl=float(g.pnl_inr.sum()), ret=float(g.pnl_inr.sum() / start * 100),
                        dd=float((1 - eq / np.maximum.accumulate(eq)).max() * 100)))
    return out


def sip_sim(tr, start, sip):
    """Reference SIP simulation (the page recomputes this in JS; test_sip.js checks they agree).
    Rs 1,500 at `start`, + `sip` on the 1st of every following month (IST), bot sizing on the running balance."""
    eq, inv, idx, peak, dd = S.CAPITAL_INR, S.CAPITAL_INR, 1.0, 1.0, 0.0
    cur = pd.Timestamp(start, tz=S.IST).to_period("M")
    flows = [(pd.Timestamp(start, tz=S.IST), -S.CAPITAL_INR)]
    for r in tr.itertuples():
        m = r.time.tz_convert(S.IST).to_period("M")
        while cur < m:
            cur += 1
            eq += sip
            inv += sip
            flows.append((cur.to_timestamp().tz_localize(S.IST), -sip))
        n = S.contracts(eq, r.e, r.sl)
        pl = S.pnl_inr(n, r.dir, r.entry_px, r.exit_px) if n else 0.0
        idx *= 1 + pl / eq
        peak = max(peak, idx)
        dd = max(dd, 1 - idx / peak)
        eq += pl
    return dict(final=eq, invested=inv, twr_dd=dd * 100, flows=flows)


def xirr(flows, final, end):
    cf = flows + [(end, final)]
    t0 = cf[0][0]
    f = lambda r: sum(v / (1 + r) ** ((t - t0).days / 365.0) for t, v in cf)
    lo, hi = -0.99, 10.0
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
    return mid * 100


_HIST = {}


def _history():
    if not _HIST:
        import train
        df = train.load_history()
        _HIST.update(df=df, cands=S.candidates(df))
    return _HIST["df"], _HIST["cands"]


def with_costs(m, taker=0.0, maker=0.0):
    """INR replay (bot sizing) of trades m, minus exchange fees on every fill."""
    eq, rows = S.CAPITAL_INR, []
    for r in m.itertuples():
        n = S.contracts(eq, r.e, r.sl)
        fee = n * S.CONTRACT_BTC * (taker * r.entry_px + (maker if r.how in (1, 3) else taker) * r.exit_px) * S.USDINR
        pl = (S.pnl_inr(n, r.dir, r.entry_px, r.exit_px) - fee) if n else 0.0
        eq += pl
        rows.append(eq)
    eqs = np.concatenate([[S.CAPITAL_INR], rows])
    return float(eqs[-1]), float((1 - eqs / np.maximum.accumulate(eqs)).max() * 100)


def sensitivity(start):
    """Same ML trades re-priced with slippage, a pessimistic fill rule, and Delta's normal fees."""
    df, cands = _history()
    ml = pd.read_csv(os.path.join(ROOT, "model", "backtest_ml_trades.csv"), parse_dates=["time"])
    ml = ml[ml.time >= pd.Timestamp(start, tz="UTC")]
    rows = []
    for rule, sf in (("TradingView fill rule", False), ("Stop always first (pessimistic)", True)):
        for slip in (0.0, 0.00001, 0.00002, 0.00003, 0.00005):
            t = S.label(df, cands, slip, sf)[["time", "prio", "R", "entry_px", "exit_px", "how"]]
            m = ml.drop(columns=["R", "entry_px", "exit_px", "how"]).merge(t, on=["time", "prio"])
            final, dd = with_costs(m)
            rows.append(dict(rule=rule, cost=f"{slip * 100:.3f}% slippage per fill", avgR=float(m.R.mean()),
                             win=float((m.R > 0).mean() * 100), final=final, dd=dd, base=(slip == 0 and not sf)))
    t = S.label(df, cands)[["time", "prio", "R", "entry_px", "exit_px", "how"]]
    m = ml.drop(columns=["R", "entry_px", "exit_px", "how"]).merge(t, on=["time", "prio"])
    fee_R = (0.0005 * m.entry_px + np.where(m.how.isin([1, 3]), 0.0002, 0.0005) * m.exit_px) / (m.e - m.sl).abs()
    final, dd = with_costs(m, 0.0005, 0.0002)
    rows.append(dict(rule="If the fee-free rule ended", cost="Delta normal fees (0.05% taker, 0.02% maker)",
                     avgR=float((m.R - fee_R).mean()), win=float((m.R - fee_R > 0).mean() * 100), final=final, dd=dd,
                     base=False))
    se = float(m.R.std() / np.sqrt(len(m)))
    return dict(rows=rows, avgR=float(m.R.mean()), se=se, n=len(m))


def period(start):
    out = {"sens": sensitivity(start)}
    for acc in ("ml", "base"):
        tr = pd.read_csv(os.path.join(ROOT, "model", f"backtest_{acc}_trades.csv"), parse_dates=["time"])
        tr = tr[tr.time >= pd.Timestamp(start, tz="UTC")].reset_index(drop=True)
        d_sip = dict(t=[int(x.timestamp() * 1000) for x in tr.time], e=tr.e.round(2).tolist(), sl=tr.sl.round(2).tolist(),
                     en=tr.entry_px.round(2).tolist(), ex=tr.exit_px.round(2).tolist(), d=tr.dir.astype(int).tolist())
        c, skipped = account(tr)
        c["ist"] = c.time.dt.tz_convert(S.IST)
        eq = np.concatenate([[S.CAPITAL_INR], c.equity_inr])
        peak = np.maximum.accumulate(eq)
        d = dict(summary=summary(c, skipped),
                 t=[int(c.time.iloc[0].timestamp() * 1000) - 86_400_000] + [int(x.timestamp() * 1000) for x in c.time],
                 eq=[round(float(v), 2) for v in eq], dd=[round(float(v), 2) for v in (eq / peak - 1) * 100],
                 years=breakdown(c, c.ist.dt.year), sessions=breakdown(c, "session"),
                 dirs=breakdown(c, np.where(c.dir == 1, "Long", "Short")),
                 kinds=breakdown(c, np.where(c.fvg == 1, "FVG retest", "EMA pullback")),
                 exits=breakdown(c, c.how.map(S.HOW)), sip=d_sip)
        if acc == "ml":
            m = c.groupby(c.ist.dt.strftime("%Y-%m")).pnl_inr.sum()
            d["months"] = [dict(k=k, v=round(float(v), 2)) for k, v in m.items()]
            d["trades"] = [dict(t=r.ist.strftime("%d %b %Y %H:%M"), s=r.session, d="Long" if r.dir == 1 else "Short",
                                k="FVG" if r.fvg else "EMA", p=round(r.p, 3), n=int(r.contracts), e=round(r.entry_px, 1),
                                sl=round(r.stop, 1), tp=round(r.tgt, 1), x=round(r.exit_px, 1), h=S.HOW[int(r.how)],
                                R=round(r.R, 2), pl=round(r.pnl_inr, 2), eq=round(r.equity_inr, 2),
                                m=round(r.margin_inr, 0), liq=round(r.liq_pct, 1))
                           for r in c.iloc[::-1].itertuples()]
        out[acc] = d
    return out


def main(path=os.path.join(ROOT, "docs", "backtest_50x.html")):
    with open(os.path.join(ROOT, "model", "backtest.json")) as fh:
        bt = json.load(fh)
    data = dict(periods={k: period(v) for k, v in PERIODS.items()},
                meta=dict(auc=bt["auc"], auc24=bt["auc_2024"], end=bt["end"], lev=S.LEVERAGE, mode=S.MARGIN_MODE,
                          risk=S.RISK_PCT, cap=S.CAPITAL_INR, fx=S.USDINR))
    with open(os.path.join(ROOT, "research", "templates", "backtest_50x.html"), encoding="utf-8") as fh:
        html = fh.read().replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html)
    for k, start in PERIODS.items():
        for acc in ("ml", "base"):
            tr = pd.read_csv(os.path.join(ROOT, "model", f"backtest_{acc}_trades.csv"), parse_dates=["time"])
            tr = tr[tr.time >= pd.Timestamp(start, tz="UTC")]
            r = sip_sim(tr, start, 500)
            end = pd.Timestamp(bt["end"], tz=S.IST)
            print(f"SIP 500 from {k} {acc:4s}: invested Rs {r['invested']:,.0f} -> Rs {r['final']:,.0f} "
                  f"(profit Rs {r['final'] - r['invested']:,.0f}, XIRR {xirr(r['flows'], r['final'], end):.1f}%/yr, "
                  f"worst fall {r['twr_dd']:.1f}%)")
    for k, p in data["periods"].items():
        for acc in ("ml", "base"):
            s = p[acc]["summary"]
            print(f"from {k} {acc:4s}: {s['n']} trades, win {s['win']:.1f}%, PF {s['PF']:.2f}, Rs 1500 -> Rs {s['final']:,.0f} "
                  f"({s['ret']:+.0f}%, CAGR {s['cagr']:.0f}%), maxDD {s['maxDD']:.1f}%, lots med {s['med_lots']:.0f} "
                  f"max {s['max_lots']}, margin med Rs {s['med_margin']:.0f}, liq min {s['min_liq']:.1f}%")
    print("->", path, f"{os.path.getsize(path) / 1e3:.0f} kB")


if __name__ == "__main__":
    main()
