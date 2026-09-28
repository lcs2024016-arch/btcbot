"""No-sugar-coating equity outlook at the bot's settings (1% risk, 50x cross, Delta lots, Rs 1,500).

1. Past equity since Jan 2024 re-priced with realistic costs: 0.003% slippage per market fill and the
   pessimistic fill rule (stop first when stop and target are inside one candle), plus a harsher 0.005% case.
2. The next 12 months as a distribution: 5,000 bootstrap paths drawn from those realistic trades at the
   observed trade rate, under three edge assumptions (as measured, half, none), with and without a monthly
   Rs 500 SIP. Sizing per path uses the same rules as strategy.contracts() (vectorised; checked below).

    python research/realistic_report.py      # -> docs/realistic.html
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import strategy as S  # noqa: E402
import train  # noqa: E402

START = pd.Timestamp("2024-01-01", tz="UTC")
REAL = dict(slip=0.00003, stop_first=True)       # realistic: 0.003% per fill, pessimistic fills
HARSH = dict(slip=0.00005, stop_first=True)
PATHS, MONTHS, SEED = 5000, 12, 7


def priced(df, cands, ml, slip, stop_first):
    t = S.label(df, cands, slip, stop_first)[["time", "prio", "R", "entry_px", "exit_px", "how"]]
    return ml.drop(columns=["R", "entry_px", "exit_px", "how"]).merge(t, on=["time", "prio"]).sort_values("time")


def curve(m):
    c = S.inr_curve(m)
    eq = np.concatenate([[S.CAPITAL_INR], c.equity_inr])
    return c, eq


def lots_vec(eq, e, sl):
    """Vectorised strategy.contracts(): 1% risk, 1-lot minimum up to 2%, 50x margin cap, liquidation guard."""
    risk1 = S.CONTRACT_BTC * np.abs(e - sl) * S.USDINR
    n = np.floor(eq * S.RISK_PCT / 100 / risk1)
    n = np.where((n == 0) & (risk1 <= eq * S.MAX_RISK_PCT / 100), 1, n)
    n = np.minimum(n, np.floor(eq / S.USDINR * S.LEVERAGE / (S.CONTRACT_BTC * e)))
    stop = np.abs(e - sl) / e
    n = np.minimum(n, np.floor(eq / S.USDINR / (S.CONTRACT_BTC * e * (S.LIQ_SAFETY * stop + S.MAINT_MARGIN))))
    return np.where(eq > 0, np.maximum(n, 0), 0)


def monte_carlo(m, per_month, shift_R, sip, rng):
    """Bootstrap PATHS x 12 months. shift_R is subtracted from every trade's R (edge haircut)."""
    e, sl, d = m.e.to_numpy(), m.sl.to_numpy(), m.dir.to_numpy()
    move = d * (m.exit_px.to_numpy() - m.entry_px.to_numpy()) - shift_R * np.abs(e - sl)   # per BTC, after haircut
    eq = np.full(PATHS, S.CAPITAL_INR)
    inv = S.CAPITAL_INR
    peak, idx, dd = np.ones(PATHS), np.ones(PATHS), np.zeros(PATHS)
    months = [eq.copy()]
    for _ in range(MONTHS):
        n_tr = rng.poisson(per_month, PATHS)
        for k in range(n_tr.max()):
            active = k < n_tr
            j = rng.integers(0, len(m), PATHS)
            n = lots_vec(eq, e[j], sl[j]) * active
            pl = n * S.CONTRACT_BTC * move[j] * S.USDINR
            idx = idx * np.where(eq > 0, 1 + pl / np.where(eq > 0, eq, 1), 1)
            eq = np.maximum(eq + pl, 0)
            peak = np.maximum(peak, idx)
            dd = np.maximum(dd, 1 - idx / peak)
        months.append(eq.copy())
        eq = eq + sip
        inv += sip
    mat = np.array(months)                        # (13, PATHS)
    q = np.percentile(mat, [5, 25, 50, 75, 95], axis=1)
    invested = S.CAPITAL_INR + sip * (MONTHS - 1)
    end = mat[-1]
    return dict(q=[[round(float(v), 1) for v in row] for row in q], invested=invested,
                p_loss=float((end < invested).mean() * 100), p_dd20=float((dd > 0.20).mean() * 100),
                p_dd30=float((dd > 0.30).mean() * 100), med=float(np.median(end)), p5=float(np.percentile(end, 5)),
                p95=float(np.percentile(end, 95)), med_dd=float(np.median(dd) * 100),
                p_double=float((end >= 2 * invested).mean() * 100))


def main(path=os.path.join(ROOT, "docs", "realistic.html")):
    df = train.load_history()
    cands = S.candidates(df)
    ml = pd.read_csv(os.path.join(ROOT, "model", "backtest_ml_trades.csv"), parse_dates=["time"])
    ml = ml[ml.time >= START]

    head = priced(df, cands, ml, 0.0, False)
    real = priced(df, cands, ml, **REAL)
    harsh = priced(df, cands, ml, **HARSH)
    # sanity: vectorised sizing == strategy.contracts
    eqs = np.linspace(300, 60000, 400)
    r0 = real.iloc[np.arange(400) % len(real)]
    assert (lots_vec(eqs, r0.e.to_numpy(), r0.sl.to_numpy()) ==
            [S.contracts(q, a, b) for q, a, b in zip(eqs, r0.e, r0.sl)]).all(), "vectorised sizing mismatch"

    hist = {}
    for name, m in (("headline", head), ("realistic", real), ("harsh", harsh)):
        c, eq = curve(m)
        dd = (1 - eq / np.maximum.accumulate(eq)).max() * 100
        yrs = {}
        for y, g in c.groupby(c.time.dt.tz_convert(S.IST).dt.year):
            yrs[str(y)] = dict(pnl=float(g.pnl_inr.sum()), avgR=float(g.R.mean()), n=len(g))
        hist[name] = dict(t=[int(c.time.iloc[0].timestamp() * 1000) - 86_400_000] + [int(x.timestamp() * 1000) for x in c.time],
                          eq=[round(float(v), 2) for v in eq], final=float(eq[-1]), dd=float(dd), avgR=float(m.R.mean()),
                          win=float((m.R > 0).mean() * 100), n=len(m), years=yrs)
    months = max((real.time.max() - real.time.min()).days / 30.44, 1)
    per_month = len(real) / months
    # Edge in rupees, not R: at Rs 1,500 whole-lot rounding makes the real risk per trade 0.5-2%, and it is
    # larger on trades that did better, so a plain R shift would leave a hidden rupee edge. Weight each trade's
    # R by its rupee risk at the starting balance; subtracting this weighted mean makes the expected rupee
    # P&L per trade zero ("none") or half ("half").
    w = np.array([S.contracts(S.CAPITAL_INR, a, b) for a, b in zip(real.e, real.sl)]) *         S.CONTRACT_BTC * (real.e - real.sl).abs().to_numpy() * S.USDINR
    mean_R = float(np.average(real.R, weights=w))
    print(f"plain mean R {real.R.mean():+.3f} | rupee-weighted mean R {mean_R:+.3f}")
    se = float(real.R.std() / np.sqrt(len(real)))
    plain_R = float(real.R.mean())

    mc = {}
    rng = np.random.default_rng(SEED)
    for edge, shift in (("measured", 0.0), ("half", mean_R / 2), ("none", mean_R)):
        for sip in (0, 500):
            mc[f"{edge}_{sip}"] = monte_carlo(real, per_month, shift, sip, rng)
            r = mc[f"{edge}_{sip}"]
            print(f"edge {edge:8s} sip {sip:3d}: median Rs {r['med']:,.0f} (5% {r['p5']:,.0f} / 95% {r['p95']:,.0f}) "
                  f"invested {r['invested']:,} P(loss) {r['p_loss']:.0f}% P(DD>20%) {r['p_dd20']:.0f}% P(DD>30%) {r['p_dd30']:.0f}%")
    for k, h in hist.items():
        print(f"{k:9s}: avg R {h['avgR']:+.3f} win {h['win']:.1f}% Rs 1500 -> Rs {h['final']:,.0f} DD {h['dd']:.1f}%")

    data = dict(hist=hist, mc=mc, meta=dict(per_month=per_month, mean_R=mean_R, plain_R=plain_R, se=se, paths=PATHS,
                                            start=str(real.time.min().tz_convert(S.IST))[:10],
                                            end=str(real.time.max().tz_convert(S.IST))[:10],
                                            real_slip=REAL["slip"] * 100, harsh_slip=HARSH["slip"] * 100))
    with open(os.path.join(ROOT, "research", "templates", "realistic.html"), encoding="utf-8") as fh:
        html = fh.read().replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html)
    print("->", path, f"{os.path.getsize(path) / 1e3:.0f} kB")


if __name__ == "__main__":
    main()
