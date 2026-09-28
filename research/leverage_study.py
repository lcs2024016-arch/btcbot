"""What does 100x / 200x leverage do to the 9/15 EMA + ML strategy on Delta India BTCUSD?

Uses the walk-forward ML trades (model/backtest_ml_trades.csv, zero slippage, fee-free) and
re-sizes them with Delta's margin rules: initial margin = 1/leverage, maintenance 0.25% of notional,
1 lot = 0.001 BTC, Rs 85 / USD. Adverse excursion per trade comes from the 5m candles, so a position is
liquidated if price reaches its liquidation level before the stop / target / time exit.

    python research/leverage_study.py   # prints the scenario table (page: research/leverage_report.py)
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import strategy as S  # noqa: E402
import train  # noqa: E402

MM = 0.0025                      # Delta BTCUSD maintenance margin (api: maintenance_margin 0.25%)
LOT = S.CONTRACT_BTC
FX = S.USDINR
START = pd.Timestamp("2024-01-01", tz="UTC")


def trades_with_mae():
    """ML walk-forward trades + max adverse excursion (fraction of entry) while the trade was open."""
    ml = pd.read_csv(os.path.join(ROOT, "model", "backtest_ml_trades.csv"), parse_dates=["time"])
    ml = ml[ml.time >= START].copy()
    df = train.load_history()
    t = S.label(df, S.candidates(df))
    ml = ml.merge(t[["time", "prio", "bar", "exit_bar"]], on=["time", "prio"], how="left")
    H, L, O = df.high.to_numpy(), df.low.to_numpy(), df.open.to_numpy()
    mae = []
    for r in ml.itertuples():
        i0, xb = int(r.bar) + 1, int(r.exit_bar)
        last = xb if r.how in (0, 1) else xb - 1          # exits at an open: that bar never traded against us
        lo = L[i0:last + 1].min() if last >= i0 else O[xb]
        hi = H[i0:last + 1].max() if last >= i0 else O[xb]
        if r.how not in (0, 1):
            lo, hi = min(lo, O[xb]), max(hi, O[xb])
        mae.append((r.entry_px - lo) / r.entry_px if r.dir == 1 else (hi - r.entry_px) / r.entry_px)
    ml["mae"] = np.maximum(mae, 0)
    ml["stop_dist"] = (ml.entry_px - ml.stop).abs() / ml.entry_px
    return ml


def simulate(tr, lev, mode, risk_pct=1.0, margin="cross"):
    """mode 'risk': lots for risk_pct of equity (1-lot minimum up to 2x risk, as the bot);
       mode 'allin': every rupee of equity is margin at `lev`."""
    eq, out, ruined = S.CAPITAL_INR, [], None
    for r in tr.itertuples():
        px = r.entry_px
        lot_usd = LOT * px
        eq_usd = eq / FX
        max_lots = int(eq_usd * lev // lot_usd)                    # margin available
        if max_lots < 1:
            ruined = ruined or r.time
            out.append((r.time, eq, 0, 0, False))
            continue
        if mode == "allin":
            n = max_lots
        else:
            risk_1 = LOT * abs(r.e - r.sl) * FX
            n = int(eq * risk_pct / 100 // risk_1)
            if n == 0 and risk_1 <= eq * 2 * risk_pct / 100:
                n = 1
            n = min(n, max_lots)
            if n == 0:
                out.append((r.time, eq, 0, 0, False))
                continue
        notional = n * lot_usd
        pos_margin = notional / lev if margin == "isolated" else eq_usd
        liq_dist = (pos_margin - MM * notional) / notional       # adverse move that wipes the margin
        liquidated = r.mae >= liq_dist and (r.stop_dist >= liq_dist or r.how in (2,))
        if liquidated:
            pl = -pos_margin * FX
        else:
            pl = n * LOT * r.dir * (r.exit_px - r.entry_px) * FX
        eq = max(eq + pl, 0.0)
        out.append((r.time, eq, n, pl, liquidated))
        if eq <= 0:
            ruined = ruined or r.time
    o = pd.DataFrame(out, columns=["time", "equity", "lots", "pnl", "liq"])
    peak = np.maximum.accumulate(np.concatenate([[S.CAPITAL_INR], o.equity]))[1:]
    return o, dict(final=float(o.equity.iloc[-1]), ret=float(o.equity.iloc[-1] / S.CAPITAL_INR * 100 - 100),
                   maxDD=float((1 - o.equity / peak).max() * 100), liqs=int(o.liq.sum()),
                   taken=int((o.lots > 0).sum()), wins=int((o.pnl > 0).sum()),
                   med_lots=float(o.lots[o.lots > 0].median()) if (o.lots > 0).any() else 0.0,
                   min_eq=float(o.equity.min()),
                   ruined=str(ruined.tz_convert(S.IST))[:10] if ruined is not None else None)


def main():
    tr = trades_with_mae()
    print(len(tr), "ML trades since 2024 | median stop %.3f%% | stop > 0.25%%: %.0f%% | stop > 0.75%%: %.1f%%"
          % (tr.stop_dist.median() * 100, (tr.stop_dist > .0025).mean() * 100, (tr.stop_dist > .0075).mean() * 100))
    print("MAE median %.3f%%, 90th %.3f%%, max %.3f%%" % (tr.mae.median() * 100, tr.mae.quantile(.9) * 100, tr.mae.max() * 100))
    for lev in (10, 25, 50, 100, 200):
        for mode, margin in (("risk", "cross"), ("risk", "isolated"), ("allin", "cross")):
            _, s = simulate(tr, lev, mode, margin=margin)
            print(f"{lev:3d}x {mode:5s} {margin:8s} final Rs {s['final']:>12,.0f} ret {s['ret']:+9.0f}% DD {s['maxDD']:5.1f}% "
                  f"liq {s['liqs']:3d} lots~{s['med_lots']:.0f} ruined {s['ruined']}")
    for rp in (1, 2, 3, 5, 7.5, 10, 15, 20):
        _, s = simulate(tr, 100, "risk", rp)
        print(f"risk {rp:4.1f}% @100x cross: final Rs {s['final']:>12,.0f} DD {s['maxDD']:5.1f}% liq {s['liqs']} ruined {s['ruined']}")


if __name__ == "__main__":
    main()
