"""Replay a period through bot.py bar by bar and compare with the vectorised backtest (strategy.py).

    python tests/test_parity.py 2026-08-01 2026-09-27
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd

import bot
import strategy as S


def main(a, b):
    subprocess.run([sys.executable, os.path.join(ROOT, "bot.py"), "--replay", a, b], check=True, cwd=ROOT)
    got = pd.read_csv(os.path.join(ROOT, "state", "replay_trades.csv"), parse_dates=["signal_time"])
    import train
    hist = train.load_history()
    t0, t1 = pd.Timestamp(a, tz="UTC"), pd.Timestamp(b, tz="UTC")
    df = hist[hist.index < t1]
    t = S.label(df, S.candidates(df))
    t = t[(t.time >= t0) & (t.exit_bar < len(df) - 1)].copy()
    m = bot.Model()
    t["p"] = m.score(df, t)
    for acc, take in (("base", t.strategy.to_numpy()), ("ml", (t.strategy & (t.p >= m.cut)).to_numpy())):
        exp = S.execute(t, take)
        g = got[got.account == acc]
        k = pd.merge(exp[["time", "dir", "R"]], g[["signal_time", "dir", "R"]], left_on=["time", "dir"],
                     right_on=["signal_time", "dir"], how="outer", indicator=True)
        both = k[k._merge == "both"]
        print(f"{acc}: backtest {len(exp)} trades, bot {len(g)} | matched {len(both)}, only backtest "
              f"{(k._merge == 'left_only').sum()}, only bot {(k._merge == 'right_only').sum()} | "
              f"max |R diff| {np.abs(both.R_x - both.R_y).max():.4f} | avg R {exp.R.mean():+.3f} vs {g.R.mean():+.3f}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
