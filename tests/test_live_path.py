"""Exercise the live-quote code path (entries at bid/ask, 2 s stop/target polling, time exits, INR books)
on history: quotes are simulated along each bar's open -> high/low -> close path.

    python tests/test_live_path.py 2026-09-01 2026-09-27
"""
import os
import sys
import tempfile

os.environ["FT_STATE"] = tempfile.mkdtemp()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import bot  # noqa: E402
import train  # noqa: E402


class FakeQuotes(bot.Replay):
    name = "test:simulated-quotes"
    px = None

    def quote(self):
        return self.px - 0.5, self.px + 0.5, self.px            # 1-tick spread around "last"


def main(a, b):
    hist = train.load_history()
    t0, t1 = pd.Timestamp(a, tz="UTC"), pd.Timestamp(b, tz="UTC")
    src = FakeQuotes(hist)
    b_ = bot.Bot(src, bot.Model(), mode="live")
    b_.df = hist[(hist.index >= t0 - bot.WARMUP) & (hist.index < t1)]
    b_.publish = lambda why: None
    df = b_.df
    for i in np.flatnonzero(df.index >= t0):
        src.px = df.close.iat[i]
        b_.on_bar(i, live=True)                                  # decisions at the close
        if i + 1 < len(df):
            o, h, l, c = df.open.iat[i + 1], df.high.iat[i + 1], df.low.iat[i + 1], df.close.iat[i + 1]
            path = (o, h, l, c) if (h - o) < (o - l) else (o, l, h, c)
            for p in path:                                       # polls during the next bar
                src.px = p
                b_.poll_shadows()
    tr = pd.read_csv(b_.trades_path)
    for acc, g in tr.groupby("account"):
        print(f"{acc}: {len(g)} trades | candle avg R {g.R.mean():+.3f} -> Rs {g.equity_inr.iloc[-1]:,.2f} | "
              f"live avg R {g.R_live.mean():+.3f} -> Rs {g.equity_inr_live.dropna().iloc[-1]:,.2f} | "
              f"live missing {g.R_live.isna().sum()}")
        print("   live exit reasons:", g.live_how.value_counts().to_dict(), "| candle:", g.how.value_counts().to_dict())
    print("open shadows left:", len(b_.shadows))


if __name__ == "__main__":
    main(*sys.argv[1:3])
