"""HTML page: what 100x / 200x leverage does to the 9/15 EMA + ML strategy on Delta India.

    python research/leverage_report.py     # -> docs/leverage.html
"""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import leverage_study as L
from leverage_study import FX, LOT, MM, ROOT, S


def build(path=os.path.join(ROOT, "docs", "leverage.html")):
    tr = L.trades_with_mae()
    x = [int(pd.Timestamp(t).timestamp() * 1000) for t in tr.time]
    x = [x[0] - 86_400_000] + x                                            # start point one day earlier

    def curve(o):
        return [S.CAPITAL_INR] + [round(float(v), 2) for v in o.equity]

    rows, cache = [], {}
    for lev in (10, 25, 50, 100, 200):
        for mode, margin, label in (("risk", "cross", "1% risk · cross margin"),
                                    ("risk", "isolated", "1% risk · isolated margin"),
                                    ("allin", "cross", "All-in (full balance as margin)")):
            o, s = L.simulate(tr, lev, mode, margin=margin)
            cache[(lev, mode, margin)] = o
            rows.append(dict(lev=lev, sizing=label, **s))
    risk = []
    for rp in (1, 2, 3, 5, 7.5, 10, 15, 20):
        _, s = L.simulate(tr, 100, "risk", rp)
        risk.append(dict(risk=rp, **s))
    margin_rows = []
    for lev in (10, 25, 50, 100, 200):
        liq = 1 / lev - MM
        margin_rows.append(dict(lev=lev, im=100 / lev, liq=liq * 100,
                                lot_margin=LOT * tr.entry_px.median() / lev * FX,
                                stop_beyond=float((tr.stop_dist >= liq).mean() * 100),
                                mae_beyond=float(((tr.mae >= liq) & (tr.stop_dist >= liq)).mean() * 100)))
    data = dict(
        x=x,
        c1=[dict(name="1% risk · 100x or 200x cross", y=curve(cache[(200, "risk", "cross")])),
            dict(name="1% risk · 100x isolated", y=curve(cache[(100, "risk", "isolated")])),
            dict(name="1% risk · 200x isolated", y=curve(cache[(200, "risk", "isolated")]))],
        c2=[dict(name=f"All-in {lev}x", y=curve(cache[(lev, "allin", "cross")])) for lev in (10, 25, 50, 100, 200)],
        risk=risk, rows=rows, margin=margin_rows,
        meta=dict(n=len(tr), start=str(tr.time.min().tz_convert(S.IST))[:10],
                  end=str(tr.time.max().tz_convert(S.IST))[:10], win=float((tr.R > 0).mean() * 100),
                  stop_med=float(tr.stop_dist.median() * 100), px=float(tr.entry_px.median())))
    with open(os.path.join(ROOT, "research", "templates", "leverage.html"),
              encoding="utf-8") as fh:
        html = fh.read().replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html)
    print("->", path, f"{os.path.getsize(path) / 1e3:.0f} kB")


if __name__ == "__main__":
    build()
