"""Build README.md + charts/*.svg from the forward-test log (state/) and the walk-forward backtest (model/).

    python report.py
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import strategy as S  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CH = os.path.join(HERE, "charts")
THEME = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9",
                  axis="#c3c2b7", s1="#2a78d6", s2="#eb6834", pos="#2a78d6", neg="#e34948"),
    "dark": dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a",
                 axis="#383835", s1="#3987e5", s2="#d95926", pos="#3987e5", neg="#e66767"),
}
NAMES = {"ml": "ML filter (top 35%)", "base": "No ML (Pine rules)"}


def fmt_ist(ts):
    return pd.Timestamp(ts).tz_convert(S.IST).strftime("%d %b %Y, %H:%M")


def money(v, nd=2):
    return f"{'−' if v < 0 else ''}₹{abs(v):,.{nd}f}"


def inr(x, sign=False):
    return (f"{'+' if sign and x > 0 else ''}{'-' if x < 0 else ''}₹{abs(x):,.2f}") if pd.notna(x) else "–"


def r_(x):
    return f"{x:+.2f}R" if pd.notna(x) else "–"


# ---------------------------------------------------------------- charts

def _axes(th, w=9.0, h=3.2):
    fig, ax = plt.subplots(figsize=(w, h), dpi=100)
    fig.patch.set_facecolor(th["surface"])
    ax.set_facecolor(th["surface"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(th["axis"])
    ax.tick_params(colors=th["muted"], labelsize=9, length=0)
    ax.grid(axis="y", color=th["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    return fig, ax


def _save(fig, name, mode):
    fig.tight_layout()
    fig.savefig(os.path.join(CH, f"{name}_{mode}.svg"), facecolor=fig.get_facecolor())
    plt.close(fig)


def equity_chart(name, series, title, xlabel_fmt="%d %b"):
    """series: list of (label, times, equity_inr, slot, dashed)."""
    for mode, th in THEME.items():
        fig, ax = _axes(th)
        for label, x, y, slot, dashed in series:
            ax.plot(x, y, color=th[slot], linewidth=2, linestyle=(0, (4, 3)) if dashed else "-",
                    drawstyle="steps-post")
        ax.axhline(S.CAPITAL_INR, color=th["axis"], linewidth=1)
        # end labels, pushed apart so they never overlap (>= 14 px)
        fig.canvas.draw()
        ends = sorted(((ax.transData.transform((matplotlib.dates.date2num(pd.Timestamp(x[-1])), y[-1]))[1], k)
                       for k, (_, x, y, _, _) in enumerate(series)))
        placed = []
        for py, k in ends:
            py = max(py, placed[-1] + 14) if placed else py
            placed.append(py)
            label, x, y, _, _ = series[k]
            ax.annotate(f"{label}  {money(y[-1], 0)}", (x[-1], y[-1]), xycoords="data",
                        xytext=(8, py - ax.transData.transform(
                            (matplotlib.dates.date2num(pd.Timestamp(x[-1])), y[-1]))[1]),
                        textcoords="offset pixels", va="center", fontsize=9, color=th["ink2"])
        ax.set_title(title, loc="left", fontsize=11, color=th["ink"], pad=10)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: money(v, 0)))
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter(xlabel_fmt))
        ax.margins(x=0.02)
        fig.subplots_adjust(right=0.78)
        _save(fig, name, mode)


def daily_chart(name, days, pnl, title):
    for mode, th in THEME.items():
        fig, ax = _axes(th, h=2.6)
        cols = [th["pos"] if v >= 0 else th["neg"] for v in pnl]
        ax.bar(days, pnl, color=cols, width=0.6)
        ax.axhline(0, color=th["axis"], linewidth=1)
        ax.set_title(title, loc="left", fontsize=11, color=th["ink"], pad=10)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: money(v, 0)))
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d %b"))
        _save(fig, name, mode)


def picture(name, alt):
    return (f'<picture>\n  <source media="(prefers-color-scheme: dark)" srcset="charts/{name}_dark.svg">\n'
            f'  <img alt="{alt}" src="charts/{name}_light.svg">\n</picture>')


# ---------------------------------------------------------------- tables

def summary_row(label, g, live=False):
    s = S.stats(g.R)
    row = [label, f"{s['n']}", f"{s['win']:.1f}%" if s["n"] else "–", r_(s["avgR"]) if s["n"] else "–",
           f"{s['PF']:.2f}" if s["n"] and np.isfinite(s["PF"]) else "–"]
    if "pnl_inr" in g:
        eq = S.CAPITAL_INR + g.pnl_inr.cumsum()
        dd = (1 - eq / np.maximum.accumulate(np.concatenate([[S.CAPITAL_INR], eq]))[1:]).max() * 100 if len(g) else 0
        row += [inr(g.pnl_inr.sum(), True), inr(S.CAPITAL_INR + g.pnl_inr.sum()), f"{dd:.1f}%"]
    if live:
        gl = g[g.R_live.notna()]
        row += [r_(gl.R_live.mean()) if len(gl) else "–",
                inr(gl.pnl_inr_live.sum(), True) if len(gl) else "–"]
    return row


def md_table(header, rows, align=None):
    align = align or ["---"] + ["---:"] * (len(header) - 1)
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join(align) + " |"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def forward_section(tr):
    if tr.empty:
        return ("_No forward-test trades yet. The bot trades Monday–Friday, London 13:30–17:30 IST and "
                "New York 18:30–22:30 IST; results appear here after the first closed trade._\n")
    for c in ("signal_time", "entry_time", "exit_time"):
        tr[c] = pd.to_datetime(tr[c], utc=True)
    out = []
    head = ["Account", "Trades", "Win rate", "Avg R", "Profit factor", "P&L", "Equity", "Max DD",
            "Avg R @ live prices", "P&L @ live prices"]
    rows = [summary_row(NAMES[a], tr[tr.account == a], live=True) for a in ("ml", "base")]
    first, last = tr.entry_time.min(), tr.exit_time.max()
    out.append(f"Paper trading since **{fmt_ist(first)} IST** · last trade closed {fmt_ist(last)} IST · "
               f"start capital ₹{S.CAPITAL_INR:,.0f} per account\n")
    out.append(md_table(head, rows))

    series = []
    for a, slot, dashed, lab in (("ml", "s1", False, "ML"), ("base", "s2", False, "No ML"),
                                 ("ml", "s1", True, "ML @ live")):
        g = tr[tr.account == a].sort_values("exit_time")
        col = "equity_inr_live" if dashed else "equity_inr"
        g = g[g[col].notna()]
        if len(g):
            x = np.concatenate([[g.entry_time.iloc[0]], g.exit_time])
            y = np.concatenate([[S.CAPITAL_INR], g[col]])
            series.append((lab, x, y, slot, dashed))
    equity_chart("forward_equity", series, "Forward test · paper equity (INR)")
    out.append("\n" + picture("forward_equity", "Forward test equity") + "\n")

    ml = tr[tr.account == "ml"].copy()
    if len(ml):
        d = ml.groupby(ml.exit_time.dt.tz_convert(S.IST).dt.normalize()).pnl_inr.sum()
        daily_chart("forward_daily", d.index.tz_localize(None), d.to_numpy(), "ML account · P&L per day (INR)")
        out.append(picture("forward_daily", "ML daily P&L") + "\n")

        out.append("#### ML account by session\n")
        rows = [summary_row(f"{s} ({'13:30–17:30' if s == 'London' else '18:30–22:30'} IST)", ml[ml.session == s])
                for s in ("London", "New York")]
        out.append(md_table(["Session", "Trades", "Win rate", "Avg R", "Profit factor", "P&L", "Equity*",
                             "Max DD*"], rows) + "\n\n<sub>* as if the session were traded on its own</sub>\n")

        lv = ml[ml.R_live.notna()]
        if len(lv):
            ent = (lv.dir * (lv.live_entry - lv.entry_px)).mean()
            stops = lv[lv.live_how == "stop"]
            stop_slip = (stops.dir * (stops.stop - stops.live_exit)).mean() if len(stops) else np.nan
            out.append("#### Execution check (backtest prices vs live quotes)\n")
            out.append(md_table(["Measure", "Value"], [
                ["Trades with live quotes", f"{len(lv)}"],
                ["Avg entry cost vs 5m open", f"${ent:+.2f} per BTC ({ent / lv.entry_px.mean() * 1e4:+.2f} bp)"],
                ["Avg stop slippage", f"${stop_slip:+.2f} per BTC" if pd.notna(stop_slip) else "–"],
                ["Avg decision delay after bar close", f"{lv.latency_s.mean():.1f} s"],
                ["Avg R: candle model → live prices", f"{r_(lv.R.mean())} → {r_(lv.R_live.mean())}"]],
                ["---", "---:"]) + "\n")

    out.append("#### Latest trades (ML account)\n")
    rows = []
    for r in ml.sort_values("exit_time", ascending=False).head(20).itertuples():
        rows.append([fmt_ist(r.entry_time), r.session, "Long" if r.dir == 1 else "Short", r.kind.split()[0],
                     f"{r.p:.2f}", r.contracts, f"{r.entry_px:,.1f}", f"{r.stop:,.1f}", f"{r.tgt:,.1f}",
                     f"{r.exit_px:,.1f}", r.how, r_(r.R), inr(r.pnl_inr, True), r_(r.R_live)])
    out.append(md_table(["Entry (IST)", "Session", "Side", "Type", "p(win)", "Lots", "Entry", "Stop", "TP", "Exit",
                         "Exit reason", "R", "P&L", "R @ live"], rows,
                        ["---"] * 5 + ["---:"] * 9))
    return "\n".join(out) + "\n"


def backtest_section():
    with open(os.path.join(HERE, "model", "backtest.json")) as fh:
        bt = json.load(fh)
    P = bt["periods"]
    rows = []
    for per, lab in (("2024+", "Jan 2024 →"), ("all", "Jan 2021 →")):
        for a in ("ml", "base"):
            v = P[f"{a}_{per}"]
            rows.append([f"{NAMES[a]}, {lab}", f"{v['n']:.0f}", f"{v['per_mo']:.0f}", f"{v['win']:.1f}%",
                         r_(v["avgR"]), f"{v['PF']:.2f}", f"₹{v['inr_end']:,.0f}", f"{v['inr_maxDD']:.1f}%"])
    out = [f"Walk-forward {bt['start'][:7]} → {bt['end'][:7]}: the model is retrained every month on past data "
           f"only and scores the next month. Out-of-sample AUC {bt['auc']:.3f} (since 2024: {bt['auc_2024']:.3f}). "
           f"Same fills as the bot's candle model, zero slippage, whole Delta contracts from ₹1,500.\n",
           md_table(["Backtest", "Trades", "Per month", "Win rate", "Avg R", "Profit factor", "₹1,500 →",
                     "Max DD"], rows)]
    yrs = bt["years"]
    rows = [[y, r_(yrs[y].get("ml", {}).get("avgR", np.nan)), f"{yrs[y].get('ml', {}).get('win', np.nan):.1f}%",
             r_(yrs[y].get("base", {}).get("avgR", np.nan)), f"{yrs[y].get('base', {}).get('win', np.nan):.1f}%"]
            for y in sorted(yrs)]
    out.append("\n" + md_table(["Year", "ML avg R", "ML win rate", "No-ML avg R", "No-ML win rate"], rows))

    series = []
    for a, slot, lab in (("ml", "s1", "ML"), ("base", "s2", "No ML")):
        g = pd.read_csv(os.path.join(HERE, "model", f"backtest_{a}_trades.csv"), parse_dates=["time"])
        g = S.inr_curve(g[g.time >= "2024-01-01"])
        series.append((lab, np.concatenate([[g.time.iloc[0]], g.time]),
                       np.concatenate([[S.CAPITAL_INR], g.equity_inr]), slot, False))
    equity_chart("backtest_equity", series, "Walk-forward backtest since Jan 2024 · ₹1,500 start (INR)", "%b %Y")
    out.append("\n" + picture("backtest_equity", "Backtest equity since 2024"))
    return "\n".join(out) + "\n"


def status_line():
    st = {}
    p = os.path.join(HERE, "state", "state.json")
    if os.path.exists(p):
        with open(p) as fh:
            st = json.load(fh)
    with open(os.path.join(HERE, "model", "meta.json")) as fh:
        meta = json.load(fh)
    upd = fmt_ist(pd.Timestamp(st["updated"], tz="UTC")) + " IST" if st.get("updated") else "not started yet"
    pos = []
    for a, d in st.get("accounts", {}).items():
        if d.get("pos"):
            pos.append(f"{a.upper()}: {'long' if d['pos']['dir'] == 1 else 'short'} {d['pos']['contracts']} lot(s)")
    return (f"**Mode:** paper trading · **Last update:** {upd} · **Data:** {st.get('source', '–')} · "
            f"**Model:** trained {meta['trained_at'][:10]}, cut p ≥ {meta['cut']:.3f} · "
            f"**Open positions:** {', '.join(pos) if pos else 'none'}")


def build():
    os.makedirs(CH, exist_ok=True)
    tp = os.path.join(HERE, "state", "trades.csv")
    tr = pd.read_csv(tp) if os.path.exists(tp) else pd.DataFrame()
    with open(os.path.join(HERE, "README_TEMPLATE.md"), encoding="utf-8") as fh:
        tpl = fh.read()
    md = (tpl.replace("{{STATUS}}", status_line()).replace("{{FORWARD}}", forward_section(tr))
          .replace("{{BACKTEST}}", backtest_section()))
    with open(os.path.join(HERE, "README.md"), "w", encoding="utf-8") as fh:
        fh.write(md)


if __name__ == "__main__":
    build()
