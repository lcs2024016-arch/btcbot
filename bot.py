"""Paper-trading forward test of the 9/15 EMA Scalper v3 (+ ML filter) on BTC 5m.

    python bot.py                       # run the current/next session (London or New York), then exit
    python bot.py --session london      # force a session
    python bot.py --replay 2026-09-01 2026-09-27   # replay history bar by bar (parity test, no network)

Two paper accounts trade the same signals side by side:
  ml    the model's top 35% of signals (the strategy under test)
  base  every signal the Pine script takes (no ML) - the control group
Each trade is priced twice:
  candle  exactly like the backtest: market fill at the next 5m open, SL/TP from bar highs/lows
  live    real quotes: ask/bid when the order would be sent, stop filled at the bid/ask seen when
          the last price crosses it (polled every second), TP at its limit price, time exit at bid/ask
          -> the gap between the two is the slippage the backtest does not see.

Sessions (entries): London 08:00-12:00 UTC = 13:30-17:30 IST, New York 13:00-17:00 UTC = 18:30-22:30 IST.
Monday-Friday only. Every position is closed 25 min after entry, so every trade is fee-free on Delta (< 30 min).
Sizing: Delta India BTCUSD, 1 contract = 0.001 BTC, whole contracts for 1% risk of the INR paper account
(start Rs 1,500, USD/INR fixed at 85); 1 contract minimum unless that risks > 2%, then the trade is skipped.
Margin: 50x cross (strategy.LEVERAGE / MARGIN_MODE); lots are capped so liquidation is >= 2x the stop distance.
Missed bars (late GitHub start, crash) are caught up from candles and flagged mode=catchup.
"""
import argparse
import json
import os
import subprocess
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import requests

import strategy as S

HERE = os.path.dirname(os.path.abspath(__file__))
STATE = os.environ.get("FT_STATE", os.path.join(HERE, "state"))
BAR = pd.Timedelta(S.TF)
WARMUP = pd.Timedelta("35D")
POLL_S = 1.0
TRADE_COLS = ["account", "mode", "source", "session", "signal_time", "entry_time", "exit_time", "dir", "kind", "p",
              "cut", "e", "sl", "stop", "tgt", "entry_px", "exit_px", "how", "R", "live_entry", "live_exit",
              "live_how", "R_live", "latency_s", "contracts", "pnl_inr", "equity_inr", "pnl_inr_live",
              "equity_inr_live", "leverage", "margin_inr", "liq_dist_pct"]


def now_utc():
    return pd.Timestamp.now(tz="UTC")


def ist(ts):
    return pd.Timestamp(ts).tz_convert(S.IST).strftime("%Y-%m-%d %H:%M IST")


def log(msg):
    print(f"[{now_utc():%H:%M:%S} UTC | {now_utc().tz_convert(S.IST):%H:%M:%S} IST] {msg}", flush=True)


# ---------------------------------------------------------------- market data

class Source:
    name = ""

    def candles(self, start, end):
        raise NotImplementedError

    def quote(self):
        """(bid, ask, last)"""
        raise NotImplementedError


class Delta(Source):
    name, url = "delta:BTCUSD", "https://api.india.delta.exchange"

    def candles(self, start, end):
        rows, cur = [], int(start.timestamp())
        end_s = int(end.timestamp())
        while cur < end_s:
            stop = min(cur + 300 * 1999, end_s)
            r = requests.get(f"{self.url}/v2/history/candles", timeout=15,
                             params=dict(resolution="5m", symbol="BTCUSD", start=cur, end=stop)).json()
            rows += r["result"]
            cur = stop + 300
        d = pd.DataFrame(rows).drop_duplicates("time")
        d.index = pd.to_datetime(d.pop("time"), unit="s", utc=True)
        return d.sort_index()[["open", "high", "low", "close", "volume"]].astype(float)

    def quote(self):
        r = requests.get(f"{self.url}/v2/tickers/BTCUSD", timeout=5).json()["result"]
        q = r["quotes"]
        return float(q["best_bid"]), float(q["best_ask"]), float(r["close"])


class Binance(Source):
    name, url = "binance:BTCUSDT", "https://fapi.binance.com"

    def candles(self, start, end):
        rows, cur = [], int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)
        while cur < end_ms:
            r = requests.get(f"{self.url}/fapi/v1/klines", timeout=15,
                             params=dict(symbol="BTCUSDT", interval="5m", startTime=cur, endTime=end_ms,
                                         limit=1500))
            r.raise_for_status()
            b = r.json()
            if not b:
                break
            rows += b
            cur = b[-1][0] + 300_000
        d = pd.DataFrame([x[:6] for x in rows], columns=["time", "open", "high", "low", "close", "volume"])
        d = d.drop_duplicates("time")
        d.index = pd.to_datetime(d.pop("time"), unit="ms", utc=True)
        return d.astype(float)

    def quote(self):
        b = requests.get(f"{self.url}/fapi/v1/ticker/bookTicker", params=dict(symbol="BTCUSDT"), timeout=5).json()
        p = requests.get(f"{self.url}/fapi/v1/ticker/price", params=dict(symbol="BTCUSDT"), timeout=5).json()
        return float(b["bidPrice"]), float(b["askPrice"]), float(p["price"])


class Coinbase(Source):
    name, url = "coinbase:BTC-USD", "https://api.exchange.coinbase.com"

    def candles(self, start, end):
        rows, cur = [], start
        while cur < end:
            stop = min(cur + BAR * 299, end)
            r = requests.get(f"{self.url}/products/BTC-USD/candles", timeout=15,
                             params=dict(granularity=300, start=cur.isoformat(), end=stop.isoformat()))
            r.raise_for_status()
            rows += r.json()
            cur = stop + BAR
            time.sleep(0.15)
        d = pd.DataFrame(rows, columns=["time", "low", "high", "open", "close", "volume"]).drop_duplicates("time")
        d.index = pd.to_datetime(d.pop("time"), unit="s", utc=True)
        return d.sort_index()[["open", "high", "low", "close", "volume"]].astype(float)

    def quote(self):
        r = requests.get(f"{self.url}/products/BTC-USD/ticker", timeout=5).json()
        return float(r["bid"]), float(r["ask"]), float(r["price"])


class Replay(Source):
    name = "replay:binance-history"

    def __init__(self, df):
        self.df = df

    def candles(self, start, end):
        return self.df[(self.df.index >= start) & (self.df.index < end)]

    def quote(self):
        raise RuntimeError("no quotes in replay")


def pick_source():
    for src in (Delta(), Binance(), Coinbase()):
        try:
            end = now_utc().floor(S.TF)
            d = src.candles(end - BAR * 20, end)
            src.quote()
            if len(d) >= 10:
                log(f"data source: {src.name}")
                return src
        except Exception as e:                                        # noqa: BLE001 - try the next one
            log(f"source {src.name} unavailable: {type(e).__name__}: {str(e)[:120]}")
    raise SystemExit("no market data source reachable")


# ---------------------------------------------------------------- model

class Model:
    def __init__(self):
        with open(os.path.join(HERE, "model", "meta.json")) as fh:
            self.meta = json.load(fh)
        self.booster = lgb.Booster(model_file=os.path.join(HERE, "model", "model.txt"))
        self.cut = self.meta["cut"]

    def score(self, df, cands):
        F = S.features(df, cands)[self.meta["features"]]
        return self.booster.predict(F.to_numpy(dtype=float))


# ---------------------------------------------------------------- paper accounts

class Account:
    def __init__(self, name, d=None):
        self.name = name
        d = d or {}
        self.pos = d.get("pos")              # open position (candle model)
        self.pending = d.get("pending")      # entry decided at a bar close, fills at the next open
        self.equity = d.get("equity", S.CAPITAL_INR)          # INR, candle model
        self.equity_live = d.get("equity_live", S.CAPITAL_INR)  # INR, live-quote model

    def to_dict(self):
        return dict(pos=self.pos, pending=self.pending, equity=self.equity, equity_live=self.equity_live)


class Bot:
    def __init__(self, src, model, mode="live", commit=False):
        self.src, self.model, self.mode, self.commit = src, model, mode, commit
        os.makedirs(STATE, exist_ok=True)
        self.state_path = os.path.join(STATE, "state.json")
        st = {}
        if os.path.exists(self.state_path) and mode == "live":
            with open(self.state_path) as fh:
                st = json.load(fh)
        self.accounts = {k: Account(k, st.get("accounts", {}).get(k)) for k in ("ml", "base")}
        self.last_bar = pd.Timestamp(st["last_bar"]) if st.get("last_bar") else None
        self.shadows = st.get("shadows", [])     # live-quote twins of open trades
        if self.last_bar is not None and now_utc() - self.last_bar > pd.Timedelta("1D"):
            for a in self.accounts.values():         # a crashed run left a position open: drop it
                if a.pos is not None or a.pending is not None:
                    log(f"{a.name.upper()}: dropping stale open position from {self.last_bar}")
                a.pos = a.pending = None
        self.df = None
        self.trades_path = os.path.join(STATE, "trades.csv" if mode == "live" else "replay_trades.csv")
        self.signals_path = os.path.join(STATE, "signals.csv" if mode == "live" else "replay_signals.csv")
        self.closed = 0
        self.flush_shadows()                         # live tracking cannot resume across runs

    # ---------- persistence
    def save(self):
        if self.mode != "live":
            return
        st = dict(last_bar=str(self.last_bar) if self.last_bar is not None else None, source=self.src.name,
                  updated=str(now_utc())[:19], accounts={k: a.to_dict() for k, a in self.accounts.items()},
                  shadows=self.shadows)
        with open(self.state_path, "w") as fh:
            json.dump(st, fh, indent=1, default=str)

    def append(self, path, row, cols=None):
        d = pd.DataFrame([row], columns=cols or list(row))
        d.to_csv(path, mode="a", header=not os.path.exists(path), index=False)

    # ---------- data
    def load_window(self, end):
        start = end - WARMUP
        if self.last_bar is not None and self.last_bar > start:
            start = min(start, self.last_bar - WARMUP)
        self.df = self.src.candles(start, end)
        self.df = self.df[self.df.index + BAR <= now_utc()] if self.mode == "live" else self.df

    def refresh(self):
        """Fetch the latest closed bars and merge them into the window."""
        end = now_utc().floor(S.TF)
        new = self.src.candles(end - BAR * 12, end + BAR)
        new = new[new.index + BAR <= now_utc()]
        self.df = pd.concat([self.df, new])
        self.df = self.df[~self.df.index.duplicated(keep="last")].sort_index()
        self.df = self.df[self.df.index >= self.df.index[-1] - WARMUP - pd.Timedelta("2D")]

    # ---------- core: one closed bar (same order of operations as the backtest engine)
    def on_bar(self, i, live):
        df = self.df
        t = df.index[i]
        O, H, L = df.open.iat[i], df.high.iat[i], df.low.iat[i]
        for a in self.accounts.values():
            p = a.pos
            if p is not None and p.get("close_next"):                         # time exit at this open
                self.close(a, i, O, 4)
            if a.pending is not None:                                           # entry fill at this open
                a.pos = dict(a.pending, entry_time=str(t), entry_px=O, i0=str(t))
                a.pending = None
            p = a.pos
            if p is not None:
                d, stop, tgt = p["dir"], p["stop"], p["tgt"]
                px, how = None, None
                if d == 1:
                    if O <= stop:
                        px, how = O, 2
                    elif O >= tgt:
                        px, how = O, 3
                    elif L <= stop and H >= tgt:
                        px, how = (tgt, 1) if (H - O) < (O - L) else (stop, 0)
                    elif L <= stop:
                        px, how = stop, 0
                    elif H >= tgt:
                        px, how = tgt, 1
                else:
                    if O >= stop:
                        px, how = O, 2
                    elif O <= tgt:
                        px, how = O, 3
                    elif H >= stop and L <= tgt:
                        px, how = (tgt, 1) if (O - L) < (H - O) else (stop, 0)
                    elif H >= stop:
                        px, how = stop, 0
                    elif L <= tgt:
                        px, how = tgt, 1
                if px is not None:
                    self.close(a, i, px, how)
                elif (t - pd.Timestamp(p["i0"])) // BAR >= S.CAP:
                    p["close_next"] = True                                      # exit at the next open
        # bar close: new signals
        self.new_signals(i, live)
        self.last_bar = t
        if live:
            self.shadow_timeouts(t + BAR)
        self.save()

    def new_signals(self, i, live):
        df = self.df
        t = df.index[i]
        if S.session_of(np.array([t.hour]))[0] == "" or t.dayofweek not in S.WEEKDAYS:
            return
        c = S.candidates(df.iloc[: i + 1])
        c = c[(c.bar == i) & c.strategy]
        if c.empty:
            return
        c = c.copy()
        c["p"] = self.model.score(df.iloc[: i + 1], c)
        quote, latency = (None, np.nan)
        if live:
            try:
                quote = self.src.quote()
                latency = (now_utc() - (t + BAR)).total_seconds()
            except Exception as e:                                            # noqa: BLE001
                log(f"quote failed: {e}")
        taken = {}
        for name, a in self.accounts.items():
            if a.pos is not None or a.pending is not None:
                continue
            ok = c if name == "base" else c[c.p >= self.model.cut]
            if ok.empty:
                continue
            r = ok.iloc[0]
            n = S.contracts(a.equity, r.e, r.sl)
            if n == 0:
                log(f"{name.upper():4s} skip: 1 lot would risk more than {S.MAX_RISK_PCT}% of Rs {a.equity:,.0f} "
                    f"or exceed {S.LEVERAGE}x {S.MARGIN_MODE} margin")
                continue
            a.pending = dict(signal_time=str(t), session=r.session, dir=int(r.dir), kind=S.PRIO[int(r.prio)],
                             p=float(r.p), cut=self.model.cut, e=float(r.e), sl=float(r.sl),
                             stop=float(S.round_tick(r.stop)), tgt=float(S.round_tick(r.tgt)), contracts=n,
                             margin_inr=n * S.CONTRACT_BTC * float(r.e) / S.LEVERAGE * S.USDINR,
                             liq_dist_pct=S.liq_distance(n, float(r.e), a.equity) * 100,
                             latency_s=latency,
                             mode=self.mode if not (self.mode == "live" and not live) else "catchup")
            taken[name] = int(r.prio)
            if quote is not None:
                bid, ask, _ = quote
                a.pending["live_entry"] = ask if r.dir == 1 else bid
                self.shadows.append(dict(account=name, signal_time=str(t), dir=int(r.dir),
                                         stop=a.pending["stop"], tgt=a.pending["tgt"], entry=a.pending["live_entry"],
                                         deadline=str(t + BAR * (S.CAP + 2)), exit=None, how=None))
            log(f"{name.upper():4s} {'LONG' if r.dir == 1 else 'SHORT'} {n} lot(s) @ {S.LEVERAGE}x {S.MARGIN_MODE} "
                f"(margin Rs {a.pending['margin_inr']:,.0f}, liq {a.pending['liq_dist_pct']:.1f}% away) {S.PRIO[int(r.prio)]} "
                f"p={r.p:.3f} (cut {self.model.cut:.3f}) stop {a.pending['stop']:.1f} tp {a.pending['tgt']:.1f} "
                f"[{r.session}, {ist(t + BAR)}]")
        for r in c.itertuples():
            self.append(self.signals_path, dict(time=str(t), session=r.session, kind=S.PRIO[int(r.prio)],
                                                dir=int(r.dir), p=round(float(r.p), 4), cut=round(self.model.cut, 4),
                                                ml_pass=bool(r.p >= self.model.cut),
                                                taken_ml=taken.get("ml") == int(r.prio),
                                                taken_base=taken.get("base") == int(r.prio)))

    def close(self, a, i, px, how):
        p = a.pos
        t = self.df.index[i]
        R = p["dir"] * (px - p["entry_px"]) / abs(p["e"] - p["sl"])
        sh = next((s for s in self.shadows if s["account"] == a.name and s["signal_time"] == p["signal_time"]), None)
        live_exit, live_how, R_live = np.nan, None, np.nan
        if sh is not None and sh.get("exit") is not None:
            live_exit, live_how = sh["exit"], sh["how"]
            R_live = p["dir"] * (live_exit - sh["entry"]) / abs(p["e"] - p["sl"])
            self.shadows.remove(sh)
        elif sh is not None:
            sh["candle_closed"] = dict(exit_time=str(t), px=px, how=how, R=R)   # finish when the shadow exits
        n = p["contracts"]
        pl = S.pnl_inr(n, p["dir"], p["entry_px"], px)
        a.equity += pl
        row = dict(account=a.name, mode=p.get("mode", self.mode), source=self.src.name, session=p["session"],
                   signal_time=p["signal_time"], entry_time=p["entry_time"], exit_time=str(t), dir=p["dir"],
                   kind=p["kind"], p=p["p"], cut=p["cut"], e=p["e"], sl=p["sl"], stop=p["stop"], tgt=p["tgt"],
                   entry_px=p["entry_px"], exit_px=px, how=S.HOW[how], R=R, live_entry=p.get("live_entry", np.nan),
                   live_exit=live_exit, live_how=live_how, R_live=R_live, latency_s=p.get("latency_s", np.nan),
                   contracts=n, pnl_inr=pl, equity_inr=a.equity, pnl_inr_live=np.nan, equity_inr_live=np.nan,
                   leverage=f"{S.LEVERAGE}x {S.MARGIN_MODE}", margin_inr=p.get("margin_inr", np.nan),
                   liq_dist_pct=p.get("liq_dist_pct", np.nan))
        if not np.isnan(R_live):
            self.book_live(a, row, sh["entry"], live_exit)
        if sh is None or sh.get("exit") is not None or not np.isnan(R_live) or "live_entry" not in p:
            self.append(self.trades_path, row, TRADE_COLS)
            self.closed += 1
        else:
            sh["row"] = row
        a.pos = None
        log(f"{a.name.upper():4s} closed {S.HOW[how]} R={R:+.2f} Rs {pl:+.2f} -> Rs {a.equity:,.2f} [{ist(t)}]")

    # ---------- live-quote shadows
    def poll_shadows(self):
        open_ = [s for s in self.shadows if s.get("exit") is None]
        if not open_:
            return
        try:
            bid, ask, last = self.src.quote()
        except Exception:                                                     # noqa: BLE001
            return
        for s in open_:
            d = s["dir"]
            if (d == 1 and last <= s["stop"]) or (d == -1 and last >= s["stop"]):
                s["exit"], s["how"] = (bid if d == 1 else ask), "stop"
            elif (d == 1 and last >= s["tgt"]) or (d == -1 and last <= s["tgt"]):
                s["exit"], s["how"] = s["tgt"], "target"
            if s.get("exit") is not None:
                self.finish_shadow(s)

    def shadow_timeouts(self, now_bar):
        """Time exit for live twins at the same moment as the candle model (open of entry + 5 bars)."""
        due = [s for s in self.shadows if s.get("exit") is None and now_bar >= pd.Timestamp(s["deadline"])]
        if not due:
            return
        try:
            bid, ask, _ = self.src.quote()
        except Exception:                                                     # noqa: BLE001
            return
        for s in due:
            s["exit"], s["how"] = (bid if s["dir"] == 1 else ask), "time exit"
            self.finish_shadow(s)

    def finish_shadow(self, s):
        if "row" in s:                                                        # candle side already closed
            row = s["row"]
            row["live_exit"], row["live_how"] = s["exit"], s["how"]
            row["R_live"] = row["dir"] * (s["exit"] - s["entry"]) / abs(row["e"] - row["sl"])
            self.book_live(self.accounts[row["account"]], row, s["entry"], s["exit"])
            self.append(self.trades_path, row, TRADE_COLS)
            self.closed += 1
            self.shadows.remove(s)
            self.publish(f"trade closed ({row['account']})")

    def flush_shadows(self):
        """Write out trades still waiting on their live twin (live exit unknown) and drop all twins."""
        for s in self.shadows:
            if "row" in s:
                self.append(self.trades_path, s["row"], TRADE_COLS)
                self.closed += 1
        self.shadows = []

    def book_live(self, a, row, entry, exit_):
        pl = S.pnl_inr(row["contracts"], row["dir"], entry, exit_)
        a.equity_live += pl
        row["pnl_inr_live"], row["equity_inr_live"] = pl, a.equity_live

    # ---------- publishing
    def publish(self, why):
        if self.mode != "live":
            return
        self.save()
        try:
            import report
            report.build()
        except Exception as e:                                                # noqa: BLE001
            log(f"report failed: {e}")
        if self.commit:
            git_commit(f"forward test: {why} [{now_utc().tz_convert(S.IST):%d %b %H:%M} IST]")

    # ---------- loops
    def catch_up(self, until):
        """Process every closed bar after last_bar (or from `until` - lookback) without live quotes."""
        idx = self.df.index
        start = self.last_bar if self.last_bar is not None and now_utc() - self.last_bar < pd.Timedelta("1D") \
            else None
        todo = np.flatnonzero((idx > start) & (idx <= until)) if start is not None else np.array([], int)
        if len(todo):
            log(f"catching up {len(todo)} bars from candles ({idx[todo[0]]} -> {idx[todo[-1]]})")
        for i in todo:
            self.on_bar(i, live=False)
        if start is None and len(idx):
            self.last_bar = idx[idx <= until][-1]

    def run_live(self, sess_end, minutes=None):
        hard_stop = sess_end + pd.Timedelta("35min")
        if minutes:
            hard_stop = min(hard_stop, now_utc() + pd.Timedelta(minutes=minutes))
        for attempt in range(10):
            try:
                self.load_window(now_utc().floor(S.TF))
                break
            except Exception as e:                                            # noqa: BLE001
                log(f"history fetch failed ({attempt + 1}/10): {e}")
                time.sleep(30)
        else:
            raise SystemExit("could not load candle history")
        self.catch_up(self.df.index[-1])
        before = self.closed
        while True:
            nxt = self.last_bar + 2 * BAR                   # close time of the next bar
            while now_utc() < nxt + pd.Timedelta("2s"):
                self.poll_shadows()
                time.sleep(POLL_S if any(s.get("exit") is None for s in self.shadows) else 1.0)
            for attempt in range(20):
                try:
                    self.refresh()
                except Exception as e:                                        # noqa: BLE001
                    log(f"candle fetch failed: {e}")
                if self.df.index[-1] >= self.last_bar + BAR:
                    break
                time.sleep(3)
            new = np.flatnonzero(self.df.index > self.last_bar)
            for k, i in enumerate(new):
                self.on_bar(i, live=(k == len(new) - 1))
            if self.closed > before:
                before = self.closed
                self.publish("trade closed")
            flat = all(a.pos is None and a.pending is None for a in self.accounts.values())
            done = self.last_bar + BAR >= sess_end + pd.Timedelta("30min") and flat and \
                not any(s.get("exit") is None for s in self.shadows)
            if done or now_utc() > hard_stop:
                break
        self.flush_shadows()
        self.publish("session end")


def git_commit(msg):
    cmds = [["git", "add", "state", "README.md", "charts"], ["git", "commit", "-q", "-m", msg]]
    for c in cmds:
        subprocess.run(c, cwd=HERE, check=False)
    for _ in range(3):
        if subprocess.run(["git", "pull", "-q", "--rebase", "--autostash", "-X", "theirs"], cwd=HERE).returncode == 0 and \
                subprocess.run(["git", "push", "-q"], cwd=HERE).returncode == 0:
            return
        time.sleep(5)
    log("git push failed")


def session_window(name, now):
    lo, hi = S.SESSIONS[name]
    day = now.normalize()
    return day + pd.Timedelta(hours=lo), day + pd.Timedelta(hours=hi)


def pick_session(now):
    return "London" if now.hour < 12 or (now.hour == 12 and now.minute < 30) else "New York"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", choices=["london", "newyork", "auto"], default="auto")
    ap.add_argument("--replay", nargs=2, metavar=("START", "END"))
    ap.add_argument("--commit", action="store_true", help="git commit + push results (GitHub Actions)")
    ap.add_argument("--minutes", type=float, help="stop after N minutes (testing)")
    a = ap.parse_args()
    model = Model()

    if a.replay:
        import train
        hist = train.load_history()
        t0, t1 = pd.Timestamp(a.replay[0], tz="UTC"), pd.Timestamp(a.replay[1], tz="UTC")
        bot = Bot(Replay(hist), model, mode="replay")
        for p in (bot.trades_path, bot.signals_path):
            if os.path.exists(p):
                os.remove(p)
        bot.df = hist[(hist.index >= t0 - WARMUP) & (hist.index < t1)]
        for i in np.flatnonzero(bot.df.index >= t0):
            bot.on_bar(i, live=False)
        tr = pd.read_csv(bot.trades_path) if os.path.exists(bot.trades_path) else pd.DataFrame()
        for acc, g in tr.groupby("account"):
            s = S.stats(g.R)
            print(f"{acc}: {s['n']} trades, win {s['win']:.1f}%, avg R {s['avgR']:+.3f}")
        return

    now = now_utc()
    if now.dayofweek not in S.WEEKDAYS:
        log(f"{now.tz_convert(S.IST):%A} - no trading on weekends")
        return
    name = pick_session(now) if a.session == "auto" else {"london": "London", "newyork": "New York"}[a.session]
    s0, s1 = session_window(name, now)
    log(f"{name} session: {ist(s0)} -> {ist(s1)} (entries), model cut {model.cut:.3f}")
    if now > s1 + pd.Timedelta("30min"):
        log("session already over")
        return
    wait = (s0 - pd.Timedelta("10min") - now).total_seconds()
    if wait > 0:
        log(f"waiting {wait / 60:.0f} min for the session")
        time.sleep(wait)
    bot = Bot(pick_source(), model, commit=a.commit)
    t_start = now_utc()
    bot.run_live(s1, a.minutes)
    with open(os.path.join(STATE, "runs.csv"), "a") as fh:
        if fh.tell() == 0:
            fh.write("session,start_utc,end_utc,source,late_min\n")
        fh.write(f"{name},{t_start:%Y-%m-%d %H:%M},{now_utc():%Y-%m-%d %H:%M},{bot.src.name},"
                 f"{max(0, (t_start - s0).total_seconds() / 60):.0f}\n")
    bot.publish("session end")


if __name__ == "__main__":
    main()
