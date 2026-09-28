"""Exercise live.py against a fake Delta exchange (no network, no real orders).

    python tests/test_live_mock.py

Scenarios: stop hit at the exchange, target hit, 25-minute time exit, missing bracket -> position closed,
test-mode risk cap, 3-trades-a-day limit, foreign position blocks entries, dry run sends nothing,
restart with an open position, and the bot.py hook (every live ML signal reaches LiveTrader.enter).
"""
import os
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
TMP = tempfile.mkdtemp()
os.environ.update(FT_STATE=os.path.join(TMP, "paper"), DELTA_API_KEY="k", DELTA_API_SECRET="s",
                  DELTA_FORCE_IPV4="no", TEST_MODE="yes", LIVE_TRADING="yes")

import live  # noqa: E402

live.LOG_DIR = os.path.join(TMP, "logs")
live.STATE_DIR = os.path.join(TMP, "live_state")
live.public_ips = lambda: {"ipv4": "test", "ipv6": "test"}
live.time.sleep = lambda s: None                                  # no waiting in tests
CLOCK = [time.time()]
live.time.time = lambda: CLOCK[0]


class FakeDelta:
    """Tiny exchange: market orders fill at the quote, brackets become stop orders that trigger on tick()."""

    def __init__(self, balance=1.34, price=84000.0, honour_brackets=True, bracket_endpoint=True):
        self.bal, self.px, self.size, self.entry = balance, price, 0, None
        self.orders, self.fill_log, self.placed, self.lev = [], [], [], 200
        self.honour, self.bracket_ok = honour_brackets, bracket_endpoint

    # reads
    def balances(self):
        return [{"asset_symbol": "USD", "balance": str(self.bal), "available_balance": str(self.bal)}]

    def position(self):
        return {"size": self.size, "entry_price": self.entry}

    def open_orders(self):
        return list(self.orders)

    def order_history(self, n=30):
        return []

    def fills(self, n=50):
        return list(reversed(self.fill_log))

    def leverage(self):
        return {"leverage": str(self.lev)}

    def product(self):
        return {"contract_value": "0.001", "tick_size": "0.5"}

    def ticker(self):
        return self.px - 0.5, self.px, self.px

    # writes
    def set_leverage(self, lev):
        self.lev = float(lev)

    def _fill(self, side, n, price):
        d = 1 if side == "buy" else -1
        if self.size and (self.size > 0) != (d > 0):                   # closing
            self.bal += abs(n) * 0.001 * (price - self.entry) * (1 if self.size > 0 else -1)
        else:
            self.entry = price
        self.size += d * n
        self.fill_log.append({"side": side, "size": n, "price": str(price), "commission": "0",
                              "created_at": int(CLOCK[0] * 1e6)})

    def place(self, body):
        self.placed.append(body)
        if body.get("reduce_only") and not self.size:
            return {"id": len(self.placed), "size": 0, "unfilled_size": 0}
        px = self.px if body["side"] == "buy" else self.px - 0.5
        self._fill(body["side"], body["size"], px)
        if self.honour and "bracket_stop_loss_price" in body:
            self.orders = [{"id": 90, "stop_order_type": "stop_loss_order", "stop_price": body["bracket_stop_loss_price"]},
                           {"id": 91, "stop_order_type": "take_profit_order", "stop_price": body["bracket_take_profit_price"]}]
        if not self.size:
            self.orders = []
        return {"id": len(self.placed), "size": body["size"], "unfilled_size": 0, "average_fill_price": str(px),
                "client_order_id": body.get("client_order_id")}

    def bracket(self, body):
        if not self.bracket_ok:
            raise live.DeltaError(400, "bracket_rejected", body)
        self.orders = [{"id": 92, "stop_order_type": "stop_loss_order", "stop_price": body["stop_loss_order"]["stop_price"]}]

    def cancel_all(self):
        self.orders = []

    def tick(self, price):
        """Move the market; trigger bracket legs like Delta would."""
        self.px = price
        for o in list(self.orders):
            sp = float(o["stop_price"])
            long_ = self.size > 0
            hit = (o["stop_order_type"] == "stop_loss_order" and ((long_ and price <= sp) or (not long_ and price >= sp))) or \
                  (o["stop_order_type"] == "take_profit_order" and ((long_ and price >= sp) or (not long_ and price <= sp)))
            if hit and self.size:
                self._fill("sell" if long_ else "buy", abs(self.size), price)
                self.orders = []


def trader(ex, armed=True):
    os.environ["LIVE_TRADING"] = "yes" if armed else "no"
    for f in ("live.json",):
        p = os.path.join(live.STATE_DIR, f)
        if os.path.exists(p):
            os.remove(p)
    t = live.LiveTrader()
    t.api = ex
    return t


def sig(d=1, px=84000.0, stop_dist=200.0):
    sl = px - d * stop_dist
    return dict(dir=d, e=px, sl=sl, stop=sl - d * 1, tgt=px + d * 2 * stop_dist - d * 1,
                signal_time=f"2026-09-29 08:{int(CLOCK[0]) % 60:02d}:00+00:00", kind="normal long", p=0.5,
                session="London")


def trades():
    import pandas as pd
    p = os.path.join(live.LOG_DIR, "live_trades.csv")
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()


checks = []


def check(name, cond):
    checks.append((name, bool(cond)))
    print(("PASS " if cond else "FAIL ") + name)


# 1. stop hit at the exchange
ex = FakeDelta(); t = trader(ex)
check("preflight passes and sets 100x in test mode", t.preflight() and ex.lev == 100)
t.enter(sig())
check("entry: 1 lot market buy with bracket", len(ex.placed) == 1 and ex.placed[0]["size"] == 1 and
      "bracket_stop_loss_price" in ex.placed[0] and ex.size == 1)
ex.tick(83700); CLOCK[0] += 3; t.poll()
tr = trades()
check("stop at Delta detected and logged", t.pos is None and len(tr) == 1 and tr.exit_reason.iloc[-1].startswith("stop"))
check("loss booked ~ -1R", tr.R.iloc[-1] < -0.9)

# 2. target hit
CLOCK[0] += 60; ex.tick(84000); t.enter(sig()); ex.tick(84500); CLOCK[0] += 3; t.poll()
check("target at Delta detected", t.pos is None and trades().R.iloc[-1] > 1.5)

# 3. time exit after 25 min
CLOCK[0] += 60; ex.tick(84000); t.enter(sig()); ex.tick(84050); CLOCK[0] += 25 * 60 + 5; t.poll(force=True)
tr = trades()
check("time exit closes with reduce-only market order", t.pos is None and ex.size == 0 and
      ex.placed[-1].get("reduce_only") is True and tr.exit_reason.iloc[-1] == "time exit")

# 4. daily limit: 3 trades done -> 4th skipped
CLOCK[0] += 60; ex.tick(84000); n0 = len(ex.placed); t.enter(sig())
check("4th trade of the day skipped (test limit 3)", len(ex.placed) == n0 and t.pos is None)

# 5. missing bracket + bracket endpoint fails -> position closed immediately
ex = FakeDelta(honour_brackets=False, bracket_endpoint=False); t = trader(ex); t.preflight()
t.enter(sig())
check("unprotected position is closed at once", t.pos is None and ex.size == 0 and
      trades().exit_reason.iloc[-1] == "unprotected")

# 6. missing bracket but bracket endpoint works -> stays open, protected
ex = FakeDelta(honour_brackets=False); t = trader(ex); t.preflight(); t.enter(sig())
check("missing bracket gets attached", t.pos is not None and any(o["id"] == 92 for o in ex.orders))
t.shutdown()
check("shutdown flattens an open position", t.pos is None and ex.size == 0)

# 2b. entry refused when price is already beyond the stop
ex = FakeDelta(); t = trader(ex); t.preflight(); ex.tick(83700); t.enter(sig())
check("entry refused when price already beyond the stop", not ex.placed and t.pos is None)

# 7. risk cap: 1 lot with a 600$ stop risks $0.60 = 45% of $1.34 -> skip
ex = FakeDelta(); t = trader(ex); t.preflight(); t.enter(sig(stop_dist=600))
check("trade skipped when 1 lot risks > 35%", not ex.placed and t.pos is None)

# 8. margin: $0.50 balance cannot margin 1 lot at 100x ($0.84)
ex = FakeDelta(balance=0.5); t = trader(ex); t.preflight(); t.enter(sig())
check("trade skipped when margin is short", not ex.placed)

# 9. foreign position blocks entries
ex = FakeDelta(); ex.size, ex.entry = 2, 83000; t = trader(ex); t.preflight(); t.enter(sig())
check("foreign position blocks new entries", t.blocked == "foreign position" and not ex.placed)

# 10. dry run sends nothing, still simulates the round trip
ex = FakeDelta(); t = trader(ex, armed=False); t.preflight(); t.enter(sig())
check("dry run: no order sent, leverage untouched", not ex.placed and ex.lev == 200 and t.pos and t.pos["dry"])
ex.tick(83700); CLOCK[0] += 3; t.poll()
check("dry run: stop simulated from quotes", t.pos is None and trades().dry.iloc[-1])

# 11. restart with an open position resumes management
ex = FakeDelta(); t = trader(ex); t.preflight(); t.enter(sig())
t2 = live.LiveTrader(); t2.api = ex; t2.preflight()
check("restart resumes the open position", t2.pos is not None and t2.blocked is None)
ex.tick(83700); CLOCK[0] += 3; t2.poll()
check("resumed position closes normally", t2.pos is None)

# 12. bot.py hook: every live ML signal reaches LiveTrader.enter
import bot  # noqa: E402
import train  # noqa: E402


class Recorder:
    def __init__(self):
        self.calls = []

    def enter(self, s):
        self.calls.append(s)

    def poll(self, force=False):
        pass

    def flat(self):
        return True

    def shutdown(self):
        pass


hist = train.load_history()
import pandas as pd  # noqa: E402

t0, t1 = pd.Timestamp("2026-09-14", tz="UTC"), pd.Timestamp("2026-09-19", tz="UTC")
rec = Recorder()
b = bot.Bot(bot.Replay(hist), bot.Model(), mode="replay", live=rec)
b.df = hist[(hist.index >= t0 - bot.WARMUP) & (hist.index < t1)]
b.src.quote = lambda: (0.0, 0.0, 0.0)
for i in __import__("numpy").flatnonzero(b.df.index >= t0):
    b.on_bar(i, live=True)
import strategy as S  # noqa: E402

c = S.candidates(b.df)
c = c[c.strategy & (c.time >= t0)].copy()
c["p"] = b.model.score(b.df, c)
expected = c[c.p >= b.model.cut].groupby("bar").size().shape[0]
check(f"bot.py sends every ML signal bar to the live trader ({len(rec.calls)} calls, expected {expected})",
      len(rec.calls) == expected and expected > 0)

print(f"\n{sum(ok for _, ok in checks)}/{len(checks)} checks passed")
sys.exit(0 if all(ok for _, ok in checks) else 1)
