"""Live trading on Delta Exchange India (BTCUSD perpetual) for the ML account of bot.py.

Runs on YOUR machine (Delta India API keys with trading permission only work from whitelisted IPs).

    python live.py check                 # IPs, API auth, balance, leverage, open position/orders. Sends nothing.
    python live.py flatten --yes         # emergency: cancel all BTCUSD orders and close the position
    python bot.py --live --all-day       # trade today's London + New York sessions (see run_live.bat)

Real orders are sent only when LIVE_TRADING=yes (environment or .env). Otherwise --live is a dry run: every
check and API read is real, orders are logged as "DRY RUN" and simulated from live quotes.

Safety rails
  * every entry carries an exchange-side bracket: stop-loss (market on trigger) + take-profit (market on
    trigger), so a crashed PC or lost internet still leaves the stop working at Delta
  * if the bracket is missing after the fill, a separate bracket is attached; if that fails, the position is
    closed immediately (never left unprotected)
  * leverage must read back as strategy.LEVERAGE (50x) or no orders are sent
  * no new entry while any BTCUSD position exists that this bot did not open
  * max lots per trade (MAX_LOTS, default 5) and a daily loss stop (DAILY_LOSS_PCT of the day's starting
    balance, default 5%)
  * time exit 25 min after entry (reduce-only market order), same rule as the backtest and the paper bot
  * each entry has a unique client_order_id; a failed request is reconciled against order history before
    anything is retried, so a network error cannot double an order

Logs (all git-ignored)
  logs/live_YYYY-MM-DD.log   every API call (no secrets), every decision, errors with tracebacks; IST times
  logs/live_orders.csv       every order sent and Delta's reply
  logs/live_trades.csv       each round trip: fills, fees, slippage vs signal, R, P&L in USD and INR,
                             wallet balance before/after
  live_state/live.json       the open position (lets a restarted bot resume managing it)
"""
import argparse
import csv
import hashlib
import hmac
import json
import logging
import os
import socket
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone

import requests
import urllib3.util.connection

import strategy as S

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(HERE, "logs")
STATE_DIR = os.path.join(HERE, "live_state")
PRODUCT_ID, SYMBOL = 27, "BTCUSD"
HOLD_S = 25 * 60                       # time exit: 25 min after entry (fee-free rule is < 30 min)
POLL_S = 2.0
S_IST = timezone(timedelta(hours=5, minutes=30))
# TEST_MODE=yes: a tiny account (e.g. Rs 100) that cannot meet the 1-2% risk rule. Trades exactly 1 lot,
# 100x cross margin, at most TEST_MAX_TRADES a day, skips a trade only if 1 lot risks more than
# TEST_MAX_RISK_PCT of the balance or the liquidation guard fails; no daily loss stop. Purpose: check
# orders, brackets, fills, Delta's fees and slippage with real money, not to make money.
TEST = dict(leverage=100, max_trades=3, max_risk_pct=35.0)
log = logging.getLogger("live")


# ---------------------------------------------------------------- config & logging

def load_env():
    """Read KEY=VALUE lines from .env in this folder and its parent (no extra dependency)."""
    for path in (os.path.join(HERE, ".env"), os.path.join(os.path.dirname(HERE), ".env")):
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def cfg(name, default):
    v = os.environ.get(name)
    return type(default)(v) if v not in (None, "") else default


def force_ipv4():
    """Use IPv4 for every request, so only one (IPv4) address needs whitelisting at Delta."""
    urllib3.util.connection.allowed_gai_family = lambda: socket.AF_INET


class ISTFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        t = datetime.fromtimestamp(record.created, timezone.utc).astimezone(S_IST)
        return t.strftime("%Y-%m-%d %H:%M:%S") + f".{int(record.msecs):03d} IST"



def setup_logging():
    if log.handlers:
        return
    os.makedirs(LOG_DIR, exist_ok=True)
    log.setLevel(logging.DEBUG)
    fmt = ISTFormatter("%(asctime)s %(levelname)-7s %(message)s")
    day = datetime.now(S_IST).strftime("%Y-%m-%d")
    fh = logging.FileHandler(os.path.join(LOG_DIR, f"live_{day}.log"), encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(ch)


def append_csv(name, row):
    os.makedirs(LOG_DIR, exist_ok=True)
    path = os.path.join(LOG_DIR, name)
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def now_ist():
    return datetime.now(S_IST).strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------- REST client

class DeltaError(Exception):
    def __init__(self, status, code, body):
        super().__init__(f"{status} {code}: {str(body)[:300]}")
        self.status, self.code, self.body = status, code, body


class Delta:
    def __init__(self, key, secret, base_url):
        self.key, self.secret, self.base = key, secret, base_url.rstrip("/")
        self.http = requests.Session()
        self.http.headers.update({"User-Agent": "ema915-live/1.0 python-requests",
                                  "Accept": "application/json", "Content-Type": "application/json"})

    def _headers(self, method, path, qs, body):
        ts = str(int(time.time()))
        sig = hmac.new(self.secret.encode(), (method + ts + path + qs + body).encode(), hashlib.sha256).hexdigest()
        return {"api-key": self.key, "timestamp": ts, "signature": sig}

    def request(self, method, path, params=None, body=None, auth=True, retries=None):
        """GETs are retried on network/5xx/429 errors; writes are never retried here (see LiveTrader)."""
        method = method.upper()
        qs = ("?" + "&".join(f"{k}={v}" for k, v in params.items())) if params else ""
        data = json.dumps(body, separators=(",", ":")) if body is not None else ""
        retries = (4 if method == "GET" else 1) if retries is None else retries
        for attempt in range(1, retries + 1):
            t0 = time.time()
            try:
                h = self._headers(method, path, qs, data) if auth else {}
                r = self.http.request(method, self.base + path + qs, data=data or None, headers=h, timeout=10)
            except requests.RequestException as e:
                log.warning(f"API {method} {path}{qs} network error ({attempt}/{retries}): {e}")
                if attempt == retries:
                    raise
                time.sleep(attempt)
                continue
            ms = (time.time() - t0) * 1000
            try:
                js = r.json()
            except ValueError:
                js = {"success": False, "error": {"code": "non_json", "text": r.text[:200]}}
            log.debug(f"API {method} {path}{qs} body={data or '-'} -> {r.status_code} in {ms:.0f} ms: "
                      f"{json.dumps(js)[:600]}")
            if r.status_code == 429 or (r.status_code >= 500 and attempt < retries):
                wait = int(r.headers.get("X-RATE-LIMIT-RESET", 1000)) / 1000 if r.status_code == 429 else attempt
                log.warning(f"API {method} {path} -> {r.status_code}, retry in {wait:.1f}s")
                time.sleep(wait)
                continue
            if not js.get("success", False):
                err = js.get("error", {})
                code = err.get("code", "unknown") if isinstance(err, dict) else str(err)
                raise DeltaError(r.status_code, code, js)
            return js.get("result")
        raise DeltaError(0, "retries_exhausted", f"{method} {path}")

    # ---- reads
    def balances(self):
        return self.request("GET", "/v2/wallet/balances")

    def position(self):
        return self.request("GET", "/v2/positions", {"product_id": PRODUCT_ID}) or {}

    def open_orders(self):
        return self.request("GET", "/v2/orders", {"product_ids": PRODUCT_ID, "states": "open"}) or []

    def order_history(self, n=30):
        return self.request("GET", "/v2/orders/history", {"product_ids": PRODUCT_ID, "page_size": n}) or []

    def fills(self, n=50):
        return self.request("GET", "/v2/fills", {"product_ids": PRODUCT_ID, "page_size": n}) or []

    def leverage(self):
        return self.request("GET", f"/v2/products/{PRODUCT_ID}/orders/leverage")

    def product(self):
        return self.request("GET", f"/v2/products/{SYMBOL}", auth=False)

    def ticker(self):
        r = self.request("GET", f"/v2/tickers/{SYMBOL}", auth=False)
        q = r.get("quotes") or {}
        return float(q["best_bid"]), float(q["best_ask"]), float(r["close"])

    # ---- writes
    def set_leverage(self, lev):
        return self.request("POST", f"/v2/products/{PRODUCT_ID}/orders/leverage", body={"leverage": str(lev)})

    def place(self, body):
        return self.request("POST", "/v2/orders", body=body)

    def bracket(self, body):
        return self.request("POST", "/v2/orders/bracket", body=body)

    def cancel_all(self):
        return self.request("DELETE", "/v2/orders/all",
                            body={"product_id": PRODUCT_ID, "cancel_limit_orders": True, "cancel_stop_orders": True})


# ---------------------------------------------------------------- helpers

def pos_size(p):
    try:
        return int(float((p or {}).get("size") or 0))
    except (TypeError, ValueError):
        return 0


def ts_seconds(v):
    """Delta timestamps come as ISO strings or integer micro/milliseconds."""
    if v is None:
        return 0.0
    if isinstance(v, (int, float)) or (isinstance(v, str) and v.isdigit()):
        v = float(v)
        return v / 1e6 if v > 1e14 else v / 1e3 if v > 1e11 else v
    try:
        return datetime.fromisoformat(str(v).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def usd_balance(rows):
    rows = rows if isinstance(rows, list) else [rows]
    for sym in ("USD", "USDT", "INR"):
        for b in rows:
            if b.get("asset_symbol") == sym:
                return float(b.get("balance") or 0), float(b.get("available_balance") or 0), sym
    return 0.0, 0.0, "?"


def public_ips():
    out = {}
    for fam, url in (("ipv4", "https://api.ipify.org"), ("ipv6", "https://api64.ipify.org")):
        try:
            out[fam] = requests.get(url, timeout=5).text.strip()
        except requests.RequestException:
            out[fam] = "?"
    return out


# ---------------------------------------------------------------- the trader

class LiveTrader:
    def __init__(self):
        load_env()
        setup_logging()
        self.armed = os.environ.get("LIVE_TRADING", "").lower() == "yes"
        self.test = os.environ.get("TEST_MODE", "").lower() == "yes"
        self.max_lots = 1 if self.test else cfg("MAX_LOTS", 5)
        self.daily_loss_pct = 100.0 if self.test else cfg("DAILY_LOSS_PCT", 5.0)
        self.leverage = cfg("TEST_LEVERAGE", TEST["leverage"]) if self.test else S.LEVERAGE
        self.max_trades = cfg("TEST_MAX_TRADES", TEST["max_trades"]) if self.test else 10**6
        self.test_max_risk = cfg("TEST_MAX_RISK_PCT", TEST["max_risk_pct"])
        base = os.environ.get("DELTA_BASE_URL", "https://api.india.delta.exchange")
        key = os.environ.get("DELTA_API_KEY") or os.environ.get("apikey")
        secret = os.environ.get("DELTA_API_SECRET") or os.environ.get("apisecret")
        if os.environ.get("DELTA_FORCE_IPV4", "yes").lower() == "yes":
            force_ipv4()
        if not key or not secret:
            raise SystemExit("No API key: set DELTA_API_KEY and DELTA_API_SECRET in .env")
        self.api = Delta(key, secret, base)
        self.base = base
        os.makedirs(STATE_DIR, exist_ok=True)
        self.state_path = os.path.join(STATE_DIR, "live.json")
        self.st = {}
        if os.path.exists(self.state_path):
            with open(self.state_path) as fh:
                self.st = json.load(fh)
        self.pos = self.st.get("pos")
        self.blocked = None
        self.last_poll = 0.0

    def save(self):
        self.st["pos"] = self.pos
        with open(self.state_path, "w") as fh:
            json.dump(self.st, fh, indent=1, default=str)

    # ---------- start of session
    def preflight(self):
        mode = "ARMED - REAL ORDERS" if self.armed else "DRY RUN - no orders will be sent"
        if self.test:
            log.info(f"live trader: {mode} | TEST MODE: always 1 lot, {self.leverage}x cross, max "
                     f"{self.max_trades} trades/day, skip if 1 lot risks > {self.test_max_risk}% | {self.base}")
        else:
            log.info(f"live trader: {mode} | {self.base} | max {self.max_lots} lots/trade | daily loss stop "
                     f"{self.daily_loss_pct}% | {S.RISK_PCT}% risk | {self.leverage}x")
        ips = public_ips()
        log.info(f"public IPs: IPv4 {ips['ipv4']} | IPv6 {ips['ipv6']} "
                 f"(requests use {'IPv4' if os.environ.get('DELTA_FORCE_IPV4', 'yes') == 'yes' else 'default'})")
        try:
            bal, avail, asset = usd_balance(self.api.balances())
        except DeltaError as e:
            if e.code == "ip_not_whitelisted_for_api_key":
                ip = (e.body.get("error", {}).get("context") or {}).get("client_ip", "?")
                log.critical(f"Delta rejected this IP ({ip}). Whitelist it on the API key "
                             f"(Delta > Account > API keys > edit), then run: python live.py check")
            else:
                log.critical(f"API auth failed: {e}")
            self.blocked = "auth"
            return False
        log.info(f"wallet: {bal:,.2f} {asset} (available {avail:,.2f}) = Rs {bal * S.USDINR:,.0f} at Rs {S.USDINR:.0f}/$")
        day = datetime.now(S_IST).strftime("%Y-%m-%d")
        if self.st.get("day") != day:
            self.st.update(day=day, day_start_balance=bal, day_pnl=0.0, day_trades=0)
            log.info(f"new trading day {day}: start balance {bal:,.2f} {asset}")
        prod = self.api.product()
        if float(prod.get("contract_value", 0)) != S.CONTRACT_BTC or float(prod.get("tick_size", 0)) != S.TICK:
            log.critical(f"product spec changed: contract_value {prod.get('contract_value')} "
                         f"tick {prod.get('tick_size')} - refusing to trade")
            self.blocked = "product"
            return False
        if not self.ensure_leverage():
            self.blocked = "leverage"
            return False
        p = self.api.position()
        size = pos_size(p)
        if size and not self.pos:
            log.error(f"a BTCUSD position of {size} lots exists that this bot did not open "
                      f"(entry {p.get('entry_price')}). No new entries until it is closed.")
            self.blocked = "foreign position"
        elif self.pos and not size and not self.pos.get("dry"):
            log.warning("state says a position is open but Delta shows none: it was closed while the bot was "
                        "off (stop/target). Recording it from fills.")
            self.finalize("closed while offline")
        elif self.pos:
            log.info(f"resuming management of open position: {self.pos['side']} {self.pos['lots']} lots "
                     f"@ {self.pos['entry_px']}, time exit {self.pos['deadline_ist']}")
        orders = self.api.open_orders()
        log.info(f"open BTCUSD orders: {len(orders)}")
        self.save()
        return True

    def ensure_leverage(self):
        try:
            cur = self.api.leverage()
            lev = float((cur or {}).get("leverage", 0))
        except DeltaError as e:
            log.error(f"could not read leverage: {e}")
            lev = 0
        if lev == self.leverage:
            log.info(f"leverage {lev:.0f}x OK")
            return True
        if not self.armed:
            log.warning(f"leverage is {lev:.0f}x, not {self.leverage}x. Dry run: would set it to {self.leverage}x.")
            return True
        try:
            self.api.set_leverage(self.leverage)
            lev = float((self.api.leverage() or {}).get("leverage", 0))
        except DeltaError as e:
            log.critical(f"setting leverage failed: {e}")
            return False
        if lev != self.leverage:
            log.critical(f"leverage reads {lev}x after setting {self.leverage}x - refusing to trade")
            return False
        log.info(f"leverage set to {lev:.0f}x")
        return True

    # ---------- entries
    def enter(self, sig):
        """sig: dir, e, sl, stop, tgt, signal_time (bar open, UTC str), kind, p, session."""
        tag = f"{sig['kind']} {'LONG' if sig['dir'] == 1 else 'SHORT'} p={sig['p']:.3f} [{sig['session']}]"
        if self.blocked:
            log.warning(f"LIVE skip {tag}: blocked ({self.blocked})")
            return
        if self.pos:
            log.info(f"LIVE skip {tag}: already in a position")
            return
        try:
            bal, _, _ = usd_balance(self.api.balances())
            if not self.pos and pos_size(self.api.position()):
                log.error(f"LIVE skip {tag}: exchange shows an open position this bot does not track")
                self.blocked = "foreign position"
                return
            bid, ask, last = self.api.ticker()
        except (DeltaError, requests.RequestException) as e:
            log.error(f"LIVE skip {tag}: pre-trade read failed: {e}")
            return
        start = self.st.get("day_start_balance") or bal
        if self.st.get("day_pnl", 0.0) <= -start * self.daily_loss_pct / 100:
            log.warning(f"LIVE skip {tag}: daily loss stop hit ({self.st['day_pnl']:+.2f} of {start:.2f})")
            return
        d = sig["dir"]
        px = ask if d == 1 else bid
        stop, tgt = float(S.round_tick(sig["stop"])), float(S.round_tick(sig["tgt"]))
        if (d == 1 and (px <= stop or px >= tgt)) or (d == -1 and (px >= stop or px <= tgt)):
            log.warning(f"LIVE skip {tag}: price {px} already beyond stop {stop} / target {tgt}")
            return
        if self.st.get("day_trades", 0) >= self.max_trades:
            log.warning(f"LIVE skip {tag}: {self.max_trades} trades already today (test mode limit)")
            return
        lots = self.size(bal, sig, px, stop)
        if lots <= 0:
            return
        risk_usd = lots * S.CONTRACT_BTC * abs(px - stop)
        side = "buy" if d == 1 else "sell"
        coid = f"e915{sig['signal_time'][2:16].replace('-', '').replace(' ', '').replace(':', '')}{'L' if d == 1 else 'S'}"
        body = {"product_id": PRODUCT_ID, "product_symbol": SYMBOL, "size": lots, "side": side,
                "order_type": "market_order", "client_order_id": coid,
                "bracket_stop_loss_price": f"{stop:.1f}", "bracket_take_profit_price": f"{tgt:.1f}",
                "bracket_stop_trigger_method": "last_traded_price"}
        log.info(f"LIVE ENTRY {tag}: {side} {lots} lot(s) ~{px} | stop {stop} | target {tgt} | "
                 f"risk ~${risk_usd:.2f} (Rs {risk_usd * S.USDINR:.0f}, {risk_usd / bal * 100:.2f}% of balance)")
        deadline = time.time() + HOLD_S
        self.pos = dict(side=side, dir=d, lots=lots, e=sig["e"], sl=sig["sl"], stop=stop, tgt=tgt, coid=coid,
                        signal_time=sig["signal_time"], kind=sig["kind"], p=sig["p"], session=sig["session"],
                        quote_px=px, entry_px=px, entry_ts=time.time(), entry_ist=now_ist(),
                        deadline=deadline, deadline_ist=datetime.fromtimestamp(deadline, S_IST).strftime("%H:%M:%S"),
                        balance_before=bal, dry=not self.armed, order_id=None)
        if not self.armed:
            self.st["day_trades"] = self.st.get("day_trades", 0) + 1
            log.info(f"DRY RUN: would POST /v2/orders {json.dumps(body)}")
            append_csv("live_orders.csv", dict(time_ist=now_ist(), action="entry", dry=True, client_order_id=coid,
                                               request=json.dumps(body), response=""))
            self.save()
            return
        try:
            res = self.api.place(body)
        except (DeltaError, requests.RequestException) as e:
            log.error(f"entry order failed: {e}")
            res = self.reconcile(coid)
            if res is None:
                append_csv("live_orders.csv", dict(time_ist=now_ist(), action="entry", dry=False,
                                                   client_order_id=coid, request=json.dumps(body), response=str(e)))
                self.pos = None
                self.save()
                return
        append_csv("live_orders.csv", dict(time_ist=now_ist(), action="entry", dry=False, client_order_id=coid,
                                           request=json.dumps(body), response=json.dumps(res)[:1500]))
        filled = int(res.get("size", lots)) - int(res.get("unfilled_size") or 0)
        avg = float(res.get("average_fill_price") or px)
        self.pos.update(order_id=res.get("id"), lots=filled or lots, entry_px=avg)
        self.st["day_trades"] = self.st.get("day_trades", 0) + 1
        log.info(f"LIVE FILLED {filled}/{lots} @ {avg} (quote {px}, slippage "
                 f"{d * (avg - px):+.1f} $/BTC; signal close {sig['e']})")
        self.save()
        self.ensure_protection()

    def size(self, bal, sig, px, stop):
        """Lots for this trade, or 0 (with the reason logged)."""
        eq_inr = bal * S.USDINR
        lot_usd = S.CONTRACT_BTC * px
        risk1 = S.CONTRACT_BTC * abs(px - stop)
        if lot_usd / self.leverage > bal * 0.95:
            log.warning(f"LIVE skip: 1 lot needs ${lot_usd / self.leverage:.2f} margin at {self.leverage}x, "
                        f"balance ${bal:.2f}")
            return 0
        if self.test:
            if risk1 > bal * self.test_max_risk / 100:
                log.warning(f"LIVE skip: 1 lot risks ${risk1:.2f} = {risk1 / bal * 100:.0f}% of ${bal:.2f} "
                            f"(test limit {self.test_max_risk}%)")
                return 0
            if S.liq_distance(1, px, eq_inr) < S.LIQ_SAFETY * abs(px - stop) / px:
                log.warning("LIVE skip: liquidation would be closer than 2x the stop distance")
                return 0
            return 1
        lots = min(S.contracts(eq_inr, sig["e"], sig["sl"]), self.max_lots)
        if lots <= 0:
            log.warning(f"LIVE skip: 1 lot would risk more than {S.MAX_RISK_PCT}% of ${bal:.2f}")
        return lots

    def reconcile(self, coid):
        """After a failed order request: did Delta accept it anyway? Never re-send blindly."""
        time.sleep(1.5)
        try:
            for o in self.api.order_history() + self.api.open_orders():
                if o.get("client_order_id") == coid:
                    log.warning(f"order {coid} was accepted despite the error: state {o.get('state')}")
                    return o
        except (DeltaError, requests.RequestException) as e:
            log.error(f"reconcile failed: {e}")
        if pos_size(self.api.position()):
            log.critical("position exists after a failed entry request and no matching order was found - "
                         "managing it as ours")
            return {"id": None, "size": abs(pos_size(self.api.position())), "unfilled_size": 0,
                    "average_fill_price": (self.api.position() or {}).get("entry_price")}
        return None

    def ensure_protection(self):
        """The position must have a stop at Delta. Attach a bracket if missing; close if that fails."""
        time.sleep(1.0)
        try:
            orders = self.api.open_orders()
        except (DeltaError, requests.RequestException) as e:
            log.error(f"cannot list orders to verify the stop: {e}")
            orders = []
        stops = [o for o in orders if o.get("stop_order_type") == "stop_loss_order" or
                 (o.get("stop_price") and abs(float(o["stop_price"]) - self.pos["stop"]) < 1)]
        if stops:
            log.info(f"protection OK: {len(orders)} open order(s) incl. stop-loss at "
                     f"{stops[0].get('stop_price')}")
            return
        log.warning("no stop-loss order found after entry - attaching a bracket")
        body = {"product_id": PRODUCT_ID, "product_symbol": SYMBOL,
                "stop_loss_order": {"order_type": "market_order", "stop_price": f"{self.pos['stop']:.1f}"},
                "take_profit_order": {"order_type": "market_order", "stop_price": f"{self.pos['tgt']:.1f}"},
                "bracket_stop_trigger_method": "last_traded_price"}
        try:
            res = self.api.bracket(body)
            append_csv("live_orders.csv", dict(time_ist=now_ist(), action="bracket", dry=False,
                                               client_order_id=self.pos["coid"], request=json.dumps(body),
                                               response=json.dumps(res)[:1500]))
            log.info("bracket attached")
        except (DeltaError, requests.RequestException) as e:
            log.critical(f"could not protect the position ({e}) - closing it now")
            self.close_position("unprotected")

    # ---------- exits
    def poll(self, force=False):
        if not self.pos or (not force and time.time() - self.last_poll < POLL_S):
            return
        self.last_poll = time.time()
        p = self.pos
        try:
            if p.get("dry"):
                bid, ask, last = self.api.ticker()
                d = p["dir"]
                if (d == 1 and last <= p["stop"]) or (d == -1 and last >= p["stop"]):
                    self.finalize("stop", exit_px=bid if d == 1 else ask)
                elif (d == 1 and last >= p["tgt"]) or (d == -1 and last <= p["tgt"]):
                    self.finalize("target", exit_px=bid if d == 1 else ask)
                elif time.time() >= p["deadline"]:
                    self.finalize("time exit", exit_px=bid if d == 1 else ask)
                return
            if time.time() >= p["deadline"]:
                self.close_position("time exit")
                return
            if not pos_size(self.api.position()):
                self.finalize("stop/target at Delta")
        except (DeltaError, requests.RequestException) as e:
            log.warning(f"poll failed (will retry): {e}")

    def close_position(self, why):
        p = self.pos
        log.info(f"LIVE EXIT ({why}): closing {p['lots']} lot(s)")
        try:
            self.api.cancel_all()
        except (DeltaError, requests.RequestException) as e:
            log.warning(f"cancel orders failed: {e}")
        for attempt in range(1, 4):
            try:
                size = pos_size(self.api.position())
                if not size:
                    break
                body = {"product_id": PRODUCT_ID, "product_symbol": SYMBOL, "size": abs(size),
                        "side": "sell" if size > 0 else "buy", "order_type": "market_order", "reduce_only": True,
                        "client_order_id": f"{p['coid']}X{attempt}"}
                res = self.api.place(body)
                append_csv("live_orders.csv", dict(time_ist=now_ist(), action=f"exit:{why}", dry=False,
                                                   client_order_id=body["client_order_id"],
                                                   request=json.dumps(body), response=json.dumps(res)[:1500]))
                time.sleep(1.0)
            except (DeltaError, requests.RequestException) as e:
                log.error(f"close attempt {attempt} failed: {e}")
                time.sleep(2)
        else:
            if pos_size(self.api.position()):
                log.critical("POSITION STILL OPEN after 3 close attempts - close it manually on Delta "
                             "or run: python live.py flatten --yes")
                return
        try:
            self.api.cancel_all()                         # leftover bracket legs
        except (DeltaError, requests.RequestException):
            pass
        self.finalize(why)

    def finalize(self, how, exit_px=None):
        p = self.pos
        d, lots = p["dir"], p["lots"]
        commission, bal_after = 0.0, None
        if not p.get("dry"):
            time.sleep(1.0)
            try:
                fills = [f for f in self.api.fills() if ts_seconds(f.get("created_at")) >= p["entry_ts"] - 5]
                exits = [f for f in fills if f.get("side") != p["side"]]
                commission = sum(float(f.get("commission") or 0) for f in fills)
                if exits:
                    q = sum(abs(float(f.get("size") or 0)) for f in exits)
                    exit_px = sum(float(f["price"]) * abs(float(f.get("size") or 0)) for f in exits) / (q or 1)
                bal_after, _, _ = usd_balance(self.api.balances())
            except (DeltaError, requests.RequestException) as e:
                log.warning(f"could not read fills/balance: {e}")
        exit_px = exit_px if exit_px is not None else p["entry_px"]
        gross = lots * S.CONTRACT_BTC * d * (exit_px - p["entry_px"])
        pnl = gross - commission
        R = d * (exit_px - p["entry_px"]) / abs(p["e"] - p["sl"])
        wallet_pnl = (bal_after - p["balance_before"]) if bal_after is not None else None
        self.st["day_pnl"] = self.st.get("day_pnl", 0.0) + (wallet_pnl if wallet_pnl is not None else pnl)
        row = dict(signal_time_utc=p["signal_time"], entry_ist=p["entry_ist"], exit_ist=now_ist(),
                   hold_min=round((time.time() - p["entry_ts"]) / 60, 1), session=p["session"], kind=p["kind"],
                   side=p["side"], lots=lots, p=round(p["p"], 4), signal_close=p["e"], stop=p["stop"], target=p["tgt"],
                   quote_at_entry=p["quote_px"], entry_px=round(p["entry_px"], 2), exit_px=round(exit_px, 2),
                   exit_reason=how, R=round(R, 3), gross_usd=round(gross, 4), commission_usd=round(commission, 4),
                   pnl_usd=round(pnl, 4), pnl_inr=round(pnl * S.USDINR, 2),
                   wallet_pnl_usd=round(wallet_pnl, 4) if wallet_pnl is not None else "",
                   balance_before=round(p["balance_before"], 4),
                   balance_after=round(bal_after, 4) if bal_after is not None else "",
                   entry_slip_usd_per_btc=round(d * (p["entry_px"] - p["quote_px"]), 2), dry=p.get("dry", False))
        append_csv("live_trades.csv", row)
        log.info(f"LIVE CLOSED {how}: {p['side']} {lots} lot(s) {p['entry_px']:.1f} -> {exit_px:.1f} | "
                 f"R {R:+.2f} | P&L ${pnl:+.4f} (Rs {pnl * S.USDINR:+.2f}) | fees ${commission:.4f}"
                 + (f" | wallet {wallet_pnl:+.4f}" if wallet_pnl is not None else "") + (" | DRY" if p.get("dry") else ""))
        if commission > 0 and row["hold_min"] < 30:
            log.warning(f"fees of ${commission:.4f} were charged on a trade held {row['hold_min']} min - "
                        "check Delta's fee-free rule; without it this strategy loses money")
        self.pos = None
        self.save()

    def flat(self):
        return self.pos is None

    def shutdown(self):
        """End of session: never leave a position the bot is no longer watching."""
        if self.pos:
            log.warning("session ending with a live position open - closing it")
            if self.pos.get("dry"):
                self.poll(force=True)
                if self.pos:
                    self.finalize("session end")
            else:
                self.close_position("session end")


# ---------------------------------------------------------------- CLI

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["check", "flatten"])
    ap.add_argument("--yes", action="store_true", help="confirm flatten")
    a = ap.parse_args()
    t = LiveTrader()
    if a.cmd == "check":
        ok = t.preflight()
        try:
            b, s, last = t.api.ticker()
            log.info(f"BTCUSD bid {b} ask {s} last {last} (spread {(s - b) / last * 1e4:.2f} bp)")
            p = t.api.position()
            log.info(f"position: {pos_size(p)} lots" + (f" @ {p.get('entry_price')}" if pos_size(p) else ""))
            for o in t.api.open_orders():
                log.info(f"open order {o.get('id')}: {o.get('side')} {o.get('size')} {o.get('order_type')} "
                         f"stop {o.get('stop_price')} limit {o.get('limit_price')} state {o.get('state')}")
        except (DeltaError, requests.RequestException) as e:
            log.error(f"read failed: {e}")
        log.info("CHECK " + ("PASSED" if ok and not t.blocked else f"FAILED ({t.blocked})") +
                 (" - armed: real orders will be sent" if t.armed else " - dry run (set LIVE_TRADING=yes to arm)"))
    elif a.cmd == "flatten":
        if not a.yes:
            raise SystemExit("add --yes to cancel all BTCUSD orders and close the position")
        t.api.cancel_all()
        size = pos_size(t.api.position())
        if size:
            res = t.api.place({"product_id": PRODUCT_ID, "product_symbol": SYMBOL, "size": abs(size),
                               "side": "sell" if size > 0 else "buy", "order_type": "market_order",
                               "reduce_only": True, "client_order_id": f"e915flat{int(time.time())}"})
            log.info(f"flatten order: {json.dumps(res)[:300]}")
        time.sleep(1)
        log.info(f"position now {pos_size(t.api.position())} lots; open orders {len(t.api.open_orders())}")
        if t.pos:
            t.finalize("manual flatten")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        setup_logging()
        log.critical("live.py crashed:\n" + traceback.format_exc())
        raise
