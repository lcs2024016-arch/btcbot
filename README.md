# BTC 9/15 EMA Scalper + ML · forward test

Paper-trading forward test of the **"9/15 EMA Scalper Strategy v3"** (TradingView Pine, uzairblaoch) on
**Delta Exchange India BTCUSD perpetual**, 5-minute candles, with a LightGBM filter that keeps only the
highest-probability 35% of signals. Runs automatically on GitHub Actions and rewrites this page after
every trade.

**Mode:** paper trading · **Last update:** not started yet · **Data:** – · **Model:** trained 2026-09-28, cut p ≥ 0.441 · **Open positions:** none

> ⚠️ Paper trading only. No orders are sent to any exchange. Past and simulated results do not
> guarantee future returns.

## Schedule (Indian Standard Time)

| Session | Entries (IST) | Entries (UTC) | Bot running (IST) | Days |
|---|---|---|---|---|
| London | 13:30 – 17:30 | 08:00 – 12:00 | ~12:50 – 18:00 | Mon – Fri |
| New York | 18:30 – 22:30 | 13:00 – 17:00 | ~18:00 – 23:00 | Mon – Fri |

No trading on Saturday and Sunday. IST has no daylight saving, so these times never shift.
Every position is closed at most **25 minutes** after entry, so every trade qualifies for Delta's
fee-free rule (closed within 30 minutes).

## Forward test

_No forward-test trades yet. The bot trades Monday–Friday, London 13:30–17:30 IST and New York 18:30–22:30 IST; results appear here after the first closed trade._


## Backtest reference (what the forward test should look like)

Walk-forward 2021-01 → 2026-09: the model is retrained every month on past data only and scores the next month. Out-of-sample AUC 0.588 (since 2024: 0.585). Same fills as the bot's candle model, zero slippage, whole Delta contracts from ₹1,500.

| Backtest | Trades | Per month | Win rate | Avg R | Profit factor | ₹1,500 → | Max DD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ML filter (top 35%), Jan 2024 → | 501 | 15 | 52.5% | +0.11R | 1.22 | ₹2,789 | 12.4% |
| No ML (Pine rules), Jan 2024 → | 1357 | 41 | 43.8% | +0.05R | 1.09 | ₹2,875 | 25.1% |
| ML filter (top 35%), Jan 2021 → | 1015 | 15 | 53.8% | +0.16R | 1.34 | ₹4,558 | 15.2% |
| No ML (Pine rules), Jan 2021 → | 2860 | 42 | 44.6% | +0.09R | 1.17 | ₹10,588 | 34.2% |

| Year | ML avg R | ML win rate | No-ML avg R | No-ML win rate |
| --- | ---: | ---: | ---: | ---: |
| 2021 | +0.18R | 55.2% | +0.11R | 46.4% |
| 2022 | +0.33R | 61.1% | +0.22R | 48.3% |
| 2023 | +0.12R | 50.3% | +0.06R | 41.8% |
| 2024 | +0.13R | 52.5% | +0.10R | 45.3% |
| 2025 | -0.04R | 48.3% | -0.06R | 40.4% |
| 2026 | +0.26R | 58.0% | +0.11R | 45.7% |

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="charts/backtest_equity_dark.svg">
  <img alt="Backtest equity since 2024" src="charts/backtest_equity_light.svg">
</picture>


## Strategy

| | Rule |
|---|---|
| Timeframe | 5 minutes, BTCUSD perpetual |
| Trend | EMA 9 above/below EMA 15, EMA 9 sloping ≥ 0.3 pts over 3 bars, gap ≥ 0.5 pts |
| Normal entry | EMA 9 inside the candle body + strong candle (body ≥ 60%, small opposite wick), pin bar or engulfing |
| FVG entry | Displacement candle (body > 2× 20-bar average, touching an EMA, in trend); next bar retraces 30–70% of it without breaking its extreme |
| 1h bias | 1h EMA 9 vs 15, 1h EMA 9 slope over 5 bars, 1h close on the right side |
| Stop / target | Signal candle low/high ∓ 1 pt · target 2R ∓ 1 pt |
| Exit | Stop, target, or market exit 25 minutes after entry |
| ML filter | LightGBM P(win) from ~50 OHLCV features (multi-timeframe trend, volatility, candle shape, stop size, session); trade only if p ≥ cut. Cut = 65th percentile of recent signals (keeps the top 35%). Retrained monthly on the full Binance BTCUSDT 5m history. |
| Days / hours | Monday – Friday, London 13:30–17:30 IST and New York 18:30–22:30 IST |

**Two paper accounts** trade side by side: *ML filter* (the strategy) and *No ML* (every signal the Pine
rules take), so the forward test shows whether the model adds value.

**Two prices per trade:**
- *Candle model*: filled exactly as in the backtest, at the next 5-minute open, with stop and target taken from candle highs/lows.
- *Live prices*: Delta's actual best bid/ask when the order would be sent, and the price seen when the stop or target triggers (checked every second).

The gap between the two is the slippage the backtest cannot see.

### Position sizing (Delta Exchange India)

| | |
|---|---|
| Contract | BTCUSD perpetual, 1 lot = 0.001 BTC, minimum 1 lot, tick 0.5 |
| Capital | ₹1,500 per paper account (USD/INR fixed at ₹85 by Delta India) |
| Risk | whole lots for 1% of equity; 1 lot if that rounds to 0 but risks ≤ 2%, otherwise skip |
| Fees | 0 (every trade closes within 30 min); standard 0.02% maker / 0.05% taker otherwise |
| Leverage | 1 lot ≈ $80 notional ≈ 4–5× on ₹1,500; Delta allows up to 200× |

## Pine Script (TradingView)

[`pine/ema915_v3_london_ny.pine`](pine/ema915_v3_london_ny.pine) is the same rules for TradingView's
Strategy Tester: IST sessions, Monday–Friday, 25-minute exit, 1% risk sizing, zero commission. Open a
`BINANCE:BTCUSDT.P` or `DELTAIN:BTCUSD.P` 5-minute chart, paste the script into the Pine Editor, and click
**Add to chart**. The ML filter cannot run inside Pine, so the TradingView backtest corresponds to the
**No ML** account.

## How it runs

| Workflow | When (IST) | What |
|---|---|---|
| `session.yml` | Mon–Fri ~12:50 and ~18:10 | Runs `bot.py` for the London or New York session (≈ 5 h each), commits `state/`, `charts/`, `README.md` after every trade |
| `retrain.yml` | 1st of each month, 08:30 | Downloads new Binance 5m candles, retrains the model, commits `model/` |

Run locally: `pip install -r requirements.txt`, then `python bot.py` (current session),
`python report.py` (rebuild this page), `python train.py --walkforward` (re-run the backtest),
`python test_parity.py 2026-09-01 2026-09-27` (bot vs backtest on history).

**Note:** the two session jobs use about 300 runner-hours a month. That's free on a **public**
repository; on a private repository it is far beyond the free 2,000 minutes.
