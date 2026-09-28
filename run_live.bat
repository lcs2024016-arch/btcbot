@echo off
REM Live/dry-run trading for today: London 13:30-17:30 IST then New York 18:30-22:30 IST.
REM Settings come from btcbot\.env (LIVE_TRADING, TEST_MODE). Logs: ema915-forward\logs\r
cd /d "%~dp0"
python live.py check
python bot.py --live --all-day
