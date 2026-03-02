import yfinance as yf
import pandas as pd
from datetime import time, timedelta

def backtest_sqqq_strategy(period="60d", interval="5m"):
    print(f"--- Backtesting SQQQ Strategy (Signal: NQ=F London Low) ---")

    # Symbols
    nq_symbol = "NQ=F"
    sqqq_symbol = "SQQQ"

    try:
        nq_df = yf.download(nq_symbol, period=period, interval=interval, progress=False)
        sqqq_df = yf.download(sqqq_symbol, period=period, interval=interval, progress=False)
    except Exception as e:
        print(f"Error fetching data: {e}")
        return

    # Process Data
    for df in [nq_df, sqqq_df]:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        df.index = df.index.tz_convert("America/New_York")

    grouped_nq = nq_df.groupby(nq_df.index.date)
    grouped_sqqq = sqqq_df.groupby(sqqq_df.index.date)

    london_start = time(2, 0)
    london_end = time(8, 0)

    trades = []
    missed_trades = 0
    total_signals = 0

    dates = sorted(list(set(grouped_nq.groups.keys()) & set(grouped_sqqq.groups.keys())))

    for date in dates:
        try:
            day_nq = grouped_nq.get_group(date)
            day_sqqq = grouped_sqqq.get_group(date)
        except KeyError:
            continue

        # Analyze London (NQ)
        london_nq = day_nq.between_time(london_start, london_end, inclusive="left")
        if london_nq.empty:
            continue

        london_high_nq = london_nq['High'].max()
        london_low_nq = london_nq['Low'].min()
        midpoint_nq = (london_high_nq + london_low_nq) / 2

        # 08:00 Filter (NQ)
        try:
            at_8am_nq = day_nq.between_time(time(8,0), time(8,5)).iloc[0]
            price_8am_nq = at_8am_nq['Open']
        except IndexError:
            continue

        if price_8am_nq >= midpoint_nq:
            continue # Signal Invalid

        total_signals += 1

        # Check if Target Hit BEFORE 09:30 Open
        pre_market_nq = day_nq.between_time(time(8,0), time(9,30), inclusive="left")

        if not pre_market_nq.empty:
            pre_low_nq = pre_market_nq['Low'].min()
            if pre_low_nq <= london_low_nq:
                # Target Hit in Pre-Market -> Missed Trade for SQQQ Open
                missed_trades += 1
                # Could potentially trade pre-market, but usually illiquid/hard for retail.
                # Let's count as missed for strict "Wait for Open" strategy.
                continue

        # Trade SQQQ at 09:30 Open
        # Find 09:30 Candle
        try:
            at_930_sqqq = day_sqqq.between_time(time(9,30), time(9,35)).iloc[0]
            entry_sqqq = at_930_sqqq['Open']
        except IndexError:
            continue

        # Exit at 16:00 Close
        try:
            at_1600_sqqq = day_sqqq.between_time(time(15,55), time(16,0)).iloc[-1]
            exit_sqqq = at_1600_sqqq['Close']
        except IndexError:
            # Fallback to last available
            exit_sqqq = day_sqqq.iloc[-1]['Close']

        pnl_pct = ((exit_sqqq - entry_sqqq) / entry_sqqq) * 100

        trades.append({
            'date': date,
            'entry': entry_sqqq,
            'exit': exit_sqqq,
            'pnl_pct': pnl_pct
        })

    if not trades and missed_trades == 0:
        print("No trades found.")
        return

    print(f"\n--- SQQQ Strategy Results (60 Days) ---")
    print(f"Total Signals (NQ 08:00 Setup): {total_signals}")
    print(f"Missed Trades (Hit Target Pre-Market): {missed_trades} ({(missed_trades/total_signals)*100:.1f}%)")
    print(f"Executed Trades (SQQQ 09:30 Open): {len(trades)}")

    if trades:
        df = pd.DataFrame(trades)
        wins = df[df['pnl_pct'] > 0]
        win_rate = (len(wins) / len(df)) * 100
        avg_pnl = df['pnl_pct'].mean()
        total_pnl = df['pnl_pct'].sum() # Simple sum of % returns (assuming reinvestment or fixed size)

        # Annualized ROI (approx)
        # Using compounding? Or simple sum?
        # Let's use simple sum for "Asset Increase" metric requested.
        # 60 days -> ~4.2x for year.
        annualized_roi = total_pnl * (252 / 60) # Rough scaler

        print(f"Win Rate: {win_rate:.2f}% ({len(wins)}/{len(df)})")
        print(f"Avg PnL per Trade: {avg_pnl:.2f}%")
        print(f"Total Return (Simple): {total_pnl:.2f}%")
        print(f"Projected Annual Return: {annualized_roi:.2f}%")
    else:
        print("No executed trades (all missed or data missing).")

if __name__ == "__main__":
    backtest_sqqq_strategy()
