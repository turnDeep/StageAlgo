import yfinance as yf
import pandas as pd
from datetime import time, timedelta

def compare_simple_strategies(period="60d", interval="5m"):
    print(f"--- Comparing Simple Strategies (NQ=F, {period}) ---")

    # Fetch Data
    nq_symbol = "NQ=F"
    try:
        nq_df = yf.download(nq_symbol, period=period, interval=interval, progress=False)
    except Exception as e:
        print(f"Error fetching data: {e}")
        return

    # Process Data
    if isinstance(nq_df.columns, pd.MultiIndex):
        nq_df.columns = nq_df.columns.get_level_values(0)
    if nq_df.index.tz is None:
        nq_df.index = nq_df.index.tz_localize("UTC")
    nq_df.index = nq_df.index.tz_convert("America/New_York")

    grouped = nq_df.groupby(nq_df.index.date)

    london_start = time(2, 0)
    london_end = time(8, 0)
    ny_start = time(8, 0)
    ny_end = time(16, 0)

    # Strategy A: Short (Target London Low)
    trades_short = []

    # Strategy B: Reversal (Fixed 2:1 RR)
    trades_reversal = []

    for date, day_df in grouped:
        # Define Sessions
        london_data = day_df.between_time(london_start, london_end, inclusive="left")
        if london_data.empty:
            continue

        london_high = london_data['High'].max()
        london_low = london_data['Low'].min()
        midpoint = (london_high + london_low) / 2

        # 08:00 Filter
        try:
            at_8am = day_df.between_time(time(8,0), time(8,5)).iloc[0]
            price_8am = at_8am['Open']
        except IndexError:
            continue

        if price_8am >= midpoint:
            continue # Skip

        # --- Strategy A: Short ---
        # Entry: 08:00 Open
        # Target: London Low
        # Stop: London High

        entry_short = price_8am
        target_short = london_low
        stop_short = london_high

        trade_data = day_df.between_time(ny_start, ny_end) # 08:00 onwards
        if trade_data.empty:
            continue

        outcome_short = "Running"
        exit_short = trade_data.iloc[-1]['Close']

        # Skip first candle (entry candle) for strict simulation
        bars_short = trade_data.iloc[1:]

        for i in range(len(bars_short)):
            bar = bars_short.iloc[i]
            if bar['High'] >= stop_short:
                outcome_short = "Stop Hit"
                exit_short = stop_short
                break
            if bar['Low'] <= target_short:
                outcome_short = "Target Hit"
                exit_short = target_short
                break

        pnl_short = entry_short - exit_short # Short PnL
        trades_short.append(pnl_short)

        # --- Strategy B: Reversal (Sweep & 2:1 RR) ---
        # Check if Short actually swept the low (prerequisite for Reversal)
        ny_low = trade_data['Low'].min()
        if ny_low >= london_low:
            continue # No Sweep -> No Reversal Setup

        # Find Sweep Time
        sweep_idx = trade_data[trade_data['Low'] < london_low].index[0]
        post_sweep = trade_data.loc[sweep_idx:]

        entry_rev = None
        stop_rev = None
        target_rev = None

        # Find Reversal Candle
        candles = post_sweep.iloc[1:]
        curr_lowest = post_sweep.iloc[0]['Low']

        for i in range(len(candles)):
            curr = candles.iloc[i]
            prev = post_sweep.iloc[i]

            if curr['Low'] < curr_lowest:
                curr_lowest = curr['Low']

            if curr['Close'] > prev['High']:
                entry_rev = curr['Close']
                entry_time_rev = curr.name
                stop_rev = curr_lowest
                risk = entry_rev - stop_rev
                target_rev = entry_rev + (risk * 2.0) # 2:1 RR
                break

        if entry_rev is None:
            continue

        # Simulate Reversal Trade
        trade_data_rev = trade_data.loc[entry_time_rev:].iloc[1:]
        outcome_rev = "Running"
        exit_rev = trade_data.iloc[-1]['Close']

        for i in range(len(trade_data_rev)):
            bar = trade_data_rev.iloc[i]
            if bar['Low'] <= stop_rev:
                outcome_rev = "Stop Hit"
                exit_rev = stop_rev
                break
            if bar['High'] >= target_rev:
                outcome_rev = "Target Hit"
                exit_rev = target_rev
                break

        pnl_rev = exit_rev - entry_rev
        trades_reversal.append(pnl_rev)

    # Compare
    print("\n--- Strategy A: Short (Target London Low) ---")
    if trades_short:
        df_short = pd.DataFrame({'pnl': trades_short})
        wr_short = (len(df_short[df_short['pnl'] > 0]) / len(df_short)) * 100
        avg_short = df_short['pnl'].mean()
        total_short = df_short['pnl'].sum()
        print(f"Trades: {len(df_short)}")
        print(f"Win Rate: {wr_short:.2f}%")
        print(f"Avg Profit: {avg_short:.2f} pts")
        print(f"Total PnL: {total_short:.2f} pts")
    else:
        print("No Short trades found.")

    print("\n--- Strategy B: Simple Reversal (2:1 Fixed RR) ---")
    if trades_reversal:
        df_rev = pd.DataFrame({'pnl': trades_reversal})
        wr_rev = (len(df_rev[df_rev['pnl'] > 0]) / len(df_rev)) * 100
        avg_rev = df_rev['pnl'].mean()
        total_rev = df_rev['pnl'].sum()
        print(f"Trades: {len(df_rev)}")
        print(f"Win Rate: {wr_rev:.2f}%")
        print(f"Avg Profit: {avg_rev:.2f} pts")
        print(f"Total PnL: {total_rev:.2f} pts")
    else:
        print("No Reversal trades found.")

if __name__ == "__main__":
    compare_simple_strategies()
