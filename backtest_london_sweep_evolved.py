import yfinance as yf
import pandas as pd
import mplfinance as mpf
from datetime import time, timedelta
import os

def backtest_london_sweep_evolved(period="60d", interval="5m"):
    print(f"--- Backtesting Evolved London Sweep Strategy (SMT Highs Magnet) ---")

    # 1. Fetch Data
    nq_symbol = "NQ=F"
    es_symbol = "ES=F"

    try:
        nq_df = yf.download(nq_symbol, period=period, interval=interval, progress=False)
        es_df = yf.download(es_symbol, period=period, interval=interval, progress=False)
    except Exception as e:
        print(f"Error fetching data: {e}")
        return

    # Process Data
    for df in [nq_df, es_df]:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        df.index = df.index.tz_convert("America/New_York")

    # Group by Date
    grouped_nq = nq_df.groupby(nq_df.index.date)
    grouped_es = es_df.groupby(es_df.index.date)

    results = []
    plot_candidates = []

    london_start = time(2, 0)
    london_end = time(8, 0)
    ny_start = time(8, 0)
    ny_end = time(16, 0)

    dates = sorted(list(set(grouped_nq.groups.keys()) & set(grouped_es.groups.keys())))

    for date in dates:
        try:
            day_nq = grouped_nq.get_group(date)
            day_es = grouped_es.get_group(date)
        except KeyError:
            continue

        # Analyze London Highs (SMT Check)
        london_nq = day_nq.between_time(london_start, london_end, inclusive="left")
        london_es = day_es.between_time(london_start, london_end, inclusive="left")

        if london_nq.empty or london_es.empty:
            continue

        london_high_nq = london_nq['High'].max()
        time_high_nq = london_nq['High'].idxmax()

        london_high_es = london_es['High'].max()
        time_high_es = london_es['High'].idxmax()

        # Check SMT at Highs (Time Divergence)
        time_diff = abs((time_high_nq - time_high_es).total_seconds()) / 60
        has_smt_at_highs = time_diff > 15 # Divergence > 15 mins implies mismatched structure

        # London Lows
        london_low_nq = london_nq['Low'].min()

        # 08:00 Filter
        midpoint_nq = (london_high_nq + london_low_nq) / 2
        try:
            at_8am_nq = day_nq.between_time(time(8,0), time(8,5)).iloc[0]
            price_8am_nq = at_8am_nq['Open']
        except IndexError:
            continue

        if price_8am_nq >= midpoint_nq:
            continue # Not in lower half

        # Check for Sweep (NY Session)
        ny_nq = day_nq.between_time(ny_start, ny_end, inclusive="left")
        if ny_nq.empty:
            continue

        ny_low_nq = ny_nq['Low'].min()

        if ny_low_nq < london_low_nq:
            # Sweep Confirmed
            # Find sweep time
            sweep_idx = ny_nq[ny_nq['Low'] < london_low_nq].index[0]

            # --- Entry Logic: Reversal Candle ---
            # Look for Close > High of previous candle *after* sweep
            post_sweep = ny_nq.loc[sweep_idx:]

            entry_price = None
            entry_time = None
            stop_loss = None

            # Iterate candles after sweep
            candles = post_sweep.iloc[1:] # Need at least one previous candle
            if candles.empty:
                continue

            # Track lowest low seen SO FAR (for Stop Loss)
            curr_lowest = post_sweep.iloc[0]['Low']

            for i in range(len(candles)):
                curr = candles.iloc[i]
                prev = post_sweep.iloc[i] # previous relative to curr (using post_sweep indexing)

                # Update Lowest Low
                if curr['Low'] < curr_lowest:
                    curr_lowest = curr['Low']

                # Simple Reversal Signal: Close > Previous High
                if curr['Close'] > prev['High']:
                    entry_price = curr['Close']
                    entry_time = curr.name
                    stop_loss = curr_lowest # The lowest point reached UP TO entry
                    break

            if entry_price is None:
                continue # No reversal signal found

            # --- Trade Simulation ---
            target_price = london_high_nq # Magnet Target

            outcome = "Running"
            exit_price = None
            exit_time = None

            trade_data = ny_nq.loc[entry_time:]

            # Start loop from NEXT candle to avoid ambiguity
            bars = trade_data.iloc[1:]

            for i in range(len(bars)):
                bar = bars.iloc[i]

                # check stop hit first (safety)
                if bar['Low'] <= stop_loss:
                    outcome = "Stop Hit"
                    exit_price = stop_loss
                    exit_time = bar.name
                    break

                # Check Target Hit
                if bar['High'] >= target_price:
                    outcome = "Target Hit"
                    exit_price = target_price
                    exit_time = bar.name
                    break

            if outcome == "Running":
                # Close at end of day
                outcome = "EOD Close"
                exit_price = trade_data.iloc[-1]['Close']
                exit_time = trade_data.index[-1]

            profit = exit_price - entry_price if outcome == "Target Hit" or outcome == "EOD Close" else exit_price - entry_price

            trade_res = {
                'date': date,
                'entry_time': entry_time,
                'entry_price': entry_price,
                'exit_time': exit_time,
                'exit_price': exit_price,
                'stop_loss': stop_loss,
                'target': target_price,
                'outcome': outcome,
                'profit': profit,
                'has_smt': has_smt_at_highs,
                'df': day_nq.between_time(time(2,0), time(16,0))
            }
            results.append(trade_res)

            if outcome == "Target Hit":
                plot_candidates.append(trade_res)

    # 3. Analyze Results
    df_res = pd.DataFrame(results)

    if df_res.empty:
        print("No trades found.")
        return

    print(f"\n--- Backtest Results ({len(df_res)} Trades) ---")

    # Overall Win Rate (Target Hit)
    wins = df_res[df_res['outcome'] == "Target Hit"]
    wr = (len(wins) / len(df_res)) * 100
    print(f"Overall Win Rate (Target Hit): {wr:.2f}% ({len(wins)}/{len(df_res)})")

    # SMT Filter Analysis
    smt_trades = df_res[df_res['has_smt'] == True]
    no_smt_trades = df_res[df_res['has_smt'] == False]

    if not smt_trades.empty:
        smt_wins = smt_trades[smt_trades['outcome'] == "Target Hit"]
        smt_wr = (len(smt_wins) / len(smt_trades)) * 100
        print(f"Win Rate WITH SMT at Highs: {smt_wr:.2f}% ({len(smt_wins)}/{len(smt_trades)})")
    else:
        print("No trades with SMT at Highs found.")

    if not no_smt_trades.empty:
        no_smt_wins = no_smt_trades[no_smt_trades['outcome'] == "Target Hit"]
        no_smt_wr = (len(no_smt_wins) / len(no_smt_trades)) * 100
        print(f"Win Rate WITHOUT SMT at Highs: {no_smt_wr:.2f}% ({len(no_smt_wins)}/{len(no_smt_trades)})")

    # Avg Profit
    print(f"Average Profit per Trade: {df_res['profit'].mean():.2f} pts")

    # 4. Plot Last 5 Trades
    print("\n--- Generating Charts for Last 5 Successful Trades ---")
    plot_candidates.sort(key=lambda x: x['date'], reverse=True)
    top_5 = plot_candidates[:5]

    if not os.path.exists("charts_evolved"):
        os.makedirs("charts_evolved")

    for i, trade in enumerate(top_5):
        date_str = trade['date'].strftime("%Y-%m-%d")
        filename = f"charts_evolved/london_sweep_evolved_{i+1}_{date_str}.png"

        df_plot = trade['df']

        # Add Markers
        # Use lists with NaNs, same length as df_plot
        buy_signal = [float('nan')] * len(df_plot)
        sell_signal = [float('nan')] * len(df_plot)

        # Find index locations
        try:
            entry_loc = df_plot.index.get_loc(trade['entry_time'])

            # exit_time might be the last candle if EOD close
            exit_ts = trade['exit_time']
            if exit_ts not in df_plot.index:
                 # fallback to nearest if needed, but should be there
                 exit_ts = df_plot.index[df_plot.index.get_indexer([exit_ts], method='nearest')[0]]

            exit_loc = df_plot.index.get_loc(exit_ts)

            buy_signal[entry_loc] = trade['entry_price'] * 0.9995 # Slightly below
            sell_signal[exit_loc] = trade['exit_price'] * 1.0005 # Slightly above
        except KeyError:
            continue # Should not happen if slicing is consistent

        # Add Lines
        hlines = dict(hlines=[trade['target'], trade['stop_loss']], colors=['g', 'r'], linestyle='-.', linewidths=1.5)

        # Add Buy/Sell markers
        apds = [
            mpf.make_addplot(buy_signal, type='scatter', markersize=100, marker='^', color='g'),
            mpf.make_addplot(sell_signal, type='scatter', markersize=100, marker='v', color='r'),
        ]

        title = f"Evolved Logic ({date_str})\nEntry: {trade['entry_price']:.2f} -> Target: {trade['target']:.2f} (SMT: {trade['has_smt']})"

        mpf.plot(
            df_plot,
            type='candle',
            style='yahoo',
            title=title,
            addplot=apds,
            hlines=hlines,
            savefig=filename,
            volume=False
        )
        print(f"Saved chart: {filename}")

if __name__ == "__main__":
    backtest_london_sweep_evolved(period="60d", interval="5m")
