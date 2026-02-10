import yfinance as yf
import pandas as pd
import numpy as np
import datetime
from datetime import timedelta

def calculate_atr(df, period):
    high = df['High']
    low = df['Low']
    close = df['Close']

    # True Range
    df['tr0'] = abs(high - low)
    df['tr1'] = abs(high - close.shift(1))
    df['tr2'] = abs(low - close.shift(1))
    tr = df[['tr0', 'tr1', 'tr2']].max(axis=1)

    # ATR using RMA (Wilder's Smoothing)
    # alpha = 1 / period
    # pd.ewm(alpha=1/period, adjust=False).mean() matches Pine's ta.rma
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    return atr

def main():
    # Configuration
    sox_ticker = "^SOX"
    soxl_ticker = "SOXL"
    start_date = (datetime.datetime.now() - timedelta(days=365*4)).strftime('%Y-%m-%d')
    end_date = datetime.datetime.now().strftime('%Y-%m-%d')

    print(f"Fetching data for {sox_ticker} and {soxl_ticker} from {start_date} to {end_date}...")

    sox_data = yf.download(sox_ticker, start=start_date, end=end_date, progress=False)
    soxl_data = yf.download(soxl_ticker, start=start_date, end=end_date, progress=False)

    # Ensure timezone-naive index for merging
    if sox_data.index.tz is not None:
        sox_data.index = sox_data.index.tz_localize(None)
    if soxl_data.index.tz is not None:
        soxl_data.index = soxl_data.index.tz_localize(None)

    # Flatten columns if multi-index (common in yfinance latest versions)
    if isinstance(sox_data.columns, pd.MultiIndex):
        sox_data.columns = sox_data.columns.get_level_values(0)
    if isinstance(soxl_data.columns, pd.MultiIndex):
        soxl_data.columns = soxl_data.columns.get_level_values(0)

    # Calculate indicators on SOX
    print("Calculating indicators on SOX...")

    # Buy Settings
    buy_key = 2.0
    buy_atr_period = 300

    # Sell Settings
    sell_key = 2.0
    sell_atr_period = 1

    # Calculate ATR
    sox_data['xATR_Buy'] = calculate_atr(sox_data, buy_atr_period)
    sox_data['xATR_Sell'] = calculate_atr(sox_data, sell_atr_period)

    sox_data['nLoss_Buy'] = buy_key * sox_data['xATR_Buy']
    sox_data['nLoss_Sell'] = sell_key * sox_data['xATR_Sell']

    # Logic State Machine
    pos = 0
    xATRTrailingStop = 0.0

    positions = []
    trailing_stops = []

    # Iterate through rows
    # We need to access previous values, so iteration is easiest
    # Create numpy arrays for speed if needed, but simple iteration is fine for < 2000 rows

    # Initialize variables for the loop
    # We start from index 1 (need previous value)
    # Initialize first row
    positions.append(0)
    trailing_stops.append(0.0)

    src = sox_data['Close'].values
    nLoss_Buy = sox_data['nLoss_Buy'].values
    nLoss_Sell = sox_data['nLoss_Sell'].values

    for i in range(1, len(sox_data)):
        prev_pos = positions[-1]
        prev_stop = trailing_stops[-1]
        curr_src = src[i]

        # Check for NaN (at the beginning due to ATR calculation)
        if np.isnan(nLoss_Buy[i]) or np.isnan(nLoss_Sell[i]):
            positions.append(0)
            trailing_stops.append(0.0)
            continue

        if prev_pos == 0:
            # Initialize
            # If we don't have a previous state, let's assume Short (pos=-1) logic to start?
            # Or use logic similar to Pine:
            # In Pine, nz(pos[1]) is 0.
            # if prev_pos == 1 -> False
            # else -> check Short logic.
            # new_ceiling = src + nLoss_Buy
            # if src > prev_stop (0) -> True.
            # xATRTrailingStop = src - nLoss_Sell
            # pos = 1
            # So it initializes to Long on the first valid bar.

            new_ceiling = curr_src + nLoss_Buy[i]
            if curr_src > prev_stop:
                curr_stop = curr_src - nLoss_Sell[i]
                curr_pos = 1
            else:
                curr_stop = min(prev_stop, new_ceiling)
                curr_pos = -1

        elif prev_pos == 1:
            # Currently Long
            new_floor = curr_src - nLoss_Sell[i]
            if curr_src < prev_stop:
                curr_stop = curr_src + nLoss_Buy[i]
                curr_pos = -1
            else:
                curr_stop = max(prev_stop, new_floor)
                curr_pos = 1

        else: # prev_pos == -1
            # Currently Short
            new_ceiling = curr_src + nLoss_Buy[i]
            if curr_src > prev_stop:
                curr_stop = curr_src - nLoss_Sell[i]
                curr_pos = 1
            else:
                curr_stop = min(prev_stop, new_ceiling)
                curr_pos = -1

        positions.append(curr_pos)
        trailing_stops.append(curr_stop)

    sox_data['pos'] = positions
    sox_data['trailing_stop'] = trailing_stops

    # Generate signals
    # Buy: pos == 1 and prev_pos == -1
    # Sell: pos == -1 and prev_pos == 1
    sox_data['prev_pos'] = sox_data['pos'].shift(1)
    sox_data['Buy_Signal'] = (sox_data['pos'] == 1) & (sox_data['prev_pos'] == -1)
    sox_data['Sell_Signal'] = (sox_data['pos'] == -1) & (sox_data['prev_pos'] == 1)

    # Backtest Simulation
    print("Running backtest simulation...")

    # Join SOX signals with SOXL data
    # We use SOXL 'Open' for execution on the NEXT day after signal

    simulation_df = pd.DataFrame(index=soxl_data.index)
    simulation_df['SOXL_Open'] = soxl_data['Open']
    simulation_df['SOXL_Close'] = soxl_data['Close']

    # Align signals. Signal occurs at Close of day T. Trade at Open of day T+1.
    # So we shift signals forward by 1 day to align with trade execution day.

    # Map SOX signals to simulation_df (joining on index)
    # We need to handle potential index mismatch (trading holidays? usually same for US markets)
    # Inner join to be safe

    combined = pd.concat([
        sox_data[['Buy_Signal', 'Sell_Signal', 'pos']],
        simulation_df
    ], axis=1, join='inner')

    # Filter for the last 1 year
    one_year_ago = (datetime.datetime.now() - timedelta(days=365)).replace(hour=0, minute=0, second=0, microsecond=0)
    # Find the closest date or start date
    test_data = combined[combined.index >= one_year_ago].copy()

    if len(test_data) == 0:
        print("Error: No data for the last year.")
        return

    # Strategy 1: UT Bot on SOX -> Trade SOXL
    capital = 10000.0
    shares = 0
    in_position = False

    # We need to check the state at the beginning of the test period.
    # If the system was already Long (pos=1) coming into the period, should we be Long?
    # Standard backtest: Start flat? Or Assume ongoing position?
    # "1年前から比較して" implies we start the comparison from 1 year ago.
    # If the strategy was already Long 1 year ago, we should probably Buy at the start.
    # Let's check the 'pos' value of the day BEFORE the test_data starts.

    start_index = combined.index.get_loc(test_data.index[0])
    initial_pos = combined['pos'].iloc[start_index - 1] if start_index > 0 else 0

    # If initial_pos == 1, we start with a Buy.
    # But wait, we execute at Open.
    # So if Day -1 was Long, we buy at Day 0 Open.

    trades_log = []

    cash = capital
    equity_curve = []

    # Initial Entry if already Long
    if initial_pos == 1:
        # Buy at Open of first day
        buy_price = test_data.iloc[0]['SOXL_Open']
        shares = cash / buy_price
        cash = 0
        in_position = True
        trades_log.append(f"Initial Entry: Buy {shares:.2f} shares at {buy_price:.2f} on {test_data.index[0].date()}")

    for i in range(len(test_data) - 1): # Iterate up to second to last day
        # Current day's signal (generated at Close) executes at Next Day's Open

        # We look at the signal generated TODAY (index i)
        # Execution is TOMORROW (index i+1)

        today_signal_buy = test_data.iloc[i]['Buy_Signal']
        today_signal_sell = test_data.iloc[i]['Sell_Signal']
        today_pos = test_data.iloc[i]['pos']

        next_open = test_data.iloc[i+1]['SOXL_Open']
        next_date = test_data.index[i+1]

        if today_signal_buy and not in_position:
            # Buy
            buy_price = next_open
            shares = cash / buy_price
            cash = 0
            in_position = True
            trades_log.append(f"Buy: {shares:.2f} shares at {buy_price:.2f} on {next_date.date()}")

        elif today_signal_sell and in_position:
            # Sell
            sell_price = next_open
            cash = shares * sell_price
            shares = 0
            in_position = False
            trades_log.append(f"Sell: at {sell_price:.2f} on {next_date.date()}. Cash: {cash:.2f}")

        # Update Equity
        current_equity = cash + (shares * test_data.iloc[i]['SOXL_Close']) # Mark to market at Close
        equity_curve.append(current_equity)

    # Final Value
    last_price = test_data.iloc[-1]['SOXL_Close']
    final_equity_strat = cash + (shares * last_price)
    equity_curve.append(final_equity_strat)

    # Strategy 2: Buy and Hold SOXL
    # Buy at Open of first day
    bh_buy_price = test_data.iloc[0]['SOXL_Open']
    bh_shares = capital / bh_buy_price
    bh_final_value = bh_shares * last_price

    # Output
    print("-" * 50)
    print(f"Backtest Period: {test_data.index[0].date()} to {test_data.index[-1].date()}")
    print("-" * 50)
    print("Strategy 1: UT Bot (SOX Signals -> Trade SOXL)")
    print(f"Initial Capital: ${capital:,.2f}")
    print(f"Final Equity:    ${final_equity_strat:,.2f}")
    print(f"Return:          {((final_equity_strat - capital) / capital) * 100:.2f}%")
    print(f"Trades:          {len(trades_log)}")
    # for trade in trades_log:
    #     print(trade)

    print("-" * 50)
    print("Strategy 2: Buy and Hold SOXL")
    print(f"Initial Capital: ${capital:,.2f}")
    print(f"Final Equity:    ${bh_final_value:,.2f}")
    print(f"Return:          {((bh_final_value - capital) / capital) * 100:.2f}%")
    print("-" * 50)

    if final_equity_strat > bh_final_value:
        print("Winner: UT Bot Strategy")
    else:
        print("Winner: Buy and Hold")

if __name__ == "__main__":
    main()
