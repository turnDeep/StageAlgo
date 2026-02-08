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
    signal_ticker = "GLD"  # SPDR Gold Shares
    trade_ticker = "GDXU"  # MicroSectors Gold Miners 3X Leveraged ETN

    # Note: GDXU inception date is around 2020. If 4 years of history is requested,
    # yfinance will return what it has.
    start_date = (datetime.datetime.now() - timedelta(days=365*4)).strftime('%Y-%m-%d')
    end_date = datetime.datetime.now().strftime('%Y-%m-%d')

    print(f"Fetching data for {signal_ticker} and {trade_ticker} from {start_date} to {end_date}...")

    signal_data = yf.download(signal_ticker, start=start_date, end=end_date, progress=False)
    trade_data = yf.download(trade_ticker, start=start_date, end=end_date, progress=False)

    # Ensure timezone-naive index for merging
    if signal_data.index.tz is not None:
        signal_data.index = signal_data.index.tz_localize(None)
    if trade_data.index.tz is not None:
        trade_data.index = trade_data.index.tz_localize(None)

    # Flatten columns if multi-index
    if isinstance(signal_data.columns, pd.MultiIndex):
        signal_data.columns = signal_data.columns.get_level_values(0)
    if isinstance(trade_data.columns, pd.MultiIndex):
        trade_data.columns = trade_data.columns.get_level_values(0)

    # Calculate indicators on Signal Ticker (GLD)
    print(f"Calculating indicators on {signal_ticker}...")

    # Buy Settings
    buy_key = 2.0
    buy_atr_period = 300

    # Sell Settings
    sell_key = 2.0
    sell_atr_period = 1

    # Calculate ATR
    signal_data['xATR_Buy'] = calculate_atr(signal_data, buy_atr_period)
    signal_data['xATR_Sell'] = calculate_atr(signal_data, sell_atr_period)

    signal_data['nLoss_Buy'] = buy_key * signal_data['xATR_Buy']
    signal_data['nLoss_Sell'] = sell_key * signal_data['xATR_Sell']

    # Logic State Machine
    pos = 0
    xATRTrailingStop = 0.0

    positions = []
    trailing_stops = []

    # Initialize variables for the loop
    positions.append(0)
    trailing_stops.append(0.0)

    src = signal_data['Close'].values
    nLoss_Buy = signal_data['nLoss_Buy'].values
    nLoss_Sell = signal_data['nLoss_Sell'].values

    for i in range(1, len(signal_data)):
        prev_pos = positions[-1]
        prev_stop = trailing_stops[-1]
        curr_src = src[i]

        if np.isnan(nLoss_Buy[i]) or np.isnan(nLoss_Sell[i]):
            positions.append(0)
            trailing_stops.append(0.0)
            continue

        if prev_pos == 0:
            new_ceiling = curr_src + nLoss_Buy[i]
            if curr_src > prev_stop:
                curr_stop = curr_src - nLoss_Sell[i]
                curr_pos = 1
            else:
                curr_stop = min(prev_stop, new_ceiling)
                curr_pos = -1

        elif prev_pos == 1:
            new_floor = curr_src - nLoss_Sell[i]
            if curr_src < prev_stop:
                curr_stop = curr_src + nLoss_Buy[i]
                curr_pos = -1
            else:
                curr_stop = max(prev_stop, new_floor)
                curr_pos = 1

        else: # prev_pos == -1
            new_ceiling = curr_src + nLoss_Buy[i]
            if curr_src > prev_stop:
                curr_stop = curr_src - nLoss_Sell[i]
                curr_pos = 1
            else:
                curr_stop = min(prev_stop, new_ceiling)
                curr_pos = -1

        positions.append(curr_pos)
        trailing_stops.append(curr_stop)

    signal_data['pos'] = positions
    signal_data['trailing_stop'] = trailing_stops

    # Generate signals
    signal_data['prev_pos'] = signal_data['pos'].shift(1)
    signal_data['Buy_Signal'] = (signal_data['pos'] == 1) & (signal_data['prev_pos'] == -1)
    signal_data['Sell_Signal'] = (signal_data['pos'] == -1) & (signal_data['prev_pos'] == 1)

    # Backtest Simulation
    print("Running backtest simulation...")

    simulation_df = pd.DataFrame(index=trade_data.index)
    simulation_df['GDXU_Open'] = trade_data['Open']
    simulation_df['GDXU_Close'] = trade_data['Close']

    # Align signals
    combined = pd.concat([
        signal_data[['Buy_Signal', 'Sell_Signal', 'pos']],
        simulation_df
    ], axis=1, join='inner')

    # Filter for the last 1 year
    one_year_ago = (datetime.datetime.now() - timedelta(days=365)).replace(hour=0, minute=0, second=0, microsecond=0)
    test_data = combined[combined.index >= one_year_ago].copy()

    if len(test_data) == 0:
        print("Error: No data for the last year.")
        return

    # Strategy 1: UT Bot on GLD -> Trade GDXU
    capital = 10000.0
    shares = 0
    in_position = False

    start_index = combined.index.get_loc(test_data.index[0])
    initial_pos = combined['pos'].iloc[start_index - 1] if start_index > 0 else 0

    trades_log = []
    cash = capital
    equity_curve = []

    # Initial Entry
    if initial_pos == 1:
        buy_price = test_data.iloc[0]['GDXU_Open']
        shares = cash / buy_price
        cash = 0
        in_position = True
        trades_log.append(f"Initial Entry: Buy {shares:.2f} shares at {buy_price:.2f} on {test_data.index[0].date()}")

    for i in range(len(test_data) - 1):
        today_signal_buy = test_data.iloc[i]['Buy_Signal']
        today_signal_sell = test_data.iloc[i]['Sell_Signal']

        next_open = test_data.iloc[i+1]['GDXU_Open']
        next_date = test_data.index[i+1]

        if today_signal_buy and not in_position:
            buy_price = next_open
            shares = cash / buy_price
            cash = 0
            in_position = True
            trades_log.append(f"Buy: {shares:.2f} shares at {buy_price:.2f} on {next_date.date()}")

        elif today_signal_sell and in_position:
            sell_price = next_open
            cash = shares * sell_price
            shares = 0
            in_position = False
            trades_log.append(f"Sell: at {sell_price:.2f} on {next_date.date()}. Cash: {cash:.2f}")

        current_equity = cash + (shares * test_data.iloc[i]['GDXU_Close'])
        equity_curve.append(current_equity)

    last_price = test_data.iloc[-1]['GDXU_Close']
    final_equity_strat = cash + (shares * last_price)
    equity_curve.append(final_equity_strat)

    # Strategy 2: Buy and Hold GDXU
    bh_buy_price = test_data.iloc[0]['GDXU_Open']
    bh_shares = capital / bh_buy_price
    bh_final_value = bh_shares * last_price

    # Output
    print("-" * 50)
    print(f"Backtest Period: {test_data.index[0].date()} to {test_data.index[-1].date()}")
    print("-" * 50)
    print(f"Strategy 1: UT Bot ({signal_ticker} Signals -> Trade {trade_ticker})")
    print(f"Initial Capital: ${capital:,.2f}")
    print(f"Final Equity:    ${final_equity_strat:,.2f}")
    print(f"Return:          {((final_equity_strat - capital) / capital) * 100:.2f}%")
    print(f"Trades:          {len(trades_log)}")

    print("-" * 50)
    print(f"Strategy 2: Buy and Hold {trade_ticker}")
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
