import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
import matplotlib.pyplot as plt
import mplfinance as mpf
from datetime import datetime, timedelta
import os
import sys

# Constants
BREADTH_FILE = 'market_breadth_history.csv'
SOX_TICKER = '^SOX'
SOXL_TICKER = 'SOXL'
PMAX_LENGTH = 10
PMAX_MULTIPLIER = 3

# DAILY LOGIC PARAMETERS
CLIMAX_THRESHOLD_DAILY = 20.0
BLOODBATH_THRESHOLD_DAILY = 4.0
ENTRY_LAG_DAYS = 22

# Backtest Start Date
START_DATE_STR = '2016-07-22'

def load_breadth_data():
    if not os.path.exists(BREADTH_FILE):
        print(f"Error: {BREADTH_FILE} not found.")
        return None

    df = pd.read_csv(BREADTH_FILE)
    if 'Unnamed: 0' in df.columns:
        df.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
    if 'Date' not in df.columns and df.index.name != 'Date':
         df.rename(columns={df.columns[0]: 'Date'}, inplace=True)

    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)

    df.sort_index(inplace=True)
    return df

def resample_to_weekly(df):
    """
    Resamples daily data to weekly bars (Friday Close).
    Used for PMax calculation.
    """
    logic = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }
    logic = {k: v for k, v in logic.items() if k in df.columns}
    df_w = df.resample('W-FRI').agg(logic).dropna()
    return df_w

def calculate_pmax(df, length=10, multiplier=3):
    st = ta.supertrend(df['High'], df['Low'], df['Close'], length=length, multiplier=multiplier)
    if st is None:
        return pd.Series(0, index=df.index), pd.Series(0, index=df.index)

    col_dir = [c for c in st.columns if c.startswith("SUPERTd")][0]
    return st[col_dir], st[col_dir] # Return Dir twice as val not needed for logic

def run_backtest():
    # 1. Load Breadth (Daily)
    print("Loading Market Breadth Data...")
    breadth_df = load_breadth_data()
    if breadth_df is None: return

    # Filter Start Date
    start_dt = pd.to_datetime(START_DATE_STR)

    # 2. Process Daily Breadth Signals
    if 'New_Lows_Ratio' not in breadth_df.columns:
        breadth_df['New_Lows_Ratio'] = (breadth_df['New_Lows'] / breadth_df['Total_Issues']) * 100

    # Signals
    breadth_df['Is_Climax'] = breadth_df['New_Lows_Ratio'] >= CLIMAX_THRESHOLD_DAILY
    breadth_df['Is_Bloodbath'] = breadth_df['New_Lows_Ratio'] >= BLOODBATH_THRESHOLD_DAILY

    # Entry Signal (22 Days Lag)
    breadth_df['Climax_Entry'] = breadth_df['Is_Climax'].shift(ENTRY_LAG_DAYS).fillna(False)

    # 3. Load Price Data
    print("Fetching ^SOX and SOXL Data...")
    # Fetch ample history to cover start date logic
    fetch_start = start_dt - timedelta(days=365)

    sox = yf.download(SOX_TICKER, start=fetch_start, progress=False)
    soxl = yf.download(SOXL_TICKER, start=fetch_start, progress=False)

    if isinstance(sox.columns, pd.MultiIndex): sox.columns = sox.columns.get_level_values(0)
    if isinstance(soxl.columns, pd.MultiIndex): soxl.columns = soxl.columns.get_level_values(0)

    if sox.index.tz is not None: sox.index = sox.index.tz_localize(None)
    if soxl.index.tz is not None: soxl.index = soxl.index.tz_localize(None)
    if breadth_df.index.tz is not None: breadth_df.index = breadth_df.index.tz_localize(None)

    # 4. Integrate Weekly PMax with Daily Signals
    # We run the simulation on DAILY bars to capture the exact entry day.
    # But we check PMax status from Weekly bars.

    # Calculate Weekly PMax
    sox_w = resample_to_weekly(sox)
    pmax_dir_w, _ = calculate_pmax(sox_w)
    sox_w['PMax_Dir'] = pmax_dir_w

    # Resample PMax back to Daily (Forward Fill)
    # This simulates "Weekly state persists through the next week"
    pmax_daily = sox_w['PMax_Dir'].resample('D').ffill()

    # Merge everything into a Daily Simulation DF
    df = pd.DataFrame(index=soxl.index)
    df = df.join(soxl[['Open', 'Close']], how='inner')
    df = df.join(breadth_df[['New_Lows_Ratio', 'Climax_Entry', 'Is_Bloodbath']], how='left')
    df = df.join(pmax_daily.rename('PMax_Dir'), how='left')

    # Filter by requested Start Date
    df = df[df.index >= start_dt].copy()
    df.ffill(inplace=True) # Fill gaps if any
    df.dropna(inplace=True) # Drop initial NaNs if PMax not ready

    # 5. Simulation Loop
    pos = 0
    equity = 1.0
    equity_curve = [1.0]
    log_data = []

    # Initial PMax State might be NaN if start date is too early?
    # We filtered, so should be fine.

    # Logic:
    # 1. Climax Entry (True) -> Force Buy.
    # 2. Bloodbath (True) -> Sell/Cash.
    # 3. PMax (1) -> Buy.
    # 4. PMax (-1) -> Sell.
    # Priority: Climax Entry > Bloodbath > PMax.

    for i in range(1, len(df)):
        date = df.index[i]
        price = df['Close'].iloc[i]
        prev_price = df['Close'].iloc[i-1]

        climax_entry = df['Climax_Entry'].iloc[i]
        is_bloodbath = df['Is_Bloodbath'].iloc[i]
        pmax_dir = df['PMax_Dir'].iloc[i]

        # Signal
        signal = 0

        # Note: If Climax Entry happens ON a Bloodbath day?
        # Entry logic usually assumes "panic is over" (22 days later).
        # If 22 days later is ALSO a Bloodbath day (New lows > 4%), it's a double bottom or crash continuation.
        # Strict Vince rule: "Sidestep if > 4%".
        # But "Re-entry rule" is "22 days after > 20%".
        # We assume Re-entry overrides because it's a specific setup.

        if climax_entry:
            signal = 1
        elif is_bloodbath:
            signal = 0
        elif pmax_dir == 1:
            signal = 1
        else:
            signal = 0

        # Execute
        prev_pos = 0 if i==1 else log_data[-1]['Position']
        ret = (price - prev_price) / prev_price

        if prev_pos == 1:
            equity *= (1 + ret)

        equity_curve.append(equity)

        log_data.append({
            'Date': date,
            'Price': price,
            'Position': signal,
            'Equity': equity
        })

    # Results
    results_df = pd.DataFrame(log_data)
    if results_df.empty:
        print("No data in range.")
        return

    results_df.set_index('Date', inplace=True)

    # Buy & Hold
    bh_equity = (df['Close'] / df['Close'].iloc[0]).iloc[1:]

    total_ret = (results_df['Equity'].iloc[-1] - 1) * 100
    bh_ret = (bh_equity.iloc[-1] - 1) * 100

    print("\n" + "="*50)
    print(f"BACKTEST: Daily Logic (Start: {START_DATE_STR})")
    print("="*50)
    print(f"Strategy Return: {total_ret:.2f}%")
    print(f"Buy & Hold     : {bh_ret:.2f}%")
    print("="*50)

    # Plot
    plt.figure(figsize=(12,6))
    plt.plot(results_df.index, results_df['Equity'], label='Strategy')
    plt.plot(results_df.index, bh_equity, label='Buy & Hold', alpha=0.5)
    plt.title(f'Daily Climax + Weekly PMax ({START_DATE_STR} - Present)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.savefig('pmax_daily_2016.png')

if __name__ == "__main__":
    run_backtest()
