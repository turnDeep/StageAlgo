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
CLIMAX_THRESHOLD_WEEKLY = 13.0 # Optimized to catch 2016-01-04
BLOODBATH_THRESHOLD_WEEKLY = 4.0 # Should I adjust this? User asked to adjust Climax. Bloodbath usually 4%.
# If Weekly Climax is 13%, Daily Bloodbath (4%) is much lower.
# Weekly New Lows Aggregation (Max) means if ANY day > 4%, it's Bloodbath?
# Let's keep 4% for Bloodbath (defensive) as it's standard.
ENTRY_LAG_WEEKS = 4 # Approx 22 trading days

def load_breadth_data():
    if not os.path.exists(BREADTH_FILE):
        print(f"Error: {BREADTH_FILE} not found. Run market_breadth_generator.py first.")
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

def resample_to_weekly(df, method='last'):
    """
    Resamples daily data to weekly bars (Monday start).
    """
    logic = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }
    logic = {k: v for k, v in logic.items() if k in df.columns}

    # Resample W-MON, label='left' puts the date at the start of the week (Monday)
    df_w = df.resample('W-MON', closed='left', label='left').agg(logic).dropna()
    return df_w

def calculate_pmax(df, length=10, multiplier=3):
    st = ta.supertrend(df['High'], df['Low'], df['Close'], length=length, multiplier=multiplier)
    if st is None:
        return pd.Series(0, index=df.index), pd.Series(0, index=df.index)

    col_dir = [c for c in st.columns if c.startswith("SUPERTd")][0]
    col_val = [c for c in st.columns if c.startswith("SUPERT_")][0]

    return st[col_dir], st[col_val]

def run_backtest():
    # 1. Load Data
    print("Loading Market Breadth Data...")
    breadth_df = load_breadth_data()
    if breadth_df is None: return

    print("Fetching ^SOX and SOXL Data...")
    start_date = breadth_df.index[0]
    end_date = breadth_df.index[-1]

    sox = yf.download(SOX_TICKER, start=start_date, end=end_date, progress=False)
    soxl = yf.download(SOXL_TICKER, start=start_date, end=end_date, progress=False)

    if isinstance(sox.columns, pd.MultiIndex): sox.columns = sox.columns.get_level_values(0)
    if isinstance(soxl.columns, pd.MultiIndex): soxl.columns = soxl.columns.get_level_values(0)

    if sox.index.tz is not None: sox.index = sox.index.tz_localize(None)
    if soxl.index.tz is not None: soxl.index = soxl.index.tz_localize(None)
    if breadth_df.index.tz is not None: breadth_df.index = breadth_df.index.tz_localize(None)

    # 2. Resample Everything to Weekly (Monday)
    # Breadth Logic: 'New_Lows_Ratio': 'max' (If any day in week was Climax, week is Climax)
    if 'New_Lows_Ratio' not in breadth_df.columns:
        breadth_df['New_Lows_Ratio'] = (breadth_df['New_Lows'] / breadth_df['Total_Issues']) * 100

    breadth_w = breadth_df.resample('W-MON', closed='left', label='left').agg({
        'New_Lows_Ratio': 'max'
    })

    sox_w = resample_to_weekly(sox)
    soxl_w = resample_to_weekly(soxl)

    # 3. Calculate Indicators (Weekly)

    # PMax
    pmax_dir, pmax_val = calculate_pmax(sox_w)
    sox_w['PMax_Dir'] = pmax_dir

    # Breadth Signals
    breadth_w['Is_Climax'] = breadth_w['New_Lows_Ratio'] >= CLIMAX_THRESHOLD_WEEKLY
    breadth_w['Is_Bloodbath'] = breadth_w['New_Lows_Ratio'] >= BLOODBATH_THRESHOLD_WEEKLY

    # Entry Signal (Shift 4 Weeks)
    breadth_w['Climax_Entry'] = breadth_w['Is_Climax'].shift(ENTRY_LAG_WEEKS).fillna(False)

    # Merge
    # Use sox_w index as base
    df = pd.concat([sox_w, breadth_w], axis=1).dropna()

    # Align SOXL
    # We need SOXL Open/Close for trading
    # Assuming soxl_w index aligns (Mon-Mon)
    # Join
    df = df.join(soxl_w[['Open', 'Close']], rsuffix='_SOXL')
    df.rename(columns={'Open_SOXL': 'SOXL_Open', 'Close_SOXL': 'SOXL_Close'}, inplace=True)
    df.dropna(inplace=True)

    # 4. Simulation

    pos = 0
    equity = 1.0
    equity_curve = [1.0]
    log_data = []

    for i in range(1, len(df)):
        date = df.index[i]
        price_soxl = df['SOXL_Close'].iloc[i]
        prev_price_soxl = df['SOXL_Close'].iloc[i-1]

        pmax_dir = df['PMax_Dir'].iloc[i]
        climax_entry = df['Climax_Entry'].iloc[i]
        is_bloodbath = df['Is_Bloodbath'].iloc[i]

        # Signal Logic
        signal = 0

        if climax_entry:
            signal = 1 # Force Buy
        elif is_bloodbath:
            signal = 0 # Step Aside
        elif pmax_dir == 1:
            signal = 1
        else:
            signal = 0

        # Execute (from prev close to this close)
        prev_pos = 0 if i==1 else log_data[-1]['Position']
        ret = (price_soxl - prev_price_soxl) / prev_price_soxl

        if prev_pos == 1:
            equity *= (1 + ret)

        equity_curve.append(equity)

        log_data.append({
            'Date': date,
            'Price': price_soxl,
            'Position': signal,
            'Equity': equity,
            'Climax': climax_entry,
            'Bloodbath': is_bloodbath
        })

    results_df = pd.DataFrame(log_data)
    results_df.set_index('Date', inplace=True)

    # Buy & Hold
    soxl_bh = (df['SOXL_Close'] / df['SOXL_Close'].iloc[0])

    # Reporting
    total_ret_strat = (results_df['Equity'].iloc[-1] - 1) * 100
    total_ret_bh = (soxl_bh.iloc[-1] - 1) * 100

    print("\n" + "="*50)
    print(f"BACKTEST RESULTS (Weekly PMax + Weekly Climax)")
    print(f"Period: {df.index[0].date()} to {df.index[-1].date()}")
    print("="*50)
    print(f"Strategy Return: {total_ret_strat:.2f}%")
    print(f"Buy & Hold     : {total_ret_bh:.2f}%")
    print("="*50)
    print(f"Climax Threshold (Weekly Max): {CLIMAX_THRESHOLD_WEEKLY}%")

    # Plotting
    plt.figure(figsize=(14, 8))
    plt.plot(results_df.index, results_df['Equity'], label='Strategy')
    plt.plot(soxl_bh.index, soxl_bh, label='Buy & Hold', alpha=0.5)
    plt.yscale('log')
    plt.title('Weekly Strategy vs Buy & Hold')
    plt.legend()
    plt.grid(True)
    plt.savefig('pmax_vince_williams_weekly_result.png')

if __name__ == "__main__":
    run_backtest()
