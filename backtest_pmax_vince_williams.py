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
CLIMAX_THRESHOLD = 20.0
BLOODBATH_THRESHOLD = 4.0
ENTRY_LAG = 22 # days

def load_breadth_data():
    if not os.path.exists(BREADTH_FILE):
        print(f"Error: {BREADTH_FILE} not found. Run market_breadth_generator.py first.")
        return None

    df = pd.read_csv(BREADTH_FILE)
    if 'Unnamed: 0' in df.columns: # Assuming previous script might save index as column
        df.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
    if 'Date' not in df.columns and df.index.name != 'Date':
         # Check if first column is date
         df.rename(columns={df.columns[0]: 'Date'}, inplace=True)

    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)

    df.sort_index(inplace=True)
    return df

def resample_to_weekly(df):
    """
    Resamples daily data to weekly bars (Friday close).
    """
    logic = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }
    # Check if columns exist
    logic = {k: v for k, v in logic.items() if k in df.columns}

    df_w = df.resample('W-FRI').agg(logic).dropna()
    return df_w

def calculate_pmax(df, length=10, multiplier=3):
    # Using pandas_ta supertrend
    st = ta.supertrend(df['High'], df['Low'], df['Close'], length=length, multiplier=multiplier)
    if st is None:
        return pd.Series(0, index=df.index), pd.Series(0, index=df.index)

    # ST columns: SUPERT_7_3.0 (trend), SUPERTd_7_3.0 (direction 1/-1)
    # Find columns dynamically
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

    # Flatten MultiIndex
    if isinstance(sox.columns, pd.MultiIndex): sox.columns = sox.columns.get_level_values(0)
    if isinstance(soxl.columns, pd.MultiIndex): soxl.columns = soxl.columns.get_level_values(0)

    # Timezone remove
    if sox.index.tz is not None: sox.index = sox.index.tz_localize(None)
    if soxl.index.tz is not None: soxl.index = soxl.index.tz_localize(None)
    if breadth_df.index.tz is not None: breadth_df.index = breadth_df.index.tz_localize(None)

    # 2. Process Breadth Signals (Daily)
    # Calculate Ratio if not present
    if 'New_Lows_Ratio' not in breadth_df.columns:
        breadth_df['New_Lows_Ratio'] = (breadth_df['New_Lows'] / breadth_df['Total_Issues']) * 100

    # Detect Climax Days (Daily)
    breadth_df['Is_Climax'] = breadth_df['New_Lows_Ratio'] >= CLIMAX_THRESHOLD

    # Detect Bloodbath Days (Daily)
    breadth_df['Is_Bloodbath'] = breadth_df['New_Lows_Ratio'] >= BLOODBATH_THRESHOLD

    # Calculate Entry Signal (Climax + 22 Days)
    # We shift the Climax signal FORWARD by 22 trading days.
    # .shift(22) means the value at T is the value from T-22.
    # So if T-22 was Climax, T is Entry.
    breadth_df['Climax_Entry'] = breadth_df['Is_Climax'].shift(ENTRY_LAG).fillna(False)

    # 3. Strategy Logic (Weekly PMax + Daily Breadth Integration)
    # The user wants "Formal Implementation into PMax".
    # PMax is Weekly. Breadth is Daily.
    # Option A: Run everything on Daily. (PMax on Daily is choppy).
    # Option B: Run PMax on Weekly, but check Daily Breadth signals during the week?
    # Option C: Resample Breadth to Weekly? (Loss of precision: Climax might happen on Tuesday).
    # Option D: Keep PMax Weekly, but allow Climax Entry to trigger a "Weekly Buy" if the Entry Date falls within that week?

    # Let's stick to Weekly PMax as the base frame because it performs best.
    # We will map Daily Breadth signals to the Weekly bars.
    # For a given Week (ending Friday), if ANY day in that week (Mon-Fri) had a 'Climax_Entry' signal, we treat the Week as having an Entry Signal.
    # For Bloodbath: If the Week ends with Bloodbath active? Or if any day was Bloodbath?
    # Probably if the most recent status (Friday) is Bloodbath?
    # Or average?
    # Let's say: If 'Climax_Entry' triggers on Tuesday, we buy. In a Weekly backtest, we can assume we buy at that week's Close (or next week's Open).
    # To be precise, let's look at Weekly bars.

    sox_w = resample_to_weekly(sox)
    soxl_w = resample_to_weekly(soxl)

    # Calculate PMax (Weekly)
    pmax_dir, pmax_val = calculate_pmax(sox_w)
    sox_w['PMax_Dir'] = pmax_dir
    sox_w['PMax_Val'] = pmax_val

    # Map Breadth to Weekly
    # 1. Climax Entry: If any day in the week is an entry day.
    # 2. Bloodbath: If the ratio at the END of the week (Friday) > 4%? Or average > 4%?
    # Vince/Williams says "Stay out". If the market is in Bloodbath mode (Breadth > 4%), we should be out.
    # Let's take the Breadth Ratio of the last day of the week (Friday close).

    # Resample Breadth to Weekly
    # Logic: 'New_Lows_Ratio': 'last' (Status at end of week)
    # 'Climax_Entry': 'max' (True if any day was True)
    breadth_w = breadth_df.resample('W-FRI').agg({
        'New_Lows_Ratio': 'last',
        'Climax_Entry': 'max' # boolean max is OR
    })

    # Merge
    df = pd.concat([sox_w, breadth_w], axis=1).dropna()

    # Align SOXL
    df['SOXL_Open'] = soxl_w['Open']
    df['SOXL_Close'] = soxl_w['Close']
    df.dropna(inplace=True)

    # 4. Simulation Loop

    # PMax Logic:
    # Buy: PMax_Dir flips -1 -> 1
    # Sell: PMax_Dir flips 1 -> -1

    # Enhanced Logic:
    # Buy Trigger 1: PMax Buy Signal AND (NOT Bloodbath)
    # Buy Trigger 2: Climax Entry Signal (Overrides Bloodbath? Overrides PMax Sell?) -> "Ultimate Buy".
    # Sell Trigger 1: PMax Sell Signal
    # Sell Trigger 2: Bloodbath Start? (If New Lows > 4%, exit immediately?)
    # "Bloodbath Sidestepping Rule": Avoid Longs. If Long, Sell.

    # Priority:
    # 1. Climax Entry (Strongest Buy) -> Enter Long.
    # 2. Bloodbath (>4%) -> Exit Long / Stay Cash (Unless Climax Entry just happened? Climax Entry happens 22 days AFTER panic. Usually panic is >20%. 22 days later it might be calm (<4%). If it's still >4%, maybe wait? But Rule says "22 days later". Let's assume Time Filter is sufficient).
    #    However, if Breadth > 4% generally, we step aside.
    #    Conflict: What if T+22 arrives and Breadth is 5%?
    #    Vince: "Wait for it to drop below 4%?" or "Time filter handles it".
    #    Let's assume Climax Entry (Time Based) overrides Bloodbath check for that specific entry.
    # 3. PMax Trend -> If Green and Not Bloodbath -> Long.

    pos = 0 # 0 or 1
    equity = 1.0
    equity_curve = [1.0]

    log_data = []

    for i in range(1, len(df)):
        # Data for TODAY (deciding for Tomorrow/Close)
        # We trade at Close of 'i' (simplification)

        date = df.index[i]
        price_soxl = df['SOXL_Close'].iloc[i]
        prev_price_soxl = df['SOXL_Close'].iloc[i-1]

        pmax_dir = df['PMax_Dir'].iloc[i] # 1 or -1
        breadth_ratio = df['New_Lows_Ratio'].iloc[i]
        climax_entry = df['Climax_Entry'].iloc[i] # True/False

        # Determine Desired Position
        signal = 0 # Cash

        is_bloodbath = breadth_ratio > BLOODBATH_THRESHOLD

        # Logic Tree
        if climax_entry:
            signal = 1 # Force Buy (Climax Bottom)
        elif is_bloodbath:
            signal = 0 # Step Aside (Danger Zone)
        elif pmax_dir == 1:
            signal = 1 # PMax Trend Following
        else:
            signal = 0 # PMax Bearish

        # Execute Strategy (Calculate Return from prev close to this close if we held)
        # Actually, we decide position at i, realize return at i+1.
        # So we update equity based on POS from i-1.

        prev_pos = 0 if i==1 else log_data[-1]['Position']

        # Return of holding from i-1 to i
        ret = (price_soxl - prev_price_soxl) / prev_price_soxl

        if prev_pos == 1:
            equity *= (1 + ret)

        equity_curve.append(equity)

        log_data.append({
            'Date': date,
            'Price': price_soxl,
            'PMax': pmax_dir,
            'Breadth': breadth_ratio,
            'Climax': climax_entry,
            'Bloodbath': is_bloodbath,
            'Position': signal, # Desired position for NEXT period
            'Equity': equity
        })

    results_df = pd.DataFrame(log_data)
    results_df.set_index('Date', inplace=True)

    # Calculate Buy & Hold
    soxl_bh = (df['SOXL_Close'] / df['SOXL_Close'].iloc[0])

    # 5. Reporting
    total_ret_strat = (results_df['Equity'].iloc[-1] - 1) * 100
    total_ret_bh = (soxl_bh.iloc[-1] - 1) * 100

    print("\n" + "="*50)
    print(f"BACKTEST RESULTS (Weekly PMax + Vince/Williams)")
    print(f"Period: {df.index[0].date()} to {df.index[-1].date()}")
    print("="*50)
    print(f"Strategy Return: {total_ret_strat:.2f}%")
    print(f"Buy & Hold     : {total_ret_bh:.2f}%")
    print("="*50)

    # 6. Visualization
    plt.figure(figsize=(14, 10))

    # Ax1: Equity
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(results_df.index, results_df['Equity'], label='Strategy', color='green')
    ax1.plot(soxl_bh.index, soxl_bh, label='Buy & Hold', color='gray', alpha=0.5)
    ax1.set_title('Equity Curve')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True)

    # Ax2: Breadth
    ax2 = plt.subplot(3, 1, 2, sharex=ax1)
    ax2.plot(results_df.index, results_df['Breadth'], label='New Lows %', color='blue')
    ax2.axhline(CLIMAX_THRESHOLD, color='red', linestyle='--', label='Climax (20%)')
    ax2.axhline(BLOODBATH_THRESHOLD, color='orange', linestyle='--', label='Bloodbath (4%)')
    ax2.fill_between(results_df.index, 0, 100, where=results_df['Bloodbath'], color='orange', alpha=0.2, label='Bloodbath Zone')
    ax2.scatter(results_df[results_df['Climax']].index, results_df[results_df['Climax']]['Breadth'], color='purple', marker='*', s=100, label='Climax Entry Trigger')
    ax2.set_title('Market Breadth (New Lows %)')
    ax2.legend()
    ax2.grid(True)

    # Ax3: PMax
    ax3 = plt.subplot(3, 1, 3, sharex=ax1)
    ax3.plot(df.index, df['Close'], label='^SOX Price', color='black')
    # Overlay signals?
    # Simple color bar for position
    # Create a collection for position
    # If Pos=1 Green, Pos=0 Red/White

    ax3.fill_between(results_df.index, df['Close'].min(), df['Close'].max(), where=(results_df['Position']==1), color='green', alpha=0.1, label='Long Position')
    ax3.set_title('Strategy Position')
    ax3.legend()

    plt.tight_layout()
    plt.savefig('pmax_vince_williams_result.png')
    print("Chart saved to pmax_vince_williams_result.png")

if __name__ == "__main__":
    run_backtest()
