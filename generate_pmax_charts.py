import yfinance as yf
import pandas as pd
import numpy as np
import datetime
from datetime import timedelta
import pandas_ta as ta
import mplfinance as mpf

# --- Helper Functions ---

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
    if 'Volume' not in df.columns:
        if 'Volume' in logic: del logic['Volume']
    df_w = df.resample('W-FRI').agg(logic).dropna()
    return df_w

def calculate_indicators(df):
    # PMax (SuperTrend)
    st = ta.supertrend(df['High'], df['Low'], df['Close'], length=10, multiplier=3)
    col_dir = [c for c in st.columns if c.startswith("SUPERTd")][0]
    col_val = [c for c in st.columns if c.startswith("SUPERT_")][0]

    df['PMax_Dir'] = st[col_dir]
    df['PMax_Line'] = st[col_val]

    # Vix Fix (Climax)
    period = 22
    highest_close = df['Close'].rolling(window=period).max()
    wvf = ((highest_close - df['Low']) / highest_close) * 100

    wvf_mean = wvf.rolling(window=20).mean()
    wvf_std = wvf.rolling(window=20).std()
    wvf_upper = wvf_mean + (2.0 * wvf_std)

    is_panic = (wvf >= wvf_upper)
    # Buy when Panic recedes (Yesterday was Panic, Today is not)
    # Note: Weekly bars.
    climax_buy = (is_panic.shift(1) == True) & (is_panic == False)
    df['Climax_Buy'] = climax_buy

    return df

def generate_charts():
    print("Fetching Daily data for ^SOX (10 years)...")
    start_date = (datetime.datetime.now() - timedelta(days=365*12)).strftime('%Y-%m-%d') # Extra buffer
    end_date = datetime.datetime.now().strftime('%Y-%m-%d')

    sox_d = yf.download("^SOX", start=start_date, end=end_date, progress=False)

    if isinstance(sox_d.columns, pd.MultiIndex): sox_d.columns = sox_d.columns.get_level_values(0)
    if sox_d.index.tz is not None: sox_d.index = sox_d.index.tz_localize(None)

    print("Resampling to Weekly...")
    sox_w = resample_to_weekly(sox_d)

    print("Calculating Indicators...")
    df = calculate_indicators(sox_w.copy())

    # Filter last 5 years for clearer chart visibility
    plot_start = (datetime.datetime.now() - timedelta(days=365*5))
    df = df[df.index >= plot_start].copy()

    # --- Prepare Plots ---

    # 1. PMax Signals
    # Buy: Dir flips -1 -> 1
    # Sell: Dir flips 1 -> -1
    df['Prev_Dir'] = df['PMax_Dir'].shift(1)

    buy_sigs = np.where((df['PMax_Dir'] == 1) & (df['Prev_Dir'] == -1), df['Low']*0.95, np.nan)
    sell_sigs = np.where((df['PMax_Dir'] == -1) & (df['Prev_Dir'] == 1), df['High']*1.05, np.nan)

    # 2. Climax Signals
    # Only valid if PMax is -1 (Bearish)
    climax_sigs = np.where((df['Climax_Buy'] == True) & (df['PMax_Dir'] == -1), df['Low']*0.90, np.nan)

    # Plot 1: Pure PMax
    apds_1 = [
        mpf.make_addplot(df['PMax_Line'], color='orange', width=1.5),
        mpf.make_addplot(buy_sigs, type='scatter', markersize=100, marker='^', color='green', label='Buy'),
        mpf.make_addplot(sell_sigs, type='scatter', markersize=100, marker='v', color='red', label='Sell')
    ]

    print("Generating Chart 1: Pure PMax...")
    mpf.plot(df, type='candle', style='yahoo', addplot=apds_1,
             title='SOX Weekly - Pure PMax (SuperTrend)',
             ylabel='Price', volume=False,
             savefig='pmax_weekly_sox.png',
             figsize=(12, 8))

    # Plot 2: PMax + Climax
    # Add Climax markers
    apds_2 = [
        mpf.make_addplot(df['PMax_Line'], color='orange', width=1.5),
        mpf.make_addplot(buy_sigs, type='scatter', markersize=100, marker='^', color='green'),
        mpf.make_addplot(sell_sigs, type='scatter', markersize=100, marker='v', color='red'),
        mpf.make_addplot(climax_sigs, type='scatter', markersize=150, marker='*', color='blue', label='Climax Buy')
    ]

    print("Generating Chart 2: PMax + Climax...")
    mpf.plot(df, type='candle', style='yahoo', addplot=apds_2,
             title='SOX Weekly - PMax + Selling Climax',
             ylabel='Price', volume=False,
             savefig='pmax_climax_weekly_sox.png',
             figsize=(12, 8))

    print("Done.")

if __name__ == "__main__":
    generate_charts()
