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

# PINE SCRIPT PARAMS
ST_PERIOD = 10
ST_MULTIPLIER = 3.0

# CLIMAX LOGIC (Daily)
CLIMAX_THRESHOLD_DAILY = 20.0
BLOODBATH_THRESHOLD_DAILY = 4.0
ENTRY_LAG_DAYS = 22

# Start Date: Use earliest available breadth data (approx 2011) for max history
# Or 2016 if preferred? User didn't specify date, but previous was 2016.
# Let's use 2011 to show full potential as this is a new strategy test.
START_DATE_STR = '2011-01-01'

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

def calculate_kivanc_supertrend(df, period=10, multiplier=3.0):
    """
    Implements KivancOzbilgic's SuperTrend Logic (Pine Script v4).
    src = hl2
    atr = atr(period)
    up = src - (multiplier * atr)
    dn = src + (multiplier * atr)
    """
    # 1. Calculate ATR and Source
    # pandas_ta atr uses RMA smoothing by default, matching Pine
    atr = ta.atr(df['High'], df['Low'], df['Close'], length=period)
    src = (df['High'] + df['Low']) / 2.0

    # 2. Arrays for iteration
    n = len(df)
    close = df['Close'].values
    atr_val = atr.values
    src_val = src.values

    up = np.zeros(n)
    dn = np.zeros(n)
    trend = np.zeros(n) # 1 for Up, -1 for Down

    # Initialize (avoid NaN issues at start)
    # Forward fill ATR NaNs or skip
    start_idx = period # Roughly

    trend[0:start_idx] = 1 # Default

    for i in range(1, n):
        if np.isnan(atr_val[i]):
            continue

        # Basic Bands
        basic_up = src_val[i] - (multiplier * atr_val[i])
        basic_dn = src_val[i] + (multiplier * atr_val[i])

        # Recursive Bands
        # Pine: up1 = nz(up[1], up) -> prev_up
        # Pine: up := close[1] > up1 ? max(up, up1) : up
        prev_up = up[i-1]
        prev_close = close[i-1]

        if prev_close > prev_up:
            up[i] = max(basic_up, prev_up)
        else:
            up[i] = basic_up

        # Pine: dn1 = nz(dn[1], dn) -> prev_dn
        # Pine: dn := close[1] < dn1 ? min(dn, dn1) : dn
        prev_dn = dn[i-1]

        if prev_close < prev_dn:
            dn[i] = min(basic_dn, prev_dn)
        else:
            dn[i] = basic_dn

        # Trend
        # Pine: trend := trend == -1 and close > dn1 ? 1 : trend == 1 and close < up1 ? -1 : trend
        # Note: dn1 is prev_dn (from previous iteration of THIS loop structure in Pine,
        # but in Pine 'dn' is a series. dn[1] is yesterday's final dn).
        # In our loop, dn[i-1] is exactly that.

        prev_trend = trend[i-1]
        curr_close = close[i]

        if prev_trend == -1 and curr_close > prev_dn:
            trend[i] = 1
        elif prev_trend == 1 and curr_close < prev_up:
            trend[i] = -1
        else:
            trend[i] = prev_trend

    return pd.Series(trend, index=df.index), pd.Series(up, index=df.index), pd.Series(dn, index=df.index)

def run_backtest():
    # 1. Load Data
    print("Loading Breadth Data...")
    breadth_df = load_breadth_data()
    if breadth_df is None: return

    print("Fetching Price Data...")
    start_dt = pd.to_datetime(START_DATE_STR)
    fetch_start = start_dt - timedelta(days=365) # Warmup

    sox = yf.download(SOX_TICKER, start=fetch_start, progress=False)
    soxl = yf.download(SOXL_TICKER, start=fetch_start, progress=False)

    # Fix MultiIndex/Timezone
    if isinstance(sox.columns, pd.MultiIndex): sox.columns = sox.columns.get_level_values(0)
    if isinstance(soxl.columns, pd.MultiIndex): soxl.columns = soxl.columns.get_level_values(0)
    if sox.index.tz is not None: sox.index = sox.index.tz_localize(None)
    if soxl.index.tz is not None: soxl.index = soxl.index.tz_localize(None)
    if breadth_df.index.tz is not None: breadth_df.index = breadth_df.index.tz_localize(None)

    # 2. Climax Logic (Daily)
    if 'New_Lows_Ratio' not in breadth_df.columns:
        breadth_df['New_Lows_Ratio'] = (breadth_df['New_Lows'] / breadth_df['Total_Issues']) * 100

    breadth_df['Is_Climax'] = breadth_df['New_Lows_Ratio'] >= CLIMAX_THRESHOLD_DAILY
    breadth_df['Is_Bloodbath'] = breadth_df['New_Lows_Ratio'] >= BLOODBATH_THRESHOLD_DAILY
    breadth_df['Climax_Entry'] = breadth_df['Is_Climax'].shift(ENTRY_LAG_DAYS).fillna(False)

    # 3. SuperTrend Logic (Weekly)
    sox_w = resample_to_weekly(sox)
    st_dir_w, st_up_w, st_dn_w = calculate_kivanc_supertrend(sox_w, period=ST_PERIOD, multiplier=ST_MULTIPLIER)
    sox_w['ST_Dir'] = st_dir_w

    # Resample to Daily
    st_daily = sox_w['ST_Dir'].resample('D').ffill()

    # 4. Simulation Setup
    df = pd.DataFrame(index=soxl.index)
    df = df.join(soxl[['Open', 'Close']], how='inner')
    df = df.join(breadth_df[['New_Lows_Ratio', 'Climax_Entry', 'Is_Bloodbath']], how='left')
    df = df.join(st_daily.rename('ST_Dir'), how='left')

    # Filter Start
    df = df[df.index >= start_dt].copy()
    df.ffill(inplace=True)
    df.dropna(inplace=True)

    pos = 0
    equity = 1.0
    equity_curve = [1.0]
    log_data = []

    # 5. Logic Loop
    # Priority:
    # 1. Climax Entry -> Force Buy
    # 2. Bloodbath -> Force Sell/Avoid (unless Climax Entry just triggered? Assuming Climax Entry overrides Bloodbath for "Catch Knife" logic, but Bloodbath generally overrides Trend)
    #    Let's stick to the previous successful logic: Climax Entry > Bloodbath > Trend.
    #    Because Climax Entry implies "Wait 22 days after panic".
    # 3. SuperTrend Buy -> Buy
    # 4. SuperTrend Sell -> Sell

    # Persistence:
    # If entered via Climax (while ST is Sell), hold until ST turns Buy then Sell?
    # Or hold until ST flips 1->-1?
    # Previous logic: "Exit only if Prev_ST=1 and Curr_ST=-1".
    # This keeps us in the trade if ST is currently -1 (Bearish) but hasn't "flipped" to Bearish (already was).
    # It waits for ST to become Bullish, then eventually flip Bearish.

    if not df.empty and df['ST_Dir'].iloc[0] == 1:
        pos = 1

    for i in range(1, len(df)):
        date = df.index[i]
        price = df['Close'].iloc[i]
        prev_price = df['Close'].iloc[i-1]

        # Return
        ret = (price - prev_price) / prev_price
        if pos == 1:
            equity *= (1 + ret)
        equity_curve.append(equity)

        # Signals
        climax = df['Climax_Entry'].iloc[i]
        bloodbath = df['Is_Bloodbath'].iloc[i]
        st_dir = df['ST_Dir'].iloc[i]
        prev_st = df['ST_Dir'].iloc[i-1]

        next_pos = pos

        if bloodbath:
            next_pos = 0 # Safety first
        elif pos == 0:
            if climax:
                next_pos = 1 # Catch Bottom
            elif st_dir == 1:
                next_pos = 1 # Trend Follow
        elif pos == 1:
            # Exit Logic
            # Exit if ST flips 1 -> -1
            if prev_st == 1 and st_dir == -1:
                next_pos = 0
            # Note: If we entered on Climax and ST is -1, we ignore the state -1 until it flips 1->-1.
            # So we implicitly hold.

        pos = next_pos

        log_data.append({
            'Date': date,
            'Price': price,
            'Position': pos,
            'Equity': equity,
            'Climax': climax,
            'Bloodbath': bloodbath
        })

    # 6. Report
    results = pd.DataFrame(log_data)
    if results.empty:
        print("No results.")
        return

    results.set_index('Date', inplace=True)

    bh = (df['Close'] / df['Close'].iloc[0]).iloc[1:]

    strat_ret = (results['Equity'].iloc[-1] - 1) * 100
    bh_ret = (bh.iloc[-1] - 1) * 100

    print("\n" + "="*50)
    print(f"BACKTEST: Daily Climax + Weekly Kivanc SuperTrend")
    print(f"Period: {results.index[0].date()} to {results.index[-1].date()}")
    print("="*50)
    print(f"Strategy: {strat_ret:.2f}%")
    print(f"Buy & Hold: {bh_ret:.2f}%")
    print("="*50)

    # 7. Chart
    # Plotting Equity
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    # Ax0: Equity
    axes[0].plot(results.index, results['Equity'], label='Strategy', color='green')
    axes[0].plot(results.index, bh, label='Buy & Hold', color='gray', alpha=0.6)
    axes[0].set_title('Equity Curve')
    axes[0].set_yscale('log')
    axes[0].legend()
    axes[0].grid(True)

    # Ax1: Drawdowns / Climax Markers
    # Let's plot Price with ST background
    axes[1].plot(df.index[1:], df['Close'][1:], color='black', label='Price')
    # Climax Markers
    climax_dates = results[results['Climax']].index
    axes[1].scatter(climax_dates, df.loc[climax_dates]['Close'], color='purple', marker='^', s=100, label='Climax Buy', zorder=5)
    # Bloodbath Zones
    # Identify spans where bloodbath is true
    # Use fill_between
    axes[1].fill_between(results.index, df['Close'].min(), df['Close'].max(), where=results['Bloodbath'], color='red', alpha=0.3, label='Bloodbath')
    axes[1].set_title('Price & Signals')
    axes[1].legend()
    axes[1].grid(True)

    # Ax2: Breadth
    axes[2].plot(df.index[1:], df['New_Lows_Ratio'][1:], color='blue', label='New Lows %')
    axes[2].axhline(20, color='red', linestyle='--')
    axes[2].axhline(4, color='orange', linestyle='--')
    axes[2].set_title('Market Breadth')
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig('pmax_vince_williams_result.png')
    print("Chart saved.")

if __name__ == "__main__":
    run_backtest()
