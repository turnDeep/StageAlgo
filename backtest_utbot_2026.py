import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
from datetime import datetime, timedelta

def run_backtest():
    # 1. Configuration
    SIGNAL_TICKER = "^SOX"
    TRADE_TICKER = "SOXL"

    # Parameters from Pine Script
    BUY_KEY = 2.0
    BUY_ATR_PERIOD = 300
    SELL_KEY = 2.0
    SELL_ATR_PERIOD = 1

    # Timeframes
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 3) # 3 years for warm up
    comparison_start_date = end_date - timedelta(days=365) # 1 year backtest

    print(f"Fetching data for {SIGNAL_TICKER} and {TRADE_TICKER}...")

    # 2. Fetch Data
    sox = yf.download(SIGNAL_TICKER, start=start_date, end=end_date, progress=False)
    soxl = yf.download(TRADE_TICKER, start=start_date, end=end_date, progress=False)

    # Align data
    # Ensure both have the same index
    # Flatten multi-index columns if present (yfinance update)
    if isinstance(sox.columns, pd.MultiIndex):
        sox.columns = sox.columns.get_level_values(0)
    if isinstance(soxl.columns, pd.MultiIndex):
        soxl.columns = soxl.columns.get_level_values(0)

    # Use Adjusted Close for calculations to account for splits/dividends in SOXL
    # But for the Signal (SOX Index), Close is fine (indexes don't split usually, but let's use Close)

    df = pd.DataFrame(index=sox.index)
    df['SOX_Close'] = sox['Close']
    df['SOX_High'] = sox['High']
    df['SOX_Low'] = sox['Low']
    df['SOXL_Close'] = soxl['Close'] # Use adj close if accurate returns needed, but yf 'Close' is often adj depending on settings. auto_adjust=True is default in recent versions? Let's check returns.

    # Drop NaNs
    df.dropna(inplace=True)

    # 3. Calculate Indicators (on SOX)
    # ATR
    # We need to manually calculate ATR because pandas_ta ATR might expect 'High', 'Low', 'Close' columns in the df passed to it
    # We'll create a temporary dataframe for SOX ta
    sox_ta = pd.DataFrame()
    sox_ta['high'] = df['SOX_High']
    sox_ta['low'] = df['SOX_Low']
    sox_ta['close'] = df['SOX_Close']

    # Calculate ATRs
    df['xATR_Buy'] = ta.atr(sox_ta['high'], sox_ta['low'], sox_ta['close'], length=BUY_ATR_PERIOD)
    df['xATR_Sell'] = ta.atr(sox_ta['high'], sox_ta['low'], sox_ta['close'], length=SELL_ATR_PERIOD)

    # 4. Implement UT Bot Logic (State Machine)
    # Variables
    pos = 0 # 0: None, 1: Long, -1: Short
    xATRTrailingStop = 0.0

    # Lists to store results for plotting/analysis
    pos_list = []
    stop_list = []

    # Iterate
    # We need to iterate row by row. Vectorization is hard for this recursive logic.
    # Convert necessary columns to numpy arrays for speed
    close_arr = df['SOX_Close'].values
    atr_buy_arr = df['xATR_Buy'].values
    atr_sell_arr = df['xATR_Sell'].values

    # Initialize arrays
    n = len(df)
    pos_arr = np.zeros(n, dtype=int)
    stop_arr = np.zeros(n, dtype=float)

    # Variables for the loop
    curr_pos = 0 # Init as 0
    curr_stop = 0.0

    # Pre-calculate nLoss
    # nLoss_Buy = BUY_KEY * xATR_Buy
    # nLoss_Sell = SELL_KEY * xATR_Sell
    n_loss_buy_arr = BUY_KEY * atr_buy_arr
    n_loss_sell_arr = SELL_KEY * atr_sell_arr

    for i in range(n):
        src = close_arr[i]

        # Skip if ATR is NaN
        if np.isnan(atr_buy_arr[i]) or np.isnan(atr_sell_arr[i]):
            pos_arr[i] = 0
            stop_arr[i] = src # Default to price
            continue

        prev_pos = pos_arr[i-1] if i > 0 else 0
        prev_stop = stop_arr[i-1] if i > 0 else 0

        # Logic
        if prev_pos == 1: # Currently Long
            new_floor = src - n_loss_sell_arr[i]
            if src < prev_stop: # Stop Hit! Switch to Short
                curr_pos = -1
                curr_stop = src + n_loss_buy_arr[i] # Init Ceiling
            else: # Still Long
                curr_pos = 1
                curr_stop = max(prev_stop, new_floor)
        elif prev_pos == -1: # Currently Short
            new_ceiling = src + n_loss_buy_arr[i]
            if src > prev_stop: # Stop Hit! Switch to Long
                curr_pos = 1
                curr_stop = src - n_loss_sell_arr[i] # Init Floor
            else: # Still Short
                curr_pos = -1
                curr_stop = min(prev_stop, new_ceiling)
        else: # Initialization (first valid bar)
            curr_pos = 1 # Assume Long to start or strictly wait?
            # Let's verify Pine Script: 'var int pos = 0'. 'prev_pos = nz(pos[1])'.
            # If prev_pos is 0 (start), it goes to 'else' block (Currently Short or Start).
            # So it treats 0 as Short context initially.
            new_ceiling = src + n_loss_buy_arr[i]
            # If src > prev_stop (0), which is true?
            # nz(xATRTrailingStop[1]) is 0. src > 0 is True.
            # So it switches to Long immediately?
            # Yes, usually.
            curr_pos = 1
            curr_stop = src - n_loss_sell_arr[i]

        pos_arr[i] = curr_pos
        stop_arr[i] = curr_stop

    df['Signal_Pos'] = pos_arr
    df['Trailing_Stop'] = stop_arr

    # 5. Backtest Execution (Comparison Window)
    # Filter for the last 1 year
    compare_df = df[df.index >= pd.Timestamp(comparison_start_date)].copy()

    # Calculate SOXL Returns
    compare_df['SOXL_Ret'] = compare_df['SOXL_Close'].pct_change()

    # Strategy Position (Shifted by 1 because we trade AT CLOSE of the signal bar,
    # effectively realizing returns from the NEXT bar, OR we trade Next Open.
    # Standard backtest: Signal at Close -> Return is from Close to Next Close.
    # So Strategy Return = Position[i] * Return[i+1].
    # In pandas: Strategy_Ret = Position.shift(1) * Returns.

    # Interpretation:
    # "SOXでbuyが出ればSOXLを買い" (If SOX Buy appears, Buy SOXL)
    # "SOXでSellが出ればSOXLを売る" (If SOX Sell appears, Sell SOXL)
    # Logic:
    # If Signal_Pos == 1 (Long), we hold SOXL.
    # If Signal_Pos == -1 (Short), we hold Cash (0).

    compare_df['Strat_Pos'] = compare_df['Signal_Pos'].apply(lambda x: 1 if x == 1 else 0)

    # Shift position by 1 day to simulate entering on the Close (or next Open) and holding for the day
    compare_df['Strat_Ret'] = compare_df['Strat_Pos'].shift(1) * compare_df['SOXL_Ret']

    # Cumulative Returns
    compare_df['Cum_SOXL'] = (1 + compare_df['SOXL_Ret']).cumprod()
    compare_df['Cum_Strat'] = (1 + compare_df['Strat_Ret']).cumprod()

    # Performance Metrics
    soxl_total_ret = (compare_df['Cum_SOXL'].iloc[-1] - 1) * 100
    strat_total_ret = (compare_df['Cum_Strat'].iloc[-1] - 1) * 100

    print("\n" + "="*50)
    print(f"BACKTEST RESULTS (Last 1 Year: {compare_df.index[0].date()} to {compare_df.index[-1].date()})")
    print("="*50)
    print(f"Strategy (UT Bot 2026): {strat_total_ret:.2f}%")
    print(f"Buy & Hold (SOXL)     : {soxl_total_ret:.2f}%")
    print("="*50)

    # 6. Visualization
    # Plotting using mplfinance for professional look or matplotlib for custom

    # We will use matplotlib for the Equity Curve
    plt.figure(figsize=(12, 8))

    # Subplot 1: Equity Curve
    plt.subplot(2, 1, 1)
    plt.plot(compare_df.index, compare_df['Cum_Strat'], label='UT Bot Strategy (Long/Cash)', color='green')
    plt.plot(compare_df.index, compare_df['Cum_SOXL'], label='Buy & Hold SOXL', color='blue', alpha=0.6)
    plt.title(f'Strategy vs Buy & Hold (1 Year) - Signal: SOX, Trade: SOXL')
    plt.ylabel('Normalized Equity (Start=1.0)')
    plt.legend()
    plt.grid(True)

    # Subplot 2: SOX Price with Trailing Stop and Signals
    plt.subplot(2, 1, 2)
    plt.plot(compare_df.index, compare_df['SOX_Close'], label='SOX Price', color='black', alpha=0.7)
    plt.plot(compare_df.index, compare_df['Trailing_Stop'], label='Trailing Stop', color='red', linestyle='--', alpha=0.7)

    # Plot Signals
    # Buy: Pos 1, Prev -1
    # Sell: Pos -1, Prev 1
    # We need to detect changes in the sliced dataframe 'compare_df'
    # Actually, we should use the original 'pos_arr' logic but mapped to compare_df

    buy_signals = compare_df[(compare_df['Signal_Pos'] == 1) & (compare_df['Signal_Pos'].shift(1) == -1)]
    sell_signals = compare_df[(compare_df['Signal_Pos'] == -1) & (compare_df['Signal_Pos'].shift(1) == 1)]

    plt.scatter(buy_signals.index, buy_signals['SOX_Close'], marker='^', color='green', s=100, label='Buy Signal', zorder=5)
    plt.scatter(sell_signals.index, sell_signals['SOX_Close'], marker='v', color='red', s=100, label='Sell Signal', zorder=5)

    plt.title('UT Bot 2026 Signals on SOX Index')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('utbot_2026_sox.png')
    print("Chart saved to utbot_2026_sox.png")

if __name__ == "__main__":
    run_backtest()
