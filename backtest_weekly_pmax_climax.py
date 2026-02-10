import yfinance as yf
import pandas as pd
import numpy as np
import datetime
from datetime import timedelta
import pandas_ta as ta

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

def calculate_pmax_supertrend(df, length=10, multiplier=3):
    st = ta.supertrend(df['High'], df['Low'], df['Close'], length=length, multiplier=multiplier)
    if st is None: return pd.Series(0, index=df.index), pd.Series(0, index=df.index)
    col_dir = [c for c in st.columns if c.startswith("SUPERTd")][0]
    col_val = [c for c in st.columns if c.startswith("SUPERT_")][0]
    return st[col_dir], st[col_val]

def calculate_vix_fix_climax(df, period=22):
    """
    CM Williams Vix Fix as a proxy for Selling Climax.
    Detects when the 'Synthetic VIX' pierces the Upper Bollinger Band (Panic)
    and then recedes (Climax confirmed).
    """
    # WVF Calculation
    highest_close = df['Close'].rolling(window=period).max()
    wvf = ((highest_close - df['Low']) / highest_close) * 100

    # Bollinger Bands on WVF
    wvf_mean = wvf.rolling(window=20).mean()
    wvf_std = wvf.rolling(window=20).std()
    wvf_upper = wvf_mean + (2.0 * wvf_std)

    # Climax Logic
    # 1. WVF was > Upper Band (Panic state)
    # 2. WVF is now < Previous WVF (Calming down / Reversal)
    # OR simpler: WVF crosses under Upper Band?
    # The "Original Setup" used: "Vix Fix pierces upper band, then starts to fall".

    # Let's define "Climax Day" as WVF >= WVF_Upper.
    # "Buy Signal" as per user prompt: isClimaxDay[1] and not isClimaxDay.
    # (Panic occurred yesterday, today it stopped).

    is_climax = (wvf >= wvf_upper)
    buy_signal = (is_climax.shift(1) == True) & (is_climax == False)

    return buy_signal

def run_backtest_pmax_climax():
    print("Fetching Daily data for ^SOX and SOXL (14 years)...")
    start_date = (datetime.datetime.now() - timedelta(days=365*14)).strftime('%Y-%m-%d')
    end_date = datetime.datetime.now().strftime('%Y-%m-%d')

    sox_d = yf.download("^SOX", start=start_date, end=end_date, progress=False)
    soxl_d = yf.download("SOXL", start=start_date, end=end_date, progress=False)

    if isinstance(sox_d.columns, pd.MultiIndex): sox_d.columns = sox_d.columns.get_level_values(0)
    if isinstance(soxl_d.columns, pd.MultiIndex): soxl_d.columns = soxl_d.columns.get_level_values(0)
    if sox_d.index.tz is not None: sox_d.index = sox_d.index.tz_localize(None)
    if soxl_d.index.tz is not None: soxl_d.index = soxl_d.index.tz_localize(None)

    print("Resampling to Weekly...")
    sox_w = resample_to_weekly(sox_d)
    soxl_w = resample_to_weekly(soxl_d)

    print("Calculating Indicators on Weekly ^SOX...")
    pmax_dir, pmax_val = calculate_pmax_supertrend(sox_w.copy())
    climax_sig = calculate_vix_fix_climax(sox_w.copy())

    # Simulation Dataframe
    simulation_df = pd.DataFrame(index=soxl_w.index)
    simulation_df['Open'] = soxl_w['Open']
    simulation_df['Close'] = soxl_w['Close']

    # Align Signals
    combined = pd.concat([
        pmax_dir.rename('PMax_Dir'),
        climax_sig.rename('Climax_Buy'),
        simulation_df
    ], axis=1, join='inner')

    # Filter 10 Years
    ten_years_ago = (datetime.datetime.now() - timedelta(days=365*10)).replace(hour=0, minute=0, second=0, microsecond=0)
    test_data = combined[combined.index >= ten_years_ago].copy()

    if len(test_data) == 0: return

    print(f"Backtest Period: {test_data.index[0].date()} to {test_data.index[-1].date()}")

    # --- Simulation Strategies ---

    # 1. Pure PMax
    cap_pmax = 10000.0
    cash_pmax = cap_pmax
    shares_pmax = 0
    in_pos_pmax = False
    trades_pmax = 0

    # 2. PMax + Climax
    cap_mix = 10000.0
    cash_mix = cap_mix
    shares_mix = 0
    in_pos_mix = False
    trades_mix = 0

    # 3. Buy & Hold
    cap_bh = 10000.0
    shares_bh = cap_bh / test_data['Open'].iloc[0]
    final_bh = shares_bh * test_data['Close'].iloc[-1]

    pmax_dirs = test_data['PMax_Dir'].values
    climax_sigs = test_data['Climax_Buy'].values
    opens = test_data['Open'].values
    closes = test_data['Close'].values

    # Loop
    for i in range(len(test_data) - 1):
        p_dir = pmax_dirs[i]      # 1 or -1
        c_sig = climax_sigs[i]    # True or False
        next_o = opens[i+1]

        # --- Pure PMax Logic ---
        if p_dir == 1 and not in_pos_pmax:
            # Buy
            shares_pmax = cash_pmax / next_o
            cash_pmax = 0
            in_pos_pmax = True
            trades_pmax += 1
        elif p_dir == -1 and in_pos_pmax:
            # Sell
            cash_pmax = shares_pmax * next_o
            shares_pmax = 0
            in_pos_pmax = False
            trades_pmax += 1

        # --- PMax + Climax Logic ---
        # State Machine:
        # If PMax is Green (1) -> Must be Long.
        # If PMax is Red (-1) -> Generally Short/Cash, UNLESS Climax Triggered.

        # We need to track if we entered via Climax ("Early Entry").
        # If we entered via Climax, we hold UNTIL PMax turns Green then Red.
        # Simplified:
        # We want to be Long if (PMax == 1) OR (We caught a Climax recently and PMax hasn't sold yet).
        # Actually, PMax Sell signal is "Transition from 1 to -1".
        # If we enter on Climax while PMax is -1, we are now Long.
        # When do we sell?
        # If PMax stays -1... do we hold forever? No.
        # We need a stop logic for Climax entry if PMax doesn't turn Green?
        # Or, we just hold until the NEXT PMax Sell Signal (1 -> -1).
        # This implies we hold through the -1 phase until it becomes 1, then waits for -1.
        # Basically: Climax Buy overrides PMax Sell State.

        should_be_long = False

        # 1. Standard PMax Condition
        if p_dir == 1:
            should_be_long = True

        # 2. Climax Condition (Override)
        if c_sig and not in_pos_mix:
            # Panic Buy Trigger!
            # We initiate a Long position even if PMax is Red.
            should_be_long = True

        # 3. Persistence
        # If we are already Long, do we stay Long?
        # If PMax is Green -> Yes.
        # If PMax is Red -> Only if we haven't received a "New" Sell Signal?
        # Standard PMax Sell Signal is (Prev=1, Curr=-1).
        # If we entered via Climax (while Red), PMax is Red (Prev=-1, Curr=-1).
        # So "Standard Sell Signal" won't trigger. We act as if we are "in the trend".
        # We wait for PMax to eventually turn Green, then eventually turn Red.
        # So: If In_Pos, stay In_Pos UNLESS (Prev_PMax=1 and Curr_PMax=-1).

        if in_pos_mix:
            # Check for Sell Signal
            # Logic: Sell ONLY if PMax flips from 1 to -1.
            # Warning: If we enter on Climax (Red), and PMax stays Red for 10 years, we never sell?
            # Correct. That's the risk of "Early Entry". We assume PMax *will* turn Green eventually.
            prev_p_dir = pmax_dirs[i-1] if i > 0 else 0
            pmax_sell_sig = (prev_p_dir == 1) and (p_dir == -1)

            if pmax_sell_sig:
                should_be_long = False
            else:
                should_be_long = True

        # Execute Mix
        if should_be_long and not in_pos_mix:
            shares_mix = cash_mix / next_o
            cash_mix = 0
            in_pos_mix = True
            trades_mix += 1
        elif not should_be_long and in_pos_mix:
            cash_mix = shares_mix * next_o
            shares_mix = 0
            in_pos_mix = False
            trades_mix += 1

    # Finals
    final_pmax = cash_pmax + (shares_pmax * closes[-1])
    final_mix = cash_mix + (shares_mix * closes[-1])

    ret_pmax = (final_pmax - cap_pmax) / cap_pmax * 100
    ret_mix = (final_mix - cap_mix) / cap_mix * 100
    ret_bh = (final_bh - cap_bh) / cap_bh * 100

    print("\n===============================================")
    print("STRATEGY COMPARISON: PMax vs PMax + Climax")
    print("===============================================")
    print(f"{'Strategy':<20} | {'Return':>15} | {'Trades':>6}")
    print("-" * 50)
    print(f"{'Pure PMax':<20} | {ret_pmax:>14.2f}% | {trades_pmax:>6}")
    print(f"{'PMax + Climax':<20} | {ret_mix:>14.2f}% | {trades_mix:>6}")
    print(f"{'Buy & Hold':<20} | {ret_bh:>14.2f}% | {1:>6}")
    print("-" * 50)

if __name__ == "__main__":
    run_backtest_pmax_climax()
