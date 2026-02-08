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
    # Check if Volume exists
    if 'Volume' not in df.columns:
        if 'Volume' in logic:
            del logic['Volume']

    # Resample
    df_w = df.resample('W-FRI').agg(logic).dropna()
    return df_w

def calculate_supertrend(df, length, multiplier):
    # Standard SuperTrend logic using pandas_ta
    # We need to handle the column names dynamically
    st = ta.supertrend(df['High'], df['Low'], df['Close'], length=length, multiplier=multiplier)
    if st is None: return pd.Series(0, index=df.index)

    # Find direction column (usually SUPERTd_...)
    col_dir = [c for c in st.columns if c.startswith("SUPERTd")][0]
    return st[col_dir]

# --- MMXM Strategy (Weekly) ---
def strat_mmxm(df, swing_len=5):
    """
    Weekly MMXM Logic.
    """
    # 1. Swings (Weekly)
    # Use simple rolling max/min to detect structure
    # Lookback 10 weeks for structure?
    # MMXM simplified:
    # Bullish MSS: Close > Highest(High, swing_len) of previous structure

    # Identify FVGs
    high_prev2 = df['High'].shift(2)
    low_prev2 = df['Low'].shift(2)
    bull_fvg = (df['Low'] > high_prev2)
    bear_fvg = (df['High'] < low_prev2)

    # Structure (Break of Structure)
    # A simplified "Structure" indicator:
    # Use Donchian Channel breakout for Trend Bias
    hh = df['High'].rolling(swing_len*2).max().shift(1)
    ll = df['Low'].rolling(swing_len*2).min().shift(1)

    mss_bull = df['Close'] > hh
    mss_bear = df['Close'] < ll

    bias = 0
    sigs = np.zeros(len(df))

    mss_bull_v = mss_bull.values
    mss_bear_v = mss_bear.values
    bull_fvg_v = bull_fvg.values
    bear_fvg_v = bear_fvg.values
    low_v = df['Low'].values
    high_v = df['High'].values
    high_prev2_v = high_prev2.values
    low_prev2_v = low_prev2.values

    active_bull = []
    active_bear = []

    for i in range(len(df)):
        # Update Structure
        if mss_bull_v[i]: bias = 1
        elif mss_bear_v[i]: bias = -1

        # Add FVG
        if bull_fvg_v[i]: active_bull.append((low_v[i], high_prev2_v[i])) # Top, Bot
        if bear_fvg_v[i]: active_bear.append((low_prev2_v[i], high_v[i])) # Top, Bot

        # Prune old FVGs (Weekly context changes slowly, maybe keep longer? let's keep 5)
        if len(active_bull) > 5: active_bull.pop(0)
        if len(active_bear) > 5: active_bear.pop(0)

        sig = 0
        if bias == 1:
            # Retest Bull FVG
            for fvg in active_bull:
                if low_v[i] <= fvg[0]: # Mitigation
                    sig = 1
                    break
        elif bias == -1:
            # Retest Bear FVG
            for fvg in active_bear:
                if high_v[i] >= fvg[1]:
                    sig = -1
                    break

        sigs[i] = sig

    return pd.Series(sigs, index=df.index)

# --- PMax Strategy (Weekly) ---
def strat_pmax(df, period=10, multiplier=3):
    """
    Weekly PMax (SuperTrend Proxy).
    """
    direction = calculate_supertrend(df, period, multiplier)
    return direction # 1 or -1

def run_backtest_weekly():
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

    print("Calculating Strategies on Weekly ^SOX...")
    # Calculate signals on Index (SOX)
    sig_mmxm = strat_mmxm(sox_w.copy())
    sig_pmax = strat_pmax(sox_w.copy())

    # Merge signals to SOXL weekly data
    # Trade execution: Next Week Open

    simulation_df = pd.DataFrame(index=soxl_w.index)
    simulation_df['Open'] = soxl_w['Open']
    simulation_df['Close'] = soxl_w['Close']

    # Align
    combined = pd.concat([
        sig_mmxm.rename('MMXM'),
        sig_pmax.rename('PMax'),
        simulation_df
    ], axis=1, join='inner')

    # Filter 10 years
    ten_years_ago = (datetime.datetime.now() - timedelta(days=365*10)).replace(hour=0, minute=0, second=0, microsecond=0)
    test_data = combined[combined.index >= ten_years_ago].copy()

    if len(test_data) == 0:
        print("No data for 10 years.")
        return

    print(f"Backtest Period: {test_data.index[0].date()} to {test_data.index[-1].date()}")

    # --- Simulation Loop ---
    # Strategies: MMXM, PMax, BuyHold

    strats = ["MMXM", "PMax", "BuyHold"]
    results = {}

    for s in strats:
        capital = 10000.0
        cash = capital
        shares = 0
        in_pos = False
        trades = 0

        opens = test_data['Open'].values
        closes = test_data['Close'].values

        if s == "BuyHold":
            # Buy on first open
            shares = cash / opens[0]
            cash = 0
            trades = 1
            final = shares * closes[-1]
        else:
            signals = test_data[s].values

            for i in range(len(test_data)-1):
                sig = signals[i]
                next_open = opens[i+1]

                # Logic: Long Only
                # MMXM: 1 (Buy), -1 (Sell/Cash)
                # PMax: 1 (Buy), -1 (Sell/Cash)

                if sig == 1 and not in_pos:
                    shares = cash / next_open
                    cash = 0
                    in_pos = True
                    trades += 1
                elif sig == -1 and in_pos:
                    cash = shares * next_open
                    shares = 0
                    in_pos = False
                    trades += 1

            final = cash + (shares * closes[-1])

        ret = (final - capital) / capital * 100
        results[s] = {"Return": ret, "Final": final, "Trades": trades}

    print("\n========================================")
    print("WEEKLY STRATEGY RESULTS (SOX -> SOXL)")
    print("========================================")
    print(f"{'Strategy':<10} | {'Return':>10} | {'Trades':>6}")
    print("-" * 35)
    for s, res in results.items():
        print(f"{s:<10} | {res['Return']:>9.2f}% | {res['Trades']:>6}")

if __name__ == "__main__":
    run_backtest_weekly()
