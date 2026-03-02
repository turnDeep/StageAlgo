import yfinance as yf
import pandas as pd
import numpy as np
import datetime
from datetime import timedelta
import pandas_ta as ta

# --- Indicator Functions ---

def calculate_prc(df, length=50, degree=2, mult=3.0):
    """
    Polynomial Regression Channel.
    Calculates the regression curve and standard deviation bands.
    """
    # y = ax^2 + bx + c
    x = np.arange(len(df))
    y = df['Close'].values

    # We need a rolling calculation to avoid lookahead bias.
    # Fitting a polynomial on the WHOLE dataset and checking past points is wrong.
    # We must fit on [i-length : i] for each i. This is computationally heavy in Python loop.
    # Optimization: Use a simpler linear regression channel or accept the loop cost?
    # Or use numpy polyfit in a rolling apply?

    prc_mid = np.full(len(df), np.nan)
    prc_upper = np.full(len(df), np.nan)
    prc_lower = np.full(len(df), np.nan)

    # Pre-calculate indices
    closes = df['Close'].values

    # Only calculate if we have enough data
    for i in range(length, len(df)):
        window_y = closes[i-length:i]
        window_x = np.arange(length)

        # Fit polynomial
        coeffs = np.polyfit(window_x, window_y, degree)

        # Predict current point (end of window, x = length-1? No, next point x=length?)
        # Standard PRC usually plots the curve for the window.
        # We want the value at the current bar 'i'.
        # The fit is on past 'length' bars. The current bar is the "next" step?
        # Or do we fit including current bar? Usually indicators use current Close.
        # Let's use window including current point: closes[i-length+1 : i+1]

        window_y_inc = closes[i-length+1 : i+1]
        coeffs = np.polyfit(window_x, window_y_inc, degree)

        # The 'current' value is at x = length - 1 (last point in window)
        val = np.polyval(coeffs, length - 1)

        # Calculate Stdev of residuals in the window
        # fitted values for the whole window
        fitted = np.polyval(coeffs, window_x)
        residuals = window_y_inc - fitted
        std = np.std(residuals)

        prc_mid[i] = val
        prc_upper[i] = val + mult * std
        prc_lower[i] = val - mult * std

    return prc_upper, prc_lower

def calculate_td_sequential(df):
    """
    TD Sequential Setup (Simplified).
    Buy Setup: 9 consecutive closes < close 4 bars earlier.
    Sell Setup: 9 consecutive closes > close 4 bars earlier.
    Returns: 9 (Sell Signal), -9 (Buy Signal), 0 (Neutral)
    """
    close = df['Close']
    close_shift4 = close.shift(4)

    td_buy = (close < close_shift4)
    td_sell = (close > close_shift4)

    # Count consecutive
    # Vectorized approach: Groupby consecutive logic
    # But pure loop is safer for "9 count" logic reset

    setup = np.zeros(len(df))
    count_buy = 0
    count_sell = 0

    c_vals = close.values
    c4_vals = close_shift4.values

    for i in range(4, len(df)):
        # Buy Count
        if c_vals[i] < c4_vals[i]:
            count_buy += 1
        else:
            count_buy = 0

        # Sell Count
        if c_vals[i] > c4_vals[i]:
            count_sell += 1
        else:
            count_sell = 0

        if count_buy == 9:
            setup[i] = -9 # Buy Signal
            # count_buy = 0 # TD Sequential often continues to 13, but let's stick to Setup 9 triggers
        elif count_sell == 9:
            setup[i] = 9 # Sell Signal

    return setup

def calculate_wavetrend(df, chlen=14, avg=21): # Using slower weekly settings
    """
    WaveTrend Oscillator.
    """
    # 1. HLC3
    ap = (df['High'] + df['Low'] + df['Close']) / 3
    # 2. ESA = EMA(AP, 10)
    esa = ta.ema(ap, length=chlen)
    # 3. D = EMA(Abs(AP - ESA), 10)
    d = ta.ema((ap - esa).abs(), length=chlen)
    # 4. CI = (AP - ESA) / (0.015 * D)
    ci = (ap - esa) / (0.015 * d)
    # 5. TC1 = EMA(CI, 21)
    wt1 = ta.ema(ci, length=avg)
    # 6. TC2 = SMA(TC1, 4)
    wt2 = ta.sma(wt1, length=4)

    return wt1, wt2

def calculate_vix_fix(df, length=22):
    """
    CM Williams Vix Fix.
    WVF = ((Highest(Close, 22) - Low) / Highest(Close, 22)) * 100
    """
    highest_close = df['Close'].rolling(window=length).max()
    wvf = ((highest_close - df['Low']) / highest_close) * 100

    # Bollinger Bands on WVF to detect "High"
    # Standard deviation of WVF
    # Standard BB: SMA(20), Std(20)
    # WVF BB length often 20
    wvf_sma = wvf.rolling(window=20).mean()
    wvf_std = wvf.rolling(window=20).std()
    wvf_upper = wvf_sma + (2.0 * wvf_std)

    return wvf, wvf_upper

# --- Strategy Logic ---

def original_setup_strategy(df):
    """
    1. Context: Price > PRC Upper (Sell Zone) or Price < PRC Lower (Buy Zone).
    2. Trigger: TD 9 (+/-) OR WaveTrend Cross.
    3. Confirmation: WVF Reversal (Buy) / WT Cross (Sell).

    Simplified Implementation for Backtest:
    - BUY SIGNAL:
        (Price < PRC Lower) AND (TD == -9 OR (WVF crosses under WVF Upper))
    - SELL SIGNAL:
        (Price > PRC Upper) AND (TD == 9 OR (WT1 crosses under WT2))
    """

    # Calculate Indicators
    # PRC (Heavy calc)
    print("Calculating PRC...")
    prc_upper, prc_lower = calculate_prc(df, length=50, degree=2, mult=2.0) # Using 2.0 Sigma for "Tradeable" zone start, 3.0 is extreme

    # TD
    print("Calculating TD Sequential...")
    td_setup = calculate_td_sequential(df)

    # WaveTrend
    print("Calculating WaveTrend...")
    wt1, wt2 = calculate_wavetrend(df)

    # Vix Fix
    print("Calculating Vix Fix...")
    wvf, wvf_upper = calculate_vix_fix(df)

    # Logic Loop
    signals = np.zeros(len(df))

    c_vals = df['Close'].values
    l_vals = df['Low'].values
    h_vals = df['High'].values

    wt1_vals = wt1.values
    wt2_vals = wt2.values

    wvf_vals = wvf.values
    wvf_up_vals = wvf_upper.values

    # State
    # 1 = Long, 0 = Cash
    # The user implies a "Setup" logic. "Monitor" then "Trigger".
    # We will simulate a signal that stays 1 (Buy) until Sell signal.

    for i in range(1, len(df)):
        # Context
        in_buy_zone = (l_vals[i] <= prc_lower[i])
        in_sell_zone = (h_vals[i] >= prc_upper[i])

        # Triggers
        td_buy = (td_setup[i] == -9)
        td_sell = (td_setup[i] == 9)

        # Confirmation Proxies
        # WT Cross Under (Sell)
        wt_cross_down = (wt1_vals[i] < wt2_vals[i]) and (wt1_vals[i-1] >= wt2_vals[i-1])
        # WVF Reversal (Buy): WVF was above Upper, now below? Or just high?
        # "Vix Fix spikes > upper then falls" -> WVF[i] < WVF[i-1] AND WVF[i-1] > Upper[i-1]
        wvf_reversal = (wvf_vals[i] < wvf_vals[i-1]) and (wvf_vals[i-1] > wvf_up_vals[i-1])

        # Signals
        # Buy: In Zone + (TD or VixFix)
        if in_buy_zone and (td_buy or wvf_reversal):
            signals[i] = 1

        # Sell: In Zone + (TD or WT Cross)
        # Note: If prices goes parabolic above 3 sigma, we want to sell.
        elif in_sell_zone and (td_sell or wt_cross_down):
            signals[i] = -1

        # Alternative Exit: Trend breakdown?
        # The prompt implies "Top Selling" and "Bottom Buying".
        # It's a mean reversion strategy.
        # If we buy at bottom, when do we sell? At Top? Or Trail?
        # "Target: Channel Center" is common for Mean Reversion.
        # Let's add a "Neutral" exit if Price hits PRC Mid?
        # Or just hold until Sell Signal (Top)?
        # Holding until Top in a bull market is good (Ride trend).
        # Holding until Top in a bear market is bad (Round trip).
        # Let's stick to the explicit "Sell Signal" logic provided.

    return pd.Series(signals, index=df.index)

def resample_to_weekly(df):
    logic = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
    if 'Volume' not in df.columns and 'Volume' in logic: del logic['Volume']
    return df.resample('W-FRI').agg(logic).dropna()

def main():
    # 1. Fetch Data
    print("Fetching Daily data for ^SOX and SOXL (14 years)...")
    start_date = (datetime.datetime.now() - timedelta(days=365*14)).strftime('%Y-%m-%d')
    end_date = datetime.datetime.now().strftime('%Y-%m-%d')

    sox_d = yf.download("^SOX", start=start_date, end=end_date, progress=False)
    soxl_d = yf.download("SOXL", start=start_date, end=end_date, progress=False)

    if isinstance(sox_d.columns, pd.MultiIndex): sox_d.columns = sox_d.columns.get_level_values(0)
    if isinstance(soxl_d.columns, pd.MultiIndex): soxl_d.columns = soxl_d.columns.get_level_values(0)
    if sox_d.index.tz is not None: sox_d.index = sox_d.index.tz_localize(None)
    if soxl_d.index.tz is not None: soxl_d.index = soxl_d.index.tz_localize(None)

    # 2. Resample to Weekly
    print("Resampling to Weekly...")
    sox_w = resample_to_weekly(sox_d)
    soxl_w = resample_to_weekly(soxl_d)

    # 3. Calculate Strategy on Index (^SOX)
    print("Running 'Original Setup' Strategy on ^SOX...")
    signals = original_setup_strategy(sox_w.copy())

    # 4. Simulation
    simulation_df = pd.DataFrame(index=soxl_w.index)
    simulation_df['Open'] = soxl_w['Open']
    simulation_df['Close'] = soxl_w['Close']

    combined = pd.concat([signals.rename('Signal'), simulation_df], axis=1, join='inner')

    # Filter 10 years
    ten_years_ago = (datetime.datetime.now() - timedelta(days=365*10)).replace(hour=0, minute=0, second=0, microsecond=0)
    test_data = combined[combined.index >= ten_years_ago].copy()

    if len(test_data) == 0: return

    print(f"Backtest Period: {test_data.index[0].date()} to {test_data.index[-1].date()}")

    # PMax for Comparison (Need to calc or fetch from previous logic? Let's just hardcode result or recalc simple one)
    # Recalc simple PMax for fairness in same script
    st_dir = ta.supertrend(sox_w['High'], sox_w['Low'], sox_w['Close'], length=10, multiplier=3)
    col_dir = [c for c in st_dir.columns if c.startswith("SUPERTd")][0]
    pmax_sigs = st_dir[col_dir]
    combined_pmax = pd.concat([pmax_sigs.rename('Signal'), simulation_df], axis=1, join='inner')
    test_data_pmax = combined_pmax[combined_pmax.index >= ten_years_ago].copy()

    # Run Both
    results = {}

    for name, data in [("Original Setup", test_data), ("PMax (Weekly)", test_data_pmax)]:
        cap = 10000.0
        cash = cap
        shares = 0
        in_pos = False
        trades = 0

        sigs_arr = data['Signal'].values
        opens = data['Open'].values
        closes = data['Close'].values

        for i in range(len(data)-1):
            s = sigs_arr[i]
            nxt_o = opens[i+1]

            if s == 1 and not in_pos:
                shares = cash / nxt_o
                cash = 0
                in_pos = True
                trades += 1
            elif s == -1 and in_pos:
                cash = shares * nxt_o
                shares = 0
                in_pos = False
                trades += 1

        final = cash + (shares * closes[-1])
        ret = (final - cap) / cap * 100
        results[name] = {"Return": ret, "Trades": trades}

    # Buy Hold
    opens = test_data['Open'].values
    closes = test_data['Close'].values
    bh_ret = ((10000 / opens[0]) * closes[-1] - 10000) / 10000 * 100
    results["Buy & Hold"] = {"Return": bh_ret, "Trades": 1}

    print("\n===========================================")
    print("FINAL SHOWDOWN (SOX -> SOXL, 10 Years)")
    print("===========================================")
    print(f"{'Strategy':<20} | {'Return':>15} | {'Trades':>6}")
    print("-" * 50)
    for k, v in sorted(results.items(), key=lambda x: x[1]['Return'], reverse=True):
        print(f"{k:<20} | {v['Return']:>14.2f}% | {v['Trades']:>6}")

if __name__ == "__main__":
    main()
