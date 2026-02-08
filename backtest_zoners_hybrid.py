import yfinance as yf
import pandas as pd
import numpy as np
import datetime
from datetime import timedelta
import pandas_ta as ta

# --- ZoneRS Logic ---
def calculate_zoners_weekly(df_daily, benchmark_daily, rs_length=50, mom_length=20, vol_lookback=50):
    """
    Calculates ZoneRS based on WEEKLY data, then forward fills to daily.
    """
    # Resample to Weekly
    logic = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}

    df_w = df_daily.resample('W-FRI').agg(logic).dropna()
    bench_w = benchmark_daily.resample('W-FRI').agg(logic).dropna()

    # Align Weekly
    common_idx = df_w.index.intersection(bench_w.index)
    df_w = df_w.loc[common_idx]
    bench_w = bench_w.loc[common_idx]

    # RS Calc (Weekly)
    rs = df_w['Close'] / bench_w['Close']
    rs_sma = rs.rolling(window=rs_length).mean()
    rs_ratio = rs / rs_sma

    # Momentum (Weekly ROC)
    rs_momentum = (rs_ratio.diff(mom_length) / rs_ratio.shift(mom_length)) * 100

    # Volume Filter (Weekly)
    avg_vol = df_w['Volume'].rolling(window=vol_lookback).mean()
    vol_boost = df_w['Volume'] > avg_vol

    # Determine Quadrant
    cond_power = (rs_ratio >= 1) & (rs_momentum >= 0)
    cond_drift = (rs_ratio >= 1) & (rs_momentum < 0)
    cond_dead  = (rs_ratio < 1)  & (rs_momentum < 0)
    cond_lift  = (rs_ratio < 1)  & (rs_momentum >= 0)

    quadrants = []
    prev_quad = "Neutral"

    c_power = cond_power.values
    c_drift = cond_drift.values
    c_dead  = cond_dead.values
    c_lift  = cond_lift.values
    v_boost = vol_boost.values

    for i in range(len(df_w)):
        new_q = "Neutral"
        if c_power[i]: new_q = "Power"
        elif c_drift[i]: new_q = "Drift"
        elif c_dead[i]:  new_q = "Dead"
        elif c_lift[i]:  new_q = "Lift"

        final_q = prev_quad
        if new_q == "Power":
            if v_boost[i]:
                final_q = "Power"
            else:
                final_q = prev_quad
        else:
            final_q = new_q

        quadrants.append(final_q)
        prev_quad = final_q

    df_w['Zone'] = quadrants

    # Merge back to Daily
    # Reindex df_daily to match weekly zones (forward fill)
    df_daily_zones = df_daily.copy()

    # Create a Series indexed by weekly dates, then reindex to daily
    zone_series = df_w['Zone']

    # Reindex to daily, ffill
    # Note: Weekly date is Friday. Daily dates Mon-Fri should get the value of the *previous* Friday?
    # Or the *current* week's incomplete value?
    # Standard Backtest: You only know the weekly close on Friday Close.
    # So Monday-Friday of Week T should use the Zone determined at Week T-1 Friday.
    # Shift Weekly zones by 1 week forward to avoid lookahead?
    # Yes, we use *closed* weekly bars to determine the state for the *next* week.

    zone_series = zone_series.shift(1) # Shift 1 week

    # Resample to daily (ffill)
    zone_daily = zone_series.reindex(df_daily.index, method='ffill')
    df_daily_zones['Zone'] = zone_daily.fillna("Neutral")

    return df_daily_zones

# --- Strategies (Reused) ---

def strat_pmax_supertrend(df, period=10, multiplier=3):
    # Standard SuperTrend as proxy
    # Returns 1 (Bull) or -1 (Bear)
    import pandas_ta as ta
    st = ta.supertrend(df['High'], df['Low'], df['Close'], length=period, multiplier=multiplier)
    # Find direction column
    col = [c for c in st.columns if c.startswith("SUPERTd")][0]
    return st[col]

def strat_mmxm(df, swing_len=5):
    # Simplified MMXM (Swing + FVG)
    # Recalculate basic logic
    # 1. Swings
    df['High_Roll'] = df['High'].rolling(swing_len*2+1, center=True).max()
    df['Low_Roll'] = df['Low'].rolling(swing_len*2+1, center=True).min()
    # Confirm lags?
    # Simplified: Use rolling max of past N bars for real-time
    # Real-time approximation:
    # Bullish Struct: Close > Highest(High, 20)
    # Bearish Struct: Close < Lowest(Low, 20)
    # FVG Entry

    # Let's reuse the 'mmxm_strategy' logic from before but vectorized/simplified for speed if possible
    # Or just copy the function logic.

    # Identify FVGs
    high_prev2 = df['High'].shift(2)
    low_prev2 = df['Low'].shift(2)
    bull_fvg = (df['Low'] > high_prev2)
    bear_fvg = (df['High'] < low_prev2)

    # Structure
    # Bull MSS: Close > Highest(High, 10)[1]
    # Bear MSS: Close < Lowest(Low, 10)[1]
    hh = df['High'].rolling(10).max().shift(1)
    ll = df['Low'].rolling(10).min().shift(1)

    mss_bull = df['Close'] > hh
    mss_bear = df['Close'] < ll

    # State
    bias = 0
    sigs = np.zeros(len(df))

    b_vals = bias
    mss_bull_v = mss_bull.values
    mss_bear_v = mss_bear.values
    bull_fvg_v = bull_fvg.values
    bear_fvg_v = bear_fvg.values
    low_v = df['Low'].values
    high_v = df['High'].values
    # FVG lists
    active_bull = [] # (top, bot)
    active_bear = []

    # Iterate
    # Simplified logic:
    # If Bias Bull -> Check FVG retest -> Buy
    for i in range(len(df)):
        # Update Structure
        if mss_bull_v[i]: bias = 1
        elif mss_bear_v[i]: bias = -1

        # Add FVG
        if bull_fvg_v[i]: active_bull.append((low_v[i], high_prev2.iloc[i])) # Low[i] is top
        if bear_fvg_v[i]: active_bear.append((low_prev2.iloc[i], high_v[i])) # High[i] is bot

        # Clean FVGs (keep last 5 for speed)
        if len(active_bull) > 5: active_bull.pop(0)
        if len(active_bear) > 5: active_bear.pop(0)

        sig = 0
        if bias == 1:
            # Check retest of any bull FVG
            # Current Low touches FVG top?
            for fvg in active_bull:
                if low_v[i] <= fvg[0]: # Retest
                    sig = 1
                    break
        elif bias == -1:
            for fvg in active_bear:
                if high_v[i] >= fvg[1]:
                    sig = -1
                    break

        sigs[i] = sig

    return pd.Series(sigs, index=df.index)

# --- Hybrid Engine ---

def run_hybrid_backtest(ticker, benchmark="^GSPC"):
    print(f"--- Hybrid Backtest: {ticker} vs {benchmark} ---")

    # Fetch Data (Daily, 10y)
    start_date = (datetime.datetime.now() - timedelta(days=365*14)).strftime('%Y-%m-%d')
    data = yf.download(ticker, start=start_date, progress=False)
    bench = yf.download(benchmark, start=start_date, progress=False)

    if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
    if isinstance(bench.columns, pd.MultiIndex): bench.columns = bench.columns.get_level_values(0)

    # Calculate ZoneRS (Weekly Logic)
    df = calculate_zoners_weekly(data, bench)

    # Calculate Sub-Strategies
    df['Sig_PMax'] = strat_pmax_supertrend(df)
    df['Sig_MMXM'] = strat_mmxm(df)

    # Simulation Logic
    # 1. ZoneRS + PMax
    # 2. ZoneRS + MMXM
    # 3. Buy & Hold

    strategies = ["Zone_PMax", "Zone_MMXM", "BuyHold"]
    results = {}

    for strat in strategies:
        cap = 10000.0
        cash = cap
        shares = 0
        in_pos = False
        trades = 0

        opens = df['Open'].values
        closes = df['Close'].values
        zones = df['Zone'].values
        sig_pmax = df['Sig_PMax'].values
        sig_mmxm = df['Sig_MMXM'].values

        for i in range(len(df)-1):
            zone = zones[i]
            next_open = opens[i+1]

            # Logic Selector
            action = 0 # 0: Hold/Neutral, 1: Buy/Long, -1: Sell/Cash

            if strat == "BuyHold":
                action = 1
            else:
                # Hybrid Logic
                # Power -> Force Long (Buy & Hold)
                if zone == "Power":
                    action = 1
                # Dead -> Force Cash
                elif zone == "Dead":
                    action = -1
                # Lift/Drift -> Use Sub-Strategy
                else:
                    sub_sig = sig_pmax[i] if strat == "Zone_PMax" else sig_mmxm[i]
                    # Map Sub-Sig to Action
                    # PMax: 1 or -1
                    # MMXM: 1 or -1 or 0
                    if sub_sig == 1: action = 1
                    elif sub_sig == -1: action = -1
                    else: action = 0 # Neutral, maintain previous state

            # Execution
            # State Management:
            # If Action 1: Ensure Long
            # If Action -1: Ensure Cash
            # If Action 0: Do nothing (Maintain)

            if action == 1 and not in_pos:
                shares = cash / next_open
                cash = 0
                in_pos = True
                trades += 1
            elif action == -1 and in_pos:
                cash = shares * next_open
                shares = 0
                in_pos = False
                trades += 1

        final = cash + (shares * closes[-1])
        ret = (final - cap) / cap * 100
        results[strat] = {"Return": ret, "Trades": trades, "Final": final}

    return results

def main():
    # Target 1: SOXL
    res_soxl = run_hybrid_backtest("SOXL", "^GSPC")

    # Target 2: GDXU (Gold Miners)
    # Benchmark? GDXU vs SPX is okay, or Gold?
    # ZoneRS usually compares to broad market (SPX). Let's stick to SPX.
    res_gdxu = run_hybrid_backtest("GDXU", "^GSPC")

    print("\n========================================")
    print("HYBRID STRATEGY RESULTS (ZoneRS + X)")
    print("========================================")

    for ticker, res in [("SOXL", res_soxl), ("GDXU", res_gdxu)]:
        print(f"\nTicker: {ticker}")
        print(f"{'Strategy':<15} | {'Return':>10} | {'Trades':>6}")
        print("-" * 35)
        for s, data in res.items():
            print(f"{s:<15} | {data['Return']:>9.2f}% | {data['Trades']:>6}")

if __name__ == "__main__":
    main()
