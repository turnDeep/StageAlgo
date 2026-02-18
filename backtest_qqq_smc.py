import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def main():
    print("Fetching QQQ data...")
    # 1. Fetch Data
    # 1H data (2 years)
    df_1h = yf.download(tickers='QQQ', period='730d', interval='1h', progress=False)

    # 5M data (60 days max via yfinance)
    df_5m = yf.download(tickers='QQQ', period='59d', interval='5m', progress=False)

    # Clean data (handle multi-index columns if any)
    def flatten_columns(df):
        if isinstance(df.columns, pd.MultiIndex):
            # If 'Close' is in the levels, drop the Ticker level
            # yfinance often returns (Price, Ticker)
            if 'Close' in df.columns.get_level_values(0):
                 df.columns = df.columns.get_level_values(0)
            elif 'Close' in df.columns.get_level_values(1):
                 df.columns = df.columns.get_level_values(1)
        return df

    df_1h = flatten_columns(df_1h)
    df_5m = flatten_columns(df_5m)

    # Ensure index is datetime and localized/tz-aware
    if df_1h.index.tz is None:
        df_1h.index = df_1h.index.tz_localize('UTC')
    else:
        df_1h.index = df_1h.index.tz_convert('UTC')

    if df_5m.index.tz is None:
        df_5m.index = df_5m.index.tz_localize('UTC')
    else:
        df_5m.index = df_5m.index.tz_convert('UTC')

    print(f"1H Data: {len(df_1h)} rows")
    print(f"5M Data: {len(df_5m)} rows")

    # 2. Resample 1H to 4H for Direction
    df_4h = df_1h.resample('4h').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()

    # Calculate 4H Trend (SMA 50)
    df_4h['SMA_50'] = df_4h['Close'].rolling(window=50).mean()
    df_4h['Trend'] = np.where(df_4h['Close'] > df_4h['SMA_50'], 'Bullish', 'Bearish')

    # IMPORTANT: Shift trend by 1 because the Close of the 4H bar is only known
    # at the END of the 4 hours. We cannot use it for the beginning of that period.
    df_4h['Trend'] = df_4h['Trend'].shift(1)

    # Map 4H Trend back to 1H (forward fill)
    df_1h['Trend_4H'] = df_4h['Trend'].reindex(df_1h.index, method='ffill')

    # 3. Identify 1H Structure (Swing Highs/Lows)
    # Using Fractal logic: High > Left/Right neighbors (Window=2)
    def identify_swings_vectorized(df, window=2):
        is_high = pd.Series([True]*len(df), index=df.index)
        is_low = pd.Series([True]*len(df), index=df.index)

        for k in range(1, window + 1):
            is_high &= (df['High'] > df['High'].shift(k)) & (df['High'] > df['High'].shift(-k))
            is_low &= (df['Low'] < df['Low'].shift(k)) & (df['Low'] < df['Low'].shift(-k))

        return is_high, is_low

    swing_highs, swing_lows = identify_swings_vectorized(df_1h, window=2)
    df_1h['Swing_High'] = swing_highs
    df_1h['Swing_Low'] = swing_lows

    # 4. Identify Setups (Sweep + FVG)
    trades = []

    # Keep track of last confirmed Swing High/Low
    last_swing_high = None
    last_swing_low = None

    print("Running backtest loop...")

    # Start loop after we have some data for SMA and initial swings
    for i in range(200, len(df_1h) - 2):
        current_time = df_1h.index[i]

        trend = df_1h['Trend_4H'].iloc[i]

        # CRITICAL: We use 'last_swing_high/low' which are the CONFIRMED swings
        # from PREVIOUS steps. We do NOT update them yet.

        setup_found = False
        direction = None
        entry_price = 0
        stop_loss = 0
        take_profit = 0

        # Bullish Setup
        # 1. Trend Bullish
        # 2. Sweep of last_swing_low (Price dipped below it but FVG formed)
        # 3. Bullish FVG at i (Gap between High[i-2] and Low[i])

        if trend == 'Bullish' and last_swing_low is not None:
            # Check sweep: Did we dip below the PREVIOUS structure low recently?
            swept = False
            for offset in range(3):
                if df_1h['Low'].iloc[i-offset] < last_swing_low:
                    swept = True
                    break

            if swept:
                # Check Bullish FVG
                prev_high = df_1h['High'].iloc[i-2]
                curr_low = df_1h['Low'].iloc[i]

                if curr_low > prev_high:
                    # Valid FVG
                    setup_found = True
                    direction = 'Long'
                    fvg_mid = (curr_low + prev_high) / 2
                    entry_price = fvg_mid
                    stop_loss = prev_high # FVG Boundary
                    take_profit = entry_price + 2 * (entry_price - stop_loss)

        # Bearish Setup
        elif trend == 'Bearish' and last_swing_high is not None:
            # Check sweep
            swept = False
            for offset in range(3):
                if df_1h['High'].iloc[i-offset] > last_swing_high:
                    swept = True
                    break

            if swept:
                # Check Bearish FVG
                prev_low = df_1h['Low'].iloc[i-2]
                curr_high = df_1h['High'].iloc[i]

                if curr_high < prev_low:
                    setup_found = True
                    direction = 'Short'
                    fvg_mid = (curr_high + prev_low) / 2
                    entry_price = fvg_mid
                    stop_loss = prev_low # FVG Boundary
                    take_profit = entry_price - 2 * (stop_loss - entry_price)

        if setup_found:
            # Execute on 5M
            # We must start execution AFTER the 1H candle 'i' has closed.
            # Candle 'i' confirms the FVG.
            exec_start_time = current_time + timedelta(hours=1)

            if not df_5m.empty and exec_start_time >= df_5m.index[0]:
                result = execute_trade(df_5m, exec_start_time, direction, entry_price, stop_loss, take_profit)
                if result:
                    trades.append(result)

        # UPDATE SWINGS
        # Check if i-2 is a confirmed swing.
        # Logic: We are at 'i'. If 'i-2' is a swing, it is confirmed by 'i-1' and 'i'.
        # We update the 'last_swing' vars AFTER checking for setups to avoid comparing
        # a new low against itself.
        if df_1h['Swing_High'].iloc[i-2]:
            last_swing_high = df_1h['High'].iloc[i-2]
        if df_1h['Swing_Low'].iloc[i-2]:
            last_swing_low = df_1h['Low'].iloc[i-2]

    analyze_results(trades)

def execute_trade(df_5m, start_time, direction, entry, stop, target):
    # Get 5M data from start_time
    subset = df_5m[df_5m.index >= start_time]

    if len(subset) == 0:
        return None

    # Limit max duration (3 hours = 36 bars of 5m)
    max_bars = 36
    subset = subset.iloc[:max_bars]

    filled = False
    fill_time = None
    exit_time = None
    pnl = 0
    outcome = 'Expired'

    for j in range(len(subset)):
        bar = subset.iloc[j]
        time = subset.index[j]

        if not filled:
            # Check Fill (Limit Order)
            if direction == 'Long':
                if bar['Low'] <= entry:
                    filled = True
                    fill_time = time
            else: # Short
                if bar['High'] >= entry:
                    filled = True
                    fill_time = time

        if filled:
            # Check Exit
            if direction == 'Long':
                if bar['Low'] <= stop:
                    outcome = 'Loss'
                    pnl = -1
                    exit_time = time
                    break
                if bar['High'] >= target:
                    outcome = 'Win'
                    pnl = 2
                    exit_time = time
                    break
            else: # Short
                if bar['High'] >= stop:
                    outcome = 'Loss'
                    pnl = -1
                    exit_time = time
                    break
                if bar['Low'] <= target:
                    outcome = 'Win'
                    pnl = 2
                    exit_time = time
                    break

    if filled and outcome == 'Expired':
        # Close at end
        last_bar = subset.iloc[-1]
        exit_time = subset.index[-1]
        if direction == 'Long':
            final_price = last_bar['Close']
            denom = entry - stop
            if abs(denom) > 1e-9:
                pnl = (final_price - entry) / denom
        else:
            final_price = last_bar['Close']
            denom = stop - entry
            if abs(denom) > 1e-9:
                pnl = (entry - final_price) / denom

    if filled:
        return {
            'Entry Time': fill_time,
            'Exit Time': exit_time,
            'Direction': direction,
            'Outcome': outcome,
            'PnL_R': pnl
        }
    return None

def analyze_results(trades):
    if not trades:
        print("No trades found/executed.")
        return

    df = pd.DataFrame(trades)
    print("\nBacktest Results:")
    print(f"Total Trades: {len(df)}")
    wins = len(df[df['Outcome'] == 'Win'])
    losses = len(df[df['Outcome'] == 'Loss'])
    win_rate = (wins / len(df)) * 100 if len(df) > 0 else 0
    total_r = df['PnL_R'].sum()
    expectancy = total_r / len(df) if len(df) > 0 else 0

    print(f"Wins: {wins} ({win_rate:.2f}%)")
    print(f"Losses: {losses}")
    print(f"Total R: {total_r:.2f}R")
    print(f"Expectancy: {expectancy:.2f}R per trade")

    # Capital Growth (Compounding)
    capital = 10000
    risk_pct = 0.02
    final_capital = capital

    for r in df['PnL_R']:
        # Add Profit/Loss amount
        # Risk amount is 2% of CURRENT capital
        risk_amount = final_capital * risk_pct
        pnl_amount = risk_amount * r
        final_capital += pnl_amount

    print(f"Final Capital (est, start $10k, compounding 2% risk): ${final_capital:.2f}")

    # Print trade list
    print("\nTrade List:")
    print(df[['Entry Time', 'Direction', 'Outcome', 'PnL_R']].to_string())

if __name__ == "__main__":
    main()
