import yfinance as yf
import pandas as pd
import numpy as np
import datetime
from datetime import timedelta
import pandas_ta as ta

def calculate_swing_points(df, length=5):
    """
    Identifies swing highs and lows.
    """
    df['Swing_High'] = df['High'].rolling(window=length*2+1, center=True).max()
    df['Swing_Low'] = df['Low'].rolling(window=length*2+1, center=True).min()

    # Identify local maxima/minima
    df['Is_Swing_High'] = (df['High'] == df['Swing_High'])
    df['Is_Swing_Low'] = (df['Low'] == df['Swing_Low'])

    return df

def identify_fvgs(df):
    """
    Identifies Fair Value Gaps (FVG).
    Bullish FVG: Low[i] > High[i-2] -> Gap is between High[i-2] and Low[i].
    Bearish FVG: High[i] < Low[i-2] -> Gap is between Low[i-2] and High[i].
    """
    # Shift to align i-2 logic
    high_prev2 = df['High'].shift(2)
    low_prev2 = df['Low'].shift(2)

    df['Bull_FVG_Top'] = np.where(df['Low'] > high_prev2, df['Low'], np.nan)
    df['Bull_FVG_Bot'] = np.where(df['Low'] > high_prev2, high_prev2, np.nan)

    df['Bear_FVG_Top'] = np.where(df['High'] < low_prev2, low_prev2, np.nan)
    df['Bear_FVG_Bot'] = np.where(df['High'] < low_prev2, df['High'], np.nan)

    return df

def mmxm_strategy(df):
    """
    Implements a simplified MMXM strategy logic:
    1. Identify Market Structure Shifts (MSS).
       - Bullish MSS: Close breaks above the most recent significant Swing High.
       - Bearish MSS: Close breaks below the most recent significant Swing Low.
    2. Identify FVGs.
    3. Entry Signal:
       - Buy: After Bullish MSS, price retraces into a Bullish FVG.
       - Sell: After Bearish MSS, price retraces into a Bearish FVG.
    """

    # Add indicators
    df = calculate_swing_points(df, length=10) # Using 10 for "HTF" significance on Daily
    df = identify_fvgs(df)

    # State variables
    market_structure = 0 # 0: Neutral, 1: Bullish, -1: Bearish
    last_swing_high = df['High'].iloc[0]
    last_swing_low = df['Low'].iloc[0]

    active_bull_fvgs = [] # List of (top, bot) tuples
    active_bear_fvgs = []

    signals = [] # 1: Buy, -1: Sell, 0: Hold

    # Iterate through rows
    # Pre-calculate numpy arrays for speed
    highs = df['High'].values
    lows = df['Low'].values
    closes = df['Close'].values
    is_sh = df['Is_Swing_High'].values
    is_sl = df['Is_Swing_Low'].values

    bull_fvg_top = df['Bull_FVG_Top'].values
    bull_fvg_bot = df['Bull_FVG_Bot'].values
    bear_fvg_top = df['Bear_FVG_Top'].values
    bear_fvg_bot = df['Bear_FVG_Bot'].values

    dates = df.index

    for i in range(len(df)):
        current_signal = 0
        current_high = highs[i]
        current_low = lows[i]
        current_close = closes[i]

        # 1. Update Market Structure
        # Check for Break of Structure (MSS)
        if market_structure != 1 and current_close > last_swing_high:
            market_structure = 1
            # Clear old conflicting FVGs on MSS? usually yes, context shift.
            active_bear_fvgs = []

        elif market_structure != -1 and current_close < last_swing_low:
            market_structure = -1
            active_bull_fvgs = []

        # Update Swing Points (Lagging indicator, so we look back to see if i-length was a swing)
        # Note: 'Is_Swing_High' is forward-looking in rolling calculation if centered=True.
        # But here we use rolling(window, centered=True).
        # In real-time backtest, we only know a swing point formed 'length' bars ago.
        # So we should check if index i-length was a swing point.
        # However, purely for "Last Swing High" reference, we use the most recent *confirmed* swing.

        # Let's simplify: A Swing High is confirmed when price drops below it for N bars?
        # Or just use the pre-calculated column, assuming we are looking at *historical* swings that were valid at time i?
        # Actually, `Is_Swing_High` at index `i` is only known at `i + length`.
        # So we must look at `i - length` to update `last_swing_high`.

        length = 10
        if i > length:
            if is_sh[i-length]:
                last_swing_high = highs[i-length]
            if is_sl[i-length]:
                last_swing_low = lows[i-length]

        # 2. Update Active FVGs
        # If today formed a FVG, add to list
        if not np.isnan(bull_fvg_top[i]):
            active_bull_fvgs.append((bull_fvg_top[i], bull_fvg_bot[i]))

        if not np.isnan(bear_fvg_top[i]):
            active_bear_fvgs.append((bear_fvg_top[i], bear_fvg_bot[i]))

        # Expire FVGs? (e.g. if mitigated/filled or too old).
        # Simplified: Remove if price closes below Bull FVG or above Bear FVG (Invalidated)
        active_bull_fvgs = [fvg for fvg in active_bull_fvgs if current_close >= fvg[1]]
        active_bear_fvgs = [fvg for fvg in active_bear_fvgs if current_close <= fvg[0]]

        # 3. Check Entry Signals (Mitigation)
        # Buy: Bullish Structure + Price dips into a Bullish FVG
        if market_structure == 1:
            # Check if Low touched any active Bull FVG
            mitigated = False
            for fvg in active_bull_fvgs:
                if current_low <= fvg[0] and current_low >= fvg[1]: # Touched the top or inside
                    mitigated = True
                    break
                if current_low < fvg[1]: # Went through it (but might have wicked back up)
                     # Still counts as mitigation if it didn't close below? We filtered invalid ones above based on Close.
                     mitigated = True
                     break

            if mitigated:
                current_signal = 1

        # Sell: Bearish Structure + Price rallies into a Bearish FVG
        elif market_structure == -1:
            mitigated = False
            for fvg in active_bear_fvgs:
                if current_high >= fvg[1] and current_high <= fvg[0]:
                    mitigated = True
                    break
                if current_high > fvg[0]:
                    mitigated = True
                    break

            if mitigated:
                current_signal = -1

        signals.append(current_signal)

    df['Signal'] = signals
    return df

def main():
    # Configuration
    signal_ticker = "^SOX"
    bull_ticker = "SOXL"
    bear_ticker = "SOXS"

    start_date = (datetime.datetime.now() - timedelta(days=365*14)).strftime('%Y-%m-%d')
    end_date = datetime.datetime.now().strftime('%Y-%m-%d')

    print(f"Fetching data for {signal_ticker}, {bull_ticker}, and {bear_ticker} from {start_date} to {end_date}...")

    signal_data = yf.download(signal_ticker, start=start_date, end=end_date, progress=False)
    bull_data = yf.download(bull_ticker, start=start_date, end=end_date, progress=False)
    bear_data = yf.download(bear_ticker, start=start_date, end=end_date, progress=False)

    # Ensure timezone-naive index
    if signal_data.index.tz is not None:
        signal_data.index = signal_data.index.tz_localize(None)
    if bull_data.index.tz is not None:
        bull_data.index = bull_data.index.tz_localize(None)
    if bear_data.index.tz is not None:
        bear_data.index = bear_data.index.tz_localize(None)

    # Flatten columns
    if isinstance(signal_data.columns, pd.MultiIndex):
        signal_data.columns = signal_data.columns.get_level_values(0)
    if isinstance(bull_data.columns, pd.MultiIndex):
        bull_data.columns = bull_data.columns.get_level_values(0)
    if isinstance(bear_data.columns, pd.MultiIndex):
        bear_data.columns = bear_data.columns.get_level_values(0)

    # Implement MMXM Logic
    print(f"Calculating MMXM Strategy on {signal_ticker}...")
    signal_data = mmxm_strategy(signal_data)

    # Backtest Simulation
    print("Running backtest simulation...")

    simulation_df = pd.DataFrame(index=bull_data.index)
    simulation_df['Bull_Open'] = bull_data['Open']
    simulation_df['Bull_Close'] = bull_data['Close']
    simulation_df['Bear_Open'] = bear_data['Open']
    simulation_df['Bear_Close'] = bear_data['Close']

    # Align signals
    combined = pd.concat([
        signal_data[['Signal']],
        simulation_df
    ], axis=1, join='inner')

    # Filter last 10 years
    ten_years_ago = (datetime.datetime.now() - timedelta(days=365*10)).replace(hour=0, minute=0, second=0, microsecond=0)
    test_data = combined[combined.index >= ten_years_ago].copy()

    if len(test_data) == 0:
        print("Error: No data for the last 10 years.")
        return

    # Strategy Execution
    capital = 10000.0
    shares = 0
    position_type = 0 # 0: Flat, 1: Long (SOXL), -1: Short (SOXS)

    trades_log = []
    cash = capital
    equity_curve = []

    # Initial Entry logic?
    # If the first signal was Buy before the period, we might start Long?
    # For now, let's start Flat.

    for i in range(len(test_data) - 1):
        today_signal = test_data.iloc[i]['Signal'] # 1 Buy, -1 Sell

        next_bull_open = test_data.iloc[i+1]['Bull_Open']
        next_bear_open = test_data.iloc[i+1]['Bear_Open']
        next_date = test_data.index[i+1]

        # Entry Logic
        if today_signal == 1:
            if position_type == 1:
                pass # Already Long
            else:
                # Switch to Long
                if position_type == -1:
                    # Close Short (Sell SOXS)
                    cash = shares * next_bear_open
                    trades_log.append(f"Close Short (Sell SOXS): at {next_bear_open:.2f} on {next_date.date()}. Cash: {cash:.2f}")

                # Open Long (Buy SOXL)
                buy_price = next_bull_open
                shares = cash / buy_price
                cash = 0
                position_type = 1
                trades_log.append(f"Open Long (Buy SOXL): {shares:.2f} shares at {buy_price:.2f} on {next_date.date()}")

        elif today_signal == -1:
            if position_type == -1:
                pass # Already Short
            else:
                # Switch to Short
                if position_type == 1:
                    # Close Long (Sell SOXL)
                    cash = shares * next_bull_open
                    trades_log.append(f"Close Long (Sell SOXL): at {next_bull_open:.2f} on {next_date.date()}. Cash: {cash:.2f}")

                # Open Short (Buy SOXS)
                buy_price = next_bear_open
                shares = cash / buy_price
                cash = 0
                position_type = -1
                trades_log.append(f"Open Short (Buy SOXS): {shares:.2f} shares at {buy_price:.2f} on {next_date.date()}")

        # Update Equity
        if position_type == 1:
            current_equity = cash + (shares * test_data.iloc[i]['Bull_Close'])
        elif position_type == -1:
            current_equity = cash + (shares * test_data.iloc[i]['Bear_Close'])
        else:
            current_equity = cash

        equity_curve.append(current_equity)

    # Final Value
    if position_type == 1:
        final_equity_strat = cash + (shares * test_data.iloc[-1]['Bull_Close'])
    elif position_type == -1:
        final_equity_strat = cash + (shares * test_data.iloc[-1]['Bear_Close'])
    else:
        final_equity_strat = cash

    equity_curve.append(final_equity_strat)

    # Strategy 2: Buy and Hold SOXL
    bh_buy_price = test_data.iloc[0]['Bull_Open']
    bh_shares = capital / bh_buy_price
    bh_final_value = bh_shares * test_data.iloc[-1]['Bull_Close']

    # Output
    print("-" * 50)
    print(f"Backtest Period: {test_data.index[0].date()} to {test_data.index[-1].date()}")
    print("-" * 50)
    print(f"Strategy 1: MMXM (ICT) Long/Short ({signal_ticker} -> {bull_ticker}/{bear_ticker})")
    print(f"Initial Capital: ${capital:,.2f}")
    print(f"Final Equity:    ${final_equity_strat:,.2f}")
    print(f"Return:          {((final_equity_strat - capital) / capital) * 100:.2f}%")
    print(f"Trades:          {len(trades_log)}")

    print("-" * 50)
    print(f"Strategy 2: Buy and Hold {bull_ticker}")
    print(f"Initial Capital: ${capital:,.2f}")
    print(f"Final Equity:    ${bh_final_value:,.2f}")
    print(f"Return:          {((bh_final_value - capital) / capital) * 100:.2f}%")
    print("-" * 50)

    if final_equity_strat > bh_final_value:
        print("Winner: MMXM Strategy")
    else:
        print("Winner: Buy and Hold")

if __name__ == "__main__":
    main()
