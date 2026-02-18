import yfinance as yf
import pandas as pd
import numpy as np

def fetch_data(symbol="GC=F", period="60d", interval="5m"):
    """
    Fetches data from yfinance.
    Note: 1m data is limited to 7 days. 5m data is limited to 60 days.
    """
    print(f"Fetching {symbol} data for {period} with {interval} interval...")
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False, multi_level_index=False)
        if df.empty:
            print(f"Warning: No data found for {symbol}. Trying XAUUSD=X...")
            df = yf.download("XAUUSD=X", period=period, interval=interval, progress=False, multi_level_index=False)

        if df.empty:
            raise ValueError("No data fetched.")

        # Ensure index is datetime
        df.index = pd.to_datetime(df.index)
        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()

def resample_data(df, timeframe="10min"):
    """
    Resamples data to the target timeframe.
    """
    agg_dict = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }
    # Resample
    df_resampled = df.resample(timeframe).agg(agg_dict)
    # Drop NaN rows created by resampling (if any)
    df_resampled.dropna(inplace=True)
    return df_resampled

def calculate_indicators(df):
    """
    Adds technical indicators.
    """
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    return df

def run_backtest(df, mode='confirmed', risk_reward_ratio=1.0, use_time_filter=False):
    """
    Executes the strategy.

    Modes:
    - 'confirmed': Entry on Close of Bounce Candle. (Close > EMA).
    - 'limit': Entry at EMA touch. (Limit Order).

    Exit:
    - SL: High/Low of signal candle (confirmed) or Fixed buffer (limit).
    - TP: R:R * Risk.
    - Time/Trend Exit: Close crosses EMA.
    """
    trades = []
    position = None # 'LONG' or 'SHORT'
    entry_price = 0.0
    stop_loss = 0.0
    take_profit = 0.0

    print(f"Running backtest (Mode: {mode}, R:R: {risk_reward_ratio}, TimeFilter: {use_time_filter})...")

    for i in range(1, len(df)):
        current_idx = df.index[i]

        # Time Filter: Only trade between 08:00 and 18:00 UTC (London/NY Session)
        if use_time_filter:
            if not (8 <= current_idx.hour < 18):
                # If we have a position, we manage it, but we don't enter new ones.
                # Logic below handles management first.
                pass

        close = df['Close'].iloc[i]
        high = df['High'].iloc[i]
        low = df['Low'].iloc[i]
        open_price = df['Open'].iloc[i]
        ema = df['EMA_20'].iloc[i]

        prev_close = df['Close'].iloc[i-1]
        prev_ema = df['EMA_20'].iloc[i-1]

        # Check for open position management
        if position == 'LONG':
            # Check SL
            if low <= stop_loss:
                trades.append({'Type': 'Long', 'Entry': entry_price, 'Exit': stop_loss, 'Result': 'Loss', 'Reason': 'SL', 'Time': current_idx})
                position = None
            # Check TP
            elif high >= take_profit:
                trades.append({'Type': 'Long', 'Entry': entry_price, 'Exit': take_profit, 'Result': 'Win', 'Reason': 'TP', 'Time': current_idx})
                position = None
            # Check Trend Invalidation (Close below EMA)
            elif close < ema:
                 trades.append({'Type': 'Long', 'Entry': entry_price, 'Exit': close, 'Result': 'Loss' if close < entry_price else 'Win', 'Reason': 'TrendBreak', 'Time': current_idx})
                 position = None

        elif position == 'SHORT':
            # Check SL
            if high >= stop_loss:
                trades.append({'Type': 'Short', 'Entry': entry_price, 'Exit': stop_loss, 'Result': 'Loss', 'Reason': 'SL', 'Time': current_idx})
                position = None
            # Check TP
            elif low <= take_profit:
                trades.append({'Type': 'Short', 'Entry': entry_price, 'Exit': take_profit, 'Result': 'Win', 'Reason': 'TP', 'Time': current_idx})
                position = None
            # Check Trend Invalidation (Close above EMA)
            elif close > ema:
                 trades.append({'Type': 'Short', 'Entry': entry_price, 'Exit': close, 'Result': 'Loss' if close > entry_price else 'Win', 'Reason': 'TrendBreak', 'Time': current_idx})
                 position = None

        # Look for new entry if no position
        if position is None:
            if use_time_filter and not (8 <= current_idx.hour < 18):
                continue

            # Long Setup
            trend_up = prev_close > prev_ema

            if trend_up:
                if mode == 'confirmed':
                    # Touch: Low <= EMA
                    # Bounce: Close > EMA
                    if low <= ema and close > ema:
                        entry_price = close
                        stop_loss = low
                        risk = entry_price - stop_loss
                        if risk == 0: continue
                        take_profit = entry_price + (risk * risk_reward_ratio)
                        position = 'LONG'

                elif mode == 'limit':
                    # Limit at EMA
                    # If Low <= EMA, we are filled at EMA (approx)
                    # Ideally, entry is max(Low, EMA) but since Low <= EMA, we filled at EMA.
                    if low <= ema:
                        entry_price = ema
                        # Stop Loss? Hard to define without a swing low.
                        # Let's use ATR or fixed %? Or just Low of this candle?
                        # If we assume the bounce happens on this candle, SL is Low of this candle.
                        stop_loss = low
                        risk = entry_price - stop_loss
                        if risk <= 0: # Candle closed lower or wick is small
                             # If Close < EMA, it's a trend break immediately?
                             if close < ema:
                                 # Failed immediately
                                 trades.append({'Type': 'Long', 'Entry': entry_price, 'Exit': close, 'Result': 'Loss', 'Reason': 'TrendBreak', 'Time': current_idx})
                                 continue
                             # Otherwise, risk is small
                             risk = entry_price * 0.001 # Min risk 0.1% fallback

                        take_profit = entry_price + (risk * risk_reward_ratio)
                        position = 'LONG'

            # Short Setup
            trend_down = prev_close < prev_ema

            if trend_down:
                if mode == 'confirmed':
                    if high >= ema and close < ema:
                        entry_price = close
                        stop_loss = high
                        risk = stop_loss - entry_price
                        if risk == 0: continue
                        take_profit = entry_price - (risk * risk_reward_ratio)
                        position = 'SHORT'

                elif mode == 'limit':
                    if high >= ema:
                        entry_price = ema
                        stop_loss = high
                        risk = stop_loss - entry_price
                        if risk <= 0:
                            if close > ema:
                                trades.append({'Type': 'Short', 'Entry': entry_price, 'Exit': close, 'Result': 'Loss', 'Reason': 'TrendBreak', 'Time': current_idx})
                                continue
                            risk = entry_price * 0.001

                        take_profit = entry_price - (risk * risk_reward_ratio)
                        position = 'SHORT'

    return pd.DataFrame(trades)

def analyze_results(results, title="Backtest"):
    if not results.empty:
        total_trades = len(results)
        wins = len(results[results['Result'] == 'Win'])
        losses = len(results[results['Result'] == 'Loss'])
        win_rate = (wins / total_trades) * 100

        print(f"\n=== {title} (60 Days) ===")
        print(f"Total Trades: {total_trades}")
        print(f"Wins: {wins}")
        print(f"Losses: {losses}")
        print(f"Win Rate: {win_rate:.2f}%")

        # Analyze Last 35 Trades
        last_35 = results.tail(35)
        l35_wins = len(last_35[last_35['Result'] == 'Win'])
        l35_losses = len(last_35[last_35['Result'] == 'Loss'])
        l35_wr = (l35_wins / 35) * 100 if len(last_35) > 0 else 0

        print(f"--- Last 35 Trades ---")
        print(f"Win Rate: {l35_wr:.2f}% ({l35_wins} W / {l35_losses} L)")
    else:
        print(f"\n=== {title} ===")
        print("No trades generated.")

def main():
    # 1. Fetch
    df = fetch_data()
    if df.empty:
        print("Exiting...")
        return

    # 2. Resample
    df_10m = resample_data(df)

    # 3. Indicators
    df_10m = calculate_indicators(df_10m)

    # 4. Run Backtests
    # Scenario 1: Confirmed Entry, 1.0 RR
    results_1 = run_backtest(df_10m, mode='confirmed', risk_reward_ratio=1.0)
    analyze_results(results_1, "Confirmed Entry (1.0 R:R)")

    # Scenario 2: Confirmed Entry, 0.5 RR (Scalping)
    results_2 = run_backtest(df_10m, mode='confirmed', risk_reward_ratio=0.5)
    analyze_results(results_2, "Confirmed Entry (0.5 R:R)")

    # Scenario 3: Limit Entry, 1.0 RR
    results_3 = run_backtest(df_10m, mode='limit', risk_reward_ratio=1.0)
    analyze_results(results_3, "Limit Entry (1.0 R:R)")

    # Scenario 4: Confirmed Entry, 0.5 RR + Time Filter
    results_4 = run_backtest(df_10m, mode='confirmed', risk_reward_ratio=0.5, use_time_filter=True)
    analyze_results(results_4, "Confirmed Entry (0.5 R:R, 08-18 UTC)")

if __name__ == "__main__":
    main()
