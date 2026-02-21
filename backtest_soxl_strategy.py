import yfinance as yf
import pandas as pd
import numpy as np
import math

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean() # RMA logic in Pine is similar to EMA(alpha=1/len)
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_stoch_rsi(series, period=14, smooth_k=3, smooth_d=3):
    rsi = calculate_rsi(series, period)
    stoch_rsi = (rsi - rsi.rolling(period).min()) / (rsi.rolling(period).max() - rsi.rolling(period).min()) * 100
    k = stoch_rsi.rolling(smooth_k).mean() # SMA
    d = k.rolling(smooth_d).mean() # SMA
    return k, d

def calculate_atr(high, low, close, period=14):
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean() # RMA logic
    return atr

def calculate_atr_trailing_stop(high, low, close, period=3, multiplier=1.0):
    atr = calculate_atr(high, low, close, period)

    ts = pd.Series(index=close.index, dtype=float)
    ts[:] = np.nan

    offset = multiplier * atr

    # Iterate because TS depends on previous TS
    prev_ts = np.nan
    prev_close = np.nan

    ts_values = []

    # Convert to numpy arrays for speed
    close_np = close.values
    offset_np = offset.values

    current_ts = np.nan

    for i in range(len(close)):
        c = close_np[i]
        off = offset_np[i]

        if i == 0:
            current_ts = c - off
        else:
            prev_c = close_np[i-1]

            if np.isnan(current_ts):
                 current_ts = c - off
            else:
                # Logic from Pine Script:
                # if close > prev_ts and close[1] > prev_ts:
                #     ts := max(prev_ts, close - offset)
                # else if close < prev_ts and close[1] < prev_ts:
                #     ts := min(prev_ts, close + offset)
                # else:
                #     if close > prev_ts:
                #         ts := close - offset
                #     else:
                #         ts := close + offset

                if c > current_ts and prev_c > current_ts:
                    current_ts = max(current_ts, c - off)
                elif c < current_ts and prev_c < current_ts:
                    current_ts = min(current_ts, c + off)
                else:
                    if c > current_ts:
                        current_ts = c - off
                    else:
                        current_ts = c + off

        ts_values.append(current_ts)

    return pd.Series(ts_values, index=close.index)

def run_backtest():
    print("Fetching data...")
    # Fetch 2 years of 1h data
    # SOX index is ^SOX, SOXL is the trade instrument
    tickers = ["^SOX", "SOXL"]
    data = yf.download(tickers, period="730d", interval="1h", group_by='ticker', auto_adjust=True, progress=False)

    # Handle multi-index columns if present
    if isinstance(data.columns, pd.MultiIndex):
        sox_data = data['^SOX'].copy()
        soxl_data = data['SOXL'].copy()
    else:
        # Fallback if structure is different
        print("Data structure unexpected. Exiting.")
        return

    # Drop NaN rows to align
    sox_data = sox_data.dropna()
    soxl_data = soxl_data.dropna()

    # Align indices (intersection)
    common_index = sox_data.index.intersection(soxl_data.index)
    sox_data = sox_data.loc[common_index]
    soxl_data = soxl_data.loc[common_index]

    print(f"Data points: {len(common_index)}")

    # --- Calculate Indicators on Signal Ticker (^SOX) ---
    print("Calculating indicators...")

    # Stoch RSI Settings
    rsi_len = 14
    stoch_len = 14
    smooth_k = 3
    smooth_d = 3

    # ATR TS Settings
    atr_period = 3
    atr_mult = 1.0

    # Calculate StochRSI
    k, d = calculate_stoch_rsi(sox_data['Close'], rsi_len, smooth_k, smooth_d)

    # Calculate ATR Trailing Stop
    ts = calculate_atr_trailing_stop(sox_data['High'], sox_data['Low'], sox_data['Close'], atr_period, atr_mult)

    # --- Generate Signals ---
    # Condition 1: ATR Breakout (Close > ATS today, but was <= ATS yesterday)
    # Condition 2: StochRSI Bullish (%K > %D)

    sox_close = sox_data['Close']
    prev_sox_close = sox_close.shift(1)
    prev_ts = ts.shift(1)

    # Using 'ts' from previous bar logic in the loop is equivalent to ts[1] in Pine when calculating current ts?
    # Wait, the Pine script calculates current ts based on current close and prev ts.
    # The condition `sig_close > sig_ts` compares current close to current ts.
    # `sig_close[1] <= sig_ts_prev` compares prev close to prev ts.

    cond_bull_cross = (sox_close > ts) & (prev_sox_close <= prev_ts)
    cond_stoch_bull = k > d

    enter_long = cond_bull_cross & cond_stoch_bull

    # Exit: ATR breakdown (Close falls below ATS)
    # exit_long  = sig_close < sig_ts and sig_close[1] >= sig_ts_prev
    # Note: The Pine script exit condition includes `sig_close[1] >= sig_ts_prev` to detect the cross UNDER.
    # However, if we are already long, ANY close below TS should act as a stop, technically.
    # But let's stick to the Pine logic: "Exit on Cross Under".
    exit_long = (sox_close < ts) & (prev_sox_close >= prev_ts)

    # --- Simulation Loop ---
    print("Simulating trades...")

    capital = 10000.0
    initial_capital = capital
    position = 0 # 0 or shares
    entry_price = 0.0
    trades = []

    # Access numpy arrays for speed
    enter_signals = enter_long.values
    exit_signals = exit_long.values
    price_soxl = soxl_data['Close'].values
    timestamps = soxl_data.index

    equity_curve = []

    for i in range(len(common_index)):
        if i < 20: continue # Skip warmup

        current_price = price_soxl[i]
        date = timestamps[i]

        # Check Exit First
        if position > 0:
            if exit_signals[i]:
                # Sell
                exit_price = current_price
                pnl = (exit_price - entry_price) * position
                pnl_percent = (exit_price - entry_price) / entry_price * 100
                capital += pnl
                position = 0
                trades.append({
                    'Entry Date': entry_date,
                    'Exit Date': date,
                    'Entry Price': entry_price,
                    'Exit Price': exit_price,
                    'PnL': pnl,
                    'PnL %': pnl_percent
                })
                # print(f"Sell on {date} at {exit_price:.2f} PnL: {pnl:.2f}")

        # Check Entry
        if position == 0:
            if enter_signals[i]:
                # Buy
                entry_price = current_price
                position = capital / entry_price # Use all capital (simulating 100% equity)
                entry_date = date
                # print(f"Buy on {date} at {entry_price:.2f}")

        # Record equity (if in position, mark to market)
        current_equity = capital
        if position > 0:
             current_equity = entry_price * position + (current_price - entry_price) * position

        equity_curve.append(current_equity)

    # Finalize (close open position)
    if position > 0:
        exit_price = price_soxl[-1]
        pnl = (exit_price - entry_price) * position
        pnl_percent = (exit_price - entry_price) / entry_price * 100
        capital += pnl
        trades.append({
            'Entry Date': entry_date,
            'Exit Date': timestamps[-1],
            'Entry Price': entry_price,
            'Exit Price': exit_price,
            'PnL': pnl,
            'PnL %': pnl_percent
        })

    trades_df = pd.DataFrame(trades)

    if len(trades_df) == 0:
        print("No trades generated.")
        return

    total_return = (capital - initial_capital) / initial_capital * 100
    win_rate = len(trades_df[trades_df['PnL'] > 0]) / len(trades_df) * 100
    avg_trade = trades_df['PnL %'].mean()
    max_drawdown = 0 # simplified

    # Calculate Max Drawdown from equity curve
    equity_series = pd.Series(equity_curve)
    rolling_max = equity_series.cummax()
    drawdown = (equity_series - rolling_max) / rolling_max
    max_drawdown = drawdown.min() * 100

    print("-" * 30)
    print(f"Backtest Results ({tickers[1]} using {tickers[0]} signals)")
    print(f"Period: {timestamps[0]} to {timestamps[-1]}")
    print(f"Initial Capital: ${initial_capital:.2f}")
    print(f"Final Capital:   ${capital:.2f}")
    print(f"Total Return:    {total_return:.2f}%")
    print(f"Buy & Hold Return: {(soxl_data['Close'].iloc[-1] - soxl_data['Close'].iloc[0]) / soxl_data['Close'].iloc[0] * 100:.2f}%")
    print(f"Total Trades:    {len(trades_df)}")
    print(f"Win Rate:        {win_rate:.2f}%")
    print(f"Avg Trade:       {avg_trade:.2f}%")
    print(f"Max Drawdown:    {max_drawdown:.2f}%")
    print("-" * 30)

    # Show last 5 trades
    print("\nLast 5 Trades:")
    print(trades_df.tail(5))

if __name__ == "__main__":
    run_backtest()
