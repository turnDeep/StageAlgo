import yfinance as yf
import pandas as pd
import numpy as np
import datetime
from datetime import timedelta

def calculate_sma(series, window):
    return series.rolling(window=window).mean()

def up_down_strategy(df, long_ma_period=77, short_ma_period=7):
    """
    Implements the 'Up Down' strategy with use_ma=True.
    """
    # 1. Indicators
    df['SMA_Long'] = calculate_sma(df['Close'], long_ma_period)
    df['SMA_Short'] = calculate_sma(df['Close'], short_ma_period)

    # 2. Price Patterns
    # Buy: Close > Open and Open > Close[1]
    # Sell: Close < Open and Open < Close[1]
    prev_close = df['Close'].shift(1)
    df['Pat_Buy'] = (df['Close'] > df['Open']) & (df['Open'] > prev_close)
    df['Pat_Sell'] = (df['Close'] < df['Open']) & (df['Open'] < prev_close)

    # 3. Logic Conditions
    # mabuy = crossover(short, long) or (buy and short > long)
    # masell = crossunder(short, long) or sell

    # Crossover/Crossunder
    # Check if Short crosses Long
    # Short[i] > Long[i] and Short[i-1] <= Long[i-1]
    prev_sma_short = df['SMA_Short'].shift(1)
    prev_sma_long = df['SMA_Long'].shift(1)

    crossover = (df['SMA_Short'] > df['SMA_Long']) & (prev_sma_short <= prev_sma_long)
    crossunder = (df['SMA_Short'] < df['SMA_Long']) & (prev_sma_short >= prev_sma_long)

    # MABuy / MASell
    df['MA_Buy'] = crossover | (df['Pat_Buy'] & (df['SMA_Short'] > df['SMA_Long']))
    df['MA_Sell'] = crossunder | df['Pat_Sell']

    # 4. Signal Generation (State Machine)
    # xbuy = crossover(num_bars_sell, num_bars_buy)
    # Effectively: Switch to Buy if MA_Buy happens while state is Sell.
    # We maintain a state: 1 (Buy Mode), -1 (Sell Mode)

    signals = np.zeros(len(df))
    state = 0 # 0: Neutral, 1: Buy Zone, -1: Sell Zone

    ma_buy_arr = df['MA_Buy'].values
    ma_sell_arr = df['MA_Sell'].values

    for i in range(1, len(df)):
        # If both happen same bar? Pine executes mabuy then masell then barssince.
        # But signals are based on barssince crossover.
        # If mabuy triggers, bars_buy becomes 0.
        # If masell triggers, bars_sell becomes 0.
        # If both trigger, both are 0. crossover(0, 0) is False.
        # So no signal change if both trigger?
        # Actually, if both trigger, the state depends on the previous state and exact inequality.
        # Let's assume standard sequential processing:
        # If MA_Buy triggers -> We want to be Long.
        # If MA_Sell triggers -> We want to be Short/Flat.
        # Priority? The script defines xbuy and xsell independently.
        # Let's track the 'last signal type'.

        is_buy = ma_buy_arr[i]
        is_sell = ma_sell_arr[i]

        if is_buy and not is_sell:
            state = 1
        elif is_sell and not is_buy:
            state = -1
        elif is_buy and is_sell:
            # Conflict. In Pine, if both happen:
            # num_bars_buy = 0, num_bars_sell = 0.
            # xbuy = crossover(0, 0) -> False.
            # xsell = crossunder(0, 0) -> False.
            # So no signal occurs. State remains same?
            pass

        signals[i] = state

    df['State'] = signals

    # Entry Trigger: State changes from -1/0 to 1
    # Exit Trigger: State changes from 1 to -1

    df['prev_state'] = df['State'].shift(1)
    df['Signal'] = 0
    df['Signal'] = np.where((df['State'] == 1) & (df['prev_state'] != 1), 1, df['Signal'])
    df['Signal'] = np.where((df['State'] == -1) & (df['prev_state'] == 1), -1, df['Signal'])

    return df

def run_backtest(ticker, start_date=None, period=None, interval="1d"):
    """
    Runs a single backtest scenario.
    """
    print(f"--- Running Backtest: {ticker} ({interval}) ---")

    if period:
        data = yf.download(ticker, period=period, interval=interval, progress=False)
    else:
        end_date = datetime.datetime.now().strftime('%Y-%m-%d')
        data = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False)

    if len(data) == 0:
        print("No data found.")
        return None

    # Handle MultiIndex
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # Remove timezone
    if data.index.tz is not None:
        data.index = data.index.tz_localize(None)

    # Apply Strategy
    data = up_down_strategy(data)

    # Simulation
    capital = 10000.0
    shares = 0
    in_position = False

    # Buy & Hold
    bh_shares = capital / data['Open'].iloc[0]
    bh_final = bh_shares * data['Close'].iloc[-1]

    trades = 0
    cash = capital

    # Execution Loop (Next Open)
    opens = data['Open'].values
    signals = data['Signal'].values
    closes = data['Close'].values

    for i in range(len(data) - 1):
        sig = signals[i]
        next_open = opens[i+1]

        if sig == 1 and not in_position:
            shares = cash / next_open
            cash = 0
            in_position = True
            trades += 1
        elif sig == -1 and in_position:
            cash = shares * next_open
            shares = 0
            in_position = False
            trades += 1

    final_equity = cash + (shares * closes[-1])

    return {
        "Ticker": ticker,
        "Interval": interval,
        "Start": data.index[0],
        "End": data.index[-1],
        "Strategy Return": (final_equity - capital) / capital * 100,
        "BH Return": (bh_final - capital) / capital * 100,
        "Trades": trades
    }

def resample_1h_to_4h(df):
    logic = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
    if 'Volume' not in df.columns and 'Volume' in logic: del logic['Volume']
    return df.resample('4h').agg(logic).dropna()

def main():
    results = []

    # 1. Sector/Trend Test (Daily, Long Term)
    start_date_10y = (datetime.datetime.now() - timedelta(days=365*14)).strftime('%Y-%m-%d')

    # SOXL (Strong Trend)
    res_soxl_d = run_backtest("SOXL", start_date=start_date_10y, interval="1d")
    if res_soxl_d: results.append(res_soxl_d)

    # GDXU (Choppy/Range)
    res_gdxu_d = run_backtest("GDXU", start_date=start_date_10y, interval="1d")
    if res_gdxu_d: results.append(res_gdxu_d)

    # 2. Leverage Test (Daily, Long Term)
    # SOX (1x)
    res_sox_d = run_backtest("^SOX", start_date=start_date_10y, interval="1d")
    if res_sox_d: results.append(res_sox_d)

    # 3. Timeframe Test (Short Term: 2 Years)
    print("--- Fetching 1h Data for Timeframe Test (max 730d) ---")
    data_1h = yf.download("SOXL", period="730d", interval="1h", progress=False)
    if isinstance(data_1h.columns, pd.MultiIndex): data_1h.columns = data_1h.columns.get_level_values(0)
    if data_1h.index.tz is not None: data_1h.index = data_1h.index.tz_localize(None)

    if len(data_1h) > 0:
        # Run 1h
        df_1h = up_down_strategy(data_1h.copy())
        cap = 10000.0; sh=0; cash=cap; in_pos=False; tr=0
        for i in range(len(df_1h)-1):
            if df_1h['Signal'].iloc[i] == 1 and not in_pos:
                sh = cash / df_1h['Open'].iloc[i+1]
                cash=0; in_pos=True; tr+=1
            elif df_1h['Signal'].iloc[i] == -1 and in_pos:
                cash = sh * df_1h['Open'].iloc[i+1]
                sh=0; in_pos=False; tr+=1
        final_1h = cash + (sh * df_1h['Close'].iloc[-1])
        bh_1h = (10000 / df_1h['Open'].iloc[0]) * df_1h['Close'].iloc[-1]
        results.append({
            "Ticker": "SOXL", "Interval": "1h", "Start": df_1h.index[0], "End": df_1h.index[-1],
            "Strategy Return": (final_1h - 10000)/10000*100, "BH Return": (bh_1h - 10000)/10000*100, "Trades": tr
        })

        # Run 4h
        df_4h = resample_1h_to_4h(data_1h.copy())
        df_4h = up_down_strategy(df_4h)
        cap = 10000.0; sh=0; cash=cap; in_pos=False; tr=0
        for i in range(len(df_4h)-1):
            if df_4h['Signal'].iloc[i] == 1 and not in_pos:
                sh = cash / df_4h['Open'].iloc[i+1]
                cash=0; in_pos=True; tr+=1
            elif df_4h['Signal'].iloc[i] == -1 and in_pos:
                cash = sh * df_4h['Open'].iloc[i+1]
                sh=0; in_pos=False; tr+=1
        final_4h = cash + (sh * df_4h['Close'].iloc[-1])
        bh_4h = (10000 / df_4h['Open'].iloc[0]) * df_4h['Close'].iloc[-1]
        results.append({
            "Ticker": "SOXL", "Interval": "4h", "Start": df_4h.index[0], "End": df_4h.index[-1],
            "Strategy Return": (final_4h - 10000)/10000*100, "BH Return": (bh_4h - 10000)/10000*100, "Trades": tr
        })

        # Run Daily (Same Period)
        start_2y = df_1h.index[0].strftime('%Y-%m-%d')
        res_soxl_d_2y = run_backtest("SOXL", start_date=start_2y, interval="1d")
        if res_soxl_d_2y:
            res_soxl_d_2y["Interval"] = "1d (2y)"
            results.append(res_soxl_d_2y)

    # Print Summary
    print("\n" + "="*80)
    print(f"{'Ticker':<10} | {'Interval':<10} | {'Trades':<6} | {'Strat Return':<15} | {'BH Return':<15} | {'Winner':<10}")
    print("-" * 80)
    for r in results:
        winner = "Strat" if r['Strategy Return'] > r['BH Return'] else "Hold"
        print(f"{r['Ticker']:<10} | {r['Interval']:<10} | {r['Trades']:<6} | {r['Strategy Return']:>14.2f}% | {r['BH Return']:>14.2f}% | {winner:<10}")
    print("="*80)

if __name__ == "__main__":
    main()
