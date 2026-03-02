import yfinance as yf
import pandas as pd
import numpy as np
import datetime
from datetime import timedelta

def calculate_atr(df, period):
    high = df['High']
    low = df['Low']
    close = df['Close']

    # True Range
    df['tr0'] = abs(high - low)
    df['tr1'] = abs(high - close.shift(1))
    df['tr2'] = abs(low - close.shift(1))
    tr = df[['tr0', 'tr1', 'tr2']].max(axis=1)

    # ATR using RMA (Wilder's Smoothing)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    return atr

def ut_bot_strategy(df, key=2, period=6):
    """
    Implements the UT Bot logic from the 'SMC Flow' script.
    Inputs: Key=2, Period=6 (Aggressive).
    """
    df['ATR'] = calculate_atr(df, period)
    df['nLoss'] = key * df['ATR']

    src = df['Close'].values
    nLoss = df['nLoss'].values

    xATRTrailingStop = np.zeros(len(df))
    pos = np.zeros(len(df))

    # Initialize
    xATRTrailingStop[0] = src[0]
    pos[0] = 0

    for i in range(1, len(df)):
        prev_stop = xATRTrailingStop[i-1]
        prev_src = src[i-1]
        curr_src = src[i]
        curr_nLoss = nLoss[i]

        if np.isnan(curr_nLoss):
            xATRTrailingStop[i] = curr_src
            continue

        # Trailing Stop Logic (from Pine Script)
        if curr_src > prev_stop and prev_src > prev_stop:
            xATRTrailingStop[i] = max(prev_stop, curr_src - curr_nLoss)
        elif curr_src < prev_stop and prev_src < prev_stop:
            xATRTrailingStop[i] = min(prev_stop, curr_src + curr_nLoss)
        elif curr_src > prev_stop:
            xATRTrailingStop[i] = curr_src - curr_nLoss
        else:
            xATRTrailingStop[i] = curr_src + curr_nLoss

        # Position Logic
        prev_pos = pos[i-1]
        if prev_src < prev_stop and curr_src > xATRTrailingStop[i]:
            pos[i] = 1
        elif prev_src > prev_stop and curr_src < xATRTrailingStop[i]:
            pos[i] = -1
        else:
            pos[i] = prev_pos

    df['pos'] = pos

    # Generate Signals
    # Buy: pos flips to 1
    # Sell: pos flips to -1
    df['prev_pos'] = df['pos'].shift(1)
    df['Signal'] = 0
    df['Signal'] = np.where((df['pos'] == 1) & (df['prev_pos'] != 1), 1, df['Signal'])
    df['Signal'] = np.where((df['pos'] == -1) & (df['prev_pos'] != -1), -1, df['Signal'])

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

    # Resample if needed (e.g. 4h from 1h)
    if interval == "4h_sim":
        # Hack: Fetch 1h data first (assumed handled by caller), then resample here
        # But here we assume 'data' is already the raw data.
        # Let's assume the caller passes "1h" data and asks for resampling logic if needed.
        pass # implemented differently below

    # Apply Strategy
    data = ut_bot_strategy(data, key=2, period=6)

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
    # We need 'Open' of i+1
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

    # GDXU (Choppy/Range) - Note: Inception ~2020
    res_gdxu_d = run_backtest("GDXU", start_date=start_date_10y, interval="1d")
    if res_gdxu_d: results.append(res_gdxu_d)

    # 2. Leverage Test (Daily, Long Term)
    # SOX (1x)
    res_sox_d = run_backtest("^SOX", start_date=start_date_10y, interval="1d")
    if res_sox_d: results.append(res_sox_d)

    # 3. Timeframe Test (Short Term: 2 Years)
    # Fetch 1h data once
    print("--- Fetching 1h Data for Timeframe Test (max 730d) ---")
    data_1h = yf.download("SOXL", period="730d", interval="1h", progress=False)
    if isinstance(data_1h.columns, pd.MultiIndex): data_1h.columns = data_1h.columns.get_level_values(0)
    if data_1h.index.tz is not None: data_1h.index = data_1h.index.tz_localize(None)

    if len(data_1h) > 0:
        # Run 1h
        # Create a temp wrapper to reuse logic (not efficient but clean)
        # Actually, just call logic directly
        df_1h = ut_bot_strategy(data_1h.copy(), key=2, period=6)
        # Sim 1h
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
        df_4h = ut_bot_strategy(df_4h, key=2, period=6)
        # Sim 4h
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

        # Run Daily (Same Period Comparison)
        # Resample 1h to Daily to assume perfect match? Or fetch?
        # Better to fetch daily for same period
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
