import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys

def fetch_data(ticker="QQQ"):
    print(f"Fetching data for {ticker}...")
    end_date = datetime.now()
    start_date_1h = end_date - timedelta(days=729)
    df_1h = yf.download(ticker, start=start_date_1h, end=end_date, interval="1h", progress=False)

    start_date_5m = end_date - timedelta(days=59)
    df_5m = yf.download(ticker, start=start_date_5m, end=end_date, interval="5m", progress=False)

    if df_1h.empty:
        print("Error: Empty data returned from yfinance.")
        sys.exit(1)

    if isinstance(df_1h.columns, pd.MultiIndex):
        df_1h.columns = df_1h.columns.get_level_values(0)
    if not df_5m.empty and isinstance(df_5m.columns, pd.MultiIndex):
        df_5m.columns = df_5m.columns.get_level_values(0)

    if df_1h.index.tz is None: df_1h.index = df_1h.index.tz_localize('UTC')
    else: df_1h.index = df_1h.index.tz_convert('UTC')

    if not df_5m.empty:
        if df_5m.index.tz is None: df_5m.index = df_5m.index.tz_localize('UTC')
        else: df_5m.index = df_5m.index.tz_convert('UTC')

    df_1h = df_1h.dropna()
    df_5m = df_5m.dropna()
    return df_1h, df_5m

def calculate_trend(df_source, timeframe='4h'):
    agg_dict = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
    df_tf = df_source.resample(timeframe).agg(agg_dict).dropna()
    df_tf['Trend'] = 0 # Neutral
    return df_tf

def identify_swings(df, fractal_period=1):
    df = df.copy()
    df['Swing_High'] = False
    df['Swing_Low'] = False
    is_swing_high = pd.Series(True, index=df.index)
    is_swing_low = pd.Series(True, index=df.index)
    for i in range(1, fractal_period + 1):
        is_swing_high &= (df['High'] > df['High'].shift(i)) & (df['High'] > df['High'].shift(-i))
        is_swing_low &= (df['Low'] < df['Low'].shift(i)) & (df['Low'] < df['Low'].shift(-i))
    df.loc[is_swing_high, 'Swing_High'] = True
    df.loc[is_swing_low, 'Swing_Low'] = True
    return df

def run_strategy(df_entry, df_trend, df_sweep_source, params):
    df_trend_reindexed = df_trend['Trend'].reindex(df_entry.index, method='ffill').fillna(0)

    df_entry = identify_swings(df_entry, fractal_period=params['fractal_entry'])
    df_sweep_source = identify_swings(df_sweep_source, fractal_period=params['fractal_sweep'])
    sweep_rows = df_sweep_source[df_sweep_source['Swing_High'] | df_sweep_source['Swing_Low']].copy()

    trades = []
    active_setup = None
    SWEEP_WINDOW = params['sweep_window']

    print(f"Running strategy on {len(df_entry)} bars ({params['name']})...")

    for i in range(len(df_entry)):
        if i < 50: continue

        current_time = df_entry.index[i]
        current_row = df_entry.iloc[i]
        current_trend = df_trend_reindexed.iloc[i]

        lookback_time = current_time - timedelta(days=params['sweep_lookback_days'])
        valid_swings = sweep_rows[(sweep_rows.index >= lookback_time) & (sweep_rows.index <= current_time)]

        trend_bullish = True # Always True if Neutral
        trend_bearish = True

        # --- 1. Sweep Detection ---
        if trend_bullish:
            swept_lows = valid_swings[valid_swings['Swing_Low']]['Low']
            hit_lows = swept_lows[swept_lows > current_row['Low']]
            if not hit_lows.empty:
                if active_setup is None or active_setup['type'] != 'Long':
                    active_setup = {
                        'type': 'Long',
                        'sweep_time': current_time,
                        'lowest_price': current_row['Low'],
                        'state': 'WAIT_FOR_MSS'
                    }
                else:
                    if current_row['Low'] < active_setup['lowest_price']:
                        active_setup['lowest_price'] = current_row['Low']
                        active_setup['sweep_time'] = current_time
                        if active_setup['state'] != 'WAIT_FOR_MSS':
                            active_setup['state'] = 'WAIT_FOR_MSS'
                            active_setup.pop('entry_price', None)
                            active_setup.pop('stop_loss', None)

        if trend_bearish:
            swept_highs = valid_swings[valid_swings['Swing_High']]['High']
            hit_highs = swept_highs[swept_highs < current_row['High']]
            if not hit_highs.empty:
                if active_setup is None or active_setup['type'] != 'Short':
                    active_setup = {
                        'type': 'Short',
                        'sweep_time': current_time,
                        'highest_price': current_row['High'],
                        'state': 'WAIT_FOR_MSS'
                    }
                else:
                    if current_row['High'] > active_setup['highest_price']:
                        active_setup['highest_price'] = current_row['High']
                        active_setup['sweep_time'] = current_time
                        if active_setup['state'] != 'WAIT_FOR_MSS':
                            active_setup['state'] = 'WAIT_FOR_MSS'
                            active_setup.pop('entry_price', None)
                            active_setup.pop('stop_loss', None)

        # --- 2. Active Setup Logic ---
        if active_setup:
            if (current_time - active_setup['sweep_time']).total_seconds() > (SWEEP_WINDOW * params['bar_seconds']):
                active_setup = None; continue

            if active_setup['state'] == 'WAIT_FOR_MSS':
                # Fix Look-ahead bias: Only consider swings confirmed by current time 'i'
                # If fractal=1, i-1 is confirmed by i. Slice :i includes i-1.
                # If fractal=2, i-1 is NOT confirmed (needs i+1). Slice must stop at i-1.
                safe_end_idx = i - (params['fractal_entry'] - 1)

                if active_setup['type'] == 'Long':
                    recent_swings = df_entry.iloc[i-50:safe_end_idx]
                    confirmed_highs = recent_swings[recent_swings['Swing_High']]

                    if not confirmed_highs.empty:
                        last_swing_high = confirmed_highs.iloc[-1]['High']
                        if current_row['Close'] > last_swing_high:
                            active_setup['state'] = 'WAIT_FOR_FVG_ENTRY'
                            active_setup['mss_time'] = current_time
                            active_setup['stop_loss'] = active_setup['lowest_price']

                            found_fvg = None
                            for k in range(i, i-50, -1):
                                if k-2 < 0: continue
                                c_low = df_entry.iloc[k]['Low']
                                prev_high = df_entry.iloc[k-2]['High']
                                if c_low > prev_high:
                                    found_fvg = c_low
                                    break

                            if found_fvg:
                                active_setup['entry_price'] = found_fvg
                                active_setup['state'] = 'PENDING_ENTRY'
                            else: active_setup = None

                elif active_setup['type'] == 'Short':
                    recent_swings = df_entry.iloc[i-50:safe_end_idx]
                    confirmed_lows = recent_swings[recent_swings['Swing_Low']]

                    if not confirmed_lows.empty:
                        last_swing_low = confirmed_lows.iloc[-1]['Low']
                        if current_row['Close'] < last_swing_low:
                            active_setup['state'] = 'WAIT_FOR_FVG_ENTRY'
                            active_setup['mss_time'] = current_time
                            active_setup['stop_loss'] = active_setup['highest_price']

                            found_fvg = None
                            for k in range(i, i-50, -1):
                                if k-2 < 0: continue
                                c_high = df_entry.iloc[k]['High']
                                prev_low = df_entry.iloc[k-2]['Low']
                                if c_high < prev_low:
                                    found_fvg = c_high
                                    break

                            if found_fvg:
                                active_setup['entry_price'] = found_fvg
                                active_setup['state'] = 'PENDING_ENTRY'
                            else: active_setup = None

            elif active_setup['state'] == 'PENDING_ENTRY':
                if (current_time - active_setup['mss_time']).total_seconds() > (3600 * 4):
                    active_setup = None; continue

                if active_setup['type'] == 'Long':
                    if current_row['Low'] < active_setup['stop_loss']: active_setup = None; continue
                    if current_row['Low'] <= active_setup['entry_price']:
                        entry = active_setup['entry_price']
                        sl = active_setup['stop_loss']
                        risk = entry - sl
                        if risk <= 0: active_setup = None; continue
                        tp = entry + (2 * risk)
                        trades.append({'Entry Time': current_time, 'Type': 'Long', 'Entry Price': entry, 'Stop Loss': sl, 'Take Profit': tp, 'Status': 'Open'})
                        active_setup = None

                elif active_setup['type'] == 'Short':
                    if current_row['High'] > active_setup['stop_loss']: active_setup = None; continue
                    if current_row['High'] >= active_setup['entry_price']:
                        entry = active_setup['entry_price']
                        sl = active_setup['stop_loss']
                        risk = sl - entry
                        if risk <= 0: active_setup = None; continue
                        tp = entry - (2 * risk)
                        trades.append({'Entry Time': current_time, 'Type': 'Short', 'Entry Price': entry, 'Stop Loss': sl, 'Take Profit': tp, 'Status': 'Open'})
                        active_setup = None

        for trade in trades:
            if trade['Status'] == 'Open':
                if trade['Type'] == 'Long':
                    if current_row['Low'] <= trade['Stop Loss']:
                        trade['Status'] = 'Loss'; trade['Exit Time'] = current_time; trade['PnL'] = -1.0
                    elif current_row['High'] >= trade['Take Profit']:
                        trade['Status'] = 'Win'; trade['Exit Time'] = current_time; trade['PnL'] = 2.0
                else:
                    if current_row['High'] >= trade['Stop Loss']:
                        trade['Status'] = 'Loss'; trade['Exit Time'] = current_time; trade['PnL'] = -1.0
                    elif current_row['Low'] <= trade['Take Profit']:
                        trade['Status'] = 'Win'; trade['Exit Time'] = current_time; trade['PnL'] = 2.0

    return pd.DataFrame(trades)

def summarize(trades_df):
    if trades_df.empty: return 0, 0, 0, 0
    closed = trades_df[trades_df['Status'] != 'Open']
    wins = len(closed[closed['Status'] == 'Win'])
    total = len(closed)
    win_rate = (wins/total)*100 if total > 0 else 0
    pnl = closed['PnL'].sum() if 'PnL' in closed.columns else 0
    return total, wins, win_rate, pnl

if __name__ == "__main__":
    df_1h, df_5m = fetch_data()
    print("Data fetched.")

    df_4h_trend = calculate_trend(df_1h, '4h')
    df_daily_trend = calculate_trend(df_1h, '1D')

    # Sources
    df_1h_swings = identify_swings(df_1h, fractal_period=1)
    df_4h_swings = df_1h.resample('4h').agg({'Open':'first','High':'max','Low':'min','Close':'last'}).dropna()
    df_4h_swings = identify_swings(df_4h_swings, fractal_period=1)

    # Construct 15M from 5M
    if not df_5m.empty:
        df_15m = df_5m.resample('15min').agg({'Open':'first','High':'max','Low':'min','Close':'last'}).dropna()
        df_15m_swings = identify_swings(df_15m, fractal_period=1)

    # Scenario 1: 5M
    params_5m = {
        'name': '5M Precision (Short Term)',
        'fractal_entry': 2,
        'fractal_sweep': 1,
        'sweep_window': 144,
        'bar_seconds': 300,
        'sweep_lookback_days': 5
    }
    if not df_5m.empty: trades_5m = run_strategy(df_5m, df_4h_trend, df_1h_swings, params_5m)
    else: trades_5m = pd.DataFrame()

    # Scenario 2: 1H Proxy
    params_1h = {
        'name': '1H Scaled Proxy (2 Years)',
        'fractal_entry': 2,
        'fractal_sweep': 1,
        'sweep_window': 24,
        'bar_seconds': 3600,
        'sweep_lookback_days': 10
    }
    trades_1h = run_strategy(df_1h, df_daily_trend, df_4h_swings, params_1h)

    # Scenario 3: 15M Scalp
    params_15m = {
        'name': '15M Scalp (Short Term)',
        'fractal_entry': 1, # Ultra Sensitive Entry
        'fractal_sweep': 1,
        'sweep_window': 48, # 4 hours
        'bar_seconds': 300,
        'sweep_lookback_days': 2
    }
    if not df_5m.empty: trades_15m = run_strategy(df_5m, df_4h_trend, df_15m_swings, params_15m)
    else: trades_15m = pd.DataFrame()

    t5, w5, wr5, pnl5 = summarize(trades_5m)
    t1, w1, wr1, pnl1 = summarize(trades_1h)
    t15, w15, wr15, pnl15 = summarize(trades_15m)

    report = f"""
# Backtest Report: QQQ SMC Precision Strategy

## Overview
Due to data limitations (5M data limited to 60 days), three scenarios were tested to maximize verification:
1. **5M Precision:** Standard 1H/5M setup on last 60 days.
2. **15M Scalp:** Aggressive 15M/5M setup on last 60 days (High Frequency).
3. **1H Proxy:** Scaled 4H/1H setup on last 2 years (Robustness).

**Note:** Trend Filter disabled (Neutral) to test pure Price Action mechanics.

## Scenario 1: 5M Precision (Last ~60 Days)
*Timeframes: 1H Sweep, 5M Entry*
- **Total Trades:** {t5}
- **Win Rate:** {wr5:.2f}%
- **Wins:** {w5}
- **Total Return:** {pnl5:.2f} R

## Scenario 2: 15M Scalp (Last ~60 Days)
*Timeframes: 15M Sweep, 5M Entry*
- **Total Trades:** {t15}
- **Win Rate:** {wr15:.2f}%
- **Wins:** {w15}
- **Total Return:** {pnl15:.2f} R

## Scenario 3: 1H Proxy (Last 2 Years)
*Timeframes: 4H Sweep, 1H Entry*
- **Total Trades:** {t1}
- **Win Rate:** {wr1:.2f}%
- **Wins:** {w1}
- **Total Return:** {pnl1:.2f} R

## Conclusion
The **5M Precision** setup generated **{t5} trades** with **{pnl5:.2f} R** profit.
The **15M Scalp** generated **{t15} trades** with **{pnl15:.2f} R** profit.
    """

    print(report)
    with open("qqq_smc_report.md", "w") as f:
        f.write(report)

    trades_5m.to_csv("qqq_smc_trades_5m.csv")
    trades_15m.to_csv("qqq_smc_trades_15m.csv")
    trades_1h.to_csv("qqq_smc_trades_1h.csv")
