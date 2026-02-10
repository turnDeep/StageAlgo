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
    st = ta.supertrend(df['High'], df['Low'], df['Close'], length=length, multiplier=multiplier)
    if st is None: return pd.Series(0, index=df.index)
    col_dir = [c for c in st.columns if c.startswith("SUPERTd")][0]
    return st[col_dir]

def calculate_atr(df, period):
    high = df['High']
    low = df['Low']
    close = df['Close']
    tr0 = abs(high - low)
    tr1 = abs(high - close.shift(1))
    tr2 = abs(low - close.shift(1))
    tr = pd.concat([tr0, tr1, tr2], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    return atr

# --- Strategies ---

# 1. MMXM
def strat_mmxm(df, swing_len=5):
    high_prev2 = df['High'].shift(2)
    low_prev2 = df['Low'].shift(2)
    bull_fvg = (df['Low'] > high_prev2)
    bear_fvg = (df['High'] < low_prev2)

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
        if mss_bull_v[i]: bias = 1
        elif mss_bear_v[i]: bias = -1

        if bull_fvg_v[i]: active_bull.append((low_v[i], high_prev2_v[i]))
        if bear_fvg_v[i]: active_bear.append((low_prev2_v[i], high_v[i]))

        if len(active_bull) > 5: active_bull.pop(0)
        if len(active_bear) > 5: active_bear.pop(0)

        sig = 0
        if bias == 1:
            for fvg in active_bull:
                if low_v[i] <= fvg[0]: sig = 1; break
        elif bias == -1:
            for fvg in active_bear:
                if high_v[i] >= fvg[1]: sig = -1; break

        sigs[i] = sig

    return pd.Series(sigs, index=df.index)

# 2. PMax
def strat_pmax(df, period=10, multiplier=3):
    return calculate_supertrend(df, period, multiplier)

# 3. UT Bot (Aggressive)
def strat_ut_bot(df, key=2, period=6):
    df = df.copy()
    df['ATR'] = calculate_atr(df, period)
    df['nLoss'] = key * df['ATR']

    src = df['Close'].values
    nLoss = df['nLoss'].values
    xATRTrailingStop = np.zeros(len(df))
    pos = np.zeros(len(df))
    xATRTrailingStop[0] = src[0]

    for i in range(1, len(df)):
        prev_stop = xATRTrailingStop[i-1]
        prev_src = src[i-1]
        curr_src = src[i]
        curr_nLoss = nLoss[i]

        if np.isnan(curr_nLoss): xATRTrailingStop[i] = curr_src; continue

        if curr_src > prev_stop and prev_src > prev_stop:
            xATRTrailingStop[i] = max(prev_stop, curr_src - curr_nLoss)
        elif curr_src < prev_stop and prev_src < prev_stop:
            xATRTrailingStop[i] = min(prev_stop, curr_src + curr_nLoss)
        elif curr_src > prev_stop:
            xATRTrailingStop[i] = curr_src - curr_nLoss
        else:
            xATRTrailingStop[i] = curr_src + curr_nLoss

        prev_pos = pos[i-1]
        if prev_src < prev_stop and curr_src > xATRTrailingStop[i]: pos[i] = 1
        elif prev_src > prev_stop and curr_src < xATRTrailingStop[i]: pos[i] = -1
        else: pos[i] = prev_pos

    # Convert position state to signals (1, -1)
    # If pos == 1 -> Signal 1. If pos == -1 -> Signal -1.
    # Logic in previous scripts was state based.
    return pd.Series(pos, index=df.index)

# 4. Up Down
def strat_up_down(df, long_ma=77, short_ma=7):
    df = df.copy()
    df['SMA_Long'] = df['Close'].rolling(long_ma).mean()
    df['SMA_Short'] = df['Close'].rolling(short_ma).mean()

    prev_close = df['Close'].shift(1)
    pat_buy = (df['Close'] > df['Open']) & (df['Open'] > prev_close)
    pat_sell = (df['Close'] < df['Open']) & (df['Open'] < prev_close)

    prev_s = df['SMA_Short'].shift(1)
    prev_l = df['SMA_Long'].shift(1)

    crossover = (df['SMA_Short'] > df['SMA_Long']) & (prev_s <= prev_l)
    crossunder = (df['SMA_Short'] < df['SMA_Long']) & (prev_s >= prev_l)

    ma_buy = crossover | (pat_buy & (df['SMA_Short'] > df['SMA_Long']))
    ma_sell = crossunder | pat_sell

    signals = np.zeros(len(df))
    state = 0

    buy_vals = ma_buy.values
    sell_vals = ma_sell.values

    for i in range(1, len(df)):
        if buy_vals[i] and not sell_vals[i]: state = 1
        elif sell_vals[i] and not buy_vals[i]: state = -1
        signals[i] = state

    return pd.Series(signals, index=df.index)

def run_backtest_all():
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
    sig_mmxm = strat_mmxm(sox_w.copy())
    sig_pmax = strat_pmax(sox_w.copy())
    sig_ut = strat_ut_bot(sox_w.copy())
    sig_updown = strat_up_down(sox_w.copy())

    simulation_df = pd.DataFrame(index=soxl_w.index)
    simulation_df['Open'] = soxl_w['Open']
    simulation_df['Close'] = soxl_w['Close']

    combined = pd.concat([
        sig_mmxm.rename('MMXM'),
        sig_pmax.rename('PMax'),
        sig_ut.rename('UT_Bot'),
        sig_updown.rename('Up_Down'),
        simulation_df
    ], axis=1, join='inner')

    ten_years_ago = (datetime.datetime.now() - timedelta(days=365*10)).replace(hour=0, minute=0, second=0, microsecond=0)
    test_data = combined[combined.index >= ten_years_ago].copy()

    if len(test_data) == 0:
        print("No data for 10 years.")
        return

    print(f"Backtest Period: {test_data.index[0].date()} to {test_data.index[-1].date()}")

    strats = ["MMXM", "PMax", "UT_Bot", "Up_Down", "BuyHold"]
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
            shares = cash / opens[0]
            cash = 0
            trades = 1
            final = shares * closes[-1]
        else:
            signals = test_data[s].values

            for i in range(len(test_data)-1):
                sig = signals[i]
                next_open = opens[i+1]

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
        results[s] = {"Return": ret, "Trades": trades}

    print("\n=======================================================")
    print("WEEKLY STRATEGY COMPARISON (SOX -> SOXL, 10 Years)")
    print("=======================================================")
    print(f"{'Strategy':<15} | {'Return':>15} | {'Trades':>6}")
    print("-" * 45)

    # Sort by return
    sorted_res = sorted(results.items(), key=lambda x: x[1]['Return'], reverse=True)

    for s, res in sorted_res:
        print(f"{s:<15} | {res['Return']:>14.2f}% | {res['Trades']:>6}")

if __name__ == "__main__":
    run_backtest_all()
