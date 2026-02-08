import yfinance as yf
import pandas as pd
import numpy as np
import datetime
from datetime import timedelta
import pandas_ta as ta

# --- Helper Functions ---

def calculate_hma(series, length):
    return ta.hma(series, length=length)

def calculate_supertrend(df, length, multiplier):
    st = ta.supertrend(df['High'], df['Low'], df['Close'], length=length, multiplier=multiplier)
    # pandas_ta returns columns like SUPERT_7_3.0, SUPERTd_7_3.0, etc.
    # We need the direction (1 or -1) usually in the 'd' column
    col_dir = f"SUPERTd_{length}_{float(multiplier)}"
    if col_dir not in st.columns:
        # Fallback search
        col_dir = [c for c in st.columns if c.startswith("SUPERTd")][0]
    return st[col_dir]

def calculate_squeeze(df, bb_length=20, bb_mult=2.0, kc_length=20, kc_mult=1.5):
    # Squeeze is ON when BB is inside KC
    # BB
    bb = ta.bbands(df['Close'], length=bb_length, std=bb_mult)
    # KC
    kc = ta.kc(df['High'], df['Low'], df['Close'], length=kc_length, scalar=kc_mult)

    if bb is None or kc is None: return pd.Series(0, index=df.index)

    # BB Columns: BBL_..., BBM_..., BBU_...
    # KC Columns: KCL_..., KCB_..., KCU_...
    # pandas_ta column naming might omit .0 if int, or vary.
    # We'll use fallback search.
    cols = bb.columns
    bbu_col = [c for c in cols if c.startswith("BBU")][0]
    bbl_col = [c for c in cols if c.startswith("BBL")][0]

    kcols = kc.columns
    kcu_col = [c for c in kcols if c.startswith("KCU")][0]
    kcl_col = [c for c in kcols if c.startswith("KCL")][0]

    bb_upper = bb[bbu_col]
    bb_lower = bb[bbl_col]
    kc_upper = kc[kcu_col]
    kc_lower = kc[kcl_col]

    squeeze_on = (bb_upper < kc_upper) & (bb_lower > kc_lower)
    return squeeze_on

# --- Strategies ---

def strat_pmax(df, period=10, multiplier=3):
    """
    PMax Approximation:
    Uses an EMA and ATR-based Trailing Stop (similar to SuperTrend but centered on MA).
    Logic:
    - Calculate MA (EMA 10).
    - Calculate Bands = MA +/- ATR*Multiplier.
    - If Close > Band, Trend = Bull.
    """
    # Simply use SuperTrend logic on the EMA?
    # Or simplified: Standard SuperTrend with these params is a good proxy for PMax behavior (Trailing ATR).
    # PMax often uses Moving Average as the anchor, SuperTrend uses H/L.
    # Let's use a standard SuperTrend implementation as the robust 'Trend + Volatility' representative.

    # Input from report: PMax Explorer.
    # Proxy: SuperTrend(10, 3)

    df['ST_Dir'] = calculate_supertrend(df, period, multiplier)

    # Signal: 1 if ST_Dir == 1 (Bull), -1 if ST_Dir == -1 (Bear)
    return df['ST_Dir']

def strat_hull_suite(df, length=55):
    """
    Hull Suite:
    Entry when HMA changes slope/color.
    """
    hma = calculate_hma(df['Close'], length)

    # Direction
    # 1 if HMA > HMA[1], -1 if HMA < HMA[1]
    direction = np.where(hma > hma.shift(1), 1, -1)
    return pd.Series(direction, index=df.index)

def strat_squeeze(df):
    """
    Wolf & Bear / Squeeze Momentum:
    Enter when Squeeze 'fires' (Squeeze On -> Squeeze Off).
    Direction determined by Momentum (Close vs EMA?).
    Simplification:
    - If Squeeze was ON yesterday and OFF today -> Fire.
    - Direction: If Close > EMA(20) -> Buy, else Sell.
    """
    sqz = calculate_squeeze(df)
    ema20 = ta.ema(df['Close'], length=20)

    # Fire Condition
    sqz_prev = sqz.shift(1)
    fired = (sqz_prev == True) & (sqz == False)

    # Direction
    direction = np.where(df['Close'] > ema20, 1, -1)

    # Signal: Only valid on fire bars. Hold until next fire?
    # Or Hold until Trend Change?
    # Report says: "Catch expansion phase".
    # Implementation: Position = Direction * Fired.
    # Then forward fill? Squeeze trade usually targets momentum.
    # Let's fill forward until trend reversal (Close crosses EMA20).

    signals = np.zeros(len(df))
    current_pos = 0

    fired_vals = fired.values
    close_vals = df['Close'].values
    ema_vals = ema20.values

    for i in range(1, len(df)):
        if fired_vals[i]:
            # Entry
            if close_vals[i] > ema_vals[i]:
                current_pos = 1
            else:
                current_pos = -1
        else:
            # Exit Check
            if current_pos == 1 and close_vals[i] < ema_vals[i]:
                current_pos = 0
            elif current_pos == -1 and close_vals[i] > ema_vals[i]:
                current_pos = 0

        signals[i] = current_pos

    return pd.Series(signals, index=df.index)

def strat_ut_bot(df, key=2, period=1):
    """
    UT Bot (Strategy 9).
    High Sensitivity.
    """
    # Reuse previous logic (simplified implementation)
    atr = ta.atr(df['High'], df['Low'], df['Close'], length=period)
    nLoss = key * atr

    src = df['Close'].values
    xATRTrailingStop = np.zeros(len(df))
    pos = np.zeros(len(df))

    # Init
    xATRTrailingStop[0] = src[0]

    for i in range(1, len(df)):
        prev_stop = xATRTrailingStop[i-1]
        prev_src = src[i-1]
        curr_src = src[i]
        nl = nLoss.iloc[i]
        if np.isnan(nl): nl = 0 # warmup

        if curr_src > prev_stop and prev_src > prev_stop:
            xATRTrailingStop[i] = max(prev_stop, curr_src - nl)
        elif curr_src < prev_stop and prev_src < prev_stop:
            xATRTrailingStop[i] = min(prev_stop, curr_src + nl)
        elif curr_src > prev_stop:
            xATRTrailingStop[i] = curr_src - nl
        else:
            xATRTrailingStop[i] = curr_src + nl

        prev_pos = pos[i-1]
        if prev_src < prev_stop and curr_src > xATRTrailingStop[i]:
            pos[i] = 1
        elif prev_src > prev_stop and curr_src < xATRTrailingStop[i]:
            pos[i] = -1
        else:
            pos[i] = prev_pos

    return pd.Series(pos, index=df.index)

def run_strategy(df, strategy_name):
    if strategy_name == "PMax (SuperTrend)":
        return strat_pmax(df, period=10, multiplier=3)
    elif strategy_name == "Hull Suite":
        return strat_hull_suite(df, length=55)
    elif strategy_name == "Squeeze (Wolf)":
        return strat_squeeze(df)
    elif strategy_name == "UT Bot":
        return strat_ut_bot(df, key=2, period=1) # Original aggressive settings
    return pd.Series(0, index=df.index)

def backtest_engine(ticker, interval, start_date=None, period_str=None):
    print(f"Fetch: {ticker} ({interval})")
    if period_str:
        data = yf.download(ticker, period=period_str, interval=interval, progress=False)
    else:
        end = datetime.datetime.now().strftime('%Y-%m-%d')
        data = yf.download(ticker, start=start_date, end=end, interval=interval, progress=False)

    if len(data) == 0: return {}

    if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
    if data.index.tz is not None: data.index = data.index.tz_localize(None)

    # Strategies List
    strats = ["PMax (SuperTrend)", "Hull Suite", "Squeeze (Wolf)", "UT Bot"]

    results = {}

    # Buy & Hold
    bh_ret = (data['Close'].iloc[-1] / data['Open'].iloc[0] - 1) * 100
    results['Buy & Hold'] = bh_ret

    for strat in strats:
        # Calculate Signals
        sigs = run_strategy(data.copy(), strat)

        # Simulation (Long Only, Exit to Cash)
        cap = 10000.0
        cash = cap
        shares = 0
        in_pos = False
        trades = 0

        sig_vals = sigs.values
        opens = data['Open'].values
        closes = data['Close'].values

        for i in range(len(data)-1):
            s = sig_vals[i]
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
        results[strat] = ret

    return results

def main():
    # Scenarios
    # 1. Sector: SOXL (Trend) vs GDXU (Range) - Daily
    start_10y = (datetime.datetime.now() - timedelta(days=365*14)).strftime('%Y-%m-%d')

    print("\n=== SCENARIO 1: SECTOR STRENGTH (Daily, Max 10y) ===")
    res_soxl = backtest_engine("SOXL", "1d", start_date=start_10y)
    res_gdxu = backtest_engine("GDXU", "1d", start_date=start_10y)

    # 2. Leverage: SOX (1x) vs SOXL (3x) - Daily
    print("\n=== SCENARIO 2: LEVERAGE (Daily, Max 10y) ===")
    res_sox = backtest_engine("^SOX", "1d", start_date=start_10y)

    # 3. Timeframe: SOXL 1h vs Daily (2y)
    print("\n=== SCENARIO 3: TIMEFRAME (2 Years) ===")
    res_soxl_1h = backtest_engine("SOXL", "1h", period_str="730d")

    # Compile
    print("\n" + "="*60)
    print(f"{'Context':<15} | {'Strategy':<20} | {'Return':>10}")
    print("-" * 60)

    scenarios = [
        ("SOXL (Daily)", res_soxl),
        ("GDXU (Daily)", res_gdxu),
        ("^SOX (Daily)", res_sox),
        ("SOXL (1h)", res_soxl_1h)
    ]

    for name, res in scenarios:
        if not res: continue
        print(f"--- {name} ---")
        # Sort by return
        sorted_res = sorted(res.items(), key=lambda x: x[1], reverse=True)
        for strat, ret in sorted_res:
            print(f"{'':<15} | {strat:<20} | {ret:>9.2f}%")

    print("="*60)

if __name__ == "__main__":
    main()
