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

# --- PMax Strategy (Weekly) ---
def strat_pmax(df, period=10, multiplier=3):
    return calculate_supertrend(df, period, multiplier)

def run_weekly_pmax_backtest(signal_ticker, trade_ticker, years=10):
    print(f"\nFetching Daily data for {signal_ticker} and {trade_ticker}...")
    # Fetch extra buffer for calculation
    start_date = (datetime.datetime.now() - timedelta(days=365*(years+4))).strftime('%Y-%m-%d')
    end_date = datetime.datetime.now().strftime('%Y-%m-%d')

    sig_d = yf.download(signal_ticker, start=start_date, end=end_date, progress=False)
    trd_d = yf.download(trade_ticker, start=start_date, end=end_date, progress=False)

    # Data Cleaning
    if isinstance(sig_d.columns, pd.MultiIndex): sig_d.columns = sig_d.columns.get_level_values(0)
    if isinstance(trd_d.columns, pd.MultiIndex): trd_d.columns = trd_d.columns.get_level_values(0)
    if sig_d.index.tz is not None: sig_d.index = sig_d.index.tz_localize(None)
    if trd_d.index.tz is not None: trd_d.index = trd_d.index.tz_localize(None)

    if len(trd_d) == 0:
        print(f"Error: No data for {trade_ticker}")
        return

    print("Resampling to Weekly...")
    sig_w = resample_to_weekly(sig_d)
    trd_w = resample_to_weekly(trd_d)

    print(f"Calculating PMax on Weekly {signal_ticker}...")
    signals = strat_pmax(sig_w.copy())

    # Align Data
    simulation_df = pd.DataFrame(index=trd_w.index)
    simulation_df['Open'] = trd_w['Open']
    simulation_df['Close'] = trd_w['Close']

    combined = pd.concat([signals.rename('Signal'), simulation_df], axis=1, join='inner')

    # Filter for requested years
    target_start = (datetime.datetime.now() - timedelta(days=365*years)).replace(hour=0, minute=0, second=0, microsecond=0)
    test_data = combined[combined.index >= target_start].copy()

    if len(test_data) == 0:
        # If dataset is shorter than target years (e.g. SHNY), use all available
        print(f"Data history shorter than {years} years. Using all available data.")
        test_data = combined.copy()

    print(f"Backtest Period: {test_data.index[0].date()} to {test_data.index[-1].date()}")

    # --- Simulation ---
    results = {}

    # 1. PMax Strategy
    capital = 10000.0
    cash = capital
    shares = 0
    in_pos = False
    trades = 0

    opens = test_data['Open'].values
    closes = test_data['Close'].values
    sigs = test_data['Signal'].values

    for i in range(len(test_data)-1):
        s = sigs[i]
        next_open = opens[i+1]

        if s == 1 and not in_pos:
            shares = cash / next_open
            cash = 0
            in_pos = True
            trades += 1
        elif s == -1 and in_pos:
            cash = shares * next_open
            shares = 0
            in_pos = False
            trades += 1

    final_strat = cash + (shares * closes[-1])
    ret_strat = (final_strat - capital) / capital * 100

    # 2. Buy & Hold
    bh_shares = capital / opens[0]
    final_bh = bh_shares * closes[-1]
    ret_bh = (final_bh - capital) / capital * 100

    print("-" * 50)
    print(f"RESULTS: {signal_ticker} -> {trade_ticker}")
    print("-" * 50)
    print(f"{'Strategy':<15} | {'Return':>12} | {'Trades':>6}")
    print("-" * 40)
    print(f"{'Weekly PMax':<15} | {ret_strat:>11.2f}% | {trades:>6}")
    print(f"{'Buy & Hold':<15} | {ret_bh:>11.2f}% | {'1':>6}")
    print("-" * 50)

    winner = "PMax" if final_strat > final_bh else "Buy & Hold"
    print(f"Winner: {winner}")

def main():
    # 1. GLD -> GDXU
    run_weekly_pmax_backtest("GLD", "GDXU", years=10)

if __name__ == "__main__":
    main()
