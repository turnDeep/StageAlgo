import yfinance as yf
import pandas as pd
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

STOCK_CSV = 'stock.csv'
OUTPUT_FILE = 'market_breadth_history.csv'
BATCH_SIZE = 100 # yfinance batch download
MAX_WORKERS = 4 # Adjust based on CPU/Network
HISTORY_PERIOD = "5y" # 5 years is sufficient for recent analysis, or "max" if needed. User mentioned "formal implementation", often implies checking recent years. 5y is safer for speed.
# User asked for "full implementation", maybe 10-15y is better?
# Let's try 10y.
HISTORY_PERIOD = "15y"

def get_tickers():
    """Reads tickers from stock.csv"""
    try:
        df = pd.read_csv(STOCK_CSV)
        # Assuming header exists and column 0 is Ticker
        return df.iloc[:, 0].tolist()
    except Exception as e:
        print(f"Error reading {STOCK_CSV}: {e}")
        return []

def fetch_batch_data(tickers):
    """Fetches daily Low and Close for a batch of tickers."""
    try:
        # download returns a MultiIndex DataFrame if multiple tickers
        # We only need 'Low' to determine New Lows, and maybe 'Close' if we want to check if traded.
        # Actually, 'Low' is enough. If Low is present, it traded.
        data = yf.download(tickers, period=HISTORY_PERIOD, interval="1d", group_by='ticker', progress=False, threads=True)
        return data
    except Exception as e:
        print(f"Error fetching batch: {e}")
        return None

def process_market_breadth():
    tickers = get_tickers()
    if not tickers:
        print("No tickers found.")
        return

    print(f"Found {len(tickers)} tickers. Starting data fetch...")

    # We need to aggregate New Lows counts per day.
    # To do this efficiently without keeping 5000+ columns in memory for 15 years (which might be heavy but manageable ~200MB):
    # We will process batches and aggregate "Total Issues" and "New Lows" incrementally if possible.
    # But calculating "52-week Low" requires rolling window history.
    # So we MUST have the history for each ticker.
    # Strategy:
    # 1. Fetch batch.
    # 2. For each ticker in batch:
    #    a. Calculate 52-week Low (rolling min 252).
    #    b. Check if today's Low <= 52-week Low.
    #    c. Add to a global "New Lows Count" Series (by date).
    #    d. Add to a global "Total Traded Count" Series (by date).
    # 3. Discard batch data to free memory.

    # Initialize global aggregators (Series with Date index)
    # We don't know the exact date range yet, so we'll use a dictionary of date -> count and convert to DF later.
    # Or better, find the common date range first?
    # No, just accumulate in a massive Counter-like structure or align to a master index later.
    # Let's use a dictionary of dates.

    daily_new_lows = {} # Date -> Count
    daily_total_issues = {} # Date -> Count

    # Chunk tickers
    chunks = [tickers[i:i + BATCH_SIZE] for i in range(0, len(tickers), BATCH_SIZE)]

    for i, chunk in enumerate(chunks):
        print(f"Processing Batch {i+1}/{len(chunks)} ({len(chunk)} tickers)...")

        try:
            # Fetch
            data = yf.download(chunk, period=HISTORY_PERIOD, interval="1d", progress=False, threads=False)
            # Note: threads=False here because we use ThreadPool outside or just sequential batching?
            # yfinance multithreading is internal.

            if data.empty:
                continue

            # yfinance structure:
            # If multiple tickers: Columns = (Price, Ticker) MultiIndex
            # We need to swap levels or iterate.

            # Handling single ticker vs multiple ticker return structure
            if len(chunk) == 1:
                # Add ticker level manually to uniform processing
                data.columns = pd.MultiIndex.from_product([data.columns, chunk])

            # Data usually comes as (Price, Ticker) e.g. ('Adj Close', 'AAPL')
            # Check structure
            # Recent yfinance: Columns are (Price, Ticker)

            # Accessing Lows
            # We want 'Low' for all tickers.
            # Slice the MultiIndex
            try:
                lows = data['Low']
            except KeyError:
                continue

            # Rolling 52-week low (252 trading days)
            # 52-week low is min of previous 252 days usually? Or inclusive of today?
            # "New Low" usually means hitting a level lower than the last 52 weeks.
            # Typically: min(Low[t-251:t+1]) == Low[t] ?
            # Or is it min(Low[t-252:t]) > Low[t] ? (Today is lower than past year)
            # Let's assume inclusive: Low[t] == RollingMin(252)[t]

            rolling_min = lows.rolling(window=252, min_periods=50).min()

            # Check for New Lows
            # Condition: Low == RollingMin
            # AND Low must not be NaN
            is_new_low = (lows == rolling_min) & (lows.notna())

            # Count per date
            batch_nl = is_new_low.sum(axis=1)
            batch_total = lows.notna().sum(axis=1)

            # Aggregate
            for date, count in batch_nl.items():
                date_str = date.strftime('%Y-%m-%d')
                daily_new_lows[date_str] = daily_new_lows.get(date_str, 0) + count

            for date, count in batch_total.items():
                date_str = date.strftime('%Y-%m-%d')
                daily_total_issues[date_str] = daily_total_issues.get(date_str, 0) + count

        except Exception as e:
            print(f"Batch {i+1} failed: {e}")
            time.sleep(1) # Backoff

    # Convert to DataFrame
    print("Aggregating results...")
    dates = sorted(list(set(daily_total_issues.keys()) | set(daily_new_lows.keys())))

    results = []
    for d in dates:
        nl = daily_new_lows.get(d, 0)
        ti = daily_total_issues.get(d, 0)
        results.append({'Date': d, 'New_Lows': nl, 'Total_Issues': ti})

    final_df = pd.DataFrame(results)
    final_df['Date'] = pd.to_datetime(final_df['Date'])
    final_df.set_index('Date', inplace=True)
    final_df.sort_index(inplace=True)

    # Calculate Ratio
    final_df['New_Lows_Ratio'] = (final_df['New_Lows'] / final_df['Total_Issues']) * 100

    # Save
    final_df.to_csv(OUTPUT_FILE)
    print(f"Saved market breadth data to {OUTPUT_FILE}")
    print(final_df.tail())

if __name__ == "__main__":
    process_market_breadth()
