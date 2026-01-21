import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import sys
import os
import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
import traceback
import time

# Ensure we can import data_loader
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from data_loader import AlphaSynthesisDataLoader
except ImportError:
    # Fallback if running from root and alpha_synthesis is not a package
    sys.path.append(os.path.join(os.getcwd(), 'alpha_synthesis'))
    from data_loader import AlphaSynthesisDataLoader

def calculate_zigzag(df, order=5):
    """
    Calculates ZigZag pivots.
    Returns a list of dictionaries: {'idx': int, 'date': datetime, 'price': float, 'type': 'high'/'low'}
    """
    if df is None or df.empty:
        return []

    highs = df['High'].values
    lows = df['Low'].values
    dates = df.index

    # Find indexes of local extrema
    high_idxs = argrelextrema(highs, np.greater, order=order)[0]
    low_idxs = argrelextrema(lows, np.less, order=order)[0]

    candidates = []
    for idx in high_idxs:
        candidates.append({'idx': idx, 'date': dates[idx], 'price': highs[idx], 'type': 'high'})
    for idx in low_idxs:
        candidates.append({'idx': idx, 'date': dates[idx], 'price': lows[idx], 'type': 'low'})

    candidates.sort(key=lambda x: x['idx'])

    if not candidates:
        return []

    # Filter for Alternating High/Low
    stack = [candidates[0]]

    for current in candidates[1:]:
        last = stack[-1]

        if last['type'] == current['type']:
            if last['type'] == 'high':
                if current['price'] > last['price']:
                    stack.pop()
                    stack.append(current)
            else:
                if current['price'] < last['price']:
                    stack.pop()
                    stack.append(current)
        else:
            stack.append(current)

    return stack

def find_vcp_patterns(df, pivots):
    """
    Identifies VCP patterns based on declining depth contractions.
    Returns list of patterns found.
    """
    results = []

    # We need sequences of High -> Low waves.
    # Pivots are alternating High/Low.
    # A wave is High[i] -> Low[i+1].

    # Extract all High -> Low waves
    waves = []
    for i in range(len(pivots) - 1):
        p1 = pivots[i]
        p2 = pivots[i+1]

        if p1['type'] == 'high' and p2['type'] == 'low':
            depth = (p1['price'] - p2['price']) / p1['price'] # Depth as positive percentage
            wave = {
                'start_date': p1['date'],
                'end_date': p2['date'],
                'start_price': p1['price'],
                'end_price': p2['price'],
                'depth': depth,
                'pivot_idx_start': i,
                'pivot_idx_end': i+1
            }
            waves.append(wave)

    # Iterate through waves to find sequences
    # We need strictly decreasing depths.

    # Check for 3 Contractions (4 waves): d0 > d1 > d2 > d3
    # Check for 2 Contractions (3 waves): d0 > d1 > d2

    n_waves = len(waves)

    # We iterate ending at each wave 'i'
    for i in range(n_waves):
        # We need at least 3 waves ending at i for 2 contractions
        # Wave i is the last one (smallest)

        # Check for 3 Contractions (needs 4 waves: i-3, i-2, i-1, i)
        if i >= 3:
            w0 = waves[i-3]
            w1 = waves[i-2]
            w2 = waves[i-1]
            w3 = waves[i]

            # Check they are consecutive in pivots?
            # Yes, waves list is built from consecutive pivots, but we must ensure
            # there wasn't a gap (e.g. Low->High->Low... if we skipped something, but our loop above captures all H->L).
            # However, zigzag enforces alternation. H-L-H-L-H-L.
            # So waves[i] corresponds to H-L. waves[i+1] corresponds to next H-L.
            # Between waves[i].end (Low) and waves[i+1].start (High), there is a rally.
            # This is correct for VCP.

            if w0['depth'] > w1['depth'] and w1['depth'] > w2['depth'] and w2['depth'] > w3['depth']:
                # Found 3 contractions (4 waves)
                pattern = {
                    'type': '3_Contractions', # 4 waves
                    'start_date': w0['start_date'],
                    'end_date': w3['end_date'],
                    'depths': [w0['depth'], w1['depth'], w2['depth'], w3['depth']]
                }
                results.append(pattern)
                continue # Skip checking 2 contractions ending here if we found 3 (preference for longer)

        # Check for 2 Contractions (needs 3 waves: i-2, i-1, i)
        if i >= 2:
            w0 = waves[i-2]
            w1 = waves[i-1]
            w2 = waves[i]

            if w0['depth'] > w1['depth'] and w1['depth'] > w2['depth']:
                # Found 2 contractions (3 waves)
                pattern = {
                    'type': '2_Contractions', # 3 waves
                    'start_date': w0['start_date'],
                    'end_date': w2['end_date'],
                    'depths': [w0['depth'], w1['depth'], w2['depth']]
                }
                results.append(pattern)

    return results

def process_ticker(ticker):
    loader = AlphaSynthesisDataLoader()
    try:
        # Fetch 2 years to be safe, we will slice to 1 year for detection if needed
        # User said "Use 1 year of data".
        # But VCP setup might have started > 1 year ago?
        # Usually "Use 1 year data" means look for patterns in the last year.
        # I'll fetch 2y (as per loader default) and rely on the timestamps for reporting.
        # If the pattern *ends* in the last year, it's relevant.

        df, _ = loader.fetch_data(ticker)

        if df is None or df.empty:
            return []

        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Filter for last 1 year for the *Analysis Scope*?
        # Or calculate on everything and filter results?
        # "Use 1 year of data" -> I'll slice df to last 365 days + buffer for first wave start.
        # ZigZag needs context. If I slice strictly 1 year, the first point is a start.
        # Safest is to use the fetched data (up to 2y) but filter results where End Date is within last 1 year.

        # Calculate ZigZag
        pivots = calculate_zigzag(df, order=5)

        patterns = find_vcp_patterns(df, pivots)

        # Filter: Keep patterns where end_date is within last 365 days
        one_year_ago = df.index[-1] - timedelta(days=365)

        valid_patterns = []
        for p in patterns:
            if p['end_date'] >= one_year_ago:
                p['ticker'] = ticker
                valid_patterns.append(p)

        return valid_patterns

    except Exception as e:
        # print(f"Error processing {ticker}: {e}")
        return []
    finally:
        loader.close()

def main():
    parser = argparse.ArgumentParser(description="VCP Screener")
    parser.add_argument("--file", type=str, default="stock.csv", help="Input CSV file with Tickers")
    parser.add_argument("--output", type=str, default="alpha_synthesis/vcp_patterns_found.csv", help="Output CSV file")
    parser.add_argument("--workers", type=int, default=8, help="Number of workers")
    args = parser.parse_args()

    # Read tickers
    tickers = []
    try:
        with open(args.file, 'r') as f:
            reader = csv.reader(f)
            header = next(reader, None) # Skip header if Ticker,Exchange
            # Check if header is actually header or data
            if header:
                if header[0].upper() == "TICKER":
                    pass # It was header
                else:
                    tickers.append(header[0]) # It was data

            for row in reader:
                if row:
                    tickers.append(row[0])
    except FileNotFoundError:
        print(f"File {args.file} not found.")
        return

    print(f"Loaded {len(tickers)} tickers. Starting analysis with {args.workers} workers...")

    results = []

    # Process in batches or just map
    # Using as_completed to write incrementally if needed, or just collect all.
    # Collect all is fine for CSV writing at end, but incremental is safer for crashes.

    with open(args.output, 'w', newline='') as f_out:
        writer = csv.writer(f_out)
        writer.writerow(['Ticker', 'Pattern_Type', 'Start_Date', 'End_Date', 'Depths_Percent'])

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_ticker = {executor.submit(process_ticker, t): t for t in tickers}

            count = 0
            found_count = 0

            for future in as_completed(future_to_ticker):
                t = future_to_ticker[future]
                try:
                    patterns = future.result()
                    if patterns:
                        for p in patterns:
                            # Format depths as percentage string
                            depths_str = ", ".join([f"{d*100:.2f}%" for d in p['depths']])
                            writer.writerow([
                                p['ticker'],
                                p['type'],
                                p['start_date'].strftime('%Y-%m-%d'),
                                p['end_date'].strftime('%Y-%m-%d'),
                                depths_str
                            ])
                            f_out.flush()
                        found_count += len(patterns)
                except Exception as exc:
                    print(f"{t} generated an exception: {exc}")

                count += 1
                if count % 100 == 0:
                    print(f"Processed {count}/{len(tickers)} tickers. Found {found_count} patterns so far.")

    print(f"Done. Processed {count} tickers. Total patterns found: {found_count}. Saved to {args.output}")

if __name__ == "__main__":
    main()
