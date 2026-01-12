import pandas as pd
import os
import sys
import datetime
import time
import multiprocessing
from tqdm import tqdm
import yfinance as yf

# Add current directory to path to import local modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from data_fetcher import RDTDataFetcher
from indicators import RDTIndicators

# Global variable for workers
spy_df_global = None

def init_worker(spy_df):
    """
    Initializer for worker processes to set the global SPY dataframe.
    """
    global spy_df_global
    spy_df_global = spy_df

def process_batch(batch_tickers):
    """
    Process a single batch of tickers: fetch data, calculate indicators, filter.
    Returns a list of result dictionaries.
    """
    global spy_df_global

    # Instantiate fetcher inside worker to avoid pickling session issues
    # yfinance handles session internally if not provided, or we can use curl_cffi
    # But for batch downloading, yf.download is better.
    # To avoid rate limits, we should maybe retry or sleep.

    # Random sleep to avoid thundering herd on API
    import random
    time.sleep(random.uniform(0.1, 1.0))

    fetcher = RDTDataFetcher()

    batch_results = []

    # Batch Fetch
    try:
        # Pass threads=False inside worker to avoid nesting threads in processes (which can be problematic)
        # But yf.download(..., threads=True) is fine usually.
        # Let's try fetching tickers one by one if batch fails, or use fetch_batch
        # fetch_batch uses yf.download

        batch_data = fetcher.fetch_batch(batch_tickers)
    except Exception as e:
        # print(f"Batch fetch exception: {e}")
        return []

    if not batch_data:
        return []

    for ticker in batch_tickers:
        if ticker not in batch_data:
            continue

        df = batch_data[ticker]

        if df.empty or len(df) < 50:
            continue

        try:
            # Calculate Indicators using global SPY data
            df_calc = RDTIndicators.calculate_all(df, spy_df_global)

            if df_calc.empty:
                continue

            last_row = df_calc.iloc[-1]

            # Check Filters
            check_res = RDTIndicators.check_filters(last_row)

            res_dict = {
                'Ticker': ticker,
                'Date': last_row.name.strftime('%Y-%m-%d'),
                'Close': round(last_row['Close'], 2),
                'RRS': round(last_row['RRS'], 2) if pd.notna(last_row['RRS']) else 0,
                'RVol': round(last_row['RVol'], 2) if pd.notna(last_row['RVol']) else 0,
                'ADR%': round(last_row['ADR_Percent'], 2) if pd.notna(last_row['ADR_Percent']) else 0,
                'AvgVol_10': int(last_row['Vol_SMA_10']) if pd.notna(last_row['Vol_SMA_10']) else 0,
                'All_Pass': check_res['All_Pass'],
                'RRS_Pass': check_res['RRS_Pass'],
                'RVol_Pass': check_res['RVol_Pass'],
                'ADR_Pass': check_res['ADR_Pass'],
                'Liquidity_Pass': check_res['Liquidity_Pass'],
                'Price_Pass': check_res['Price_Pass'],
                'Trend_Pass': check_res['Trend_Pass']
            }

            batch_results.append(res_dict)

        except Exception as e:
            # print(f"Error processing {ticker}: {e}")
            pass

    return batch_results

def main():
    print("Starting RDT-system Screener (Multiprocessing)...")

    # 1. Load Stock List
    stock_csv_path = os.path.join(os.path.dirname(current_dir), 'stock.csv')
    if not os.path.exists(stock_csv_path):
        print(f"Error: stock.csv not found at {stock_csv_path}")
        return

    try:
        stocks_df = pd.read_csv(stock_csv_path)
        if 'Ticker' not in stocks_df.columns:
            tickers = stocks_df.iloc[:, 0].astype(str).tolist()
        else:
            tickers = stocks_df['Ticker'].astype(str).tolist()

        tickers = [t.strip().upper() for t in tickers if isinstance(t, str) and t.strip()]
        # Filter unwanted tickers
        tickers = [t for t in tickers if not any(c in t for c in ['.', '$', '^', ' '])]
        tickers = [t for t in tickers if not (len(t) == 5 and t[-1] in ['W', 'R', 'U'])]

        print(f"Loaded {len(tickers)} tickers from stock.csv")
    except Exception as e:
        print(f"Error reading stock.csv: {e}")
        return

    # 2. Initialize Fetcher & Fetch SPY (Main Process)
    fetcher = RDTDataFetcher()
    print("Fetching SPY data...")
    # Retry logic for SPY
    spy_df = None
    for i in range(3):
        spy_df = fetcher.fetch_spy()
        if spy_df is not None and not spy_df.empty:
            break
        print("Retrying SPY fetch...")
        time.sleep(2)

    if spy_df is None or spy_df.empty:
        print("Critical Error: Could not fetch SPY data. Aborting.")
        return

    print("SPY data fetched successfully.")

    # 3. Batch Processing setup
    results_dir = os.path.join(current_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    today_str = datetime.datetime.now().strftime('%Y%m%d')
    output_filename = f"{today_str}-rdt-screener.csv"
    output_path = os.path.join(results_dir, output_filename)
    all_output_filename = f"{today_str}-rdt-all.csv"
    all_output_path = os.path.join(results_dir, all_output_filename)

    headers = [
        'Ticker', 'Date', 'Close', 'RRS', 'RVol', 'ADR%', 'AvgVol_10',
        'All_Pass', 'RRS_Pass', 'RVol_Pass', 'ADR_Pass', 'Liquidity_Pass', 'Price_Pass', 'Trend_Pass'
    ]

    # Write headers
    with open(all_output_path, 'w') as f:
        f.write(','.join(headers) + '\n')

    with open(output_path, 'w') as f:
        f.write(','.join(headers) + '\n')

    BATCH_SIZE = 50
    # Create list of batches
    batches = [tickers[i : i + BATCH_SIZE] for i in range(0, len(tickers), BATCH_SIZE)]

    # Use 2 processes to be safer with rate limits
    num_processes = 2
    print(f"Processing {len(batches)} batches using {num_processes} processes...")

    with multiprocessing.Pool(processes=num_processes, initializer=init_worker, initargs=(spy_df,)) as pool:
        # Use tqdm to show progress
        results_iterator = list(tqdm(pool.imap_unordered(process_batch, batches), total=len(batches), desc="Screening"))

    # 4. Aggregating Results
    print("Aggregating results and writing to files...")
    all_results = []
    passed_results = []

    for batch_res in results_iterator:
        if batch_res:
            all_results.extend(batch_res)
            passed_results.extend([r for r in batch_res if r['All_Pass']])

    # Sort results by Ticker
    all_results.sort(key=lambda x: x['Ticker'])
    passed_results.sort(key=lambda x: x['Ticker'])

    # Write to CSV
    if all_results:
        df_all = pd.DataFrame(all_results)
        # Ensure column order
        df_all = df_all[headers]
        df_all.to_csv(all_output_path, mode='w', index=False)

    if passed_results:
        df_passed = pd.DataFrame(passed_results)
        df_passed = df_passed[headers]
        df_passed.to_csv(output_path, mode='w', index=False)

    print(f"Screening complete.")
    print(f"Total processed: {len(all_results)}")
    print(f"Total passed: {len(passed_results)}")
    print(f"Results saved to: {output_path}")
    print(f"All data saved to: {all_output_path}")

if __name__ == "__main__":
    # Windows support for multiprocessing
    multiprocessing.freeze_support()
    main()
