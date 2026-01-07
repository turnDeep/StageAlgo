import pandas as pd
import os
import sys
import datetime
import time
from tqdm import tqdm

# Add current directory to path to import local modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from data_fetcher import RDTDataFetcher
from indicators import RDTIndicators

def main():
    print("Starting RDT-system Screener...")

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
        # Additional filter for 5-letter tickers ending in special codes (common in rights/warrants)
        # Assuming simple 1-4 letter tickers or 5 if valid.
        # But memory says: "exclude tickers that are 5 characters long, contain .U, .W, .A, .B, .R"
        # My clean above removes '.', so this covers it mostly.
        # But let's be safe.
        tickers = [t for t in tickers if not (len(t) == 5 and t[-1] in ['W', 'R', 'U'])]

        print(f"Loaded {len(tickers)} tickers from stock.csv")
    except Exception as e:
        print(f"Error reading stock.csv: {e}")
        return

    # 2. Initialize Fetcher & Fetch SPY
    fetcher = RDTDataFetcher()
    print("Fetching SPY data...")
    spy_df = fetcher.fetch_spy()
    if spy_df is None or spy_df.empty:
        print("Critical Error: Could not fetch SPY data. Aborting.")
        return

    print("SPY data fetched successfully.")

    # 3. Batch Processing
    # Create results directory
    results_dir = os.path.join(current_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    today_str = datetime.datetime.now().strftime('%Y%m%d')
    output_filename = f"{today_str}-rdt-screener.csv"
    output_path = os.path.join(results_dir, output_filename)
    all_output_filename = f"{today_str}-rdt-all.csv"
    all_output_path = os.path.join(results_dir, all_output_filename)

    # Initialize CSVs with headers
    headers = [
        'Ticker', 'Date', 'Close', 'RRS', 'RVol', 'ADR%', 'AvgVol_10',
        'All_Pass', 'RRS_Pass', 'RVol_Pass', 'ADR_Pass', 'Liquidity_Pass', 'Price_Pass', 'Trend_Pass'
    ]

    # Write headers if files don't exist
    if not os.path.exists(all_output_path):
        with open(all_output_path, 'w') as f:
            f.write(','.join(headers) + '\n')

    if not os.path.exists(output_path):
        with open(output_path, 'w') as f:
            f.write(','.join(headers) + '\n')

    BATCH_SIZE = 50
    total_tickers = len(tickers)

    # Loop
    for i in tqdm(range(0, total_tickers, BATCH_SIZE), desc="Processing Batches"):
        batch_tickers = tickers[i : i + BATCH_SIZE]

        # Batch Fetch
        try:
            batch_data = fetcher.fetch_batch(batch_tickers)
        except Exception as e:
            # print(f"Batch fetch exception: {e}")
            continue

        batch_results = []
        passed_results = []

        for ticker in batch_tickers:
            if ticker not in batch_data:
                continue

            df = batch_data[ticker]

            if df.empty or len(df) < 50:
                continue

            try:
                # Calculate Indicators
                df_calc = RDTIndicators.calculate_all(df, spy_df)

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
                if check_res['All_Pass']:
                    passed_results.append(res_dict)

            except Exception as e:
                # print(f"Error processing {ticker}: {e}")
                pass

        # Append to CSVs immediately
        if batch_results:
            df_batch = pd.DataFrame(batch_results)
            df_batch.to_csv(all_output_path, mode='a', header=False, index=False)

        if passed_results:
            df_passed = pd.DataFrame(passed_results)
            df_passed.to_csv(output_path, mode='a', header=False, index=False)

        # Reduced sleep to speed up, assuming yf.download handles pacing reasonably well internally or we rely on retries
        time.sleep(0.5)

    print(f"Screening complete.")
    print(f"Results saved to: {output_path}")
    print(f"All data saved to: {all_output_path}")

if __name__ == "__main__":
    main()
