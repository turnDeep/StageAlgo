import pandas as pd
import random
import datetime
import os
import sys
import concurrent.futures
import time
import threading

# Ensure local imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from market_regime import check_macro_environment
from data_loader import AlphaSynthesisDataLoader
import indicators

# Thread-local storage for loader
thread_local = threading.local()

def get_loader():
    if not hasattr(thread_local, "loader"):
        thread_local.loader = AlphaSynthesisDataLoader()
    return thread_local.loader

def process_ticker(ticker):
    loader = get_loader()
    try:
        hist, financials = loader.fetch_data(ticker)

        if hist is None or hist.empty:
            return None

        # Analysis
        try:
            # A. Trend Template
            is_trend = indicators.check_trend_template(hist)
            if not is_trend:
                return None

            # B. Indicators (Selection Engine)
            # SMR Rating
            smr_val = indicators.calculate_smr_rating_value(financials)
            smr_rating_value = smr_val if smr_val else 0
            smr_rating = indicators.get_smr_rating_grade(smr_val)

            # SMR Filter: A or B only
            if smr_rating not in ['A', 'B']:
                return None

            # RS
            rs = indicators.calculate_rs(hist)

            # Structure Analysis (Phase 3)
            avwap = indicators.calculate_anchored_vwap(hist)
            is_vcp, vcp_details = indicators.check_vcp(hist)

            last_close = hist['Close'].iloc[-1]

            # C. Trigger Logic
            above_avwap = False
            if avwap is not None:
                above_avwap = last_close > avwap

            # Store Result
            res = {
                'Ticker': ticker,
                'Close': last_close,
                'SMR_Rating_Value': smr_rating_value,
                'SMR_Rating': smr_rating,
                'RS_Score': rs,
                'Above_AVWAP': above_avwap,
                'AVWAP': avwap,
                'VCP_Tightness': vcp_details.get('tightness', False),
                'Volume_DryUp': vcp_details.get('dry_up', False),
                'Pass_Filter': True
            }
            return res

        except Exception as e:
            # print(f"  Error analyzing {ticker}: {e}")
            return None

    except Exception as e:
        # print(f"Error in process_ticker for {ticker}: {e}")
        return None

def run():
    print("Starting Alpha-Synthesis System (FULL RUN - SMR Rating A/B Filter)...")

    # 1. Market Regime
    is_risk_on = check_macro_environment()
    if not is_risk_on:
        print("Market is Risk-OFF. Proceeding with caution (validation mode).")
    else:
        print("Market is Risk-ON. Good for Longs.")

    # 2. Load Universe
    try:
        stock_df = pd.read_csv("stock.csv")
        all_tickers = stock_df['Ticker'].tolist()
        all_tickers = [str(t).replace('.', '-') for t in all_tickers]

        tickers = all_tickers
        print(f"Selected ALL {len(tickers)} tickers for analysis.")

    except Exception as e:
        print(f"Error reading stock.csv: {e}")
        return

    # 3. Process Tickers in Parallel (Threads)
    # Using 12 threads for better throughput
    max_workers = 12
    print(f"Using {max_workers} threads.")

    processed_count = 0
    found_count = 0

    timestamp = datetime.datetime.now().strftime('%Y%m%d')
    filename = f"alpha_synthesis_results_SMR_AB_{timestamp}.csv"

    # Write header if new file
    if not os.path.exists(filename):
        pd.DataFrame(columns=['Ticker', 'Close', 'SMR_Rating_Value', 'SMR_Rating', 'RS_Score', 'Above_AVWAP', 'AVWAP', 'VCP_Tightness', 'Volume_DryUp', 'Pass_Filter']).to_csv(filename, index=False)

    print(f"Streaming results to {filename}...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ticker = {executor.submit(process_ticker, t): t for t in tickers}

        for future in concurrent.futures.as_completed(future_to_ticker):
            processed_count += 1
            if processed_count % 100 == 0:
                print(f"Progress: {processed_count}/{len(tickers)} tickers processed... (Found: {found_count})")

            try:
                res = future.result()
                if res:
                    found_count += 1
                    # Append to CSV immediately
                    df_single = pd.DataFrame([res])
                    df_single.to_csv(filename, mode='a', header=False, index=False)
                    print(f"  [FOUND] {res['Ticker']} (SMR: {res['SMR_Rating']}, RS: {res['RS_Score']:.1f})")

            except Exception as exc:
                print(f"Generated an exception: {exc}")

    # 4. Final Sort and Cleanup
    print("\nRun complete. Finalizing results...")
    if os.path.exists(filename):
        try:
            df_final = pd.read_csv(filename)
            df_final = df_final.drop_duplicates(subset=['Ticker'])
            df_final = df_final.sort_values(by='RS_Score', ascending=False)
            df_final.to_csv(filename, index=False)

            print(f"Final results saved to {filename}")
            print(f"Total Candidates Found (SMR Rating A/B): {len(df_final)}")
            print(df_final[['Ticker', 'SMR_Rating', 'RS_Score', 'Above_AVWAP', 'VCP_Tightness']].head(20))
        except Exception as e:
            print(f"Error finalizing CSV: {e}")
    else:
        print("No results file found.")

if __name__ == "__main__":
    run()
