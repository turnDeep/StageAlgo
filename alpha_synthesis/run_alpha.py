import pandas as pd
import random
import datetime
import os
import sys
import concurrent.futures
import time
import threading

# Ensure local imports work (current dir)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# Ensure root imports work (parent dir)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from market_regime import check_macro_environment
from data_loader import AlphaSynthesisDataLoader
import indicators
from rs_calculator import RSCalculator
import yfinance as yf

# Thread-local storage for loader
thread_local = threading.local()

def get_loader():
    if not hasattr(thread_local, "loader"):
        thread_local.loader = AlphaSynthesisDataLoader()
    return thread_local.loader

def process_ticker(ticker, spy_df):
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

            # Ensure SMR Value is Int
            smr_rating_value = int(smr_rating_value)

            # RS Analysis using RSCalculator
            rs_calc = RSCalculator(hist, spy_df)
            raw_rs_score = rs_calc.calculate_ibd_rs_score().iloc[-1]

            # Blue Sky Detection
            blue_sky_res = rs_calc.detect_blue_sky_setup()

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
                'Raw_RS_Score': raw_rs_score, # Raw score for later ranking
                'Is_Blue_Sky': blue_sky_res['is_blue_sky'],
                'RS_Line_New_High': blue_sky_res['rs_breakout'],
                'Price_In_Base': blue_sky_res['price_in_base'],
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
    print("Starting Alpha-Synthesis System (FULL RUN - SMR Rating A/B & Strict RS Filter)...")

    # 1. Market Regime
    is_risk_on = check_macro_environment()
    if not is_risk_on:
        print("Market is Risk-OFF. Proceeding with caution (validation mode).")
    else:
        print("Market is Risk-ON. Good for Longs.")

    # 2. Fetch Benchmark Data (SPY) for RS Calculation
    print("Fetching Benchmark Data (SPY)...")
    try:
        spy_df = yf.download('SPY', period='2y', progress=False, auto_adjust=True, multi_level_index=False)
        if spy_df is None or spy_df.empty:
            raise ValueError("SPY data is empty")
    except Exception as e:
        print(f"Error fetching SPY data: {e}. Aborting.")
        return

    # 3. Load Universe
    try:
        stock_df = pd.read_csv("stock.csv")
        # Remove duplicates immediately
        stock_df.drop_duplicates(subset=['Ticker'], inplace=True)
        all_tickers = stock_df['Ticker'].tolist()
        all_tickers = [str(t).replace('.', '-') for t in all_tickers]

        tickers = all_tickers
        print(f"Selected ALL {len(tickers)} tickers for analysis (Unique).")

    except Exception as e:
        print(f"Error reading stock.csv: {e}")
        return

    # 4. Process Tickers in Parallel (Threads)
    max_workers = 12
    print(f"Using {max_workers} threads.")

    processed_count = 0
    candidate_results = [] # Store in memory first for ranking

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Pass spy_df to process_ticker
        future_to_ticker = {executor.submit(process_ticker, t, spy_df): t for t in tickers}

        for future in concurrent.futures.as_completed(future_to_ticker):
            processed_count += 1
            if processed_count % 100 == 0:
                print(f"Progress: {processed_count}/{len(tickers)} tickers processed... (Candidates found: {len(candidate_results)})")

            try:
                res = future.result()
                if res:
                    candidate_results.append(res)
                    # print(f"  [CANDIDATE] {res['Ticker']} (SMR: {res['SMR_Rating']}, RawRS: {res['Raw_RS_Score']:.1f})")

            except Exception as exc:
                print(f"Generated an exception: {exc}")

    # 5. Ranking and Filtering
    print(f"\nProcessing complete. Found {len(candidate_results)} initial candidates (Trend + SMR A/B).")

    if not candidate_results:
        print("No candidates found.")
        return

    df_candidates = pd.DataFrame(candidate_results)

    # Calculate RS Rating (Percentile)
    # Rank 0-99 based on Raw_RS_Score within the high-quality subset
    # Note: Ranking within the subset of SMR A/B stocks is stricter than the general market.
    # An RS 80 here is very strong.
    df_candidates['RS_Rank_Percentile'] = df_candidates['Raw_RS_Score'].rank(pct=True) * 99
    df_candidates['RS_Rating'] = df_candidates['RS_Rank_Percentile'].astype(int).clip(1, 99)

    # Strict Filter: RS Rating >= 80
    filtered_df = df_candidates[df_candidates['RS_Rating'] >= 80].copy()

    print(f"Filtered Candidates (RS >= 80): {len(filtered_df)}")

    # Remove duplicates
    filtered_df.drop_duplicates(subset=['Ticker'], inplace=True)

    # Output
    timestamp = datetime.datetime.now().strftime('%Y%m%d')
    filename = f"alpha_synthesis_results_SMR_AB_RS80_{timestamp}.csv"

    filtered_df = filtered_df.sort_values(by='RS_Rating', ascending=False)

    # Select columns
    # User requested integer values. SMR_Rating is Grade (A/B), SMR_Rating_Value is Int (0-100).
    # Providing both for clarity.
    cols = ['Ticker', 'Close', 'SMR_Rating', 'SMR_Rating_Value', 'RS_Rating', 'Raw_RS_Score', 'Is_Blue_Sky',
            'RS_Line_New_High', 'Price_In_Base', 'Above_AVWAP', 'VCP_Tightness', 'Volume_DryUp']

    filtered_df[cols].to_csv(filename, index=False)
    print(f"Final results saved to {filename}")

    if not filtered_df.empty:
        print("\nTop 20 Candidates:")
        print(filtered_df[cols].head(20))

        blue_sky_count = filtered_df['Is_Blue_Sky'].sum()
        print(f"\nBlue Sky Setups Found: {blue_sky_count}")
        if blue_sky_count > 0:
            print(filtered_df[filtered_df['Is_Blue_Sky'] == True][cols])

if __name__ == "__main__":
    run()
