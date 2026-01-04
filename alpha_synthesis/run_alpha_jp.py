import pandas as pd
import random
import datetime
import os
import sys
import concurrent.futures
import time
import threading
import pandas_datareader.data as web

# Ensure local imports work (current dir)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# Ensure root imports work (parent dir)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

def check_macro_environment_jp():
    """
    Checks the market regime for Japan.
    Monitors USD/JPY and US 10Y Yield as per instructions.
    """
    print("Checking Macro Environment (Japan)...")
    is_risk_on = True

    # Monitor USD/JPY and US 10Y (^TNX)
    try:
        # USD/JPY
        usdjpy = yf.Ticker("JPY=X")
        hist_usdjpy = usdjpy.history(period="1y")
        if not hist_usdjpy.empty:
            curr_usdjpy = hist_usdjpy['Close'].iloc[-1]
            ma_50_usdjpy = hist_usdjpy['Close'].rolling(50).mean().iloc[-1]
            print(f"  USD/JPY: {curr_usdjpy:.2f} (50MA: {ma_50_usdjpy:.2f})")
            # Analysis logic can be added here (e.g., rapid yen appreciation risk)

        # US 10Y Yield
        tnx = yf.Ticker("^TNX")
        hist_tnx = tnx.history(period="1y")
        if not hist_tnx.empty:
            curr_tnx = hist_tnx['Close'].iloc[-1]
            print(f"  US 10Y Yield: {curr_tnx:.2f}%")

    except Exception as e:
        print(f"  [Error] Macro check failed: {e}")

    return is_risk_on

def process_ticker(ticker, benchmark_df):
    loader = get_loader()
    try:
        hist, financials = loader.fetch_data(ticker)

        if hist is None or hist.empty:
            return None

        # --- Liquidity Filter (Japan Specific) ---
        # User requirement: > 500M JPY (approx 5 oku en)
        # Calculate Average Daily Trading Value (Price * Volume) over last 20 days
        try:
            avg_price = hist['Close'].tail(20).mean()
            avg_vol = hist['Volume'].tail(20).mean()
            avg_val = avg_price * avg_vol # Yen

            if avg_val < 500_000_000: # 5 Oku
                return None
        except Exception:
            return None

        # Analysis
        try:
            # A. Trend Template
            is_trend = indicators.check_trend_template(hist)
            if not is_trend:
                return None

            # B. Indicators (Selection Engine)

            # SMR Logic (Japan Specific Adjustment)
            # User requirement: Relax ROE, rely on relative if possible.
            # We will accept 'C' grade for Japan as "A/B equivalent" or just use the value.

            smr_val = indicators.calculate_smr_rating_value(financials)

            # Let's perform a custom check if financials exist
            jp_smr_grade = 'E'
            jp_smr_val = 0

            if financials is not None and not financials.empty:
                smr_rating_grade = indicators.get_smr_rating_grade(smr_val)
                jp_smr_grade = smr_rating_grade
                jp_smr_val = smr_val if smr_val else 0

                # Filter: Allow A, B for Japan (Strict)
                if jp_smr_grade not in ['A', 'B']:
                    return None
            else:
                # Alpha Synthesis usually requires fundamentals.
                return None

            # RS Analysis using RSCalculator (Benchmark is TOPIX)
            rs_calc = RSCalculator(hist, benchmark_df)
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
                'SMR_Rating_Value': int(jp_smr_val),
                'SMR_Rating': jp_smr_grade,
                'Raw_RS_Score': raw_rs_score, # Raw score for later ranking
                'Is_Blue_Sky': blue_sky_res['is_blue_sky'],
                'RS_Line_New_High': blue_sky_res['rs_breakout'],
                'Price_In_Base': blue_sky_res['price_in_base'],
                'Above_AVWAP': above_avwap,
                'AVWAP': avwap,
                'VCP_Tightness': vcp_details.get('tightness', False),
                'Volume_DryUp': vcp_details.get('dry_up', False),
                'Pass_Filter': True,
                'Trading_Value_JPY': avg_val
            }
            return res

        except Exception as e:
            # print(f"  Error analyzing {ticker}: {e}")
            return None

    except Exception as e:
        # print(f"Error in process_ticker for {ticker}: {e}")
        return None

def run():
    print("Starting Alpha-Synthesis System (JAPAN EDITION)...")

    # 1. Market Regime
    check_macro_environment_jp()

    # 2. Fetch Benchmark Data (TOPIX) for RS Calculation
    print("Fetching Benchmark Data (TOPIX)...")
    try:
        # Using 1306.T (Nomura TOPIX ETF) as proxy for TOPIX index data availability
        benchmark_ticker = '1306.T'
        spy_df = yf.download(benchmark_ticker, period='2y', progress=False, auto_adjust=True, multi_level_index=False)
        if spy_df is None or spy_df.empty:
            raise ValueError("Benchmark data is empty")
    except Exception as e:
        print(f"Error fetching Benchmark data: {e}. Aborting.")
        return

    # 3. Load Universe
    try:
        # Load generated stock_jp.csv
        stock_df = pd.read_csv("stock_jp.csv")
        # Remove duplicates immediately
        stock_df.drop_duplicates(subset=['Ticker'], inplace=True)
        tickers = stock_df['Ticker'].tolist()

        print(f"Selected {len(tickers)} tickers for analysis.")

    except Exception as e:
        print(f"Error reading stock_jp.csv: {e}")
        return

    # 4. Process Tickers in Parallel (Threads)
    max_workers = 12
    print(f"Using {max_workers} threads.")

    processed_count = 0
    candidate_results = [] # Store in memory first for ranking

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Pass spy_df (TOPIX) to process_ticker
        future_to_ticker = {executor.submit(process_ticker, t, spy_df): t for t in tickers}

        for future in concurrent.futures.as_completed(future_to_ticker):
            processed_count += 1
            if processed_count % 100 == 0:
                print(f"Progress: {processed_count}/{len(tickers)} tickers processed... (Candidates found: {len(candidate_results)})")

            try:
                res = future.result()
                if res:
                    candidate_results.append(res)

            except Exception as exc:
                # print(f"Generated an exception: {exc}")
                pass

    # 5. Ranking and Filtering
    print(f"\nProcessing complete. Found {len(candidate_results)} initial candidates.")

    if not candidate_results:
        print("No candidates found.")
        return

    df_candidates = pd.DataFrame(candidate_results)

    # Calculate RS Rating (Percentile)
    df_candidates['RS_Rank_Percentile'] = df_candidates['Raw_RS_Score'].rank(pct=True) * 99
    df_candidates['RS_Rating'] = df_candidates['RS_Rank_Percentile'].astype(int).clip(1, 99)

    # Strict Filter: RS Rating >= 80 (As per Alpha Synthesis)
    # Since we are ranking within candidates (survivors of Trend + Liquidity + SMR),
    # the RS 80 here is relative to "Good Stocks".
    # User requirement: "SMR A or B, RS >= 80".
    filtered_df = df_candidates[df_candidates['RS_Rating'] >= 80].copy()

    print(f"Filtered Candidates (RS >= 80): {len(filtered_df)}")

    # Remove duplicates
    filtered_df.drop_duplicates(subset=['Ticker'], inplace=True)

    # Output
    timestamp = datetime.datetime.now().strftime('%Y%m%d')
    filename = f"alpha_synthesis_jp_{timestamp}.csv"

    filtered_df = filtered_df.sort_values(by='RS_Rating', ascending=False)

    cols = ['Ticker', 'Close', 'SMR_Rating', 'SMR_Rating_Value', 'RS_Rating', 'Raw_RS_Score', 'Is_Blue_Sky',
            'RS_Line_New_High', 'Price_In_Base', 'Above_AVWAP', 'VCP_Tightness', 'Volume_DryUp', 'Trading_Value_JPY']

    filtered_df[cols].to_csv(filename, index=False)
    print(f"Final results saved to {filename}")

if __name__ == "__main__":
    run()
