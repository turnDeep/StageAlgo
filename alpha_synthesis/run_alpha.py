import pandas as pd
import random
import datetime
import os
import sys

# Ensure local imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from market_regime import check_macro_environment
from data_loader import AlphaSynthesisDataLoader
import indicators

def run():
    print("Starting Alpha-Synthesis System...")

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

        # Filter out bad tickers (containing dots or weird chars if any)
        # yfinance expects 'BRK-B' for 'BRK.B'
        all_tickers = [str(t).replace('.', '-') for t in all_tickers]

        # Sample 500
        sample_size = 500
        if len(all_tickers) > sample_size:
            tickers = random.sample(all_tickers, sample_size)
        else:
            tickers = all_tickers

        print(f"Selected {len(tickers)} tickers for analysis.")

    except Exception as e:
        print(f"Error reading stock.csv: {e}")
        return

    # 3. Process Tickers
    loader = AlphaSynthesisDataLoader()
    results = []

    try:
        for i, ticker in enumerate(tickers):
            print(f"Processing {i+1}/{len(tickers)}: {ticker}")

            hist, financials = loader.fetch_data(ticker)

            if hist is None or hist.empty:
                print(f"  No history for {ticker}")
                continue

            # Analysis
            try:
                # A. Trend Template
                is_trend = indicators.check_trend_template(hist)
                if not is_trend:
                    # print("  Failed Trend Template")
                    continue

                print("  Passed Trend Template!")

                # B. Indicators
                smr = indicators.calculate_smr(financials)
                rs = indicators.calculate_rs(hist)
                avwap = indicators.calculate_anchored_vwap(hist)
                is_vcp, vcp_details = indicators.check_vcp(hist)

                last_close = hist['Close'].iloc[-1]

                # C. Trigger Logic (Simplified for Screener)
                # Check if price > AVWAP
                above_avwap = False
                if avwap is not None:
                    above_avwap = last_close > avwap

                # Store Result
                res = {
                    'Ticker': ticker,
                    'Close': last_close,
                    'SMR_Score': smr if smr else 0,
                    'RS_Score': rs,
                    'Above_AVWAP': above_avwap,
                    'AVWAP': avwap,
                    'VCP_Tightness': vcp_details.get('tightness', False),
                    'Volume_DryUp': vcp_details.get('dry_up', False),
                    'Pass_Filter': True
                }
                results.append(res)
                print(f"  -> Added. SMR:{res['SMR_Score']:.1f}, RS:{res['RS_Score']:.1f}, VCP:{is_vcp}")

            except Exception as e:
                print(f"  Error analyzing {ticker}: {e}")
                continue
    except KeyboardInterrupt:
        print("Interrupted by user. Saving partial results...")
    finally:
        loader.close()

    # 4. Output
    if results:
        df_res = pd.DataFrame(results)
        # Sort by RS Score descending
        df_res = df_res.sort_values(by='RS_Score', ascending=False)

        filename = f"alpha_synthesis_results_{datetime.datetime.now().strftime('%Y%m%d')}.csv"
        df_res.to_csv(filename, index=False)
        print(f"\nResults saved to {filename}")
        print(df_res[['Ticker', 'Close', 'RS_Score', 'SMR_Score', 'Above_AVWAP', 'VCP_Tightness']].head(10))
    else:
        print("\nNo candidates found matching criteria.")

if __name__ == "__main__":
    run()
