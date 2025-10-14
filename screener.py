"""
Stage Screener

This script screens stocks from stock.csv and categorizes them into Stage 1 (sub-stages 1E, 1F) and Stage 2
based on specific criteria including scoring grade and overheat levels.
The results are saved into separate CSV and TradingView-compatible text files.
"""
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import warnings

from data_fetcher import fetch_stock_data
from indicators import calculate_all_basic_indicators
from stage_detector import StageDetector
from scoring_system import ScoringSystem

warnings.filterwarnings('ignore')

def analyze_ticker(args):
    """
    Analyzes a single ticker and returns its stage information and other metrics.
    """
    ticker, exchange, benchmark_df_dict = args
    try:
        stock_df, _ = fetch_stock_data(ticker, period='2y')
        if stock_df is None or len(stock_df) < 252:
            return None

        indicator_df = calculate_all_basic_indicators(stock_df)
        if indicator_df.empty or len(indicator_df) < 252:
            return None
        
        indicator_df = indicator_df.dropna()
        if len(indicator_df) < 252:
            return None

        stage_detector = StageDetector(indicator_df)
        stage, sub_stage = stage_detector.determine_stage()

        benchmark_df = pd.DataFrame(benchmark_df_dict)
        benchmark_df.index = pd.to_datetime(benchmark_df.index)

        scorer = ScoringSystem(indicator_df, ticker, benchmark_df)
        result = scorer.comprehensive_analysis()

        score = result.get('total_score', 0)
        judgement = result.get('grade', '')
        overheat = result.get('details', {}).get('atr', {}).get('atr_multiple_ma50', 0)

        return {
            'Ticker': ticker,
            'Exchange': exchange,
            'Stage': stage,
            'Sub Stage': sub_stage,
            'Score': score,
            'Judgement': judgement,
            'Overheat': overheat,
            'Price': stock_df['Close'].iloc[-1],
            'Volume': stock_df['Volume'].iloc[-1]
        }
    except Exception as e:
        # print(f"Could not analyze {ticker}: {e}")
        return None
    return None

def main():
    """
    Main function to run the screener.
    """
    print("Starting stock screener with new filtering conditions...")

    try:
        stock_list_df = pd.read_csv('stock.csv')
        stock_list_df.dropna(subset=['Ticker'], inplace=True)
        tickers = [(row['Ticker'], row['Exchange']) for index, row in stock_list_df.iterrows()]
    except FileNotFoundError:
        print("Error: stock.csv not found. Please run get_tickers.py first.")
        return

    print(f"Loaded {len(tickers)} tickers for analysis.")

    # Fetch benchmark data (SPY)
    print("Fetching benchmark data (SPY)...")
    _, benchmark_df = fetch_stock_data('SPY', period='2y')
    if benchmark_df is None or benchmark_df.empty:
        print("Fatal Error: Could not fetch benchmark data.")
        return
    benchmark_df = calculate_all_basic_indicators(benchmark_df)
    benchmark_df = benchmark_df.dropna()
    benchmark_df_dict = {
        'index': benchmark_df.index.astype(str).tolist(),
        **{col: benchmark_df[col].tolist() for col in benchmark_df.columns}
    }
    print("Benchmark data fetched successfully.")


    results = []
    args_list = [(ticker, exchange, benchmark_df_dict) for ticker, exchange in tickers]

    # Use multiprocessing to speed up the analysis
    with Pool(cpu_count()) as pool:
        for result in tqdm(pool.imap_unordered(analyze_ticker, args_list), total=len(tickers), desc="Analyzing Stocks"):
            if result:
                results.append(result)

    if not results:
        print("No stocks passed the initial analysis.")
        return

    results_df = pd.DataFrame(results)

    # Apply new filtering criteria
    # Stage 1: Sub Stage in ['1E', '1F'] AND Overheat >= 2
    stage1_df = results_df[
        (results_df['Stage'] == 'Stage 1') &
        (results_df['Sub Stage'].isin(['1E', '1F'])) &
        (results_df['Overheat'] >= 2)
    ].copy()

    # Stage 2: Judgement == 'A+' AND Overheat >= 2
    stage2_df = results_df[
        (results_df['Stage'] == 'Stage 2') &
        (results_df['Judgement'] == 'A+') &
        (results_df['Overheat'] >= 2)
    ].copy()

    # Save Stage 1 results
    if not stage1_df.empty:
        stage1_df.sort_values(by='Score', ascending=False, inplace=True)
        stage1_df.to_csv('stage1.csv', index=False)
        print(f"Saved {len(stage1_df)} Stage 1 stocks to stage1.csv based on new criteria.")

        stage1_tv_list = [f"{row['Exchange']}:{row['Ticker']}" for index, row in stage1_df.iterrows()]
        with open('stage1_tradingview.txt', 'w') as f:
            f.write(','.join(stage1_tv_list))
        print("Saved TradingView list for Stage 1 stocks to stage1_tradingview.txt")
    else:
        print("No Stage 1 stocks found matching the new criteria (Sub-stage 1E/1F and Overheat >= 2).")

    # Save Stage 2 results
    if not stage2_df.empty:
        stage2_df.sort_values(by='Score', ascending=False, inplace=True)
        stage2_df.to_csv('stage2.csv', index=False)
        print(f"Saved {len(stage2_df)} Stage 2 stocks to stage2.csv based on new criteria.")

        stage2_tv_list = [f"{row['Exchange']}:{row['Ticker']}" for index, row in stage2_df.iterrows()]
        with open('stage2_tradingview.txt', 'w') as f:
            f.write(','.join(stage2_tv_list))
        print("Saved TradingView list for Stage 2 stocks to stage2_tradingview.txt")
    else:
        print("No Stage 2 stocks found matching the new criteria (Grade A+ and Overheat >= 2).")

    print("Screening complete.")

if __name__ == "__main__":
    main()