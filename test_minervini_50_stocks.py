"""
Test Minervini Template Detector with 50 target stocks
Minervini Trend Template + Green Candle + $1B+ Market Cap条件の検証
"""
import pandas as pd
import numpy as np
from data_fetcher import fetch_stock_data
from indicators import calculate_all_basic_indicators
from rs_calculator import RSCalculator
from minervini_template_detector import MinerviniTemplateDetector
import yfinance as yf
import time

# 50銘柄のリスト
TARGET_TICKERS = [
    'TNK', 'CYTK', 'PSTG', 'WCC', 'TTMI', 'MIR', 'KRYS', 'FSLR', 'SYM', 'NET',
    'SID', 'W', 'FN', 'PTGX', 'IMNM', 'MNMD', 'OSIS', 'ADPT', 'QS', 'KYMR',
    'LITE', 'CDTX', 'MDB', 'IDYA', 'PLTR', 'VSAT', 'CRDO', 'CNTA', 'ZBIO', 'NKTR',
    'NBIS', 'TGB', 'ACLX', 'WFRD', 'COGT', 'WULF', 'HUT', 'DQ', 'HOOD', 'RYTM',
    'SQM', 'TSLA', 'QCOM', 'NTGR', 'INCY', 'XPO', 'VSH', 'LASR', 'WGS', 'IVZ'
]

def get_market_cap(ticker: str) -> float:
    """
    yfinanceを使って時価総額を取得

    Args:
        ticker: ティッカーシンボル

    Returns:
        float: 時価総額（ドル）、取得できない場合はNone
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        market_cap = info.get('marketCap')
        return market_cap
    except Exception as e:
        print(f"  Warning: Could not get market cap for {ticker}: {e}")
        return None


def analyze_stocks_with_criteria(tickers: list, min_rs_rating: int = 70, verbose: bool = True):
    """
    銘柄リストを分析し、条件を満たすかチェック

    Args:
        tickers: ティッカーシンボルのリスト
        min_rs_rating: 最小RS Rating
        verbose: 詳細出力するか

    Returns:
        pd.DataFrame: 分析結果
    """
    print(f"\n{'='*80}")
    print(f"Minervini Enhanced Template Analysis (RS Rating >= {min_rs_rating})")
    print(f"{'='*80}")
    print(f"Target: {len(tickers)} stocks\n")

    # ベンチマーク（SPY）データ取得
    print("Loading benchmark data (SPY)...")
    _, benchmark_df = fetch_stock_data('SPY', period='2y')
    if benchmark_df is not None:
        benchmark_df = calculate_all_basic_indicators(benchmark_df)
        print("✓ Benchmark data loaded\n")
    else:
        print("✗ Failed to load benchmark data\n")
        return None

    results = []

    for i, ticker in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}] Analyzing {ticker}...")

        try:
            # 株価データ取得
            stock_df, _ = fetch_stock_data(ticker, period='2y')

            if stock_df is None or len(stock_df) < 200:
                print(f"  ✗ Insufficient data for {ticker}")
                results.append({
                    'Ticker': ticker,
                    'Status': 'No Data',
                    'All_Criteria_Met': False,
                    'Base_Criteria_Met': 0,
                    'Additional_Criteria_Met': 0,
                    'RS_Rating': None,
                    'Market_Cap_B': None,
                    'Green_Candle': None,
                    'Score': 0
                })
                continue

            # 指標計算
            indicators_df = calculate_all_basic_indicators(stock_df)
            indicators_df = indicators_df.dropna()

            # RS Rating計算
            if len(indicators_df) >= 252:
                rs_calc = RSCalculator(indicators_df, benchmark_df)
                rs_score_series = rs_calc.calculate_ibd_rs_score()
                rs_rating_value = rs_calc.calculate_percentile_rating(rs_score_series.iloc[-1])
                indicators_df['RS_Rating'] = rs_rating_value
            else:
                indicators_df['RS_Rating'] = 0

            if len(indicators_df) < 200:
                print(f"  ✗ Insufficient data after calculation for {ticker}")
                results.append({
                    'Ticker': ticker,
                    'Status': 'Insufficient Data',
                    'All_Criteria_Met': False,
                    'Base_Criteria_Met': 0,
                    'Additional_Criteria_Met': 0,
                    'RS_Rating': None,
                    'Market_Cap_B': None,
                    'Green_Candle': None,
                    'Score': 0
                })
                continue

            # 時価総額取得
            market_cap = get_market_cap(ticker)

            # Minervini Template Detector
            detector = MinerviniTemplateDetector(indicators_df, market_cap=market_cap)
            result = detector.check_template_with_additional_criteria(min_rs_rating=min_rs_rating)

            # 結果を集計
            rs_rating = indicators_df['RS_Rating'].iloc[-1] if 'RS_Rating' in indicators_df.columns else None
            market_cap_b = market_cap / 1_000_000_000 if market_cap else None

            green_candle_chg = result['additional_checks']['green_candle_chg']['passed']
            green_candle_open_chg = result['additional_checks']['green_candle_open_chg']['passed']
            green_candle = green_candle_chg and green_candle_open_chg

            results.append({
                'Ticker': ticker,
                'Status': 'Analyzed',
                'All_Criteria_Met': result['all_criteria_met'],
                'Base_Criteria_Met': result['base_criteria_met'],
                'Additional_Criteria_Met': result['additional_criteria_met'],
                'RS_Rating': rs_rating,
                'Market_Cap_B': market_cap_b,
                'Green_Candle': green_candle,
                'Green_Candle_Chg': green_candle_chg,
                'Green_Candle_Open_Chg': green_candle_open_chg,
                'Score': result['score']
            })

            status_icon = "✓" if result['all_criteria_met'] else "○" if result['base_criteria_met'] >= 6 else "✗"
            print(f"  {status_icon} Base: {result['base_criteria_met']}/8, "
                  f"Add: {result['additional_criteria_met']}/3, "
                  f"RS: {rs_rating:.1f if rs_rating else 'N/A'}, "
                  f"MCap: ${market_cap_b:.2f}B" if market_cap_b else "N/A")

            if verbose and result['all_criteria_met']:
                print(f"  ✓✓✓ ALL CRITERIA MET! ✓✓✓")

        except Exception as e:
            print(f"  ✗ Error analyzing {ticker}: {e}")
            results.append({
                'Ticker': ticker,
                'Status': f'Error: {str(e)[:50]}',
                'All_Criteria_Met': False,
                'Base_Criteria_Met': 0,
                'Additional_Criteria_Met': 0,
                'RS_Rating': None,
                'Market_Cap_B': None,
                'Green_Candle': None,
                'Score': 0
            })

        # API制限回避のため少し待機
        if i < len(tickers):
            time.sleep(0.5)

    return pd.DataFrame(results)


def analyze_rs_rating_threshold(results_df: pd.DataFrame):
    """
    最適なRS Rating閾値を分析

    Args:
        results_df: analyze_stocks_with_criteriaの結果DataFrame
    """
    print(f"\n{'='*80}")
    print("RS Rating Threshold Analysis")
    print(f"{'='*80}\n")

    # データがある銘柄のみ抽出
    analyzed_df = results_df[results_df['Status'] == 'Analyzed'].copy()

    if len(analyzed_df) == 0:
        print("No data available for analysis")
        return

    print(f"Total analyzed stocks: {len(analyzed_df)}")
    print(f"\nRS Rating Distribution:")
    print(analyzed_df['RS_Rating'].describe())

    # RS Rating閾値ごとの検出数
    print(f"\n{'RS Rating Threshold':<20} {'Detected Stocks':<20} {'% of Total':<15}")
    print("-" * 55)

    thresholds = [50, 60, 70, 75, 80, 85, 90, 95]
    for threshold in thresholds:
        detected = len(analyzed_df[analyzed_df['RS_Rating'] >= threshold])
        pct = (detected / len(analyzed_df)) * 100 if len(analyzed_df) > 0 else 0
        print(f"{threshold:<20} {detected:<20} {pct:.1f}%")

    # Base Criteria（Minervini 8基準）を満たす銘柄のRS Rating分析
    print(f"\n{'='*80}")
    print("Stocks meeting ALL Base Criteria (Minervini 8/8)")
    print(f"{'='*80}")

    all_base_met = analyzed_df[analyzed_df['Base_Criteria_Met'] == 8]
    print(f"\nTotal stocks with 8/8 base criteria: {len(all_base_met)}")

    if len(all_base_met) > 0:
        print(f"\nRS Rating for stocks with 8/8 base criteria:")
        print(all_base_met[['Ticker', 'RS_Rating', 'Market_Cap_B', 'Green_Candle']].sort_values('RS_Rating', ascending=False))

        print(f"\nRS Rating stats for 8/8 base criteria stocks:")
        print(all_base_met['RS_Rating'].describe())

    # 全条件を満たす銘柄
    print(f"\n{'='*80}")
    print("Stocks meeting ALL Criteria (Base 8/8 + Additional 3/3)")
    print(f"{'='*80}")

    all_met = analyzed_df[analyzed_df['All_Criteria_Met'] == True]
    print(f"\nTotal stocks with all criteria met: {len(all_met)}")

    if len(all_met) > 0:
        print(f"\nDetailed breakdown:")
        print(all_met[['Ticker', 'RS_Rating', 'Market_Cap_B', 'Green_Candle', 'Score']].sort_values('RS_Rating', ascending=False))


def main():
    """
    メイン分析フロー
    """
    print("="*80)
    print("Minervini Template Detector - 50 Stocks Verification")
    print("Criteria: Minervini Trend Template + Green Candle + $1B+ Market Cap")
    print("="*80)

    # まず、RS Rating 70で分析
    print("\n\nPhase 1: Analysis with RS Rating >= 70")
    results_70 = analyze_stocks_with_criteria(TARGET_TICKERS, min_rs_rating=70, verbose=True)

    if results_70 is not None:
        # 結果をCSVに保存
        results_70.to_csv('minervini_analysis_rs70.csv', index=False)
        print(f"\n✓ Results saved to: minervini_analysis_rs70.csv")

        # サマリー表示
        print(f"\n{'='*80}")
        print("Summary (RS Rating >= 70)")
        print(f"{'='*80}")
        analyzed = results_70[results_70['Status'] == 'Analyzed']
        print(f"Total analyzed: {len(analyzed)}/{len(TARGET_TICKERS)}")
        print(f"All criteria met: {len(analyzed[analyzed['All_Criteria_Met'] == True])}")
        print(f"Base criteria 8/8: {len(analyzed[analyzed['Base_Criteria_Met'] == 8])}")
        print(f"Base criteria 7/8: {len(analyzed[analyzed['Base_Criteria_Met'] == 7])}")
        print(f"Base criteria 6/8: {len(analyzed[analyzed['Base_Criteria_Met'] == 6])}")

        # RS Rating閾値分析
        analyze_rs_rating_threshold(results_70)

        # 推奨RS Rating閾値を計算
        print(f"\n{'='*80}")
        print("Recommended RS Rating Threshold")
        print(f"{'='*80}\n")

        # Base Criteria 8/8を満たす銘柄のRS Rating最小値を見る
        base_8_stocks = analyzed[analyzed['Base_Criteria_Met'] == 8]
        if len(base_8_stocks) > 0:
            min_rs = base_8_stocks['RS_Rating'].min()
            median_rs = base_8_stocks['RS_Rating'].median()
            print(f"For stocks with 8/8 base criteria:")
            print(f"  Minimum RS Rating: {min_rs:.1f}")
            print(f"  Median RS Rating: {median_rs:.1f}")
            print(f"\nRecommended threshold: {min_rs:.0f} (to capture all 8/8 base stocks)")
            print(f"Conservative threshold: {median_rs:.0f} (median of 8/8 base stocks)")

        # 異なるRS Rating閾値でも分析
        print(f"\n\nPhase 2: Analysis with different RS Rating thresholds")

        for threshold in [60, 75, 80]:
            print(f"\n--- Testing RS Rating >= {threshold} ---")
            results_test = analyze_stocks_with_criteria(TARGET_TICKERS, min_rs_rating=threshold, verbose=False)

            if results_test is not None:
                analyzed_test = results_test[results_test['Status'] == 'Analyzed']
                all_met = len(analyzed_test[analyzed_test['All_Criteria_Met'] == True])
                print(f"All criteria met with RS >= {threshold}: {all_met} stocks")

                results_test.to_csv(f'minervini_analysis_rs{threshold}.csv', index=False)
                print(f"Results saved to: minervini_analysis_rs{threshold}.csv")


if __name__ == '__main__':
    main()
