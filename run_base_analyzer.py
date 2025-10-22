import pandas as pd
from data_fetcher import fetch_stock_data
from indicators import calculate_all_basic_indicators
from stage_detector import StageDetector
from base_minervini_analyzer import BaseMinerviniAnalyzer
import os

def run_base_analysis(output_filename='base_analysis_results.csv'):
    """
    株価データを取得し、ステージ2の銘柄に対してベース分析を実行し、結果をCSVとTradingViewリストに出力する。

    Args:
        output_filename (str, optional): 出力するCSVファイル名.
                                        Defaults to 'base_analysis_results.csv'.
    """
    try:
        # stock.csvから銘柄リストと取引所を読み込む
        stock_list = pd.read_csv('stock.csv').dropna(subset=['Ticker', 'Exchange']).drop_duplicates(subset=['Ticker'])
        tickers_to_analyze = stock_list.to_dict('records')

        results = []

        for stock_info in tickers_to_analyze:
            ticker = stock_info['Ticker']
            exchange = stock_info['Exchange']

            print(f"Analyzing {ticker}...")
            # 5年分のデータを取得
            df, _ = fetch_stock_data(ticker, period='5y')

            if df is None or len(df) < 252:
                print(f"Could not fetch sufficient data for {ticker}")
                continue

            # テクニカル指標を計算
            df = calculate_all_basic_indicators(df)

            # ステージ検出
            stage_detector = StageDetector(df)
            stage = stage_detector.determine_stage()

            # 最新のステージが2の場合のみベース分析を実行
            if stage == 2:
                print(f"{ticker} is in Stage 2. Running base analysis...")

                # ミネルヴィニのトレンドテンプレートをチェック
                template_result = stage_detector.check_minervini_template()
                criteria_met = template_result.get('criteria_met', 0)

                # ベース分析
                base_analyzer = BaseMinerviniAnalyzer(df.copy())
                events = base_analyzer.analyze()

                if not events:
                    continue

                # 最新のベース開始イベントを取得
                base_start_events = [e for e in events if e['event'] == 'BASE_START']
                if not base_start_events:
                    continue

                latest_base_start = base_start_events[-1]
                resistance_price = latest_base_start['resistance_price']

                # レジスタンスからの経過日数を計算
                resistance_date = pd.to_datetime(latest_base_start['date'])
                days_since_resistance = (pd.to_datetime('today') - resistance_date).days

                # ベースカウンティング
                base_count = len(base_start_events)

                results.append({
                    'Ticker': ticker,
                    'Exchange': exchange,
                    'Stage': 2,
                    'Base Count': base_count,
                    'Resistance Price': f"{resistance_price:.2f}",
                    'Days Since Resistance': days_since_resistance,
                    'Minervini Criteria Met': criteria_met,
                })

        # 結果をCSVファイルとTradingViewリストに出力
        if results:
            results_df = pd.DataFrame(results)

            # 1. CSVファイルに保存
            results_df.to_csv(output_filename, index=False)
            print(f"Base analysis complete. Results saved to {output_filename}")

            # 2. TradingView用リストを生成して保存
            tradingview_list = [f"{row['Exchange']}:{row['Ticker']}" for _, row in results_df.iterrows()]
            tradingview_str = ",".join(tradingview_list)

            txt_output_filename = os.path.splitext(output_filename)[0] + ".txt"
            try:
                with open(txt_output_filename, 'w', encoding='utf-8') as f:
                    f.write(tradingview_str)
                print(f"TradingView list saved to {txt_output_filename}")
            except Exception as e:
                print(f"Warning: Could not write TradingView file: {e}")
        else:
            print("No Stage 2 stocks with bases found.")

    except FileNotFoundError:
        print("Error: stock.csv not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    run_base_analysis()
