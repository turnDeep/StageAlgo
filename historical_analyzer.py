import pandas as pd
from data_fetcher import fetch_stock_data
from indicators import calculate_all_indicators
from stage_engine import StageAnalysisSystem

def find_stage_transitions(ticker: str, period_to_analyze: int = 252):
    """
    指定されたティッカーの過去のステージ移行日を特定します。

    Args:
        ticker (str): 分析対象のティッカーシンボル。
        period_to_analyze (int): 分析対象とする直近の取引日数 (約1年分)。
    """
    print(f"\n--- {ticker} の過去のステージ移行分析を開始 ---")

    # 分析には長期のデータが必要なため、3年分のデータを取得
    stock_df, benchmark_df = fetch_stock_data(ticker, period="3y")

    if stock_df is None or stock_df.empty:
        print(f"エラー: {ticker} のデータを取得できませんでした。")
        return

    # 全期間の指標を一度に計算
    stock_indicators_df = calculate_all_indicators(stock_df, benchmark_df)
    benchmark_indicators_df = calculate_all_indicators(benchmark_df, benchmark_df.copy())

    if stock_indicators_df.empty or len(stock_indicators_df) < period_to_analyze:
        print("エラー: 指標計算後のデータが不足しているため、分析を実行できません。")
        return

    # 分析対象期間のデータに絞る
    analysis_df = stock_indicators_df.tail(period_to_analyze)

    previous_stage = None
    stage_transitions = []

    # 一日ずつ進めながらステージを判定
    for i in range(len(analysis_df)):
        # その日までの全データを使って分析
        current_date = analysis_df.index[i]

        # 分析に必要な過去データを含めてスライス
        historical_stock_data = stock_indicators_df.loc[:current_date]
        historical_benchmark_data = benchmark_indicators_df.loc[:current_date]

        if len(historical_stock_data) < 252: # 1年分のデータが溜まるまでスキップ
            continue

        try:
            analyzer = StageAnalysisSystem(historical_stock_data, ticker, historical_benchmark_data)
            current_stage = analyzer._determine_current_stage()
        except (ValueError, IndexError) as e:
            # データが不足している場合などのエラーをスキップ
            # print(f"警告: {current_date.strftime('%Y-%m-%d')} の分析中にエラー: {e}")
            continue

        if previous_stage is not None and current_stage != previous_stage:
            transition_info = {
                "date": current_date.strftime('%Y-%m-%d'),
                "transition": f"ステージ {previous_stage} → ステージ {current_stage}"
            }
            stage_transitions.append(transition_info)
            print(f"移行を検出: {transition_info['date']} に {transition_info['transition']}")

        previous_stage = current_stage

    if not stage_transitions:
        print(f"過去{period_to_analyze}日間で明確なステージの移行は検出されませんでした。")
        print(f"最新のステージ: ステージ {previous_stage}")

    return stage_transitions

import sys

def main():
    """
    メインの実行関数
    """
    if len(sys.argv) > 1:
        ticker_to_analyze = sys.argv[1]
    else:
        # デフォルトのティッカー、またはエラーメッセージを表示
        print("エラー: 分析するティッカーシンボルをコマンドライン引数として指定してください。")
        print("例: python historical_analyzer.py AAPL")
        return

    find_stage_transitions(ticker_to_analyze)

if __name__ == '__main__':
    main()