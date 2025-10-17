import pandas as pd
import sys
import os
import argparse
from tqdm import tqdm
from pathlib import Path
from data_fetcher import fetch_stock_data
from indicators import calculate_all_basic_indicators
from stage_history_manager import StageHistoryManager

def run_historical_simulation(ticker: str, period_to_analyze: int = 252, data_dir: str = "./stage_history", **kwargs):
    """
    指定されたティッカーの過去のステージ変遷をシミュレートします。

    Args:
        ticker (str): 分析対象のティッカーシンボル。
        period_to_analyze (int): 分析対象とする直近の取引日数。
        data_dir (str): 履歴ファイルを保存するディレクトリ。
        **kwargs: StageHistoryManagerに渡す追加パラメータ。
    """
    print(f"--- {ticker} の履歴分析シミュレーションを開始 ---")
    print(f"パラメータ: {kwargs}")

    # 既存の履歴ファイルを削除してクリーンな状態から開始
    history_file_path = Path(data_dir) / f"{ticker}_stage_history.json"
    if history_file_path.exists():
        os.remove(history_file_path)
        print(f"既存の履歴ファイル {history_file_path} を削除しました。")

    # 3年分のデータを取得
    try:
        stock_df, benchmark_df = fetch_stock_data(ticker, period="3y")
        if stock_df is None or benchmark_df is None or stock_df.empty or benchmark_df.empty:
            print(f"エラー: {ticker} またはベンチマークのデータを取得できませんでした。")
            return
    except Exception as e:
        print(f"データ取得中にエラーが発生しました: {e}")
        return

    # 全期間の指標を計算
    stock_indicators_df = calculate_all_basic_indicators(stock_df).dropna()
    benchmark_indicators_df = calculate_all_basic_indicators(benchmark_df).dropna()

    # 分析対象期間を特定
    if len(stock_indicators_df) < period_to_analyze:
        print("エラー: 指標計算後のデータが不足しているため、分析を実行できません。")
        return

    analysis_period_df = stock_indicators_df.tail(period_to_analyze)

    # StageHistoryManagerを初期化
    manager = StageHistoryManager(ticker, data_dir=data_dir, **kwargs)

    # 1日ずつ進めながら分析を実行
    print(f"過去{period_to_analyze}日間の分析を1日ずつ実行します...")
    for i in tqdm(range(len(analysis_period_df)), desc=f"Analyzing {ticker}"):
        current_date = analysis_period_df.index[i]
        historical_stock_data = stock_indicators_df.loc[:current_date]
        historical_benchmark_data = benchmark_indicators_df.reindex(historical_stock_data.index, method='ffill')

        if len(historical_stock_data) < 200:
            continue

        try:
            manager.analyze_and_update(historical_stock_data, historical_benchmark_data)
        except Exception:
            continue

    print("\n--- シミュレーション完了 ---")
    manager.print_summary()

def main():
    """メインの実行関数：コマンドライン引数を解析して分析を実行"""
    parser = argparse.ArgumentParser(description="指定されたティッカーのステージ変遷履歴を分析します。")

    parser.add_argument('ticker', type=str, help='分析対象のティッカーシンボル。')
    parser.add_argument('--breakout-window-days', type=int, default=10, help='ブレイクアウト条件の追跡期間（日数）。')
    parser.add_argument('--breakdown-window-days', type=int, default=10, help='ブレイクダウン条件の追跡期間（日数）。')
    parser.add_argument('--topping-days-below-ma', type=int, default=20, help='天井形成と判断するMA下回りの継続日数。')
    parser.add_argument('--basing-days-above-ma', type=int, default=20, help='底固めと判断するMA上回りでの維持日数。')
    parser.add_argument('--data-dir', type=str, default="./stage_history", help='履歴ファイルを保存するディレクトリ。')

    args = parser.parse_args()

    manager_kwargs = {
        'breakout': {'window_days': args.breakout_window_days},
        'breakdown': {'window_days': args.breakdown_window_days},
        'topping': {'days_below_ma': args.topping_days_below_ma},
        'basing': {'days_above_ma': args.basing_days_above_ma}
    }

    run_historical_simulation(
        ticker=args.ticker.upper(),
        data_dir=args.data_dir,
        **manager_kwargs
    )

if __name__ == '__main__':
    main()