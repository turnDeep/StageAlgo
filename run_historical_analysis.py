"""
Historical Stage Analysis Runner（明確化版）

【役割】
バックテストの実行とシミュレーション制御

【責任】
1. データ取得とフォーマット
2. 時系列シミュレーションのループ制御
3. 結果の集約と出力

【連携】
- stage_detector.py: 各時点でのステージ判定を使用
- stage_history_manager.py: 履歴追跡と移行検出を使用
"""
import pandas as pd
import sys
import os
import argparse
from tqdm import tqdm
from pathlib import Path
from data_fetcher import fetch_stock_data
from indicators import calculate_all_basic_indicators
from stage_history_manager import StageHistoryManager
from rs_calculator import RSCalculator


def run_historical_simulation(ticker: str,
                              period_to_analyze: int = 252,
                              data_dir: str = "./stage_history",
                              **kwargs):
    """
    指定されたティッカーの過去のステージ変遷をシミュレート

    【このファイルの役割】
    - データ取得と前処理
    - 時系列ループの制御
    - 結果の出力

    【責任外（他モジュールの役割）】
    - ステージ判定: stage_detector.py
    - 移行追跡: stage_history_manager.py
    
    Args:
        ticker: 分析対象のティッカーシンボル
        period_to_analyze: 分析対象とする直近の取引日数
        data_dir: 履歴ファイルを保存するディレクトリ
        **kwargs: StageHistoryManagerに渡す追加パラメータ
    """
    print(f"--- {ticker} の履歴分析シミュレーションを開始 ---")
    print(f"パラメータ: {kwargs}")
    
    # 既存の履歴ファイルを削除（クリーンな状態から開始）
    history_file_path = Path(data_dir) / f"{ticker}_stage_history.json"
    if history_file_path.exists():
        os.remove(history_file_path)
        print(f"既存の履歴ファイル {history_file_path} を削除しました。")

    # --- データ取得 ---
    # 日足と週足の両方のデータを取得
    try:
        stock_df_daily, benchmark_df_daily = fetch_stock_data(ticker, interval="1d")
        stock_df_weekly, benchmark_df_weekly = fetch_stock_data(ticker, interval="1wk")

        if stock_df_daily is None or stock_df_weekly is None:
            print(f"エラー: {ticker} のデータ取得に失敗しました。")
            return
    except Exception as e:
        print(f"データ取得中にエラーが発生しました: {e}")
        return

    # --- 指標計算 ---
    # 日足と週足の指標をそれぞれ計算
    stock_indicators_daily = calculate_all_basic_indicators(stock_df_daily, '1d').dropna()
    benchmark_indicators_daily = calculate_all_basic_indicators(benchmark_df_daily, '1d').dropna()
    stock_indicators_weekly = calculate_all_basic_indicators(stock_df_weekly, '1wk').dropna()

    # RS Rating 計算 (日足ベース)
    print("RS Ratingを計算中...")
    rs_calculator = RSCalculator(stock_indicators_daily, benchmark_indicators_daily)
    rs_score_series = rs_calculator.calculate_ibd_rs_score()

    # 各時点でのパーセンタイルを計算
    rs_rating_series = rs_score_series.rolling(window=252, min_periods=60).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 99, raw=False
    ).fillna(50)

    stock_indicators_daily['RS_Rating'] = rs_rating_series
    print("✓ RS Rating 計算完了")
    
    # 分析対象期間を特定
    if len(stock_indicators_daily) < period_to_analyze:
        print("エラー: 指標計算後のデータが不足しているため、分析を実行できません。")
        return
    
    analysis_period_df = stock_indicators_daily.tail(period_to_analyze)
    
    # StageHistoryManagerを初期化
    manager = StageHistoryManager(ticker, data_dir=data_dir)
    
    # 時系列シミュレーション（1日ずつ進める）
    print(f"過去{period_to_analyze}日間の分析を1日ずつ実行します...")
    for i in tqdm(range(len(analysis_period_df)), desc=f"Analyzing {ticker}"):
        current_date = analysis_period_df.index[i]

        # --- その時点までの履歴データを準備 ---
        historical_daily_data = stock_indicators_daily.loc[:current_date]
        historical_benchmark_daily = benchmark_indicators_daily.reindex(
            historical_daily_data.index, method='ffill'
        )

        # 現在の日付に対応する週足データを特定 (未来のデータを含めないように)
        historical_weekly_data = stock_indicators_weekly[stock_indicators_weekly.index <= current_date]

        # --- 最低限のデータ数を確認 ---
        if len(historical_daily_data) < 200 or len(historical_weekly_data) < 40:
            continue

        try:
            # StageHistoryManagerで分析と更新
            manager.analyze_and_update(
                daily_df=historical_daily_data,
                benchmark_daily_df=historical_benchmark_daily,
                weekly_df=historical_weekly_data
            )
        except Exception as e:
            # print(f"Warning: {current_date} の分析中にエラー: {e}")
            continue

    # シミュレーション完了
    print("\n--- シミュレーション完了 ---")
    manager.print_summary()


def main():
    """
    メインの実行関数

    コマンドライン引数を解析して分析を実行
    """
    parser = argparse.ArgumentParser(
        description="指定されたティッカーのステージ変遷履歴を分析します。"
    )

    parser.add_argument(
        'ticker',
        type=str,
        help='分析対象のティッカーシンボル。'
    )

    parser.add_argument(
        '--breakout-window-days',
        type=int,
        default=15,
        help='ブレイクアウト条件の追跡期間（日数）。'
    )

    parser.add_argument(
        '--breakout-high-period',
        type=int,
        default=126,
        help='ステージ1→2の高値ブレイクアウトのルックバック期間（日数）。'
    )
    
    parser.add_argument(
        '--breakout-vol-multiplier',
        type=float,
        default=1.5,
        help='ステージ1→2のブレイクアウト時の出来高倍率。'
    )

    parser.add_argument(
        '--breakdown-window-days',
        type=int,
        default=15,
        help='ブレイクダウン条件の追跡期間（日数）。'
    )

    parser.add_argument(
        '--topping-window-days',
        type=int,
        default=20,
        help='天井形成判定の追跡期間（日数）。'
    )

    parser.add_argument(
        '--basing-window-days',
        type=int,
        default=20,
        help='底固め判定の追跡期間（日数）。'
    )

    parser.add_argument(
        '--data-dir',
        type=str,
        default="./stage_history",
        help='履歴ファイルを保存するディレクトリ。'
    )
    
    args = parser.parse_args()

    # パラメータを構築
    manager_kwargs = {
        'breakout': {
            'window_days': args.breakout_window_days,
            'min_high_period': args.breakout_high_period,
            'volume_multiplier': args.breakout_vol_multiplier
        },
        'breakdown': {'window_days': args.breakdown_window_days},
        'topping': {'window_days': args.topping_window_days},
        'basing': {'window_days': args.basing_window_days}
    }

    # シミュレーション実行
    run_historical_simulation(
        ticker=args.ticker.upper(),
        data_dir=args.data_dir,
        **manager_kwargs
    )


if __name__ == '__main__':
    main()