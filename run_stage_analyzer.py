"""
Stage Analysis Runner: Screener & Historical Simulator

【役割】
- 多数の銘柄をスクリーニングし、特定のステージにあるものを抽出（Screener機能）
- 単一銘柄のステージ変遷を時系列でシミュレーション（Historical Simulator機能）

【責任】
- 外部からの要求に応じて、適切な分析モード（Screener / Simulator）を実行
- データ取得、指標計算、分析プロセスの制御
- 結果の集約とファイル出力

【連携】
- stage_detector.py: 各時点でのステージ判定ロジックを使用
- stage_history_manager.py: 履歴追跡と移行検出ロジックを使用
- rs_calculator.py: RS Ratingの計算ロジックを使用
"""
import pandas as pd
import sys
import os
import argparse
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool, cpu_count
import warnings
from datetime import datetime
import pytz

from data_fetcher import fetch_stock_data
from indicators import calculate_all_basic_indicators
from stage_detector import StageDetector
from rs_calculator import RSCalculator
from minervini_template_detector import MinerviniTemplateDetector
from stage_history_manager import StageHistoryManager

warnings.filterwarnings('ignore')

# ==============================================================================
#
# Screener Functions (from screener.py)
#
# ==============================================================================

def analyze_ticker_for_stage2(args):
    """
    個別銘柄を分析し、ステージ2の場合は移行日と追加情報を特定

    Args:
        args: (ticker, exchange) のタプル

    Returns:
        dict: ステージ2の銘柄情報、またはNone
    """
    ticker, exchange = args

    try:
        stock_df, benchmark_df = fetch_stock_data(ticker, period='2y')
        if stock_df is None or len(stock_df) < 252:
            return None

        indicator_df = calculate_all_basic_indicators(stock_df)
        benchmark_df_calc = calculate_all_basic_indicators(benchmark_df)

        if indicator_df.empty or len(indicator_df) < 252:
            return None

        indicator_df = indicator_df.dropna()
        benchmark_df_calc = benchmark_df_calc.dropna()

        if len(indicator_df) < 252:
            return None

        rs_calculator = RSCalculator(indicator_df, benchmark_df_calc)
        rs_score_series = rs_calculator.calculate_ibd_rs_score()
        rs_rating_series = rs_score_series.rolling(window=252, min_periods=60).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 99, raw=False
        ).fillna(50)
        indicator_df['RS_Rating'] = rs_rating_series

        detector = StageDetector(indicator_df, benchmark_df_calc)
        rs_rating = indicator_df.iloc[-1].get('RS_Rating')
        current_stage = detector.determine_stage(rs_rating=rs_rating)

        if current_stage != 2:
            return None

        stage2_transition_date = find_stage2_transition_date(
            indicator_df,
            benchmark_df_calc
        )

        latest = indicator_df.iloc[-1]
        current_price = latest['Close']
        transition_close = None
        price_change_pct = None
        minervini_criteria_met = 0

        if stage2_transition_date:
            transition_date_str = stage2_transition_date.strftime('%Y-%m-%d')
            if transition_date_str in indicator_df.index:
                transition_close = indicator_df.loc[transition_date_str]['Close']
                if transition_close > 0:
                    price_change_pct = ((current_price - transition_close) / transition_close) * 100

                historical_df = indicator_df.loc[:transition_date_str]
                if len(historical_df) >= 200:
                    minervini_detector = MinerviniTemplateDetector(historical_df)
                    minervini_results = minervini_detector.check_template()
                    minervini_criteria_met = minervini_results.get('criteria_met', 0)

        result = {
            'Ticker': ticker,
            'Exchange': exchange,
            'Stage': 'Stage 2',
            'Stage2_Transition_Date': stage2_transition_date.strftime('%Y-%m-%d') if stage2_transition_date else 'Unknown',
            'Days_Since_Transition': (indicator_df.index[-1] - stage2_transition_date).days if stage2_transition_date else None,
            'Current_Price': current_price,
            'Transition_Close': transition_close,
            'Price_Change_%': price_change_pct,
            'Minervini_Indicators_Met': minervini_criteria_met,
            'RS_Rating': latest.get('RS_Rating', 50),
            'SMA_50': latest.get('SMA_50', 0),
            'SMA_150': latest.get('SMA_150', 0),
            'SMA_200': latest.get('SMA_200', 0),
            'Volume': latest['Volume'],
            'Relative_Volume': latest.get('Relative_Volume', 1.0)
        }

        return result

    except Exception as e:
        return None


def find_stage2_transition_date(df, benchmark_df):
    """
    ステージ2への移行日を特定
    """
    lookback = min(252, len(df) - 200)
    if lookback < 50:
        return None

    prev_stage = None
    for i in range(len(df) - lookback, len(df)):
        if i < 200:
            continue
        current_date = df.index[i]
        historical_data = df.loc[:current_date]
        historical_benchmark = benchmark_df.reindex(historical_data.index, method='ffill')
        if len(historical_data) < 200:
            continue
        try:
            temp_detector = StageDetector(historical_data, historical_benchmark)
            rs_rating = historical_data.iloc[-1].get('RS_Rating')
            stage = temp_detector.determine_stage(rs_rating=rs_rating, previous_stage=prev_stage)
            if stage == 2 and prev_stage is not None and prev_stage != 2:
                return current_date
            prev_stage = stage
        except Exception as e:
            continue
    return None


def run_stage_screening(input_filename='stock.csv'):
    """
    銘柄リストをスクリーニングし、ステージ2の銘柄を特定して結果をファイルに出力
    """
    print("=" * 70)
    print("Stage 2 Screener")
    print("=" * 70)
    print()

    tz = pytz.timezone('America/New_York')
    date_str = datetime.now(tz).strftime('%Y%m%d')
    output_filename = f"{date_str}-stage.csv"

    try:
        stock_list_df = pd.read_csv(input_filename, encoding='utf-8-sig')
        stock_list_df.dropna(subset=['Ticker'], inplace=True)
        tickers = [(row['Ticker'], row['Exchange']) for _, row in stock_list_df.iterrows()]
    except FileNotFoundError:
        print(f"エラー: {input_filename}が見つかりません。")
        return None
    except Exception as e:
        print(f"エラー: {input_filename}の読み込み中にエラー: {e}")
        return None

    print(f"✓ {len(tickers)}銘柄を読み込みました")
    print()

    results = []
    print("銘柄分析を開始します...")
    print()
    with Pool(cpu_count()) as pool:
        for result in tqdm(pool.imap_unordered(analyze_ticker_for_stage2, tickers), total=len(tickers), desc="Analyzing"):
            if result:
                results.append(result)

    if not results:
        print("\n" + "=" * 70)
        print("ステージ2の銘柄は見つかりませんでした。")
        print("=" * 70)
        return None

    results_df = pd.DataFrame(results)
    results_df['Stage2_Transition_Date_dt'] = pd.to_datetime(results_df['Stage2_Transition_Date'], errors='coerce')
    results_df = results_df.sort_values('Stage2_Transition_Date_dt', ascending=False, na_position='last')
    results_df_to_save = results_df.drop('Stage2_Transition_Date_dt', axis=1)
    results_df_to_save.to_csv(output_filename, index=False, encoding='utf-8-sig')

    print(f"\n✓ {len(results_df)}銘柄をステージ2として検出しました")
    print(f"✓ 結果を {output_filename} に保存しました")

    if not results_df.empty:
        most_recent_transition_date = results_df['Stage2_Transition_Date_dt'].max()
        recent_transition_df = results_df[results_df['Stage2_Transition_Date_dt'] == most_recent_transition_date]
        if not recent_transition_df.empty:
            tradingview_list = [f"{row['Exchange']}:{row['Ticker']}" for _, row in recent_transition_df.iterrows()]
            tv_filename = f"{date_str}-stage.txt"
            with open(tv_filename, 'w', encoding='utf-8') as f:
                f.write(",".join(tradingview_list))
            print(f"✓ TradingView用リストを {tv_filename} に保存しました")

    print("\n" + "=" * 70)
    print("分析結果サマリー")
    print("=" * 70)
    # (サマリー表示のロジックはscreener.pyで表示するため、ここでは省略)
    display_cols = ['Ticker', 'Exchange', 'Stage2_Transition_Date', 'Days_Since_Transition', 'Current_Price', 'Transition_Close', 'Price_Change_%', 'RS_Rating', 'Minervini_Indicators_Met']
    display_cols = [col for col in display_cols if col in results_df.columns]
    print(results_df[display_cols].head(10).to_string(index=False))
    print()

    return output_filename


# ==============================================================================
#
# Historical Simulator Functions (Original functions)
#
# ==============================================================================

def run_historical_simulation(ticker: str,
                              period_to_analyze: int = 252,
                              data_dir: str = "./stage_history",
                              **kwargs):
    """
    指定されたティッカーの過去のステージ変遷をシミュレート
    """
    print(f"--- {ticker} の履歴分析シミュレーションを開始 ---")
    history_file_path = Path(data_dir) / f"{ticker}_stage_history.json"
    if history_file_path.exists():
        os.remove(history_file_path)

    try:
        stock_df_daily, benchmark_df_daily = fetch_stock_data(ticker, interval="1d")
        stock_df_weekly, _ = fetch_stock_data(ticker, interval="1wk")
        if stock_df_daily is None or stock_df_weekly is None:
            print(f"エラー: {ticker} のデータ取得に失敗しました。")
            return
    except Exception as e:
        print(f"データ取得中にエラーが発生しました: {e}")
        return

    stock_indicators_daily = calculate_all_basic_indicators(stock_df_daily, '1d').dropna()
    benchmark_indicators_daily = calculate_all_basic_indicators(benchmark_df_daily, '1d').dropna()
    stock_indicators_weekly = calculate_all_basic_indicators(stock_df_weekly, '1wk').dropna()

    rs_calculator = RSCalculator(stock_indicators_daily, benchmark_indicators_daily)
    rs_score_series = rs_calculator.calculate_ibd_rs_score()
    rs_rating_series = rs_score_series.rolling(window=252, min_periods=60).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 99, raw=False
    ).fillna(50)
    stock_indicators_daily['RS_Rating'] = rs_rating_series
    
    if len(stock_indicators_daily) < period_to_analyze:
        print("エラー: データ不足のため分析を実行できません。")
        return
    
    analysis_period_df = stock_indicators_daily.tail(period_to_analyze)
    manager = StageHistoryManager(ticker, data_dir=data_dir)
    
    for i in tqdm(range(len(analysis_period_df)), desc=f"Analyzing {ticker}"):
        current_date = analysis_period_df.index[i]
        historical_daily_data = stock_indicators_daily.loc[:current_date]
        historical_benchmark_daily = benchmark_indicators_daily.reindex(historical_daily_data.index, method='ffill')
        historical_weekly_data = stock_indicators_weekly[stock_indicators_weekly.index <= current_date]

        if len(historical_daily_data) < 200 or len(historical_weekly_data) < 40:
            continue

        try:
            manager.analyze_and_update(
                daily_df=historical_daily_data,
                benchmark_daily_df=historical_benchmark_daily,
                weekly_df=historical_weekly_data
            )
        except Exception as e:
            continue

    print("\n--- シミュレーション完了 ---")
    manager.print_summary()


def main():
    """
    メインの実行関数 (Historical Simulator用)
    """
    parser = argparse.ArgumentParser(description="ステージ分析ツール: スクリーニングまたは履歴分析を実行します。")
    parser.add_argument('mode', nargs='?', default='screener', choices=['screener', 'simulate'], help="実行モード ('screener' または 'simulate')。デフォルトは 'screener'。")
    parser.add_argument('-t', '--ticker', type=str, help="履歴分析モードで分析するティッカーシンボル。")
    # 他の引数は省略
    
    args = parser.parse_args()

    if args.mode == 'screener':
        run_stage_screening()
    elif args.mode == 'simulate':
        if not args.ticker:
            print("エラー: 履歴分析モードでは --ticker の指定が必要です。")
            sys.exit(1)
        run_historical_simulation(ticker=args.ticker.upper())
    else:
        # デフォルトはスクリーナーを実行
        run_stage_screening()

if __name__ == '__main__':
    main()
