"""
Stage 2 Screener with Transition Date Detection

このスクリプトは、stock.csvから銘柄を読み込み、現在ステージ2にある銘柄を抽出します。
各銘柄について、ステージ2への移行日も特定して出力します。
"""
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import warnings
import os
import shutil

from data_fetcher import fetch_stock_data
from indicators import calculate_all_basic_indicators
from stage_detector import StageDetector
from rs_calculator import RSCalculator
from minervini_template_detector import MinerviniTemplateDetector

warnings.filterwarnings('ignore')


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

        if stage2_transition_date:
            transition_date_str = stage2_transition_date.strftime('%Y-%m-%d')
            if transition_date_str in indicator_df.index:
                transition_close = indicator_df.loc[transition_date_str]['Close']
                if transition_close > 0:
                    price_change_pct = ((current_price - transition_close) / transition_close) * 100

        # Minerviniテンプレートの指標数をカウント
        minervini_detector = MinerviniTemplateDetector(indicator_df)
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

    過去1年間（252営業日）を遡って、ステージ1またはステージ4から
    ステージ2に移行した日を特定します。

    Args:
        df: 指標計算済みのDataFrame
        benchmark_df: ベンチマークのDataFrame

    Returns:
        pd.Timestamp: ステージ2への移行日、見つからない場合はNone
    """
    # 過去252日（約1年）を遡る
    lookback = min(252, len(df) - 200)

    if lookback < 50:
        return None

    prev_stage = None

    # 古い日付から順に確認
    for i in range(len(df) - lookback, len(df)):
        if i < 200:
            continue

        current_date = df.index[i]
        historical_data = df.loc[:current_date]
        historical_benchmark = benchmark_df.reindex(
            historical_data.index,
            method='ffill'
        )

        if len(historical_data) < 200:
            continue

        try:
            # その時点でのステージを判定
            temp_detector = StageDetector(historical_data, historical_benchmark)
            rs_rating = historical_data.iloc[-1].get('RS_Rating')
            stage = temp_detector.determine_stage(
                rs_rating=rs_rating,
                previous_stage=prev_stage
            )

            # ステージ2への移行を検出
            if stage == 2 and prev_stage is not None and prev_stage != 2:
                return current_date

            prev_stage = stage

        except Exception as e:
            continue

    return None


def main():
    """
    メイン処理
    """
    print("=" * 70)
    print("Stage 2 Screener with Transition Date Detection")
    print("=" * 70)
    print()

    # stock.csvを読み込み、サンプリング
    try:
        stock_list_df = pd.read_csv('stock.csv', encoding='utf-8-sig')
        stock_list_df.dropna(subset=['Ticker'], inplace=True)
        # 検証用に100銘柄をランダムにサンプリング
        if len(stock_list_df) > 100:
            stock_list_df = stock_list_df.sample(n=100, random_state=42)
        tickers = [(row['Ticker'], row['Exchange']) for index, row in stock_list_df.iterrows()]
    except FileNotFoundError:
        print("エラー: stock.csvが見つかりません。")
        print("get_tickers.pyを実行してstock.csvを作成してください。")
        return
    except Exception as e:
        print(f"エラー: stock.csvの読み込み中にエラーが発生しました: {e}")
        return

    print(f"✓ {len(tickers)}銘柄を読み込みました（サンプリング済み）")
    print()

    # ベンチマークデータ（SPY）を取得
    print("ベンチマークデータ（SPY）を取得中...")
    _, benchmark_df = fetch_stock_data('SPY', period='2y')
    if benchmark_df is None or benchmark_df.empty:
        print("エラー: ベンチマークデータを取得できませんでした。")
        return

    benchmark_df = calculate_all_basic_indicators(benchmark_df)
    benchmark_df = benchmark_df.dropna()
    print("✓ ベンチマークデータを取得しました")
    print()

    # 銘柄分析
    results = []
    print("銘柄分析を開始します...")
    print()

    # マルチプロセスで並列処理
    with Pool(cpu_count()) as pool:
        for result in tqdm(
            pool.imap_unordered(analyze_ticker_for_stage2, tickers),
            total=len(tickers),
            desc="Analyzing"
        ):
            if result:
                results.append(result)

    if not results:
        print()
        print("=" * 70)
        print("ステージ2の銘柄は見つかりませんでした。")
        print("=" * 70)
        return

    # 結果をDataFrameに変換
    results_df = pd.DataFrame(results)

    # ステージ2移行日でソート（最近移行した銘柄が上位）
    results_df['Stage2_Transition_Date_dt'] = pd.to_datetime(
        results_df['Stage2_Transition_Date'],
        errors='coerce'
    )
    results_df = results_df.sort_values(
        'Stage2_Transition_Date_dt',
        ascending=False,
        na_position='last'
    )

    # 一時列を削除
    results_df = results_df.drop('Stage2_Transition_Date_dt', axis=1)

    # CSVファイルに保存
    output_filename = 'stage2_with_transition_dates.csv'
    results_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
    print()
    print(f"✓ {len(results_df)}銘柄をステージ2として検出しました")
    print(f"✓ 結果を {output_filename} に保存しました")

    # TradingView用のテキストファイルを作成（直近でステージ2になったもののみ）
    if not results_df.empty and 'Stage2_Transition_Date_dt' in results_df.columns:
        # 最新の移行日を取得
        most_recent_transition_date = results_df['Stage2_Transition_Date_dt'].max()

        # 最新の移行日に移行した銘柄のみを抽出
        recent_transition_df = results_df[results_df['Stage2_Transition_Date_dt'] == most_recent_transition_date]

        if not recent_transition_df.empty:
            tradingview_list = [
                f"{row['Exchange']}:{row['Ticker']}"
                for _, row in recent_transition_df.iterrows()
            ]
            tradingview_str = ",".join(tradingview_list)

            tv_filename = 'stage2_tradingview.txt'
            try:
                with open(tv_filename, 'w', encoding='utf-8') as f:
                    f.write(tradingview_str)
                print(f"✓ TradingView用リスト（直近移行銘柄）を {tv_filename} に保存しました")
            except Exception as e:
                print(f"警告: TradingView用ファイルの書き込み中にエラーが発生しました: {e}")
        else:
            print("✓ TradingView用リスト: 直近で移行した銘柄はありませんでした。")


    # サマリーを表示
    print()
    print("=" * 70)
    print("分析結果サマリー")
    print("=" * 70)
    print()

    # 移行日が判明している銘柄の統計
    known_transition = results_df[results_df['Stage2_Transition_Date'] != 'Unknown'].copy()
    if not known_transition.empty:
        known_transition.loc[:, 'Price_Change_%'] = known_transition['Price_Change_%'].fillna(0)
        avg_days = known_transition['Days_Since_Transition'].mean()
        avg_price_change = known_transition['Price_Change_%'].mean()

        print(f"ステージ2移行日が判明: {len(known_transition)}銘柄")
        print(f"平均経過日数: {avg_days:.0f}日")
        print(f"移行日からの平均株価変動率: {avg_price_change:.2f}%")
        print()

    # 最近移行した銘柄（Top 10）
    print("最近ステージ2に移行した銘柄（Top 10）:")
    print("-" * 70)

    display_cols = [
        'Ticker', 'Exchange', 'Stage2_Transition_Date', 'Days_Since_Transition',
        'Current_Price', 'Transition_Close', 'Price_Change_%', 'RS_Rating',
        'Minervini_Indicators_Met'
    ]
    # DataFrameに列が存在するか確認
    display_cols = [col for col in display_cols if col in results_df.columns]

    print(results_df[display_cols].head(10).to_string(index=False))
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()