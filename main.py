from data_fetcher import fetch_stock_data
from indicators import calculate_all_indicators
from stage_engine import StageAnalysisSystem
import pandas as pd

def run_analysis_for_ticker(ticker: str, benchmark_df: pd.DataFrame):
    """
    指定されたティッカーのステージ分析を実行し、結果を出力します。

    Args:
        ticker (str): 分析対象のティッカーシンボル。
        benchmark_df (pd.DataFrame): ベンチマーク（SPY）の生データ。
    """
    print(f"\n--- {ticker} のステージ分析を開始 ---")

    # 1. 株式データの取得
    # ベンチマークは既に取得済みなので、ここでは個別の株式データのみ取得
    stock_df, _ = fetch_stock_data(ticker, benchmark_ticker="SPY")

    if stock_df is None or stock_df.empty:
        print(f"エラー: {ticker} のデータを取得できませんでした。分析をスキップします。")
        return

    # 2. 株式とベンチマークの両方でテクニカル指標を計算
    try:
        # RS Ratingの計算のために、両方のデータが必要
        stock_indicators_df = calculate_all_indicators(stock_df, benchmark_df)
        # 市場環境分析のために、ベンチマーク自体の指標も計算
        benchmark_indicators_df = calculate_all_indicators(benchmark_df, benchmark_df.copy())
    except Exception as e:
        print(f"エラー: {ticker} の指標計算中にエラーが発生しました: {e}")
        return

    if stock_indicators_df.empty or benchmark_indicators_df.empty:
        print(f"エラー: {ticker} またはベンチマークの指標計算後、データが空になりました。")
        return

    # 3. 分析システムの実行（改善版）
    try:
        # 更新されたコンストラクタにベンチマーク指標データを渡す
        analyzer = StageAnalysisSystem(stock_indicators_df, ticker, benchmark_indicators_df)
        result = analyzer.analyze()
    except ValueError as e:
        print(f"エラー: {ticker} の分析中にエラーが発生しました: {e}")
        return

    # 4. 結果の表示（詳細情報も含む）
    print(f"ティッカー: {result['ticker']}")
    print(f"分析日: {result['analysis_date']}")
    print(f"現在の推定ステージ: {result['current_stage']}")

    analysis = result['transition_analysis']
    print("\n[移行分析]")
    print(f"  分析対象: {analysis['target_transition']}")
    print(f"  スコア: {analysis['score']} / 100")
    print(f"  判定: {analysis['level']}")
    print(f"  推奨アクション: {analysis['action']}")

    # スコア詳細があれば表示
    if 'details' in analysis:
        print("\n  [スコア詳細]")
        for key, value in analysis['details'].items():
            print(f"    - {key:<12}: {value}")

    print("-" * 40)


def main():
    """
    メインの実行関数。stock.csvからティッカーを読み込み、分析を実行します。
    """
    try:
        # stock.csvをpandasで読み込む
        tickers_df = pd.read_csv('stock.csv', encoding='utf-8-sig')

        # 'Ticker'列が存在するか確認し、ティッカーリストを取得
        if 'Ticker' not in tickers_df.columns:
            print("エラー: stock.csv に 'Ticker' 列が見つかりません。")
            return

        tickers_to_analyze = tickers_df['Ticker'].dropna().astype(str).tolist()

        if not tickers_to_analyze:
            print("エラー: stock.csv が空か、有効なティッカーが含まれていません。")
            return

        print(f"stock.csv から {len(tickers_to_analyze)} 個のティッカーを読み込みました。")
    except FileNotFoundError:
        print("エラー: stock.csv が見つかりません。プログラムを終了します。")
        return
    except Exception as e:
        print(f"エラー: stock.csv の読み込み中に予期せぬエラーが発生しました: {e}")
        return

    print("\nステージ分析を開始します...")
    print("ベンチマークデータ(SPY)を取得中...")

    # 最初にベンチマークデータを一度だけ取得
    benchmark_df, _ = fetch_stock_data("SPY", benchmark_ticker="SPY")

    if benchmark_df is None or benchmark_df.empty:
        print("致命的エラー: ベンチマーク(SPY)のデータを取得できませんでした。プログラムを終了します。")
        return

    for ticker in tickers_to_analyze:
        # ベンチマークの生データを渡す
        run_analysis_for_ticker(ticker, benchmark_df)

if __name__ == '__main__':
    main()