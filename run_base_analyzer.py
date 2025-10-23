import pandas as pd
from multiprocessing import Pool, cpu_count
from pathlib import Path
import pickle
from tqdm import tqdm
import os

# 必要なモジュールをインポート
from stage_detector import StageDetector
from base_minervini_analyzer import BaseMinerviniAnalyzer

# グローバル変数としてキャッシュデータを保持
worker_cache_data = None

def init_worker(cache_file):
    """
    各ワーカープロセスの初期化関数。キャッシュファイルを一度だけ読み込む。
    """
    global worker_cache_data
    if cache_file and Path(cache_file).exists():
        try:
            with open(cache_file, 'rb') as f:
                worker_cache_data = pickle.load(f)
        except Exception:
            worker_cache_data = {} # エラー時は空の辞書
    else:
        worker_cache_data = {}

def analyze_base_with_cache(args):
    """
    キャッシュデータを使用してベース分析を実行（マルチプロセスワーカー関数）。

    Args:
        args (tuple): (ticker, exchange) のタプル。

    Returns:
        dict: 分析結果。分析対象外またはエラーの場合はNone。
    """
    ticker, exchange = args
    global worker_cache_data

    try:
        if worker_cache_data is None or ticker not in worker_cache_data:
            return None

        df = worker_cache_data[ticker]['indicators_df'].copy()

        if df is None or len(df) < 252:
            return None

        stage_detector = StageDetector(df)
        template_result = stage_detector.check_minervini_template()
        criteria_met = template_result.get('criteria_met', 0)

        base_analyzer = BaseMinerviniAnalyzer(df.copy())
        events = base_analyzer.analyze()

        if not events:
            return None

        base_start_events = [e for e in events if e['event'] == 'BASE_START']
        if not base_start_events:
            return None

        latest_base_start = base_start_events[-1]
        resistance_price = latest_base_start['resistance_price']
        resistance_date = pd.to_datetime(latest_base_start['date'])
        days_since_resistance = (pd.Timestamp.now(tz='America/New_York') - resistance_date).days
        base_count = len(base_start_events)
        status = 'Rising' if base_analyzer.state == 'WAITING_FOR_SEPARATION' else '-'

        return {
            'Ticker': ticker,
            'Exchange': exchange,
            'Stage': 2,
            'Base Count': base_count,
            'Resistance Price': f"{resistance_price:.2f}",
            'Days Since Resistance': days_since_resistance,
            'Minervini Criteria Met': criteria_met,
            'Status': status,
        }

    except Exception as e:
        return None


def run_base_analysis(output_filename='base_analysis_results.csv', input_filename=None, cache_file=None):
    """
    マルチプロセスとキャッシュを利用してベース分析を実行する。
    """
    if not cache_file or not Path(cache_file).exists():
        print(f"エラー: キャッシュファイルが見つかりません: {cache_file}")
        # ここで代替処理（手動データ取得など）を実装することもできるが、今回は終了する
        return

    print(f"✓ {cache_file} のキャッシュデータを利用します")

    try:
        stock_list = pd.read_csv(input_filename).dropna(subset=['Ticker', 'Exchange']).drop_duplicates(subset=['Ticker'])
        analysis_args = [(row['Ticker'], row['Exchange']) for _, row in stock_list.iterrows()]
        print(f"✓ {input_filename} から {len(stock_list)} 銘柄を読み込みました")

    except FileNotFoundError:
        print(f"エラー: {input_filename} が見つかりません。")
        return

    results = []

    print(f"{len(analysis_args)} 銘柄のベース分析を開始します...")
    # initializer を使って各ワーカーにキャッシュをロード
    with Pool(processes=cpu_count(), initializer=init_worker, initargs=(cache_file,)) as pool:
        for result in tqdm(pool.imap_unordered(analyze_base_with_cache, analysis_args), total=len(analysis_args), desc="Base Analyzing"):
            if result:
                results.append(result)

    if results:
        results_df = pd.DataFrame(results).sort_values(by='Days Since Resistance', ascending=True)

        results_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
        print(f"\n✓ ベース分析が完了しました。{len(results_df)}銘柄の結果を {output_filename} に保存しました")

        tradingview_list = [f"{row['Exchange']}:{row['Ticker']}" for _, row in results_df.iterrows()]
        tradingview_str = ",".join(tradingview_list)

        txt_output_filename = os.path.splitext(output_filename)[0] + ".txt"
        try:
            with open(txt_output_filename, 'w', encoding='utf-8') as f:
                f.write(tradingview_str)
            print(f"✓ TradingView用リストを {txt_output_filename} に保存しました")
        except Exception as e:
            print(f"警告: TradingView用ファイルの書き込みに失敗しました: {e}")
    else:
        print("\nベース分析の対象となる銘柄は見つかりませんでした。")

if __name__ == "__main__":
    run_base_analysis()
