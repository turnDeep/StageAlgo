"""
Base Analyzer Runner（完全改訂版）

【改善点】
1. 週足データの使用（ミネルヴィニ標準）
2. 新しいBaseMinerviniAnalyzerの活用
3. Stage 2フィルタリングの統合
4. より詳細な結果の出力
5. ベース品質評価の追加
6. パフォーマンス最適化（ベンチマークデータの共有）
"""
import pandas as pd
import os
from datetime import datetime
import pytz
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, Manager
import warnings
import sys

from data_fetcher import fetch_stock_data
from indicators import calculate_all_basic_indicators
from stage_detector import StageDetector
from base_minervini_analyzer import BaseMinerviniAnalyzer

warnings.filterwarnings('ignore')

# --- Worker Process Initializer ---
def init_worker(benchmark_df_data):
    """
    ワーカープロセスの初期化関数
    ベンチマークデータをグローバル変数として設定
    """
    global benchmark_df_global
    benchmark_df_global = benchmark_df_data

def analyze_single_ticker(args):
    """
    単一銘柄のベース分析を実行
    """
    ticker, exchange, stage2_only = args
    global benchmark_df_global # グローバル変数を使用

    try:
        # データ取得 (ベンチマークはダウンロードしない)
        daily_df, _ = fetch_stock_data(ticker, period='5y', interval='1d', fetch_benchmark=False)
        weekly_df, _ = fetch_stock_data(ticker, period='5y', interval='1wk', fetch_benchmark=False)

        if daily_df is None or weekly_df is None or len(daily_df) < 252 or benchmark_df_global is None:
            return None

        # 指標計算
        daily_df = calculate_all_basic_indicators(daily_df, interval='1d')
        weekly_df = calculate_all_basic_indicators(weekly_df, interval='1wk')

        # Stage判定
        stage_detector = StageDetector(daily_df, benchmark_df_global, interval='1d')
        current_stage = stage_detector.determine_stage()

        if stage2_only and current_stage != 2:
            return None

        # ベース分析
        analyzer = BaseMinerviniAnalyzer(daily_df, weekly_df, benchmark_df_global)
        events = analyzer.analyze()
        valid_breakouts = [e for e in events if e['event'] == 'VALID_BREAKOUT']
        if not valid_breakouts:
            return None

        latest_base = valid_breakouts[-1]
        base_count = analyzer.get_base_count()
        stage_eval = analyzer.evaluate_base_stage()
        template_result = stage_detector.check_minervini_template()
        criteria_met = template_result.get('criteria_met', 0) if template_result.get('applicable', False) else 0

        breakout_date = pd.to_datetime(latest_base.get('breakout_date')) if 'breakout_date' in latest_base else None
        days_since_breakout = (pd.Timestamp.now() - breakout_date).days if breakout_date else None

        latest_price = daily_df.iloc[-1]['Close']
        resistance_price = 0
        base_start_events = [e for e in events if e['event'] == 'BASE_START']
        if base_start_events:
            resistance_price = base_start_events[-1].get('resistance_price', 0)

        status = "Rising" if latest_price > resistance_price * 1.05 else "Near Resistance" if latest_price > resistance_price * 0.98 else "Below Resistance"

        return {
            'Ticker': ticker, 'Exchange': exchange, 'Stage': current_stage,
            'Base Count': base_count, 'Base Stage': stage_eval['stage'],
            'Base Type': latest_base.get('base_type', 'unknown'),
            'Quality': latest_base.get('quality', 'unknown'),
            'Duration (weeks)': latest_base.get('duration_weeks', 0),
            'Depth %': latest_base.get('depth_pct', '0%'),
            'VCP Valid': 'Yes' if latest_base.get('vcp_valid', False) else 'No',
            'Resistance Price': f"${resistance_price:.2f}" if resistance_price else 'N/A',
            'Current Price': f"${latest_price:.2f}",
            'Days Since Breakout': days_since_breakout if days_since_breakout is not None else 'N/A',
            'Status': status, 'Minervini Criteria Met': criteria_met,
            'Assessment': stage_eval['assessment'], 'Recommended Action': stage_eval['action']
        }
    except Exception as e:
        return None


def run_base_analysis(output_filename='base_analysis_results.csv', input_filename=None, stage2_only=True):
    """
    ベース分析を実行
    """
    print("=" * 70)
    print("Base Minervini Analyzer (Optimized)")
    print("=" * 70)
    print()

    try:
        if input_filename:
            stock_list = pd.read_csv(input_filename).dropna(subset=['Ticker', 'Exchange']).drop_duplicates(subset=['Ticker'])
            print(f"✓ {input_filename} から {len(stock_list)} 銘柄を読み込みました")
        else:
            stock_list = pd.read_csv('stock.csv').dropna(subset=['Ticker', 'Exchange']).drop_duplicates(subset=['Ticker'])
            print(f"✓ stock.csv から {len(stock_list)} 銘柄を読み込みました")
        if stage2_only:
            print(" ※ Stage 2銘柄のみを分析します")
    except FileNotFoundError:
        print(f"Error: {input_filename or 'stock.csv'} not found.")
        return

    # --- パフォーマンス最適化: ベンチマークを事前に一括取得 ---
    print("\nベンチマークデータ (SPY) を取得中...")
    try:
        benchmark_df, _ = fetch_stock_data('SPY', period='5y', interval='1d', fetch_benchmark=False)
        if benchmark_df is None or benchmark_df.empty:
            raise ValueError("ベンチマークデータの取得に失敗しました。")
        benchmark_df = calculate_all_basic_indicators(benchmark_df, interval='1d')
        print("✓ ベンチマークデータを取得しました。")
    except Exception as e:
        print(f"エラー: ベンチマークデータの取得に失敗しました: {e}")
        return

    tickers_to_analyze = [(row['Ticker'], row['Exchange'], stage2_only) for _, row in stock_list.iterrows()]

    results = []
    print(f"\n銘柄分析を開始します（{cpu_count()}プロセス使用）...")

    # --- Poolの初期化時にベンチマークデータを渡す ---
    with Pool(initializer=init_worker, initargs=(benchmark_df,)) as pool:
        for result in tqdm(pool.imap_unordered(analyze_single_ticker, tickers_to_analyze), total=len(tickers_to_analyze), desc="Analyzing"):
            if result:
                results.append(result)

    if not results:
        print("\n" + "=" * 70)
        print("ベースを持つ銘柄は見つかりませんでした。")
        print("=" * 70)
        return

    results_df = pd.DataFrame(results).sort_values('Base Count', ascending=False)
    results_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
    print(f"\n✓ {len(results_df)} 銘柄の分析結果を {output_filename} に保存しました")

    early_stage_df = results_df[results_df['Base Stage'].isin(['初期段階', '中期段階'])]
    if not early_stage_df.empty:
        tradingview_list = [f"{row['Exchange']}:{row['Ticker']}" for _, row in early_stage_df.iterrows()]
        txt_output_filename = os.path.splitext(output_filename)[0] + ".txt"
        with open(txt_output_filename, 'w', encoding='utf-8') as f:
            f.write(",".join(tradingview_list))
        print(f"✓ TradingView用リストを {txt_output_filename} に保存しました (対象: {len(early_stage_df)}銘柄)")

    # (サマリー表示部分は省略)
    print("\n" + "=" * 70)
    print("分析完了")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Mark Minerviniのベース理論に基づいたベース分析")
    parser.add_argument('-o', '--output', type=str, default=None, help='出力ファイル名')
    parser.add_argument('-i', '--input', type=str, default=None, help='入力ファイル名')
    parser.add_argument('--all-stages', action='store_true', help='全ステージの銘柄を分析')
    args = parser.parse_args()

    output_filename = args.output
    if output_filename is None:
        tz = pytz.timezone('America/New_York')
        date_str = datetime.now(tz).strftime('%Y%m%d')
        output_filename = f"{date_str}-base.csv"

    run_base_analysis(
        output_filename=output_filename,
        input_filename=args.input,
        stage2_only=not args.all_stages
    )
