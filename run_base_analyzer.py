"""
Base Analyzer Runner（完全改訂版）

【改善点】
1. 週足データの使用（ミネルヴィニ標準）
2. 新しいBaseMinerviniAnalyzerの活用
3. Stage 2フィルタリングの統合
4. より詳細な結果の出力
5. ベース品質評価の追加
"""
import pandas as pd
import os
from datetime import datetime
import pytz
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import warnings

from data_fetcher import fetch_stock_data
from indicators import calculate_all_basic_indicators
from stage_detector import StageDetector
from base_minervini_analyzer import BaseMinerviniAnalyzer

warnings.filterwarnings('ignore')


def analyze_single_ticker(args):
    """
    単一銘柄のベース分析を実行

    Args:
        args: (ticker, exchange, stage2_only) のタプル

    Returns:
        dict: 分析結果、またはNone
    """
    ticker, exchange, stage2_only = args

    try:
        # データ取得
        daily_df, benchmark_df = fetch_stock_data(ticker, period='5y', interval='1d')
        weekly_df, _ = fetch_stock_data(ticker, period='5y', interval='1wk')

        if daily_df is None or weekly_df is None or len(daily_df) < 252:
            return None

        # 指標計算
        daily_df = calculate_all_basic_indicators(daily_df, interval='1d')
        weekly_df = calculate_all_basic_indicators(weekly_df, interval='1wk')
        benchmark_df = calculate_all_basic_indicators(benchmark_df, interval='1d')

        # Stage判定
        stage_detector = StageDetector(daily_df, benchmark_df, interval='1d')
        current_stage = stage_detector.determine_stage()

        # Stage 2フィルタリング（オプション）
        if stage2_only and current_stage != 2:
            return None

        # ベース分析
        analyzer = BaseMinerviniAnalyzer(daily_df, weekly_df, benchmark_df)
        events = analyzer.analyze()

        # 有効なブレイクアウトのみを抽出
        valid_breakouts = [e for e in events if e['event'] == 'VALID_BREAKOUT']

        if not valid_breakouts:
            return None

        # 最新のベース情報を取得
        latest_base = valid_breakouts[-1]
        base_count = analyzer.get_base_count()

        # ベース段階評価
        stage_eval = analyzer.evaluate_base_stage()

        # Minerviniテンプレートチェック
        template_result = stage_detector.check_minervini_template()
        criteria_met = template_result.get('criteria_met', 0) if template_result.get('applicable', False) else 0

        # 抵抗線からの経過日数
        if 'breakout_date' in latest_base:
            breakout_date = pd.to_datetime(latest_base['breakout_date'])
            days_since_breakout = (pd.Timestamp.now() - breakout_date).days
        else:
            days_since_breakout = None

        # 現在の状態判定
        latest_price = daily_df.iloc[-1]['Close']
        resistance_price = float(latest_base.get('base_start', '0').split('$')[-1]) if '$' in str(latest_base.get('base_start', '')) else 0

        # 抵抗線価格を正しく取得
        base_start_events = [e for e in events if e['event'] == 'BASE_START']
        if base_start_events:
            resistance_price = base_start_events[-1].get('resistance_price', 0)

        # 現在の状態
        if latest_price > resistance_price * 1.05:
            status = "Rising"
        elif latest_price > resistance_price * 0.98:
            status = "Near Resistance"
        else:
            status = "Below Resistance"

        result = {
            'Ticker': ticker,
            'Exchange': exchange,
            'Stage': current_stage,
            'Base Count': base_count,
            'Base Stage': stage_eval['stage'],
            'Base Type': latest_base.get('base_type', 'unknown'),
            'Quality': latest_base.get('quality', 'unknown'),
            'Duration (weeks)': latest_base.get('duration_weeks', 0),
            'Depth %': latest_base.get('depth_pct', '0%'),
            'VCP Valid': 'Yes' if latest_base.get('vcp_valid', False) else 'No',
            'Resistance Price': f"${resistance_price:.2f}" if resistance_price else 'N/A',
            'Current Price': f"${latest_price:.2f}",
            'Days Since Breakout': days_since_breakout if days_since_breakout is not None else 'N/A',
            'Status': status,
            'Minervini Criteria Met': criteria_met,
            'Assessment': stage_eval['assessment'],
            'Recommended Action': stage_eval['action']
        }

        return result

    except Exception as e:
        # エラーを静かにスキップ
        return None


def run_base_analysis(output_filename='base_analysis_results.csv',
                      input_filename=None,
                      stage2_only=True):
    """
    ベース分析を実行

    Args:
        output_filename: 出力CSVファイル名
        input_filename: 入力CSVファイル名（Noneの場合はstock.csvを使用）
        stage2_only: Stage 2銘柄のみを分析するか
    """
    print("=" * 70)
    print("Base Minervini Analyzer（完全改訂版）")
    print("=" * 70)
    print()

    # 銘柄リストの読み込み
    try:
        if input_filename:
            stock_list = pd.read_csv(input_filename).dropna(subset=['Ticker', 'Exchange'])
            stock_list = stock_list.drop_duplicates(subset=['Ticker'])
            print(f"✓ {input_filename} から {len(stock_list)} 銘柄を読み込みました")
            if stage2_only:
                print(" ※ Stage 2銘柄のみを分析します")
        else:
            stock_list = pd.read_csv('stock.csv').dropna(subset=['Ticker', 'Exchange'])
            stock_list = stock_list.drop_duplicates(subset=['Ticker'])
            print(f"✓ stock.csv から {len(stock_list)} 銘柄を読み込みました")
            if stage2_only:
                print(" ※ Stage 2銘柄のみを分析します")

    except FileNotFoundError:
        print(f"Error: {input_filename or 'stock.csv'} not found.")
        return

    # 分析対象の準備
    tickers_to_analyze = [
        (row['Ticker'], row['Exchange'], stage2_only)
        for _, row in stock_list.iterrows()
    ]

    # マルチプロセスで分析
    results = []
    print(f"\n銘柄分析を開始します（{cpu_count()}プロセス使用）...")

    with Pool(cpu_count()) as pool:
        for result in tqdm(
            pool.imap_unordered(analyze_single_ticker, tickers_to_analyze),
            total=len(tickers_to_analyze),
            desc="Analyzing"
        ):
            if result:
                results.append(result)

    # 結果の処理
    if not results:
        print("\n" + "=" * 70)
        print("ベースを持つ銘柄は見つかりませんでした。")
        print("=" * 70)
        return

    results_df = pd.DataFrame(results)

    # Base Countでソート（降順）
    results_df = results_df.sort_values('Base Count', ascending=False)

    # CSVファイルに保存
    results_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
    print(f"\n✓ {len(results_df)} 銘柄の分析結果を {output_filename} に保存しました")

    # TradingView用リストの生成（初期段階のベースのみ）
    early_stage_df = results_df[results_df['Base Stage'].isin(['初期段階', '中期段階'])]

    if not early_stage_df.empty:
        tradingview_list = [
            f"{row['Exchange']}:{row['Ticker']}"
            for _, row in early_stage_df.iterrows()
        ]
        tradingview_str = ",".join(tradingview_list)

        txt_output_filename = os.path.splitext(output_filename)[0] + ".txt"
        try:
            with open(txt_output_filename, 'w', encoding='utf-8') as f:
                f.write(tradingview_str)
            print(f"✓ TradingView用リスト（初期・中期段階）を {txt_output_filename} に保存しました")
            print(f"  対象銘柄数: {len(early_stage_df)}")
        except Exception as e:
            print(f"Warning: TradingView用ファイルの書き込み中にエラーが発生: {e}")

    # サマリー表示
    print("\n" + "=" * 70)
    print("分析結果サマリー")
    print("=" * 70)

    # ベース段階別の集計
    print("\n【ベース段階別の分布】")
    stage_counts = results_df['Base Stage'].value_counts()
    for stage, count in stage_counts.items():
        print(f"  {stage}: {count}銘柄")

    # ベース品質別の集計
    print("\n【ベース品質別の分布】")
    quality_counts = results_df['Quality'].value_counts()
    for quality, count in quality_counts.items():
        print(f"  {quality}: {count}銘柄")

    # VCP検証の集計
    vcp_counts = results_df['VCP Valid'].value_counts()
    print(f"\n【VCPパターン検証】")
    print(f"  VCP確認済み: {vcp_counts.get('Yes', 0)}銘柄")
    print(f"  VCP未確認: {vcp_counts.get('No', 0)}銘柄")

    # トップ10の表示
    print("\n【推奨銘柄 Top 10】（初期段階優先、ベースカウント少ない順）")
    print("-" * 70)

    # 初期段階でVCP確認済みの銘柄を優先
    top_candidates = results_df[
        (results_df['Base Stage'] == '初期段階') &
        (results_df['VCP Valid'] == 'Yes')
    ].head(10)

    if len(top_candidates) < 10:
        # 不足分は初期段階の銘柄で補充
        additional = results_df[
            results_df['Base Stage'] == '初期段階'
        ].head(10 - len(top_candidates))
        top_candidates = pd.concat([top_candidates, additional])

    if len(top_candidates) < 10:
        # さらに不足分は中期段階で補充
        additional = results_df[
            results_df['Base Stage'] == '中期段階'
        ].head(10 - len(top_candidates))
        top_candidates = pd.concat([top_candidates, additional])

    display_cols = [
        'Ticker', 'Exchange', 'Base Count', 'Base Stage', 'Quality',
        'VCP Valid', 'Status', 'Minervini Criteria Met', 'Assessment'
    ]

    display_cols = [col for col in display_cols if col in top_candidates.columns]

    if not top_candidates.empty:
        print(top_candidates[display_cols].to_string(index=False))
    else:
        print("  該当銘柄なし")

    print("\n" + "=" * 70)
    print("分析完了")
    print("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Mark Minerviniのベース理論に基づいたベース分析"
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='出力ファイル名（デフォルト: YYYYMMDD-base.csv）'
    )

    parser.add_argument(
        '-i', '--input',
        type=str,
        default=None,
        help='入力ファイル名（デフォルト: stock.csv）'
    )

    parser.add_argument(
        '--all-stages',
        action='store_true',
        help='全ステージの銘柄を分析（デフォルトはStage 2のみ）'
    )

    args = parser.parse_args()

    # 出力ファイル名の生成
    if args.output is None:
        tz = pytz.timezone('America/New_York')
        date_str = datetime.now(tz).strftime('%Y%m%d')
        output_filename = f"{date_str}-base.csv"
    else:
        output_filename = args.output

    # Stage 2フィルタリング
    stage2_only = not args.all_stages

    # 分析実行
    run_base_analysis(
        output_filename=output_filename,
        input_filename=args.input,
        stage2_only=stage2_only
    )
