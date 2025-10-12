"""
統合スクリーニングシステム
全銘柄を分析してStage 1/2候補を抽出
"""
import pandas as pd
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings('ignore')

from data_fetcher import fetch_stock_data
from indicators import calculate_all_basic_indicators
from rs_calculator import analyze_rs_metrics
from scoring_system import ScoringSystem


def analyze_single_ticker(args):
    """
    単一銘柄を分析（並列処理用）
    
    Args:
        args: (ticker, exchange, benchmark_df)
        
    Returns:
        dict or None: 分析結果
    """
    ticker, exchange, benchmark_df_dict = args
    
    try:
        # データ取得
        stock_df, _ = fetch_stock_data(ticker, period='2y')
        
        if stock_df is None or len(stock_df) < 252:
            return None
        
        # 指標計算
        indicators_df = calculate_all_basic_indicators(stock_df)
        
        if indicators_df.empty or len(indicators_df) < 252:
            return None
        
        # ベンチマークDataFrameを再構築
        benchmark_df = pd.DataFrame(benchmark_df_dict)
        benchmark_df.index = pd.to_datetime(benchmark_df.index)
        
        # RS Rating計算
        rs_result = analyze_rs_metrics(indicators_df, benchmark_df)
        indicators_df['RS_Rating'] = rs_result['rs_rating']
        
        indicators_df = indicators_df.dropna()
        
        if len(indicators_df) < 252:
            return None
        
        # スコアリングシステムで分析
        scorer = ScoringSystem(indicators_df, ticker, benchmark_df)
        result = scorer.comprehensive_analysis()
        
        # Stage 1または2の候補のみ
        stage = result.get('stage', 0)
        score = result.get('total_score', 0)
        
        # フィルター条件
        is_stage1_candidate = (stage == 1 and score >= 50)
        is_stage2 = (stage == 2)
        is_high_potential_stage1 = (stage == 1 and score >= 70)
        
        if is_stage1_candidate or is_stage2 or is_high_potential_stage1:
            # ステージ開始日の推定（簡易版）
            stage_start_date = estimate_stage_start_date(indicators_df, stage)
            
            # ATR Multiple（過熱感）
            atr_multiple = result['details']['atr']['atr_multiple_ma50']
            
            # 優先度設定
            if is_stage2:
                priority = 1
            elif is_high_potential_stage1:
                priority = 2
            else:
                priority = 3
            
            return {
                'Priority': priority,
                'Ticker': ticker,
                'Exchange': exchange,
                'Current Stage': f'ステージ{stage}',
                'Substage': result.get('substage', ''),
                'Stage Start Date': stage_start_date,
                'Score': score,
                'Grade': result.get('grade', ''),
                'Overheat': f"{atr_multiple:.2f}",
                'RS Rating': result['details']['rs']['rs_rating'],
                'Action': result.get('action', ''),
                # 詳細情報
                'Base Score': result['breakdown'].get('base_quality', 0) if 'breakdown' in result else 0,
                'VCP Detected': result['details'].get('vcp', {}).get('detected', False) if 'details' in result else False,
                'Volume Score': result['details'].get('volume', {}).get('total_score', 0) if 'details' in result else 0,
                'Wyckoff Phase': result['details'].get('volume', {}).get('wyckoff_phase', '') if 'details' in result else ''
            }
        
        return None
        
    except Exception as e:
        # エラーは静かに無視
        return None


def estimate_stage_start_date(df: pd.DataFrame, stage: int) -> str:
    """
    ステージ開始日を推定（簡易版）
    
    Args:
        df: 指標データ
        stage: ステージ番号
        
    Returns:
        str: 推定開始日
    """
    if stage == 1:
        # MA150が横ばいになった時点
        slope_threshold = 0.02
        for i in range(len(df)-1, max(0, len(df)-210), -1):
            if abs(df['SMA_150_Slope'].iloc[i]) >= slope_threshold:
                return df.index[i+1].strftime('%Y-%m-%d') if i+1 < len(df) else df.index[i].strftime('%Y-%m-%d')
        return df.index[0].strftime('%Y-%m-%d')
        
    elif stage == 2:
        # 50日高値を更新した日
        for i in range(len(df)-1, max(0, len(df)-60), -1):
            if i >= 50:
                high_50d = df['Close'].iloc[i-50:i].max()
                if df['Close'].iloc[i] > high_50d * 1.02:
                    return df.index[i].strftime('%Y-%m-%d')
        return df.index[-60].strftime('%Y-%m-%d') if len(df) >= 60 else df.index[0].strftime('%Y-%m-%d')
    
    return df.index[-100].strftime('%Y-%m-%d') if len(df) >= 100 else df.index[0].strftime('%Y-%m-%d')


def run_screener(use_parallel: bool = True, max_workers: int = None):
    """
    メインスクリーナーを実行
    
    Args:
        use_parallel: 並列処理を使用するか
        max_workers: 最大ワーカー数（Noneの場合はCPUコア数-1）
    """
    print("="*70)
    print("統合スクリーニングシステム - Stage 1/2候補の抽出")
    print("="*70)
    
    # 1. ティッカーリスト読み込み
    try:
        tickers_df = pd.read_csv('stock.csv', encoding='utf-8-sig')
        if 'Ticker' not in tickers_df.columns or 'Exchange' not in tickers_df.columns:
            print("エラー: stock.csv に 'Ticker' または 'Exchange' 列が見つかりません。")
            return
        
        tickers_list = tickers_df[['Ticker', 'Exchange']].values.tolist()
        print(f"✓ {len(tickers_list)}銘柄を読み込みました")
        
    except FileNotFoundError:
        print("エラー: stock.csvが見つかりません。")
        print("先に get_tickers.py を実行してください。")
        return
    except Exception as e:
        print(f"エラー: {e}")
        return
    
    # 2. ベンチマーク(SPY)データ取得
    print("\nベンチマークデータ(SPY)を取得中...")
    _, benchmark_df = fetch_stock_data('SPY', period='2y')
    
    if benchmark_df is None or benchmark_df.empty:
        print("致命的エラー: ベンチマークデータを取得できませんでした。")
        return
    
    benchmark_df = calculate_all_basic_indicators(benchmark_df)
    benchmark_df = benchmark_df.dropna()
    
    # DataFrameを辞書に変換（並列処理用）
    benchmark_df_dict = {
        'index': benchmark_df.index.astype(str).tolist(),
        **{col: benchmark_df[col].tolist() for col in benchmark_df.columns}
    }
    
    print(f"✓ ベンチマークデータ取得完了 ({len(benchmark_df)}日分)")
    
    # 3. 並列処理または逐次処理
    results = []
    
    # 分析用の引数リスト
    args_list = [(ticker, exchange, benchmark_df_dict) 
                 for ticker, exchange in tickers_list]
    
    if use_parallel:
        # 並列処理
        if max_workers is None:
            max_workers = max(1, cpu_count() - 1)
        
        print(f"\n並列処理で分析開始（{max_workers}ワーカー）...")
        
        with Pool(processes=max_workers) as pool:
            for result in tqdm(
                pool.imap_unordered(analyze_single_ticker, args_list),
                total=len(args_list),
                desc="銘柄分析中"
            ):
                if result is not None:
                    results.append(result)
    else:
        # 逐次処理
        print("\n逐次処理で分析開始...")
        for args in tqdm(args_list, desc="銘柄分析中"):
            result = analyze_single_ticker(args)
            if result is not None:
                results.append(result)
    
    # 4. 結果の整理と出力
    if not results:
        print("\n分析完了。条件に合う銘柄は見つかりませんでした。")
        return
    
    # DataFrameに変換
    df_results = pd.DataFrame(results)
    
    # ソート
    df_results = df_results.sort_values(
        ['Priority', 'Score'], 
        ascending=[True, False]
    )
    
    # Priority列を削除（出力用）
    output_df = df_results.drop('Priority', axis=1)
    
    # 5. CSV出力
    output_df.to_csv('stage1or2.csv', index=False, encoding='utf-8-sig')
    print(f"\n✓ {len(output_df)}件の結果を stage1or2.csv に出力しました")
    
    # 6. TradingView用TXT出力
    tradingview_list = [
        f"{row['Exchange']}:{row['Ticker']}" 
        for _, row in df_results.iterrows()
    ]
    tradingview_str = ",".join(tradingview_list)
    
    try:
        with open('stage1or2_tradingview.txt', 'w', encoding='utf-8') as f:
            f.write(tradingview_str)
        print(f"✓ TradingView用リストを stage1or2_tradingview.txt に保存しました")
    except Exception as e:
        print(f"エラー: TradingView用ファイルの書き込み失敗 - {e}")
    
    # 7. 統計情報の表示
    print("\n" + "="*70)
    print("スクリーニング結果サマリー")
    print("="*70)
    
    stage2_count = len(df_results[df_results['Current Stage'] == 'ステージ2'])
    stage1_count = len(df_results[df_results['Current Stage'] == 'ステージ1'])
    
    print(f"総抽出銘柄数: {len(df_results)}")
    print(f"  - ステージ2（上昇トレンド）: {stage2_count}銘柄")
    print(f"  - ステージ1（有望ベース）: {stage1_count}銘柄")
    
    # グレード別集計
    print("\nグレード別内訳:")
    grade_counts = df_results['Grade'].value_counts().sort_index()
    for grade, count in grade_counts.items():
        print(f"  {grade}評価: {count}銘柄")
    
    # トップ10表示
    print("\n" + "="*70)
    print("トップ10銘柄")
    print("="*70)
    
    top10 = output_df.head(10)
    for idx, row in top10.iterrows():
        print(f"\n{row['Ticker']} ({row['Exchange']})")
        print(f"  ステージ: {row['Current Stage']} - {row['Substage']}")
        print(f"  スコア: {row['Score']:.1f} ({row['Grade']})")
        print(f"  RS Rating: {row['RS Rating']:.0f}")
        print(f"  過熱度: {row['Overheat']}")
        print(f"  アクション: {row['Action']}")
    
    print("\n" + "="*70)
    print("スクリーニング完了！")
    print("="*70)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='統合スクリーニングシステム')
    parser.add_argument('--sequential', action='store_true', 
                       help='逐次処理を使用（デフォルトは並列処理）')
    parser.add_argument('--workers', type=int, default=None,
                       help='ワーカー数（デフォルト: CPUコア数-1）')
    
    args = parser.parse_args()
    
    run_screener(
        use_parallel=not args.sequential,
        max_workers=args.workers
    )
