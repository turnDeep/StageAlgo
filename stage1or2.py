import pandas as pd
from tqdm import tqdm
from data_fetcher import fetch_stock_data
from indicators import calculate_all_indicators
from stage_engine import StageAnalysisSystem

def find_current_stage_start_date(ticker, stock_indicators_df, benchmark_indicators_df, current_stage_num):
    """
    指定された銘柄の現在のステージがいつから始まったかを特定する。
    """
    for i in range(1, len(stock_indicators_df)):
        date_to_check_index = -i -1
        if abs(date_to_check_index) > len(stock_indicators_df) or len(stock_indicators_df.iloc[:date_to_check_index]) < 252:
            break

        historical_stock_slice = stock_indicators_df.iloc[:date_to_check_index]
        historical_benchmark_slice = benchmark_indicators_df.loc[historical_stock_slice.index]

        try:
            analyzer = StageAnalysisSystem(historical_stock_slice, ticker, historical_benchmark_slice)
            stage = analyzer._determine_current_stage()
            if stage != current_stage_num:
                start_date = stock_indicators_df.index[date_to_check_index + 1]
                return start_date.strftime('%Y-%m-%d')
        except (ValueError, IndexError):
            continue

    return stock_indicators_df.index[0].strftime('%Y-%m-%d')

def analyze_stocks():
    """
    【改善版】stock.csvの銘柄を分析し、条件に合うものをCSVに出力する。
    - B-判定も含めるように修正
    - ステージ2の銘柄を確実に抽出
    """
    try:
        with open('stock.csv', 'r', encoding='utf-8-sig') as f:
            tickers = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print("エラー: stock.csvが見つかりません。")
        return

    print("ベンチマーク(SPY)のデータを取得中...")
    benchmark_df, _ = fetch_stock_data("SPY", benchmark_ticker="SPY")
    if benchmark_df is None or benchmark_df.empty:
        print("致命的エラー: ベンチマークデータを取得できませんでした。")
        return
    benchmark_indicators_df = calculate_all_indicators(benchmark_df, benchmark_df.copy())

    results = []

    print(f"{len(tickers)}銘柄の分析を開始します...")
    for ticker in tqdm(tickers, desc="銘柄分析中"):
        stock_df, _ = fetch_stock_data(ticker, benchmark_ticker="SPY")
        if stock_df is None or stock_df.empty or len(stock_df) < 252:
            continue

        try:
            stock_indicators_df = calculate_all_indicators(stock_df, benchmark_df)
            if stock_indicators_df.empty:
                continue

            analyzer = StageAnalysisSystem(stock_indicators_df, ticker, benchmark_indicators_df)
            analysis_result = analyzer.analyze()

            current_stage_str = analysis_result.get('current_stage', '')
            transition_analysis = analysis_result.get('transition_analysis', {})
            score = transition_analysis.get('score', 0)
            level = transition_analysis.get('level', '')

            current_stage_num = int(current_stage_str.replace('ステージ', ''))

            # 【修正】フィルタリング条件を改善
            # ステージ1: スコア40点以上
            is_stage1_candidate = (current_stage_num == 1 and score >= 40)
            
            # ステージ2: 全て含める
            is_stage2 = (current_stage_num == 2)
            
            # 【追加】ステージ1でも高スコア（70点以上）またはB-判定の銘柄を優先抽出
            is_high_potential_stage1 = (
                current_stage_num == 1 and 
                (score >= 70 or 'B-' in level or 'B判定' in level)
            )

            if is_stage1_candidate or is_stage2 or is_high_potential_stage1:
                start_date = find_current_stage_start_date(
                    ticker, stock_indicators_df, benchmark_indicators_df, current_stage_num
                )

                # 優先度を設定（ソート用）
                if is_stage2:
                    priority = 1
                elif is_high_potential_stage1:
                    priority = 2
                else:
                    priority = 3

                results.append({
                    'Priority': priority,  # 内部的なソート用
                    'Ticker': ticker,
                    'Current Stage': current_stage_str,
                    'Stage Start Date': start_date,
                    'Score': score,
                    'Judgment': level,
                    'Action': transition_analysis.get('action', 'N/A')
                })

        except Exception as e:
            # エラーは静かにスキップ（tqdmの表示を乱さない）
            continue

    if results:
        df = pd.DataFrame(results)
        # 優先度でソート（ステージ2が最上位）
        df = df.sort_values(['Priority', 'Score'], ascending=[True, False])
        # Priority列は出力から除外
        df = df.drop('Priority', axis=1)
        
        df.to_csv('stage1or2.csv', index=False, encoding='utf-8-sig')
        
        # 統計情報を表示
        stage2_count = len(df[df['Current Stage'] == 'ステージ2'])
        stage1_count = len(df[df['Current Stage'] == 'ステージ1'])
        
        print(f"\n分析完了。{len(df)}件の結果をstage1or2.csvに出力しました。")
        print(f"  - ステージ2: {stage2_count}件")
        print(f"  - ステージ1（有望株）: {stage1_count}件")
    else:
        print("分析完了。条件に合う銘柄は見つかりませんでした。")

if __name__ == '__main__':
    analyze_stocks()