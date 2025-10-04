import pandas as pd
from tqdm import tqdm
from data_fetcher import fetch_stock_data
from indicators import calculate_all_indicators
from stage_engine import StageAnalysisSystem

def find_current_stage_start_date(ticker, stock_indicators_df, benchmark_indicators_df, current_stage_num):
    """
    指定された銘柄の現在のステージがいつから始まったかを特定する。
    履歴を遡ってステージが変わる日を見つける。
    """
    # 過去252日（約1年）を遡る
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
                # ステージが変わった日の翌日が開始日
                start_date = stock_indicators_df.index[date_to_check_index + 1]
                return start_date.strftime('%Y-%m-%d')
        except (ValueError, IndexError):
            continue

    # 過去1年以内にステージ移行が見つからなければ、分析期間の初日を開始日とする
    return stock_indicators_df.index[0].strftime('%Y-%m-%d')

def analyze_stocks():
    """
    stock.csvの銘柄を分析し、条件に合うものをCSVに出力する。
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

            current_stage_num = int(current_stage_str.replace('ステージ', ''))

            # フィルタリング条件
            is_stage1_candidate = (current_stage_num == 1 and score >= 40)
            is_stage2 = (current_stage_num == 2)

            if is_stage1_candidate or is_stage2:
                # tqdm.write(f"候補銘柄: {ticker} (ステージ{current_stage_num}, スコア: {score})")
                start_date = find_current_stage_start_date(ticker, stock_indicators_df, benchmark_indicators_df, current_stage_num)

                results.append({
                    'Ticker': ticker,
                    'Current Stage': current_stage_str,
                    'Stage Start Date': start_date,
                    'Score': score,
                    'Judgment': transition_analysis.get('level', 'N/A'),
                    'Action': transition_analysis.get('action', 'N/A')
                })

        except Exception as e:
            # tqdm.write(f"エラー: {ticker}の分析中にエラーが発生しました: {e}")
            continue

    if results:
        df = pd.DataFrame(results)
        df.to_csv('stage1or2.csv', index=False, encoding='utf-8-sig')
        print(f"分析完了。{len(df)}件の結果をstage1or2.csvに出力しました。")
    else:
        print("分析完了。条件に合う銘柄は見つかりませんでした。")

if __name__ == '__main__':
    analyze_stocks()