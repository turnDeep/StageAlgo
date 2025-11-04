"""
Minervini Template Detector RS Rating検証スクリプト
指定された61銘柄について、RS Ratingの閾値を検証
"""
import pandas as pd
import numpy as np
from data_fetcher import fetch_stock_data
from indicators import calculate_all_basic_indicators
from rs_calculator import RSCalculator
from minervini_template_detector import MinerviniTemplateDetector
import time

# 検証対象の61銘柄（ユーザー指定）
TARGET_TICKERS = [
    'TNK', 'CYTK', 'PSTG', 'WCC', 'TTMI', 'MIR', 'KRYS', 'FSLR', 'SYM', 'NET',
    'SID', 'W', 'FN', 'PTGX', 'IMNM', 'MNMD', 'OSIS', 'ADPT', 'QS', 'KYMR',
    'LITE', 'CDTX', 'MDB', 'IDYA', 'PLTR', 'VSAT', 'CRDO', 'CNTA', 'ZBIO', 'NKTR',
    'NBIS', 'TGB', 'ACLX', 'WFRD', 'COGT', 'WULF', 'HUT', 'DQ', 'HOOD', 'RYTM',
    'SQM', 'TSLA', 'QCOM', 'NTGR', 'INCY', 'XPO', 'VSH', 'LASR', 'WGS', 'IVZ'
]


def analyze_stock(ticker: str, benchmark_df: pd.DataFrame) -> dict:
    """
    個別銘柄のMinervini基準とRS Ratingを分析

    Args:
        ticker: ティッカーシンボル
        benchmark_df: ベンチマーク（SPY）のデータ

    Returns:
        dict: 分析結果
    """
    result = {
        'ticker': ticker,
        'success': False,
        'rs_rating': None,
        'criteria_met': 0,
        'score': 0,
        'all_criteria_met': False,
        'checks': None,
        'error': None
    }

    try:
        # データ取得
        stock_df, _ = fetch_stock_data(ticker, period='2y')

        if stock_df is None or len(stock_df) < 200:
            result['error'] = 'データ不足'
            return result

        # 指標計算
        indicators_df = calculate_all_basic_indicators(stock_df)

        # RS Rating計算
        try:
            rs_calc = RSCalculator(indicators_df, benchmark_df)
            rs_score_series = rs_calc.calculate_ibd_rs_score()
            current_rs_score = rs_score_series.iloc[-1]
            rs_rating = rs_calc.calculate_percentile_rating(current_rs_score)

            indicators_df['RS_Rating'] = rs_rating
            result['rs_rating'] = rs_rating
        except Exception as e:
            result['error'] = f'RS Rating計算失敗: {str(e)}'
            return result

        indicators_df = indicators_df.dropna()

        if len(indicators_df) < 200:
            result['error'] = '計算後データ不足'
            return result

        # Minervini Template検出
        detector = MinerviniTemplateDetector(indicators_df)
        template_result = detector.check_template()

        result['success'] = True
        result['criteria_met'] = template_result['criteria_met']
        result['score'] = template_result['score']
        result['all_criteria_met'] = template_result['all_criteria_met']
        result['checks'] = template_result['checks']

    except Exception as e:
        result['error'] = str(e)

    return result


def main():
    """メイン検証プロセス"""
    print("="*80)
    print("Minervini Template Detector RS Rating検証")
    print("="*80)
    print(f"\n検証対象: {len(TARGET_TICKERS)}銘柄")
    print("期間: 過去2年")
    print("\n処理を開始します...\n")

    # ベンチマーク（SPY）取得
    print("ベンチマーク（SPY）データを取得中...")
    _, benchmark_df = fetch_stock_data('SPY', period='2y')

    if benchmark_df is None:
        print("❌ ベンチマークデータの取得に失敗しました")
        return

    benchmark_df = calculate_all_basic_indicators(benchmark_df)
    print(f"✓ ベンチマークデータ取得完了 ({len(benchmark_df)}日分)\n")

    # 全銘柄を分析
    results = []
    total = len(TARGET_TICKERS)

    for idx, ticker in enumerate(TARGET_TICKERS, 1):
        print(f"[{idx}/{total}] {ticker} を分析中...", end=' ')

        result = analyze_stock(ticker, benchmark_df)
        results.append(result)

        if result['success']:
            print(f"✓ RS:{result['rs_rating']:.1f} 基準:{result['criteria_met']}/8")
        else:
            print(f"❌ {result['error']}")

        # API制限対策（少し待機）
        if idx < total:
            time.sleep(0.5)

    # 結果を DataFrame に変換
    df_results = pd.DataFrame(results)

    # 成功した銘柄のみフィルタ
    df_success = df_results[df_results['success'] == True].copy()

    print(f"\n{'='*80}")
    print("分析結果サマリー")
    print(f"{'='*80}")
    print(f"総銘柄数: {total}")
    print(f"分析成功: {len(df_success)}銘柄")
    print(f"分析失敗: {len(df_results) - len(df_success)}銘柄")

    if len(df_success) > 0:
        print(f"\nRS Rating 統計:")
        print(f"  平均: {df_success['rs_rating'].mean():.1f}")
        print(f"  中央値: {df_success['rs_rating'].median():.1f}")
        print(f"  最小: {df_success['rs_rating'].min():.1f}")
        print(f"  最大: {df_success['rs_rating'].max():.1f}")

        print(f"\n基準満足数 統計:")
        print(f"  平均: {df_success['criteria_met'].mean():.1f}/8")
        print(f"  8基準すべて満たす: {(df_success['criteria_met'] == 8).sum()}銘柄")
        print(f"  7基準満たす: {(df_success['criteria_met'] == 7).sum()}銘柄")
        print(f"  6基準満たす: {(df_success['criteria_met'] == 6).sum()}銘柄")

        # RS Rating閾値別の銘柄数
        print(f"\n{'='*80}")
        print("RS Rating閾値別の銘柄数")
        print(f"{'='*80}")

        rs_thresholds = [50, 60, 70, 75, 80, 85, 90, 95]
        threshold_results = []

        for threshold in rs_thresholds:
            count = (df_success['rs_rating'] >= threshold).sum()
            threshold_results.append({
                'RS閾値': f'>= {threshold}',
                '銘柄数': count,
                '割合': f'{count/len(df_success)*100:.1f}%'
            })
            print(f"  RS Rating >= {threshold}: {count}銘柄 ({count/len(df_success)*100:.1f}%)")

        # 各基準の通過率
        print(f"\n{'='*80}")
        print("各基準の通過率")
        print(f"{'='*80}")

        for i in range(1, 9):
            criterion_key = f'criterion_{i}'
            passed_count = sum(
                1 for r in df_success.itertuples()
                if r.checks and criterion_key in r.checks and r.checks[criterion_key]['passed']
            )
            pass_rate = passed_count / len(df_success) * 100
            description = df_success.iloc[0]['checks'][criterion_key]['description'] if df_success.iloc[0]['checks'] else ''
            print(f"  基準{i}: {passed_count}/{len(df_success)} ({pass_rate:.1f}%) - {description}")

        # 詳細な結果を CSV に保存
        output_data = []
        for _, row in df_success.iterrows():
            output_row = {
                'Ticker': row['ticker'],
                'RS_Rating': row['rs_rating'],
                'Criteria_Met': row['criteria_met'],
                'Score': row['score'],
                'All_Criteria_Met': row['all_criteria_met']
            }

            # 各基準の結果を追加
            if row['checks']:
                for i in range(1, 9):
                    criterion_key = f'criterion_{i}'
                    output_row[f'C{i}'] = 1 if row['checks'][criterion_key]['passed'] else 0

            output_data.append(output_row)

        df_output = pd.DataFrame(output_data)
        df_output = df_output.sort_values('RS_Rating', ascending=False)

        output_file = 'minervini_verification_results.csv'
        df_output.to_csv(output_file, index=False)
        print(f"\n✓ 詳細結果を {output_file} に保存しました")

        # 上位銘柄（RS Rating順）
        print(f"\n{'='*80}")
        print("上位20銘柄（RS Rating順）")
        print(f"{'='*80}")
        print(df_output[['Ticker', 'RS_Rating', 'Criteria_Met', 'Score']].head(20).to_string(index=False))

        # 推奨RS Rating閾値
        print(f"\n{'='*80}")
        print("推奨RS Rating閾値")
        print(f"{'='*80}")

        # 61銘柄に絞るための閾値を計算
        if len(df_success) > 0:
            # 61銘柄に近い閾値を探す
            target_count = 61
            df_sorted = df_success.sort_values('rs_rating', ascending=False)

            if len(df_sorted) >= target_count:
                recommended_threshold = df_sorted.iloc[target_count - 1]['rs_rating']
                print(f"\n✓ {target_count}銘柄に絞るには、RS Rating >= {recommended_threshold:.1f} が推奨されます")
            else:
                print(f"\n⚠ 分析成功した銘柄数（{len(df_sorted)}）が目標（{target_count}）より少ないです")

            # 8基準すべて満たす銘柄に絞る場合
            df_all_criteria = df_success[df_success['all_criteria_met'] == True]
            if len(df_all_criteria) > 0:
                min_rs_all_criteria = df_all_criteria['rs_rating'].min()
                print(f"✓ 8基準すべて満たす銘柄は {len(df_all_criteria)}銘柄 (RS Rating >= {min_rs_all_criteria:.1f})")
            else:
                print("⚠ 8基準すべて満たす銘柄はありません")

            # 7基準以上満たす銘柄
            df_7plus = df_success[df_success['criteria_met'] >= 7]
            if len(df_7plus) > 0:
                min_rs_7plus = df_7plus['rs_rating'].min()
                print(f"✓ 7基準以上満たす銘柄は {len(df_7plus)}銘柄 (RS Rating >= {min_rs_7plus:.1f})")

    else:
        print("\n❌ 分析成功した銘柄がありません")

    print(f"\n{'='*80}")
    print("検証完了")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
