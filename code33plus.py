"""
コード33 + 赤字転換パターン検出スクリプト（アーニングスサプライズ対応版）

ミネルヴィニのコード33と、赤字から黒字への転換パターン①〜④を検出します。

【コード33】
- EPS、売上高、純利益率が3四半期連続で加速
- アーニングスサプライズが継続的にポジティブ（実績が予測を上回る）

【赤字転換パターン】
- パターン①: 赤→黒→黒→黒（最も信頼性の高いターンアラウンド）
- パターン②: 赤→赤→黒→黒（黒字転換初期の加速）
- パターン③: 赤→赤→赤→黒/赤（赤字縮小が加速中）
- パターン④: 赤→赤→赤→赤（黒字化直前の可能性）

【アーニングスサプライズの重要性】
機関投資家は予測を上回る決算に強く反応し、株価の急騰を引き起こします。
コード33＋ポジティブサプライズの組み合わせは最強のシグナルです。
"""

import yfinance as yf
import pandas as pd
import numpy as np
from tqdm import tqdm
from curl_cffi.requests import Session
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


def fetch_financial_data(ticker: str) -> Optional[Dict]:
    """
    指定ティッカーの財務データを取得
    
    Returns:
        Dict containing:
        - quarterly_earnings: 四半期EPS
        - quarterly_income: 四半期損益計算書
        - quarterly_cashflow: 四半期CF計算書
        - earnings_dates: 決算日と予測EPS（アーニングスサプライズ用）
        - calendar: 次の決算予定と予測
    """
    try:
        session = Session(impersonate="chrome110")
        stock = yf.Ticker(ticker, session=session)
        
        # 四半期データを取得
        qtr_earnings = stock.quarterly_earnings
        qtr_income = stock.quarterly_income_stmt
        qtr_cashflow = stock.quarterly_cashflow
        
        if qtr_earnings is None or qtr_earnings.empty:
            return None
        
        # 予測EPSとアーニングスサプライズデータを取得
        earnings_dates = None
        calendar = None
        try:
            earnings_dates = stock.earnings_dates
            calendar = stock.calendar
        except:
            pass  # 予測データが取得できなくても続行
            
        return {
            'quarterly_earnings': qtr_earnings,
            'quarterly_income': qtr_income,
            'quarterly_cashflow': qtr_cashflow,
            'earnings_dates': earnings_dates,
            'calendar': calendar
        }
    except Exception as e:
        return None


def calculate_growth_rate(current: float, previous: float) -> float:
    """成長率を計算（前期がゼロまたは符号変化の場合は特別処理）"""
    if pd.isna(current) or pd.isna(previous):
        return np.nan
    
    # 両方とも負の場合（赤字幅の縮小率）
    if current < 0 and previous < 0:
        if abs(previous) < 1e-6:
            return 0.0
        return (abs(previous) - abs(current)) / abs(previous) * 100
    
    # 赤字→黒字の転換
    if previous < 0 and current > 0:
        return 999.0  # 特別な値（黒字転換）
    
    # 黒字→赤字の転落
    if previous > 0 and current < 0:
        return -999.0  # 特別な値（赤字転落）
    
    # 通常の成長率計算
    if abs(previous) < 1e-6:
        return 0.0 if abs(current) < 1e-6 else 100.0
    
    return (current - previous) / abs(previous) * 100


def analyze_earnings_surprises(financial_data: Dict) -> Dict:
    """
    アーニングスサプライズ（実績 vs 予測）を分析
    
    Returns:
        Dict containing:
        - recent_surprises: 直近4四半期のサプライズ率
        - avg_surprise: 平均サプライズ率
        - positive_surprises: ポジティブサプライズの回数
        - surprise_trend: サプライズが拡大傾向か
    """
    result = {
        'recent_surprises': [],
        'avg_surprise': 0.0,
        'positive_surprises': 0,
        'surprise_trend': False,
        'has_data': False
    }
    
    try:
        earnings_dates = financial_data.get('earnings_dates')
        if earnings_dates is None or earnings_dates.empty:
            return result
        
        # 'EPS Estimate'と'Reported EPS'列をチェック
        if 'EPS Estimate' not in earnings_dates.columns or 'Reported EPS' not in earnings_dates.columns:
            return result
        
        # 実績が報告されている最新4件を取得
        reported = earnings_dates.dropna(subset=['Reported EPS']).head(4)
        
        if len(reported) < 2:
            return result
        
        surprises = []
        for idx, row in reported.iterrows():
            estimate = row['EPS Estimate']
            actual = row['Reported EPS']
            
            if pd.notna(estimate) and pd.notna(actual) and estimate != 0:
                surprise_pct = ((actual - estimate) / abs(estimate)) * 100
                surprises.append(surprise_pct)
        
        if len(surprises) >= 2:
            result['has_data'] = True
            result['recent_surprises'] = surprises
            result['avg_surprise'] = np.mean(surprises)
            result['positive_surprises'] = sum(1 for s in surprises if s > 0)
            
            # サプライズが拡大傾向か（最新が平均より大きい）
            if len(surprises) >= 3:
                result['surprise_trend'] = surprises[0] > np.mean(surprises[1:])
        
        return result
        
    except Exception as e:
        return result


def get_next_earnings_estimate(financial_data: Dict) -> Optional[float]:
    """
    次の四半期の予測EPSを取得
    """
    try:
        calendar = financial_data.get('calendar')
        if calendar is not None and 'Earnings Estimate' in calendar:
            estimate = calendar.loc['Earnings Estimate', 0] if hasattr(calendar, 'loc') else None
            if estimate is not None and not pd.isna(estimate):
                return float(estimate)
    except:
        pass
    
    return None


def extract_metrics(financial_data: Dict) -> Optional[pd.DataFrame]:
    """
    財務データから必要な指標を抽出し、直近4四半期のDataFrameを作成
    
    Returns:
        DataFrame with columns: [EPS, Revenue, NetIncome, OperatingCF, NetMargin]
        Index: Latest(最新), Latest-1, Latest-2, Latest-3(最古)
    """
    try:
        qtr_earnings = financial_data['quarterly_earnings']
        qtr_income = financial_data['quarterly_income']
        qtr_cashflow = financial_data['quarterly_cashflow']
        
        # 最新4四半期を取得
        if len(qtr_earnings) < 4:
            return None
        
        # EPSを取得
        eps_data = qtr_earnings['Reported EPS'].head(4) if 'Reported EPS' in qtr_earnings.columns else None
        if eps_data is None or len(eps_data) < 4:
            return None
        
        # 収益・純利益を取得
        revenue = qtr_income.loc['Total Revenue'].head(4) if 'Total Revenue' in qtr_income.index else None
        net_income = qtr_income.loc['Net Income'].head(4) if 'Net Income' in qtr_income.index else None
        
        if revenue is None or net_income is None:
            return None
        
        # 営業CFを取得
        operating_cf = None
        if qtr_cashflow is not None and not qtr_cashflow.empty:
            if 'Operating Cash Flow' in qtr_cashflow.index:
                operating_cf = qtr_cashflow.loc['Operating Cash Flow'].head(4)
            elif 'Cash Flow From Operating Activities' in qtr_cashflow.index:
                operating_cf = qtr_cashflow.loc['Cash Flow From Operating Activities'].head(4)
        
        # 純利益率を計算
        net_margin = (net_income / revenue * 100)
        
        # DataFrameを構築（新しい順に: Latest, Latest-1, Latest-2, Latest-3）
        df = pd.DataFrame({
            'EPS': eps_data.values,
            'Revenue': revenue.values,
            'NetIncome': net_income.values,
            'NetMargin': net_margin.values,
            'OperatingCF': operating_cf.values if operating_cf is not None else [np.nan] * 4
        }, index=['Latest', 'Latest-1', 'Latest-2', 'Latest-3'])
        
        return df
        
    except Exception as e:
        return None


def check_code33(metrics_df: pd.DataFrame, financial_data: Dict) -> Tuple[bool, Dict]:
    """
    コード33の条件をチェック（アーニングスサプライズを含む）
    - EPS成長率が3期連続加速
    - 売上高成長率が3期連続加速
    - 純利益率が3期連続上昇
    - 【追加】アーニングスサプライズが継続的にポジティブ
    
    Returns:
        (is_code33, details)
    """
    details = {
        'eps_acceleration': False,
        'revenue_acceleration': False,
        'margin_improvement': False,
        'earnings_surprises': False,
        'eps_growth_rates': [],
        'revenue_growth_rates': [],
        'margins': [],
        'surprise_data': {}
    }
    
    try:
        # EPS成長率（YoY）を計算
        eps_growth = []
        for i in range(3):
            rate = calculate_growth_rate(metrics_df['EPS'].iloc[i], metrics_df['EPS'].iloc[i+1])
            eps_growth.append(rate)
        details['eps_growth_rates'] = eps_growth
        
        # 売上高成長率（YoY）を計算
        rev_growth = []
        for i in range(3):
            rate = calculate_growth_rate(metrics_df['Revenue'].iloc[i], metrics_df['Revenue'].iloc[i+1])
            rev_growth.append(rate)
        details['revenue_growth_rates'] = rev_growth
        
        # 純利益率
        margins = metrics_df['NetMargin'].head(4).tolist()
        details['margins'] = margins
        
        # 加速判定（3期連続で成長率が上昇）
        if not any(np.isnan(eps_growth)):
            # 赤字転換は特別処理（999.0を除外）
            valid_eps = [g for g in eps_growth if abs(g) < 500]
            if len(valid_eps) == 3:
                details['eps_acceleration'] = valid_eps[0] > valid_eps[1] and valid_eps[1] > valid_eps[2]
        
        if not any(np.isnan(rev_growth)):
            valid_rev = [g for g in rev_growth if abs(g) < 500]
            if len(valid_rev) == 3:
                details['revenue_acceleration'] = valid_rev[0] > valid_rev[1] and valid_rev[1] > valid_rev[2]
        
        if not any(np.isnan(margins)):
            # 純利益率が3期連続上昇
            details['margin_improvement'] = margins[0] > margins[1] and margins[1] > margins[2]
        
        # 【追加】アーニングスサプライズの分析
        surprise_analysis = analyze_earnings_surprises(financial_data)
        details['surprise_data'] = surprise_analysis
        
        if surprise_analysis['has_data']:
            # 直近2回以上ポジティブサプライズ、かつ平均が5%以上
            details['earnings_surprises'] = (
                surprise_analysis['positive_surprises'] >= 2 and
                surprise_analysis['avg_surprise'] > 5.0
            )
        
        # コード33判定（アーニングスサプライズは必須ではないが、あれば高評価）
        is_code33_core = (details['eps_acceleration'] and 
                          details['revenue_acceleration'] and 
                          details['margin_improvement'])
        
        # アーニングスサプライズがあればさらに強力
        is_code33_enhanced = is_code33_core and details['earnings_surprises']
        
        details['is_enhanced'] = is_code33_enhanced
        
        return (is_code33_core or is_code33_enhanced), details
        
    except Exception as e:
        return False, details


def check_pattern1(metrics_df: pd.DataFrame) -> Tuple[bool, Dict]:
    """
    パターン①: 赤→黒→黒→黒
    判定条件：黒字3期（Latest-2, Latest-1, Latest）の成長率が連続で増加
    """
    details = {'pattern': 'Pattern1', 'description': '赤→黒→黒→黒', 'match': False}
    
    try:
        eps = metrics_df['EPS'].values  # [Latest, Latest-1, Latest-2, Latest-3]
        
        # Latest-3が赤字、Latest-2/Latest-1/Latestが黒字
        if eps[3] < 0 and eps[2] > 0 and eps[1] > 0 and eps[0] > 0:
            # 黒字3期の成長率を計算
            g1 = calculate_growth_rate(eps[2], eps[3])  # Latest-2 vs Latest-3（黒字転換）
            g2 = calculate_growth_rate(eps[1], eps[2])  # Latest-1 vs Latest-2
            g3 = calculate_growth_rate(eps[0], eps[1])  # Latest vs Latest-1
            
            # 成長率が加速しているか
            if g2 > 0 and g3 > 0 and g3 > g2:
                details['match'] = True
                details['growth_rates'] = [g3, g2, g1]
                
        return details['match'], details
        
    except Exception as e:
        return False, details


def check_pattern2(metrics_df: pd.DataFrame) -> Tuple[bool, Dict]:
    """
    パターン②: 赤→赤→黒→黒
    判定条件：
    1. 赤字2期（Latest-3, Latest-2）の赤字幅が縮小
    2. 黒字2期（Latest-1, Latest）の黒字幅が増大
    """
    details = {'pattern': 'Pattern2', 'description': '赤→赤→黒→黒', 'match': False}
    
    try:
        eps = metrics_df['EPS'].values  # [Latest, Latest-1, Latest-2, Latest-3]
        
        # Latest-3/Latest-2が赤字、Latest-1/Latestが黒字
        if eps[3] < 0 and eps[2] < 0 and eps[1] > 0 and eps[0] > 0:
            # 条件1: 赤字幅縮小
            loss_reduction = abs(eps[2]) < abs(eps[3])
            
            # 条件2: 黒字幅増大
            profit_growth = eps[0] > eps[1]
            
            if loss_reduction and profit_growth:
                details['match'] = True
                details['loss_reduction_rate'] = calculate_growth_rate(eps[2], eps[3])
                details['profit_growth_rate'] = calculate_growth_rate(eps[0], eps[1])
                
        return details['match'], details
        
    except Exception as e:
        return False, details


def check_pattern3(metrics_df: pd.DataFrame) -> Tuple[bool, Dict]:
    """
    パターン③: 赤→赤→赤→黒/赤
    判定条件：赤字3期（Latest-3, Latest-2, Latest-1）の赤字幅縮小率が連続で加速
    """
    details = {'pattern': 'Pattern3', 'description': '赤→赤→赤→黒/赤', 'match': False}
    
    try:
        eps = metrics_df['EPS'].values  # [Latest, Latest-1, Latest-2, Latest-3]
        
        # Latest-3/Latest-2/Latest-1が赤字
        if eps[3] < 0 and eps[2] < 0 and eps[1] < 0:
            # 赤字幅縮小率を計算
            r1 = calculate_growth_rate(eps[2], eps[3])  # Latest-2 vs Latest-3（縮小率）
            r2 = calculate_growth_rate(eps[1], eps[2])  # Latest-1 vs Latest-2（縮小率）
            
            # 縮小率が加速（r2 > r1 > 0）
            if r1 > 0 and r2 > 0 and r2 > r1:
                details['match'] = True
                details['reduction_rates'] = [r2, r1]
                details['latest_status'] = 'Black' if eps[0] > 0 else 'Red'
                
        return details['match'], details
        
    except Exception as e:
        return False, details


def check_pattern4(metrics_df: pd.DataFrame) -> Tuple[bool, Dict]:
    """
    パターン④: 赤→赤→赤→赤（黒字化に接近）
    判定条件：赤字幅縮小率が連続で加速
    """
    details = {'pattern': 'Pattern4', 'description': '赤→赤→赤→赤', 'match': False}
    
    try:
        eps = metrics_df['EPS'].values  # [Latest, Latest-1, Latest-2, Latest-3]
        
        # 全て赤字
        if all(eps < 0):
            # 赤字幅縮小率を計算
            r1 = calculate_growth_rate(eps[2], eps[3])  # Latest-2 vs Latest-3
            r2 = calculate_growth_rate(eps[1], eps[2])  # Latest-1 vs Latest-2
            r3 = calculate_growth_rate(eps[0], eps[1])  # Latest vs Latest-1
            
            # 縮小率が連続加速（r3 > r2 > r1 > 0）
            if r1 > 0 and r2 > 0 and r3 > 0 and r3 > r2 and r2 > r1:
                details['match'] = True
                details['reduction_rates'] = [r3, r2, r1]
                details['approaching_breakeven'] = abs(eps[0]) < abs(eps[1]) * 0.5
                
        return details['match'], details
        
    except Exception as e:
        return False, details


def analyze_supplementary_factors(metrics_df: pd.DataFrame) -> Dict:
    """
    補足的な判定要素を分析
    1. 売上高成長トレンド
    2. 営業CF状況
    3. 粗利率改善（純利益率で代替）
    """
    factors = {
        'revenue_growing': False,
        'operating_cf_positive': False,
        'margin_improving': False
    }
    
    try:
        # 売上高が伸びているか
        rev = metrics_df['Revenue'].values
        if rev[0] > rev[3]:  # Latest > Latest-3
            factors['revenue_growing'] = True
        
        # 営業CFがプラスか
        cf = metrics_df['OperatingCF'].values
        if not np.isnan(cf[0]) and cf[0] > 0:
            factors['operating_cf_positive'] = True
        
        # 純利益率が改善しているか
        margin = metrics_df['NetMargin'].values
        if margin[0] > margin[3]:  # Latest > Latest-3
            factors['margin_improving'] = True
            
        return factors
        
    except Exception as e:
        return factors


def analyze_ticker(ticker: str) -> Optional[Dict]:
    """
    個別ティッカーの分析を実行
    """
    # 財務データ取得
    financial_data = fetch_financial_data(ticker)
    if financial_data is None:
        return None
    
    # 指標抽出
    metrics_df = extract_metrics(financial_data)
    if metrics_df is None:
        return None
    
    # コード33チェック（アーニングスサプライズ込み）
    is_code33, code33_details = check_code33(metrics_df, financial_data)
    
    # 次期予測EPSを取得
    next_eps_estimate = get_next_earnings_estimate(financial_data)
    
    # パターンチェック
    pattern1_match, pattern1_details = check_pattern1(metrics_df)
    pattern2_match, pattern2_details = check_pattern2(metrics_df)
    pattern3_match, pattern3_details = check_pattern3(metrics_df)
    pattern4_match, pattern4_details = check_pattern4(metrics_df)
    
    # 補足要素
    supplementary = analyze_supplementary_factors(metrics_df)
    
    # いずれかにマッチしたら結果を返す
    if (is_code33 or pattern1_match or pattern2_match or 
        pattern3_match or pattern4_match):
        
        result = {
            'ticker': ticker,
            'code33': is_code33,
            'code33_enhanced': code33_details.get('is_enhanced', False),
            'pattern1': pattern1_match,
            'pattern2': pattern2_match,
            'pattern3': pattern3_match,
            'pattern4': pattern4_match,
            'eps_latest': metrics_df['EPS'].iloc[0],
            'eps_latest_1': metrics_df['EPS'].iloc[1],
            'eps_latest_2': metrics_df['EPS'].iloc[2],
            'eps_latest_3': metrics_df['EPS'].iloc[3],
            'next_eps_estimate': next_eps_estimate,
            'avg_surprise': code33_details['surprise_data'].get('avg_surprise', 0.0),
            'positive_surprises': code33_details['surprise_data'].get('positive_surprises', 0),
            'revenue_growth': supplementary['revenue_growing'],
            'cf_positive': supplementary['operating_cf_positive'],
            'margin_improving': supplementary['margin_improving'],
            'details': {
                'code33': code33_details,
                'pattern1': pattern1_details,
                'pattern2': pattern2_details,
                'pattern3': pattern3_details,
                'pattern4': pattern4_details
            }
        }
        
        return result
    
    return None


def main():
    """
    メイン処理：stock.csvから銘柄を読み込み、分析を実行
    """
    print("=" * 60)
    print("コード33 + 赤字転換パターン検出スクリプト")
    print("=" * 60)
    
    # ティッカーリスト読み込み
    try:
        with open('stock.csv', 'r', encoding='utf-8-sig') as f:
            tickers = [line.strip() for line in f if line.strip()]
        print(f"\n{len(tickers)}銘柄を読み込みました。")
    except FileNotFoundError:
        print("エラー: stock.csvが見つかりません。")
        return
    
    # 分析実行
    results = []
    print("\n分析を開始します...\n")
    
    for ticker in tqdm(tickers, desc="銘柄分析中"):
        result = analyze_ticker(ticker)
        if result:
            results.append(result)
    
    # 結果をDataFrameに変換
    if not results:
        print("\n条件に合致する銘柄は見つかりませんでした。")
        return
    
    df = pd.DataFrame(results)
    
    # パターン列を作成
    def get_pattern_str(row):
        patterns = []
        if row['code33_enhanced']:
            patterns.append('Code33★')  # 強化版
        elif row['code33']:
            patterns.append('Code33')
        if row['pattern1']:
            patterns.append('Pattern1')
        if row['pattern2']:
            patterns.append('Pattern2')
        if row['pattern3']:
            patterns.append('Pattern3')
        if row['pattern4']:
            patterns.append('Pattern4')
        return ', '.join(patterns)
    
    df['Patterns'] = df.apply(get_pattern_str, axis=1)
    
    # スコアリング（優先順位）
    def calculate_priority(row):
        score = 0
        if row['code33_enhanced']:
            score += 150  # アーニングスサプライズ付きCode33は最高評価
        elif row['code33']:
            score += 100
        if row['pattern1']:
            score += 80
        if row['pattern2']:
            score += 60
        if row['pattern3']:
            score += 40
        if row['pattern4']:
            score += 20
        
        # アーニングスサプライズのボーナス
        if row['avg_surprise'] > 10:
            score += 20
        elif row['avg_surprise'] > 5:
            score += 10
        
        if row['revenue_growth']:
            score += 10
        if row['cf_positive']:
            score += 10
        if row['margin_improving']:
            score += 10
        return score
    
    df['Priority'] = df.apply(calculate_priority, axis=1)
    
    # ソート
    df = df.sort_values('Priority', ascending=False)
    
    # CSV出力用に整形
    output_df = df[[
        'ticker', 'Patterns', 'Priority',
        'eps_latest', 'eps_latest_1', 'eps_latest_2', 'eps_latest_3',
        'next_eps_estimate', 'avg_surprise', 'positive_surprises',
        'revenue_growth', 'cf_positive', 'margin_improving'
    ]].copy()
    
    output_df.columns = [
        'Ticker', 'Detected_Patterns', 'Priority_Score',
        'EPS_Latest', 'EPS_Latest-1', 'EPS_Latest-2', 'EPS_Latest-3',
        'Next_EPS_Est', 'Avg_Surprise%', 'Positive_Surprises',
        'Revenue_Growing', 'CF_Positive', 'Margin_Improving'
    ]
    
    # CSV保存
    output_df.to_csv('code33plus_results.csv', index=False, encoding='utf-8-sig')
    
    # 統計情報表示
    print(f"\n{'=' * 60}")
    print(f"分析完了！ {len(df)}銘柄が条件に合致しました。")
    print(f"{'=' * 60}")
    print(f"Code33該当: {df['code33'].sum()}銘柄")
    print(f"  └ アーニングスサプライズ付き: {df['code33_enhanced'].sum()}銘柄")
    print(f"Pattern1該当: {df['pattern1'].sum()}銘柄")
    print(f"Pattern2該当: {df['pattern2'].sum()}銘柄")
    print(f"Pattern3該当: {df['pattern3'].sum()}銘柄")
    print(f"Pattern4該当: {df['pattern4'].sum()}銘柄")
    print(f"\n結果をcode33plus_results.csvに保存しました。")
    
    # Top 10表示
    print(f"\n{'=' * 60}")
    print("Top 10 優先銘柄:")
    print(f"{'=' * 60}")
    print(output_df.head(10).to_string(index=False))


if __name__ == '__main__':
    main()
