"""
コード33 + 赤字転換パターン検出スクリプト（yfinance修正版）

【重要】yfinanceの仕様変更に対応した修正版
- quarterly_earningsは非推奨のため、quarterly_income_stmtを使用
- EPSはNet Incomeと発行済株式数から計算
- より堅牢なエラーハンドリングを実装

【コード33】
- EPS、売上高、純利益率が3四半期連続で加速
- アーニングスサプライズが継続的にポジティブ

【赤字転換パターン】
- パターン①: 赤→黒→黒→黒
- パターン②: 赤→赤→黒→黒
- パターン③: 赤→赤→赤→黒/赤
- パターン④: 赤→赤→赤→赤
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
    指定ティッカーの財務データを取得（修正版）
    """
    try:
        session = Session(impersonate="chrome110")
        stock = yf.Ticker(ticker, session=session)
        
        # 四半期損益計算書を取得（これが主要データソース）
        qtr_income = stock.quarterly_income_stmt
        qtr_cashflow = stock.quarterly_cashflow
        
        # データが空でないかチェック
        if qtr_income is None or qtr_income.empty:
            return None
        
        # 最低限必要な列が存在するかチェック
        required_fields = ['Total Revenue', 'Net Income']
        if not all(field in qtr_income.index for field in required_fields):
            return None
        
        # アーニングス関連データ（オプション）
        earnings_dates = None
        info = None
        try:
            earnings_dates = stock.get_earnings_dates(limit=8)  # 直近8期分
            info = stock.info  # 発行済株式数など
        except:
            pass
            
        return {
            'quarterly_income': qtr_income,
            'quarterly_cashflow': qtr_cashflow,
            'earnings_dates': earnings_dates,
            'info': info
        }
    except Exception as e:
        return None


def calculate_eps_from_income(quarterly_income: pd.DataFrame, info: Optional[Dict]) -> Optional[pd.Series]:
    """
    Net Incomeと発行済株式数からEPSを計算
    """
    try:
        # Net Incomeを取得
        if 'Net Income' not in quarterly_income.index:
            return None
        
        net_income = quarterly_income.loc['Net Income']
        
        # 発行済株式数を取得
        shares_outstanding = None
        if info and 'sharesOutstanding' in info:
            shares_outstanding = info['sharesOutstanding']
        
        if shares_outstanding and shares_outstanding > 0:
            # EPSを計算: Net Income / Shares Outstanding
            eps = net_income / shares_outstanding
            return eps
        else:
            # 発行済株式数が取得できない場合はNet Incomeをそのまま使用
            # （比較可能性は保たれる）
            return net_income
            
    except Exception as e:
        return None


def extract_metrics(financial_data: Dict) -> Optional[pd.DataFrame]:
    """
    財務データから必要な指標を抽出（修正版）
    """
    try:
        qtr_income = financial_data['quarterly_income']
        qtr_cashflow = financial_data['quarterly_cashflow']
        info = financial_data.get('info')
        
        # 最新4四半期を取得
        if len(qtr_income.columns) < 4:
            return None
        
        # EPSを計算
        eps_series = calculate_eps_from_income(qtr_income, info)
        if eps_series is None or len(eps_series) < 4:
            return None
        
        # 収益と純利益を取得
        revenue = qtr_income.loc['Total Revenue'].head(4)
        net_income = qtr_income.loc['Net Income'].head(4)
        
        # 純利益率を計算
        net_margin = (net_income / revenue * 100)
        
        # 営業CFを取得（オプション）
        operating_cf = None
        if qtr_cashflow is not None and not qtr_cashflow.empty:
            cf_keys = ['Operating Cash Flow', 'Total Cash From Operating Activities']
            for key in cf_keys:
                if key in qtr_cashflow.index:
                    operating_cf = qtr_cashflow.loc[key].head(4)
                    break
        
        # DataFrameを構築
        df = pd.DataFrame({
            'EPS': eps_series.head(4).values,
            'Revenue': revenue.values,
            'NetIncome': net_income.values,
            'NetMargin': net_margin.values,
            'OperatingCF': operating_cf.values if operating_cf is not None else [np.nan] * 4
        }, index=['Latest', 'Latest-1', 'Latest-2', 'Latest-3'])
        
        return df
        
    except Exception as e:
        return None


def calculate_growth_rate(current: float, previous: float) -> float:
    """成長率を計算"""
    if pd.isna(current) or pd.isna(previous):
        return np.nan
    
    if current < 0 and previous < 0:
        if abs(previous) < 1e-6:
            return 0.0
        return (abs(previous) - abs(current)) / abs(previous) * 100
    
    if previous < 0 and current > 0:
        return 999.0
    
    if previous > 0 and current < 0:
        return -999.0
    
    if abs(previous) < 1e-6:
        return 0.0 if abs(current) < 1e-6 else 100.0
    
    return (current - previous) / abs(previous) * 100


def analyze_earnings_surprises(financial_data: Dict) -> Dict:
    """
    アーニングスサプライズを分析（修正版）
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
        
        if 'EPS Estimate' not in earnings_dates.columns or 'Reported EPS' not in earnings_dates.columns:
            return result
        
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
            
            if len(surprises) >= 3:
                result['surprise_trend'] = surprises[0] > np.mean(surprises[1:])
        
        return result
        
    except Exception as e:
        return result


def check_code33(metrics_df: pd.DataFrame, financial_data: Dict) -> Tuple[bool, Dict]:
    """
    コード33の条件をチェック
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
        # EPS成長率を計算
        eps_growth = []
        for i in range(3):
            rate = calculate_growth_rate(metrics_df['EPS'].iloc[i], metrics_df['EPS'].iloc[i+1])
            eps_growth.append(rate)
        details['eps_growth_rates'] = eps_growth
        
        # 売上高成長率を計算
        rev_growth = []
        for i in range(3):
            rate = calculate_growth_rate(metrics_df['Revenue'].iloc[i], metrics_df['Revenue'].iloc[i+1])
            rev_growth.append(rate)
        details['revenue_growth_rates'] = rev_growth
        
        # 純利益率
        margins = metrics_df['NetMargin'].head(4).tolist()
        details['margins'] = margins
        
        # 加速判定
        if not any(np.isnan(eps_growth)):
            valid_eps = [g for g in eps_growth if abs(g) < 500]
            if len(valid_eps) == 3:
                details['eps_acceleration'] = valid_eps[0] > valid_eps[1] and valid_eps[1] > valid_eps[2]
        
        if not any(np.isnan(rev_growth)):
            valid_rev = [g for g in rev_growth if abs(g) < 500]
            if len(valid_rev) == 3:
                details['revenue_acceleration'] = valid_rev[0] > valid_rev[1] and valid_rev[1] > valid_rev[2]
        
        if not any(np.isnan(margins)):
            details['margin_improvement'] = margins[0] > margins[1] and margins[1] > margins[2]
        
        # アーニングスサプライズ分析
        surprise_analysis = analyze_earnings_surprises(financial_data)
        details['surprise_data'] = surprise_analysis
        
        if surprise_analysis['has_data']:
            details['earnings_surprises'] = (
                surprise_analysis['positive_surprises'] >= 2 and
                surprise_analysis['avg_surprise'] > 5.0
            )
        
        is_code33_core = (details['eps_acceleration'] and 
                          details['revenue_acceleration'] and 
                          details['margin_improvement'])
        
        is_code33_enhanced = is_code33_core and details['earnings_surprises']
        
        details['is_enhanced'] = is_code33_enhanced
        
        return (is_code33_core or is_code33_enhanced), details
        
    except Exception as e:
        return False, details


def check_pattern1(metrics_df: pd.DataFrame) -> Tuple[bool, Dict]:
    """
    パターン①: 赤→黒→黒→黒
    判定条件：黒字3期の「絶対値」が加速度的に増加
    
    赤字→黒字転換があるため、成長率ではなく絶対値の加速を評価：
    - Latest-2, Latest-1, Latest の3期すべてが黒字
    - 増加幅が加速: (Latest-1 - Latest-2) < (Latest - Latest-1)
    - または、2階差分が正: Δ²EPS > 0
    """
    details = {'pattern': 'Pattern1', 'description': '赤→黒→黒→黒', 'match': False}
    
    try:
        eps = metrics_df['EPS'].values
        
        # Latest-3が赤字、Latest-2/Latest-1/Latestが黒字
        if eps[3] < 0 and eps[2] > 0 and eps[1] > 0 and eps[0] > 0:
            # 方法1：増加幅が加速しているか
            delta1 = eps[2] - eps[3]  # Latest-2 vs Latest-3（赤→黒の変化）
            delta2 = eps[1] - eps[2]  # Latest-1 vs Latest-2（黒字の増加）
            delta3 = eps[0] - eps[1]  # Latest vs Latest-1（黒字の増加）
            
            # 黒字期間の増加幅が加速（delta3 > delta2）
            # かつ、両方とも正の増加
            if delta2 > 0 and delta3 > 0 and delta3 > delta2:
                details['match'] = True
                details['delta_acceleration'] = [delta3, delta2, delta1]
                details['acceleration_ratio'] = delta3 / delta2 if delta2 > 0 else 0
                
        return details['match'], details
        
    except Exception as e:
        return False, details


def check_pattern2(metrics_df: pd.DataFrame) -> Tuple[bool, Dict]:
    """
    パターン②: 赤→赤→黒→黒
    判定条件：3期連続で改善幅が加速
    
    各期の改善幅：
    - delta1 = Latest-2 - Latest-3（赤字縮小幅）
    - delta2 = Latest-1 - Latest-2（黒字転換時の改善）
    - delta3 = Latest - Latest-1（黒字増加幅）
    
    加速判定：delta1 < delta2 < delta3
    """
    details = {'pattern': 'Pattern2', 'description': '赤→赤→黒→黒', 'match': False}
    
    try:
        eps = metrics_df['EPS'].values
        
        # Latest-3/Latest-2が赤字、Latest-1/Latestが黒字
        if eps[3] < 0 and eps[2] < 0 and eps[1] > 0 and eps[0] > 0:
            # 各期の改善幅を計算
            delta1 = eps[2] - eps[3]  # 赤字縮小幅
            delta2 = eps[1] - eps[2]  # 黒字転換時の改善
            delta3 = eps[0] - eps[1]  # 黒字増加幅
            
            # 3期連続で改善幅が加速
            if delta1 > 0 and delta2 > 0 and delta3 > 0:
                if delta2 > delta1 and delta3 > delta2:
                    details['match'] = True
                    details['improvement_deltas'] = [delta3, delta2, delta1]
                    details['acceleration_confirmed'] = True
                
        return details['match'], details
        
    except Exception as e:
        return False, details


def check_pattern3(metrics_df: pd.DataFrame) -> Tuple[bool, Dict]:
    """
    パターン③: 赤→赤→赤→黒/赤
    判定条件：赤字3期（Latest-3, Latest-2, Latest-1）の赤字幅縮小率が連続で加速
    
    縮小率の計算：
    - r1 = (|Latest-3| - |Latest-2|) / |Latest-3|
    - r2 = (|Latest-2| - |Latest-1|) / |Latest-2|
    
    Latestが黒字の場合は追加ボーナス
    
    加速判定：r2 > r1（両方とも正）
    """
    details = {'pattern': 'Pattern3', 'description': '赤→赤→赤→黒/赤', 'match': False}
    
    try:
        eps = metrics_df['EPS'].values
        
        # Latest-3/Latest-2/Latest-1が赤字
        if eps[3] < 0 and eps[2] < 0 and eps[1] < 0:
            # 赤字幅縮小率を計算
            r1 = calculate_growth_rate(eps[2], eps[3])  # Latest-2 vs Latest-3
            r2 = calculate_growth_rate(eps[1], eps[2])  # Latest-1 vs Latest-2
            
            # 縮小率が加速（r2 > r1 > 0）
            if r1 > 0 and r2 > 0 and r2 > r1:
                details['match'] = True
                details['reduction_rates'] = [r2, r1]
                details['latest_status'] = 'Black' if eps[0] > 0 else 'Red'
                
                # Latestも赤字の場合、さらに加速しているか確認
                if eps[0] < 0:
                    r3 = calculate_growth_rate(eps[0], eps[1])
                    if r3 > 0:
                        details['reduction_rates'].insert(0, r3)
                        # 3期連続加速の確認
                        if r3 > r2:
                            details['three_period_acceleration'] = True
                
        return details['match'], details
        
    except Exception as e:
        return False, details


def check_pattern4(metrics_df: pd.DataFrame) -> Tuple[bool, Dict]:
    """パターン④: 赤→赤→赤→赤"""
    details = {'pattern': 'Pattern4', 'description': '赤→赤→赤→赤', 'match': False}
    
    try:
        eps = metrics_df['EPS'].values
        
        if all(eps < 0):
            r1 = calculate_growth_rate(eps[2], eps[3])
            r2 = calculate_growth_rate(eps[1], eps[2])
            r3 = calculate_growth_rate(eps[0], eps[1])
            
            if r1 > 0 and r2 > 0 and r3 > 0 and r3 > r2 and r2 > r1:
                details['match'] = True
                details['reduction_rates'] = [r3, r2, r1]
                details['approaching_breakeven'] = abs(eps[0]) < abs(eps[1]) * 0.5
                
        return details['match'], details
        
    except Exception as e:
        return False, details


def analyze_supplementary_factors(metrics_df: pd.DataFrame) -> Dict:
    """補足的な判定要素を分析"""
    factors = {
        'revenue_growing': False,
        'operating_cf_positive': False,
        'margin_improving': False
    }
    
    try:
        rev = metrics_df['Revenue'].values
        if rev[0] > rev[3]:
            factors['revenue_growing'] = True
        
        cf = metrics_df['OperatingCF'].values
        if not np.isnan(cf[0]) and cf[0] > 0:
            factors['operating_cf_positive'] = True
        
        margin = metrics_df['NetMargin'].values
        if margin[0] > margin[3]:
            factors['margin_improving'] = True
            
        return factors
        
    except Exception as e:
        return factors


def analyze_ticker(ticker: str) -> Optional[Dict]:
    """個別ティッカーの分析を実行"""
    financial_data = fetch_financial_data(ticker)
    if financial_data is None:
        return None
    
    metrics_df = extract_metrics(financial_data)
    if metrics_df is None:
        return None
    
    is_code33, code33_details = check_code33(metrics_df, financial_data)
    
    pattern1_match, pattern1_details = check_pattern1(metrics_df)
    pattern2_match, pattern2_details = check_pattern2(metrics_df)
    pattern3_match, pattern3_details = check_pattern3(metrics_df)
    pattern4_match, pattern4_details = check_pattern4(metrics_df)
    
    supplementary = analyze_supplementary_factors(metrics_df)
    
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
    """メイン処理"""
    print("=" * 60)
    print("コード33 + 赤字転換パターン検出（修正版）")
    print("=" * 60)
    print("\n【注意】yfinanceの仕様変更に対応した修正版です")
    print("EPSはNet Incomeから計算しています\n")
    
    try:
        # stock.csvをpandasで読み込む
        tickers_df = pd.read_csv('stock.csv', encoding='utf-8-sig')
        if 'Ticker' not in tickers_df.columns or 'Exchange' not in tickers_df.columns:
            print("エラー: stock.csv に 'Ticker' または 'Exchange' 列が見つかりません。")
            return

        tickers_to_analyze = tickers_df['Ticker'].dropna().astype(str).tolist()
        print(f"{len(tickers_to_analyze)}銘柄を読み込みました。")

    except FileNotFoundError:
        print("エラー: stock.csvが見つかりません。")
        return
    except Exception as e:
        print(f"エラー: stock.csv の読み込み中に予期せぬエラーが発生しました: {e}")
        return

    results = []
    print("\n分析を開始します...\n")
    
    # DataFrameからティッカーリストを取得して分析
    for ticker in tqdm(tickers_df['Ticker'], desc="銘柄分析中"):
        result = analyze_ticker(ticker)
        if result:
            results.append(result)
    
    if not results:
        print("\n条件に合致する銘柄は見つかりませんでした。")
        print("\n【ヒント】")
        print("- yfinanceでデータ取得できない銘柄が多い可能性があります")
        print("- stock.csvの銘柄数を減らして再試行してください")
        print("- 大型株（AAPL, MSFT, GOOGL等）でテストすることをお勧めします")
        return
    
    df = pd.DataFrame(results)
    
    # 元のtickers_dfとマージして取引所情報を付与
    df = pd.merge(df, tickers_df, left_on='ticker', right_on='Ticker', how='left')

    def get_pattern_str(row):
        patterns = []
        if row['code33_enhanced']:
            patterns.append('Code33★')
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
    
    def calculate_priority(row):
        score = 0
        if row['code33_enhanced']:
            score += 150
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
    df = df.sort_values('Priority', ascending=False)
    
    # CSV出力用のDataFrame
    output_df = df[[
        'ticker', 'Patterns', 'Priority',
        'eps_latest', 'eps_latest_1', 'eps_latest_2', 'eps_latest_3',
        'avg_surprise', 'positive_surprises',
        'revenue_growth', 'cf_positive', 'margin_improving'
    ]].copy()
    
    output_df.columns = [
        'Ticker', 'Detected_Patterns', 'Priority_Score',
        'EPS_Latest', 'EPS_Latest-1', 'EPS_Latest-2', 'EPS_Latest-3',
        'Avg_Surprise%', 'Positive_Surprises',
        'Revenue_Growing', 'CF_Positive', 'Margin_Improving'
    ]
    
    # CSVファイルに保存
    output_df.to_csv('code33plus_results.csv', index=False, encoding='utf-8-sig')
    print(f"\n結果を code33plus_results.csv に保存しました。")

    # TradingView用のTXTファイルを作成
    if not df.empty:
        # 'Exchange'列と'ticker'列が存在することを確認
        if 'Exchange' in df.columns and 'ticker' in df.columns:
            # NaNが含まれている可能性があるため、dropnaで除外
            tv_df = df.dropna(subset=['Exchange', 'ticker'])
            tradingview_list = [f"{row['Exchange']}:{row['ticker']}" for _, row in tv_df.iterrows()]
            tradingview_str = ",".join(tradingview_list)

            try:
                with open('code33plus_tradingview.txt', 'w', encoding='utf-8') as f:
                    f.write(tradingview_str)
                print(f"TradingView用のティッカーリストを code33plus_tradingview.txt に保存しました。")
            except Exception as e:
                print(f"\nエラー: TradingView用ファイルの書き込み中にエラーが発生しました: {e}")
        else:
            print("\n警告: 'Exchange'または'ticker'列が結果にないため、TradingViewファイルは作成されませんでした。")

    # サマリーを表示
    print(f"\n{'=' * 60}")
    print(f"分析完了！ {len(df)}銘柄が条件に合致しました。")
    print(f"{'=' * 60}")
    print(f"Code33該当: {df['code33'].sum()}銘柄")
    print(f"  └ アーニングスサプライズ付き: {df['code33_enhanced'].sum()}銘柄")
    print(f"Pattern1該当: {df['pattern1'].sum()}銘柄")
    print(f"Pattern2該当: {df['pattern2'].sum()}銘柄")
    print(f"Pattern3該当: {df['pattern3'].sum()}銘柄")
    print(f"Pattern4該当: {df['pattern4'].sum()}銘柄")
    
    print(f"\n{'=' * 60}")
    print("Top 10 優先銘柄:")
    print(f"{'=' * 60}")
    print(output_df.head(10).to_string(index=False))


if __name__ == '__main__':
    main()