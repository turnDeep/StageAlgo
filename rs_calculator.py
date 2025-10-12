"""
RS Rating (Relative Strength Rating) 計算モジュール
IBD (Investor's Business Daily) の手法に基づく
"""
import pandas as pd
import numpy as np
from typing import Dict


def calculate_roc(df: pd.DataFrame, period: int) -> pd.Series:
    """
    Rate of Change (変化率)を計算
    
    Args:
        df: 株価データ
        period: 期間(営業日)
        
    Returns:
        ROC (%)
    """
    if len(df) < period + 1:
        return pd.Series([0] * len(df), index=df.index)
    
    roc = (df['Close'] / df['Close'].shift(period) - 1) * 100
    return roc.fillna(0)


def calculate_strength_factor(df: pd.DataFrame) -> pd.Series:
    """
    IBD式のStrength Factorを計算
    
    加重平均:
    - 40% × ROC(63日)  - 直近3ヶ月
    - 20% × ROC(126日) - 直近6ヶ月
    - 20% × ROC(189日) - 直近9ヶ月  
    - 20% × ROC(252日) - 直近12ヶ月
    
    Args:
        df: 株価データ
        
    Returns:
        Strength Factor
    """
    roc_63 = calculate_roc(df, 63)
    roc_126 = calculate_roc(df, 126)
    roc_189 = calculate_roc(df, 189)
    roc_252 = calculate_roc(df, 252)
    
    strength_factor = (
        0.4 * roc_63 +
        0.2 * roc_126 +
        0.2 * roc_189 +
        0.2 * roc_252
    )
    
    return strength_factor


def calculate_rs_rating_single(df: pd.DataFrame, universe_strength_factors: Dict[str, float], ticker: str) -> float:
    """
    単一銘柄のRS Ratingを計算
    
    Args:
        df: 株価データ
        universe_strength_factors: 全銘柄のStrength Factorの辞書
        ticker: ティッカーシンボル
        
    Returns:
        RS Rating (1-99)
    """
    # この銘柄のStrength Factorを計算
    strength_factor = calculate_strength_factor(df).iloc[-1]
    
    # 全銘柄のStrength Factorをソート
    all_factors = list(universe_strength_factors.values())
    all_factors_sorted = sorted(all_factors)
    
    # パーセンタイルランクを計算
    if len(all_factors_sorted) == 0:
        return 50
    
    rank = sum(1 for x in all_factors_sorted if x < strength_factor)
    percentile = (rank / len(all_factors_sorted)) * 99 + 1
    
    return min(99, max(1, percentile))


def calculate_rs_rating_series(df: pd.DataFrame) -> pd.Series:
    """
    単一銘柄のRS Ratingを時系列で計算
    （他銘柄との比較なしの簡易版）
    
    過去252日のStrength Factorのローリングパーセンタイルランクを計算
    
    Args:
        df: 株価データ
        
    Returns:
        RS Rating時系列
    """
    strength_factor = calculate_strength_factor(df)
    
    # 252日のローリングウィンドウでパーセンタイルランクを計算
    def rolling_percentile(series, window=252):
        result = []
        for i in range(len(series)):
            if i < window - 1:
                result.append(np.nan)
            else:
                window_data = series.iloc[i-window+1:i+1]
                current_value = series.iloc[i]
                rank = (window_data < current_value).sum()
                percentile = (rank / len(window_data)) * 99 + 1
                result.append(percentile)
        return pd.Series(result, index=series.index)
    
    rs_rating = rolling_percentile(strength_factor)
    
    return rs_rating.fillna(50)


def calculate_rs_line(stock_df: pd.DataFrame, benchmark_df: pd.DataFrame) -> pd.Series:
    """
    RS Line (相対強度線)を計算
    
    RS Line = (株価 / ベンチマーク価格) × 100
    
    Args:
        stock_df: 株価データ
        benchmark_df: ベンチマーク(SPY)データ
        
    Returns:
        RS Line
    """
    # インデックスを揃える
    common_index = stock_df.index.intersection(benchmark_df.index)
    stock_close = stock_df.loc[common_index, 'Close']
    benchmark_close = benchmark_df.loc[common_index, 'Close']
    
    rs_line = (stock_close / benchmark_close) * 100
    
    return rs_line


def check_rs_line_new_high(rs_line: pd.Series, lookback_days: int = 252) -> bool:
    """
    RS Lineが新高値を更新しているかチェック
    
    Args:
        rs_line: RS Line時系列
        lookback_days: 確認期間
        
    Returns:
        bool: 新高値更新している場合True
    """
    if len(rs_line) < lookback_days + 1:
        return False
    
    current_rs = rs_line.iloc[-1]
    historical_max = rs_line.iloc[-lookback_days:-1].max()
    
    return current_rs > historical_max


def analyze_rs_metrics(stock_df: pd.DataFrame, benchmark_df: pd.DataFrame, 
                       universe_strength_factors: Dict[str, float] = None,
                       ticker: str = None) -> Dict:
    """
    包括的なRS分析を実行
    
    Args:
        stock_df: 株価データ
        benchmark_df: ベンチマークデータ
        universe_strength_factors: 全銘柄のStrength Factor
        ticker: ティッカーシンボル
        
    Returns:
        dict: RS分析結果
    """
    result = {}
    
    # RS Rating計算
    if universe_strength_factors and ticker:
        # 全銘柄との比較
        rs_rating = calculate_rs_rating_single(stock_df, universe_strength_factors, ticker)
    else:
        # 時系列のみ
        rs_rating_series = calculate_rs_rating_series(stock_df)
        rs_rating = rs_rating_series.iloc[-1] if not rs_rating_series.empty else 50
    
    result['rs_rating'] = rs_rating
    
    # RS Rating評価
    if rs_rating >= 90:
        result['rs_rating_grade'] = 'A+'
        result['rs_rating_category'] = 'トップ10%'
    elif rs_rating >= 85:
        result['rs_rating_grade'] = 'A'
        result['rs_rating_category'] = 'トップ15%'
    elif rs_rating >= 80:
        result['rs_rating_grade'] = 'B+'
        result['rs_rating_category'] = 'トップ20%'
    elif rs_rating >= 70:
        result['rs_rating_grade'] = 'B'
        result['rs_rating_category'] = 'トップ30%'
    else:
        result['rs_rating_grade'] = 'C'
        result['rs_rating_category'] = '平均以下'
    
    # RS Line
    rs_line = calculate_rs_line(stock_df, benchmark_df)
    result['rs_line_current'] = rs_line.iloc[-1] if not rs_line.empty else 0
    
    # RS Line新高値チェック
    rs_new_high = check_rs_line_new_high(rs_line)
    result['rs_line_new_high'] = rs_new_high
    
    # RS Line MA
    rs_line_ma52w = rs_line.rolling(window=252, min_periods=50).mean()
    if not rs_line_ma52w.empty and pd.notna(rs_line_ma52w.iloc[-1]):
        result['rs_line_above_ma52w'] = rs_line.iloc[-1] > rs_line_ma52w.iloc[-1]
    else:
        result['rs_line_above_ma52w'] = False
    
    return result


if __name__ == '__main__':
    # テスト用
    from data_fetcher import fetch_stock_data
    
    print("RS Rating計算のテストを開始...")
    
    test_tickers = ['AAPL', 'TSLA', 'NVDA']
    
    # ベンチマーク取得
    _, benchmark_df = fetch_stock_data('SPY', period='2y')
    
    for ticker in test_tickers:
        print(f"\n{ticker} のRS分析:")
        stock_df, _ = fetch_stock_data(ticker, period='2y')
        
        if stock_df is not None and benchmark_df is not None:
            # 簡易版（他銘柄との比較なし）
            result = analyze_rs_metrics(stock_df, benchmark_df)
            
            print(f"  RS Rating: {result['rs_rating']:.1f} ({result['rs_rating_grade']})")
            print(f"  カテゴリ: {result['rs_rating_category']}")
            print(f"  RS Line新高値: {result['rs_line_new_high']}")
            print(f"  RS Line > 52週MA: {result['rs_line_above_ma52w']}")
