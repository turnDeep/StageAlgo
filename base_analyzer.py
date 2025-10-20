"""
ベースパターン分析モジュール
William O'Neilのベース理論を実装
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional


class BaseAnalyzer:
    """
    ベースパターン分析システム
    
    ベースタイプ:
    - Cup with Handle
    - Double Bottom (W)
    - Flat Base
    - High Tight Flag
    - Saucer with Handle
    - Ascending Base
    - Square Box
    """
    
    def __init__(self, df: pd.DataFrame, lookback_weeks: int = 65):
        """
        Args:
            df: 指標計算済みのDataFrame
            lookback_weeks: ベース検出の最大期間
        """
        self.df = df
        self.lookback_days = min(lookback_weeks * 5, len(df))
        
    def identify_base_period(self) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
        """
        ベース期間を特定
        
        横ばい期間を検出（標準偏差が小さい期間）
        
        Returns:
            tuple: (ベース開始日, ベース終了日) or None
        """
        # 分析期間のデータ
        analysis_df = self.df.tail(self.lookback_days)
        
        if len(analysis_df) < 35:  # 最低7週必要
            return None
        
        # 価格の標準偏差を計算（21日ローリング）
        price_std = analysis_df['Close'].rolling(window=21, min_periods=21).std()
        price_mean = analysis_df['Close'].rolling(window=21, min_periods=21).mean()
        
        # 変動係数 (CV) = std / mean
        cv = price_std / price_mean
        
        # CVが低い期間がベース候補
        # MAが横ばい（傾きが小さい）期間も考慮
        ma_slope_threshold = 0.02
        
        potential_bases = []
        
        for i in range(35, len(analysis_df)):  # 最低7週後から
            window_data = analysis_df.iloc[i-35:i]
            
            # この期間のCV
            avg_cv = cv.iloc[i-35:i].mean()
            
            # MAの傾き
            ma_slope = abs(window_data['SMA_50'].iloc[-1] - window_data['SMA_50'].iloc[0]) / window_data['SMA_50'].iloc[0]
            
            if avg_cv < 0.05 and ma_slope < ma_slope_threshold:
                # ベース候補
                base_start = window_data.index[0]
                base_end = window_data.index[-1]
                potential_bases.append((base_start, base_end, i-35, i))
        
        if not potential_bases:
            return None
        
        # 最も最近のベースを返す
        return potential_bases[-1][:2]
    
    def calculate_base_depth(self, base_start: pd.Timestamp, base_end: pd.Timestamp) -> float:
        """
        ベースの深さ（修正率）を計算
        
        Args:
            base_start: ベース開始日
            base_end: ベース終了日
            
        Returns:
            float: ベースの深さ（%）
        """
        base_data = self.df.loc[base_start:base_end]
        
        if base_data.empty:
            return 0.0
        
        high_in_base = base_data['High'].max()
        low_in_base = base_data['Low'].min()
        
        if high_in_base > 0:
            depth = (high_in_base - low_in_base) / high_in_base * 100
            return depth
        
        return 0.0
    
    def detect_handle(self, base_end: pd.Timestamp, lookback_days: int = 14) -> Optional[Dict]:
        """
        ハンドル形成を検出
        
        Args:
            base_end: ベースの終了日
            lookback_days: ハンドル検出の期間（通常1-2週間）
            
        Returns:
            dict: ハンドル情報 or None
        """
        # ベース終了日から最近のデータを取得
        end_idx = self.df.index.get_loc(base_end)
        handle_data = self.df.iloc[end_idx:end_idx + lookback_days]
        
        if len(handle_data) < 5:
            return None
        
        # ハンドルの深さ
        handle_high = handle_data['High'].max()
        handle_low = handle_data['Low'].min()
        
        if handle_high > 0:
            handle_depth = (handle_high - handle_low) / handle_high * 100
        else:
            return None
        
        # ベース全体の高さと比較
        base_data = self.df.loc[:base_end]
        if len(base_data) >= 35:
            base_high = base_data.tail(35)['High'].max()
            base_low = base_data.tail(35)['Low'].min()
            base_height = base_high - base_low
            
            # ハンドルはカップの高さの1/3以下が理想
            if base_height > 0:
                handle_ratio = (handle_high - handle_low) / base_height
                
                if handle_ratio < 0.33 and handle_depth < 15:
                    # 理想的なハンドル
                    return {
                        'detected': True,
                        'depth': handle_depth,
                        'ratio_to_base': handle_ratio,
                        'quality': 'excellent' if handle_ratio < 0.25 else 'good'
                    }
        
        return None
    
    def classify_base_type(self, base_start: pd.Timestamp, base_end: pd.Timestamp) -> str:
        """
        ベースのタイプを分類
        
        Returns:
            str: ベースタイプ
        """
        base_data = self.df.loc[base_start:base_end]
        
        if base_data.empty:
            return "Unknown"
        
        depth = self.calculate_base_depth(base_start, base_end)
        duration_weeks = len(base_data) / 5
        
        # Flat Base: 浅い調整（10-15%）、比較的短期
        if depth < 15 and 5 <= duration_weeks <= 13:
            return "Flat Base"
        
        # High Tight Flag: 非常に浅く短期（3-8週、10-25%）
        if depth < 25 and 3 <= duration_weeks <= 8:
            # 前期に急激な上昇があったか確認
            prior_data = self.df.loc[:base_start].tail(40)  # 8週前
            if len(prior_data) > 0:
                prior_gain = (prior_data['Close'].iloc[-1] / prior_data['Close'].iloc[0] - 1) * 100
                if prior_gain > 100:  # 2倍以上の上昇
                    return "High Tight Flag"
        
        # Square Box: 非常に短期の統合（4-7週）
        if 4 <= duration_weeks <= 7:
            return "Square Box"
        
        # Cup with Handle: U字型、中程度の深さ
        if 12 <= depth <= 35 and duration_weeks >= 7:
            # ハンドル検出
            handle = self.detect_handle(base_end)
            if handle and handle['detected']:
                return "Cup with Handle"
            else:
                return "Cup (no handle)"
        
        # Double Bottom (W): 2つの底を持つ
        # 簡易版: 期間中に2つの明確な安値があるか
        lows = base_data['Low']
        local_mins = lows[(lows.shift(1) > lows) & (lows.shift(-1) > lows)]
        
        if len(local_mins) == 2 and 20 <= depth <= 30:
            return "Double Bottom (W)"
        
        # Saucer with Handle: 長期のU字型
        if depth >= 12 and duration_weeks > 15:
            return "Saucer with Handle"
        
        # Ascending Base: 段階的上昇
        if 10 <= depth <= 20 and 5 <= duration_weeks <= 20:
            # 右側が左側より高いか
            mid_point = len(base_data) // 2
            left_avg = base_data['Close'].iloc[:mid_point].mean()
            right_avg = base_data['Close'].iloc[mid_point:].mean()
            
            if right_avg > left_avg * 1.02:
                return "Ascending Base"
        
        return "Generic Base"
    
    def calculate_base_quality_score(self) -> Dict:
        """
        ベースの品質を100点満点で評価
        
        評価項目:
        1. 期間要件 (25点)
        2. 深さ基準 (25点)
        3. 出来高パターン (25点)
        4. 形状 (25点)
        """
        score = 0
        details = {}
        
        # ベース期間を特定
        base_period = self.identify_base_period()
        
        if base_period is None:
            return {
                'total_score': 0,
                'details': {'error': 'ベース未検出'},
                'base_detected': False
            }
        
        base_start, base_end = base_period
        base_data = self.df.loc[base_start:base_end]
        duration_weeks = len(base_data) / 5
        
        details['base_start'] = base_start.strftime('%Y-%m-%d')
        details['base_end'] = base_end.strftime('%Y-%m-%d')
        details['duration_weeks'] = duration_weeks
        
        # 1. 期間スコア (25点)
        if 7 <= duration_weeks <= 12:
            period_score = 25
            details['period_rating'] = 'A'
        elif (5 <= duration_weeks < 7) or (12 < duration_weeks <= 20):
            period_score = 20
            details['period_rating'] = 'B'
        elif 20 < duration_weeks <= 30:
            period_score = 15
            details['period_rating'] = 'C'
        else:
            period_score = 5
            details['period_rating'] = 'D'
        
        score += period_score
        details['period_score'] = period_score
        
        # 2. 深さスコア (25点)
        depth = self.calculate_base_depth(base_start, base_end)
        details['depth_pct'] = depth
        
        if 15 <= depth <= 30:
            depth_score = 25
            details['depth_rating'] = 'A'
        elif (12 <= depth < 15) or (30 < depth <= 35):
            depth_score = 20
            details['depth_rating'] = 'B'
        elif (10 <= depth < 12) or (35 < depth <= 40):
            depth_score = 15
            details['depth_rating'] = 'C'
        else:
            depth_score = 5
            details['depth_rating'] = 'D'
        
        score += depth_score
        details['depth_score'] = depth_score
        
        # 3. 出来高パターンスコア (25点)
        # Dry Up: ベース右側の出来高が平均の30-50%以下
        mid_point = len(base_data) // 2
        right_side = base_data.iloc[mid_point:]
        
        if len(right_side) > 0 and 'Volume_SMA_50' in right_side.columns:
            right_avg_volume = right_side['Volume'].mean()
            right_avg_volume_ma = right_side['Volume_SMA_50'].mean()
            
            if right_avg_volume_ma > 0:
                dry_up_ratio = right_avg_volume / right_avg_volume_ma
                details['dry_up_ratio'] = dry_up_ratio
                
                if dry_up_ratio < 0.30:
                    volume_score = 25
                    details['volume_rating'] = 'A'
                elif dry_up_ratio < 0.50:
                    volume_score = 20
                    details['volume_rating'] = 'B'
                elif dry_up_ratio < 0.70:
                    volume_score = 15
                    details['volume_rating'] = 'C'
                else:
                    volume_score = 5
                    details['volume_rating'] = 'D'
            else:
                volume_score = 10
                details['volume_rating'] = 'N/A'
        else:
            volume_score = 10
            details['volume_rating'] = 'N/A'
        
        score += volume_score
        details['volume_score'] = volume_score
        
        # 4. 形状スコア (25点)
        shape_score = 0
        
        # 右側が左側より高い（健全な蓄積）
        left_side = base_data.iloc[:mid_point]
        
        if len(left_side) > 0 and len(right_side) > 0:
            left_avg = left_side['Close'].mean()
            right_avg = right_side['Close'].mean()
            
            if right_avg > left_avg:
                shape_score += 15
                details['right_higher_than_left'] = True
            else:
                details['right_higher_than_left'] = False
        
        # ハンドル品質
        handle = self.detect_handle(base_end)
        if handle and handle['detected']:
            shape_score += 10
            details['handle_quality'] = handle['quality']
        else:
            details['handle_quality'] = 'none'
        
        score += shape_score
        details['shape_score'] = shape_score
        details['shape_rating'] = 'A' if shape_score >= 20 else ('B' if shape_score >= 10 else 'C')
        
        # ベースタイプの分類
        base_type = self.classify_base_type(base_start, base_end)
        details['base_type'] = base_type
        
        return {
            'total_score': score,
            'details': details,
            'base_detected': True,
            'base_start': base_start,
            'base_end': base_end
        }


if __name__ == '__main__':
    # テスト用
    from data_fetcher import fetch_stock_data
    from indicators import calculate_all_basic_indicators
    
    print("ベース分析のテストを開始...")
    
    test_tickers = ['AAPL', 'TSLA', 'NVDA']
    
    for ticker in test_tickers:
        print(f"\n{ticker} のベース分析:")
        stock_df, _ = fetch_stock_data(ticker, period='2y')
        
        if stock_df is not None:
            indicators_df = calculate_all_basic_indicators(stock_df)
            indicators_df = indicators_df.dropna()
            
            if len(indicators_df) >= 200:
                analyzer = BaseAnalyzer(indicators_df, lookback_weeks=65)
                result = analyzer.calculate_base_quality_score()
                
                if result['base_detected']:
                    print(f"  ベース検出: Yes")
                    print(f"  総合スコア: {result['total_score']:.0f}/100")
                    print(f"  ベースタイプ: {result['details']['base_type']}")
                    print(f"  期間: {result['details']['duration_weeks']:.1f}週")
                    print(f"  深さ: {result['details']['depth_pct']:.1f}%")
                else:
                    print(f"  ベース検出: No")
