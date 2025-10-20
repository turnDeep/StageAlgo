"""
高値固定VWAP分析モジュール
Brian Shannonの理論に基づく
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple


class VWAPAnalyzer:
    """
    高値固定VWAP分析システム
    
    重要な高値更新時点を起点として計算される出来高加重平均価格
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Args:
            df: 指標計算済みのDataFrame
        """
        self.df = df
        self.latest = df.iloc[-1]
        
    def check_system_activation(self) -> Tuple[bool, Optional[pd.Timestamp], str]:
        """
        システムアクティブ化条件をチェック
        
        条件:
        1. 過去52週で高値更新(H)
        2. その後50日間、新高値を更新できない
        
        Returns:
            tuple: (アクティブ化フラグ, アンカー日, 理由)
        """
        if len(self.df) < 252:
            return False, None, 'データ不足（252日必要）'
        
        # 過去52週(252日)のデータ
        lookback_data = self.df.tail(252)
        
        # 52週高値を見つける
        high_52w_value = lookback_data['High'].max()
        high_52w_date = lookback_data['High'].idxmax()
        
        # 高値更新日のインデックス
        high_idx = self.df.index.get_loc(high_52w_date)
        
        # 高値更新後50日以上経過しているか
        days_since_high = len(self.df) - high_idx - 1
        
        if days_since_high < 50:
            return False, None, f'高値更新後{days_since_high}日（50日未満）'
        
        # 高値更新後に新高値を更新していないか確認
        data_after_high = self.df.iloc[high_idx+1:]
        new_highs = data_after_high['High'] > high_52w_value
        
        if new_highs.any():
            return False, None, '高値更新後に新高値を更新'
        
        return True, high_52w_date, f'条件満たす（{days_since_high}日前に高値、以降未更新）'
    
    def calculate_anchored_vwap(self, anchor_date: pd.Timestamp) -> pd.Series:
        """
        アンカー日から現在までのVWAPを計算
        
        VWAP = Σ(典型価格 × 出来高) / Σ(出来高)
        典型価格 = (高値 + 安値 + 終値) / 3
        
        Args:
            anchor_date: アンカー日
            
        Returns:
            pd.Series: VWAP時系列
        """
        # アンカー日以降のデータ
        anchor_idx = self.df.index.get_loc(anchor_date)
        data_from_anchor = self.df.iloc[anchor_idx:]
        
        # 典型価格
        typical_price = (data_from_anchor['High'] + data_from_anchor['Low'] + data_from_anchor['Close']) / 3
        
        # 累積計算
        cumulative_price_volume = (typical_price * data_from_anchor['Volume']).cumsum()
        cumulative_volume = data_from_anchor['Volume'].cumsum()
        
        # VWAP
        vwap = cumulative_price_volume / cumulative_volume
        
        return vwap
    
    def calculate_vwap_slope(self, vwap_series: pd.Series, period: int = 20) -> float:
        """
        VWAPの傾き（角度）を計算
        
        Args:
            vwap_series: VWAP時系列
            period: 傾き計算期間
            
        Returns:
            float: 角度（度）
        """
        if len(vwap_series) < period:
            return 0.0
        
        recent_vwap = vwap_series.tail(period)
        
        # 線形回帰で傾きを計算
        x = np.arange(len(recent_vwap))
        y = recent_vwap.values
        
        if len(y) < 2:
            return 0.0
        
        # 正規化
        y_norm = y / np.linalg.norm(y) if np.linalg.norm(y) > 0 else y
        
        slope = np.polyfit(x, y_norm, 1)[0]
        
        # ラジアンから度に変換
        angle_deg = np.arctan(slope) * 180 / np.pi
        
        return angle_deg
    
    def count_vwap_tests(self, vwap_series: pd.Series, tolerance_pct: float = 2.0) -> int:
        """
        価格がVWAPに接近した回数をカウント
        
        Args:
            vwap_series: VWAP時系列
            tolerance_pct: 接近の許容誤差（%）
            
        Returns:
            int: テスト回数
        """
        # VWAPが計算されている期間のデータ
        vwap_start_idx = vwap_series.index[0]
        data_with_vwap = self.df.loc[vwap_start_idx:]
        
        test_count = 0
        in_test = False
        
        for i in range(len(data_with_vwap)):
            date = data_with_vwap.index[i]
            if date not in vwap_series.index:
                continue
            
            price = data_with_vwap.loc[date, 'Close']
            vwap = vwap_series.loc[date]
            
            # VWAPからの距離
            if vwap > 0:
                distance_pct = abs(price - vwap) / vwap * 100
            else:
                distance_pct = 999
            
            # 接近している
            if distance_pct <= tolerance_pct:
                if not in_test:
                    test_count += 1
                    in_test = True
            else:
                in_test = False
        
        return test_count
    
    def analyze_vwap(self) -> Dict:
        """
        包括的なVWAP分析を実行
        
        Returns:
            dict: VWAP分析結果
        """
        result = {}
        
        # システムアクティブ化チェック
        is_active, anchor_date, reason = self.check_system_activation()
        
        result['system_active'] = is_active
        result['activation_reason'] = reason
        
        if not is_active:
            result['vwap_score'] = 5  # 最低スコア
            return result
        
        result['anchor_date'] = anchor_date.strftime('%Y-%m-%d')
        
        # VWAPを計算
        vwap_series = self.calculate_anchored_vwap(anchor_date)
        current_vwap = vwap_series.iloc[-1]
        result['vwap_value'] = current_vwap
        
        # 現在価格とVWAPの関係
        current_price = self.latest['Close']
        deviation_pct = (current_price - current_vwap) / current_vwap * 100
        result['deviation_from_vwap_pct'] = deviation_pct
        
        # VWAPの状態
        if current_price > current_vwap:
            result['vwap_status'] = 'Above'
            result['interpretation'] = '買い需要優勢、機関投資家が利益圏内'
        elif current_price < current_vwap:
            result['vwap_status'] = 'Below'
            result['interpretation'] = '売り圧力優勢、供給圧力が継続'
        else:
            result['vwap_status'] = 'At'
            result['interpretation'] = '均衡点'
        
        # VWAP角度
        angle = self.calculate_vwap_slope(vwap_series, 20)
        result['vwap_angle'] = angle
        
        # VWAPテスト回数
        test_count = self.count_vwap_tests(vwap_series, tolerance_pct=2.0)
        result['vwap_test_count'] = test_count
        
        # VWAPスコア計算（20点満点）
        score = 0
        
        if current_price > current_vwap and angle > 30:
            score = 20
            result['vwap_rating'] = 'A+'
        elif current_price > current_vwap and angle > 0:
            score = 15
            result['vwap_rating'] = 'A'
        elif abs(deviation_pct) <= 2:
            score = 10
            result['vwap_rating'] = 'B'
        elif current_price < current_vwap:
            score = 5
            result['vwap_rating'] = 'C'
        else:
            score = 10
            result['vwap_rating'] = 'B'
        
        result['vwap_score'] = score
        
        # 乖離の健全性チェック
        if abs(deviation_pct) > 5:
            result['deviation_warning'] = '過度の乖離（±5%以上）、警戒'
        else:
            result['deviation_warning'] = None
        
        # 拒否パターンチェック（Stage 3の兆候）
        if test_count >= 3 and current_price < current_vwap:
            result['rejection_pattern'] = True
            result['rejection_interpretation'] = '3回以上の拒否、分配の兆候'
        else:
            result['rejection_pattern'] = False
        
        return result


if __name__ == '__main__':
    # テスト用
    from data_fetcher import fetch_stock_data
    from indicators import calculate_all_basic_indicators
    
    print("VWAP分析のテストを開始...")
    
    test_tickers = ['AAPL', 'TSLA', 'NVDA', 'COIN', 'RIVN']
    
    for ticker in test_tickers:
        print(f"\n{ticker} のVWAP分析:")
        stock_df, _ = fetch_stock_data(ticker, period='2y')
        
        if stock_df is not None:
            indicators_df = calculate_all_basic_indicators(stock_df)
            indicators_df = indicators_df.dropna()
            
            if len(indicators_df) >= 252:
                analyzer = VWAPAnalyzer(indicators_df)
                result = analyzer.analyze_vwap()
                
                print(f"  システムアクティブ: {result['system_active']}")
                if result['system_active']:
                    print(f"  VWAP値: ${result['vwap_value']:.2f}")
                    print(f"  状態: {result['vwap_status']}")
                    print(f"  乖離: {result['deviation_from_vwap_pct']:.2f}%")
                    print(f"  スコア: {result['vwap_score']}/20")
                else:
                    print(f"  理由: {result['activation_reason']}")
