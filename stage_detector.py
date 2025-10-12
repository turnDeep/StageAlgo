"""
ステージ検出モジュール
Stan Weinstein & Mark Minerviniの統合ステージ理論を実装
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple


class StageDetector:
    """
    ステージ判定システム
    Minerviniトレンドテンプレート8基準による厳格な判定
    """
    
    def __init__(self, df: pd.DataFrame, benchmark_df: pd.DataFrame = None):
        """
        Args:
            df: 指標計算済みのDataFrame
            benchmark_df: ベンチマーク(SPY)の指標計算済みDataFrame
        """
        self.df = df
        self.benchmark_df = benchmark_df
        self.latest = df.iloc[-1]
        
    def check_minervini_template(self) -> Dict:
        """
        Mark MinerviniのTrend Template 8基準をチェック
        
        Returns:
            dict: 各基準の結果とスコア
        """
        checks = {}
        
        # 現在の価格とMA
        current_price = self.latest['Close']
        sma_50 = self.latest['SMA_50']
        sma_150 = self.latest['SMA_150']
        sma_200 = self.latest['SMA_200']
        
        # 基準1: 現在価格 > 150日MA and 200日MA
        checks['criterion_1'] = (current_price > sma_150) and (current_price > sma_200)
        
        # 基準2: 150日MA > 200日MA
        checks['criterion_2'] = sma_150 > sma_200
        
        # 基準3: 200日MAが上昇トレンド (最低1ヶ月 = 20営業日)
        if len(self.df) >= 21:
            sma_200_20d_ago = self.df['SMA_200'].iloc[-21]
            checks['criterion_3'] = sma_200 > sma_200_20d_ago
        else:
            checks['criterion_3'] = False
        
        # 基準4: 50日MA > 150日MA and 200日MA
        checks['criterion_4'] = (sma_50 > sma_150) and (sma_50 > sma_200)
        
        # 基準5: 現在価格 > 50日MA
        checks['criterion_5'] = current_price > sma_50
        
        # 基準6: 52週安値から30%以上上
        low_52w = self.latest['Low_52W']
        if pd.notna(low_52w) and low_52w > 0:
            gain_from_low = (current_price - low_52w) / low_52w
            checks['criterion_6'] = gain_from_low > 0.30
        else:
            checks['criterion_6'] = False
        
        # 基準7: 52週高値の25%以内
        high_52w = self.latest['High_52W']
        if pd.notna(high_52w) and high_52w > 0:
            dist_from_high = (high_52w - current_price) / high_52w
            checks['criterion_7'] = dist_from_high < 0.25
        else:
            checks['criterion_7'] = False
        
        # 基準8: RS Rating ≥ 70 (外部で計算されることを想定)
        if 'RS_Rating' in self.df.columns:
            rs_rating = self.latest['RS_Rating']
            checks['criterion_8'] = rs_rating >= 70
        else:
            checks['criterion_8'] = False  # RS Ratingが計算されていない場合
        
        # スコア計算
        template_score = sum(checks.values()) / len(checks) * 100
        all_pass = all(checks.values())
        
        return {
            'all_criteria_met': all_pass,
            'score': template_score,
            'checks': checks,
            'criteria_met': sum(checks.values()),
            'total_criteria': len(checks)
        }
    
    def determine_stage(self) -> Tuple[int, str]:
        """
        現在のステージを判定
        
        Returns:
            tuple: (ステージ番号, サブステージ)
        """
        # Minerviniトレンドテンプレートをチェック
        template_result = self.check_minervini_template()
        
        # 全8基準を満たす場合はStage 2確定
        if template_result['all_criteria_met']:
            substage = self._determine_stage2_substage()
            return 2, substage
        
        # MAの配置と傾きを確認
        current_price = self.latest['Close']
        sma_150 = self.latest['SMA_150']
        sma_200 = self.latest['SMA_200']
        sma_150_slope = self.latest['SMA_150_Slope']
        sma_200_slope = self.latest['SMA_200_Slope']
        
        # Stage 4判定: 明確な下降トレンド
        if sma_150_slope < -0.002 and current_price < sma_150:
            substage = self._determine_stage4_substage()
            return 4, substage
        
        # Stage 1判定: ベース形成（横ばい）
        if (abs(sma_150_slope) < 0.02 and abs(sma_200_slope) < 0.02):
            # 価格が両MA の±15%以内
            if 0.85 < current_price/sma_150 < 1.15:
                substage = self._determine_stage1_substage()
                return 1, substage
        
        # Stage 3判定: 天井圏（MAは横ばい、価格は上）
        if abs(sma_150_slope) < 0.02 and current_price > sma_150:
            substage = self._determine_stage3_substage()
            return 3, substage
        
        # どれにも当てはまらない場合
        return 0, "Undefined"
    
    def _determine_stage1_substage(self) -> str:
        """Stage 1のサブステージを判定"""
        # ベース期間の推定（MAが横ばいになってからの期間）
        slope_threshold = 0.02
        base_start_idx = None
        
        for i in range(len(self.df)-1, max(0, len(self.df)-210), -1):  # 最大30週前まで
            if abs(self.df['SMA_150_Slope'].iloc[i]) >= slope_threshold:
                base_start_idx = i + 1
                break
        
        if base_start_idx is None:
            base_weeks = 30  # デフォルト
        else:
            base_weeks = (len(self.df) - base_start_idx) / 5  # 週数に変換
        
        # 出来高Dry Up率
        relative_vol = self.latest['Relative_Volume']
        
        # RS Rating
        rs_rating = self.latest.get('RS_Rating', 50)
        
        # ベース内位置（安値からの回復度）
        if 'Low_52W' in self.df.columns and pd.notna(self.latest['Low_52W']):
            recent_low = self.df['Low'].tail(int(base_weeks * 5)).min()
            recent_high = self.df['High'].tail(int(base_weeks * 5)).max()
            if recent_high > recent_low:
                base_position = (self.latest['Close'] - recent_low) / (recent_high - recent_low)
            else:
                base_position = 0.5
        else:
            base_position = 0.5
        
        # サブステージ判定
        if base_weeks < 10:
            return "1C"  # 初期安定化
        elif base_weeks < 20:
            if relative_vol < 0.50:
                return "1D"  # ベース中期
            else:
                return "1C"
        elif base_weeks < 30:
            if relative_vol < 0.30 and rs_rating > 70:
                return "1E"  # ベース後期
            else:
                return "1D"
        else:
            if relative_vol < 0.30 and rs_rating > 85 and base_position > 0.8:
                return "1F"  # ブレイク直前
            else:
                return "1E"
    
    def _determine_stage2_substage(self) -> str:
        """Stage 2のサブステージを判定"""
        # ブレイクアウト日の推定
        # 50日高値を更新した日を探す
        breakout_idx = None
        for i in range(len(self.df)-1, max(0, len(self.df)-60), -1):
            if i >= 50:
                high_50d = self.df['Close'].iloc[i-50:i].max()
                if self.df['Close'].iloc[i] > high_50d * 1.02:  # 2%以上の更新
                    breakout_idx = i
                    break
        
        if breakout_idx is None:
            # ブレイクアウト未確認
            return "2A"
        
        weeks_since_breakout = (len(self.df) - breakout_idx - 1) / 5
        
        # ブレイクアウトからの上昇率
        breakout_price = self.df['Close'].iloc[breakout_idx]
        gain_from_breakout = (self.latest['Close'] - breakout_price) / breakout_price * 100
        
        # ATR Multiple
        atr_multiple = self.latest.get('ATR_Multiple_MA50', 0)
        
        if weeks_since_breakout < 1:
            return "2A"  # ブレイクアウト
        elif weeks_since_breakout < 4:
            return "2B"  # 初期上昇
        elif weeks_since_breakout < 8:
            # 現在調整中か？
            if self.latest['Close'] < self.df['Close'].iloc[-6] * 0.95:
                return "2C"  # 第1押し目
            else:
                return "2B"
        elif weeks_since_breakout < 20:
            # ベースカウントの推定が必要（簡易版）
            if atr_multiple < 3.0:
                return "2E"  # 第2押し目
            else:
                return "2D"  # 再加速
        elif weeks_since_breakout < 40:
            return "2F"  # 主要上昇
        else:
            return "2G"  # 後期上昇
    
    def _determine_stage3_substage(self) -> str:
        """Stage 3のサブステージを判定"""
        # ATR Multiple
        atr_multiple = self.latest.get('ATR_Multiple_MA50', 0)
        
        # 出来高パターン
        recent_20d = self.df.tail(20)
        down_days_high_vol = recent_20d[
            (recent_20d['Close'] < recent_20d['Close'].shift(1)) & 
            (recent_20d['Volume'] > recent_20d['Volume_SMA_50'] * 1.5)
        ]
        distribution_days = len(down_days_high_vol)
        
        if distribution_days >= 3 and atr_multiple < 3.0:
            return "3B"  # 天井感が強まる
        elif distribution_days >= 1 or atr_multiple > 5.0:
            return "3"  # 天井エリア
        else:
            return "3A"  # 天井が形成され始め
    
    def _determine_stage4_substage(self) -> str:
        """Stage 4のサブステージを判定"""
        # 下落幅
        high_52w = self.latest['High_52W']
        current_price = self.latest['Close']
        
        if pd.notna(high_52w) and high_52w > 0:
            decline_from_high = (high_52w - current_price) / high_52w
        else:
            decline_from_high = 0
        
        # 出来高トレンド
        relative_vol = self.latest['Relative_Volume']
        
        # MA傾き
        sma_200_slope = self.latest['SMA_200_Slope']
        
        if decline_from_high > 0.5 and sma_200_slope < -0.01:
            return "4"  # 下降段階
        elif decline_from_high > 0.3:
            return "4A"  # 下降トレンド段階に入った
        else:
            # 底打ちの兆候チェック
            if abs(self.latest['SMA_150_Slope']) < 0.005 and relative_vol < 0.7:
                return "4B"  # 下降トレンド後期
            else:
                return "4"


if __name__ == '__main__':
    # テスト用
    from data_fetcher import fetch_stock_data
    from indicators import calculate_all_basic_indicators
    
    print("ステージ検出のテストを開始...")
    
    test_tickers = ['AAPL', 'TSLA', 'NVDA']
    
    for ticker in test_tickers:
        print(f"\n{ticker} の分析:")
        stock_df, benchmark_df = fetch_stock_data(ticker, period='2y')
        
        if stock_df is not None:
            indicators_df = calculate_all_basic_indicators(stock_df)
            indicators_df = indicators_df.dropna()
            
            if len(indicators_df) >= 200:
                detector = StageDetector(indicators_df)
                
                # Minerviniテンプレートチェック
                template = detector.check_minervini_template()
                print(f"  トレンドテンプレート: {template['criteria_met']}/8基準 満たす ({template['score']:.1f}点)")
                
                # ステージ判定
                stage, substage = detector.determine_stage()
                print(f"  ステージ: {stage} ({substage})")
