"""
ATR Multiple from MA 分析モジュール
移動平均線からの乖離をATRで正規化し、過熱感を測定
"""
import pandas as pd
import numpy as np
from typing import Dict


class ATRAnalyzer:
    """
    ATR Multiple分析システム
    
    計算方法:
    1. ATR% = (ATR / 終値) × 100
    2. 乖離率% = ((終値 - MA) / MA) × 100
    3. ATR Multiple = 乖離率% / ATR%
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Args:
            df: 指標計算済みのDataFrame
        """
        self.df = df
        self.latest = df.iloc[-1]
        
    def calculate_atr_multiple(self, ma_column: str = 'SMA_50') -> float:
        """
        指定したMAからのATR Multipleを計算
        
        Args:
            ma_column: 使用する移動平均線の列名
            
        Returns:
            float: ATR Multiple
        """
        if ma_column not in self.df.columns or 'ATR_14' not in self.df.columns:
            return 0.0
        
        current_price = self.latest['Close']
        ma = self.latest[ma_column]
        atr = self.latest['ATR_14']
        
        if current_price <= 0 or atr <= 0:
            return 0.0
        
        # ATR%
        atr_pct = (atr / current_price) * 100
        
        # 乖離率%
        deviation_pct = ((current_price - ma) / ma) * 100
        
        # ATR Multiple
        if atr_pct > 0:
            atr_multiple = deviation_pct / atr_pct
        else:
            atr_multiple = 0.0
        
        return abs(atr_multiple)
    
    def interpret_atr_multiple(self, atr_multiple: float, stage: int = 2) -> Dict:
        """
        ATR Multipleの解釈とアクション推奨
        
        Args:
            atr_multiple: 計算されたATR Multiple
            stage: 現在のステージ (1, 2, 3, 4)
            
        Returns:
            dict: 解釈とアクション
        """
        interpretation = {
            'value': atr_multiple,
            'status': '',
            'action': '',
            'risk_level': '',
            'profit_take_pct': 0
        }
        
        if stage == 4:
            # Stage 4での解釈
            if atr_multiple <= -3.0:
                interpretation['status'] = '極度の売られすぎ'
                interpretation['action'] = 'Stage 4末期、Stage 1移行監視'
                interpretation['risk_level'] = 'low'
            elif -3.0 < atr_multiple <= -2.0:
                interpretation['status'] = '大きな調整'
                interpretation['action'] = 'ベース形成の可能性'
                interpretation['risk_level'] = 'medium'
            else:
                interpretation['status'] = '下降継続'
                interpretation['action'] = '底打ち待ち'
                interpretation['risk_level'] = 'high'
                
        elif stage == 1:
            # Stage 1での解釈
            if -0.5 <= atr_multiple <= 0.5:
                interpretation['status'] = 'ベース内、正常範囲'
                interpretation['action'] = 'エントリー準備、ブレイクアウト待ち'
                interpretation['risk_level'] = 'low'
            elif -2.0 <= atr_multiple < -0.5:
                interpretation['status'] = '大きな調整'
                interpretation['action'] = 'ベース形成中'
                interpretation['risk_level'] = 'medium'
            else:
                interpretation['status'] = '範囲外'
                interpretation['action'] = '様子見'
                interpretation['risk_level'] = 'medium'
                
        elif stage == 2:
            # Stage 2での解釈（最重要）
            if 0 <= atr_multiple < 3.0:
                interpretation['status'] = '健全な上昇'
                interpretation['action'] = 'トレンド継続、ホールド'
                interpretation['risk_level'] = 'low'
                interpretation['profit_take_pct'] = 0
                
            elif 3.0 <= atr_multiple < 5.0:
                interpretation['status'] = '軽度の過剰拡張'
                interpretation['action'] = '注意深く監視、利確準備'
                interpretation['risk_level'] = 'low-medium'
                interpretation['profit_take_pct'] = 0
                
            elif 5.0 <= atr_multiple < 7.0:
                interpretation['status'] = '中程度の過剰拡張'
                interpretation['action'] = '部分利確を検討（20-25%）'
                interpretation['risk_level'] = 'medium'
                interpretation['profit_take_pct'] = 20
                
            elif 7.0 <= atr_multiple < 10.0:
                interpretation['status'] = '高度な過剰拡張'
                interpretation['action'] = '利確開始（25-30%）- 重要閾値！'
                interpretation['risk_level'] = 'high'
                interpretation['profit_take_pct'] = 25
                
            elif atr_multiple >= 10.0:
                interpretation['status'] = '極度の過剰拡張'
                interpretation['action'] = '積極的な利確（30-35%以上）'
                interpretation['risk_level'] = 'very-high'
                interpretation['profit_take_pct'] = 30
                
            else:  # 負の値
                interpretation['status'] = 'MA割れ'
                interpretation['action'] = 'ストップロス検討'
                interpretation['risk_level'] = 'high'
                interpretation['profit_take_pct'] = 0
                
        elif stage == 3:
            # Stage 3での解釈
            if atr_multiple < 3.0:
                interpretation['status'] = 'モメンタム減衰'
                interpretation['action'] = '分配フェーズ確認、利確推奨'
                interpretation['risk_level'] = 'medium-high'
                interpretation['profit_take_pct'] = 50
            else:
                interpretation['status'] = '天井圏'
                interpretation['action'] = '積極的利確、ポジション削減'
                interpretation['risk_level'] = 'high'
                interpretation['profit_take_pct'] = 75
        
        return interpretation
    
    def calculate_dynamic_stop_loss(self, atr_multiple: float) -> float:
        """
        ATR Multipleに基づく動的ストップロス倍数を計算
        
        過剰拡張が大きいほど、より緊密なストップが必要
        
        Args:
            atr_multiple: ATR Multiple
            
        Returns:
            float: ATR倍数（現在価格 - この倍数×ATR = ストップロス）
        """
        if atr_multiple < 3.0:
            return 2.5
        elif atr_multiple < 5.0:
            return 2.0
        elif atr_multiple < 7.0:
            return 1.5
        elif atr_multiple < 10.0:
            return 1.0
        else:
            return 0.5
    
    def adjust_threshold_for_market(self, base_threshold: float, 
                                   market_condition: str = 'neutral') -> float:
        """
        市場環境に応じてATR Multiple閾値を調整
        
        Args:
            base_threshold: 基準閾値
            market_condition: 'bull', 'bear', 'neutral'
            
        Returns:
            float: 調整後の閾値
        """
        if market_condition == 'bull':
            # 強気市場: 標準の1.2-1.5倍
            return base_threshold * 1.3
        elif market_condition == 'bear':
            # 弱気市場: 標準の0.6-0.8倍
            return base_threshold * 0.7
        else:
            return base_threshold
    
    def analyze_atr_metrics(self, stage: int = 2, 
                           market_condition: str = 'neutral') -> Dict:
        """
        包括的なATR分析を実行
        
        Args:
            stage: 現在のステージ
            market_condition: 市場環境
            
        Returns:
            dict: ATR分析結果
        """
        result = {}
        
        # 50日MAからのATR Multiple
        atr_multiple_50 = self.calculate_atr_multiple('SMA_50')
        result['atr_multiple_ma50'] = atr_multiple_50
        
        # 200日MAからのATR Multiple（参考用）
        atr_multiple_200 = self.calculate_atr_multiple('SMA_200')
        result['atr_multiple_ma200'] = atr_multiple_200
        
        # 解釈
        interpretation = self.interpret_atr_multiple(atr_multiple_50, stage)
        result.update(interpretation)
        
        # 動的ストップロス
        stop_multiplier = self.calculate_dynamic_stop_loss(atr_multiple_50)
        result['stop_loss_atr_multiplier'] = stop_multiplier
        
        # ストップロス価格
        current_price = self.latest['Close']
        atr = self.latest['ATR_14']
        stop_price = current_price - (atr * stop_multiplier)
        result['suggested_stop_loss'] = stop_price
        
        # 市場環境調整後の閾値（7倍を基準）
        adjusted_threshold = self.adjust_threshold_for_market(7.0, market_condition)
        result['adjusted_profit_take_threshold'] = adjusted_threshold
        
        # 現在の状態が調整後閾値を超えているか
        result['exceeds_adjusted_threshold'] = atr_multiple_50 >= adjusted_threshold
        
        return result


if __name__ == '__main__':
    # テスト用
    from data_fetcher import fetch_stock_data
    from indicators import calculate_all_basic_indicators
    
    print("ATR Multiple分析のテストを開始...")
    
    test_tickers = ['AAPL', 'TSLA', 'NVDA']
    
    for ticker in test_tickers:
        print(f"\n{ticker} のATR分析:")
        stock_df, _ = fetch_stock_data(ticker, period='2y')
        
        if stock_df is not None:
            indicators_df = calculate_all_basic_indicators(stock_df)
            indicators_df = indicators_df.dropna()
            
            if len(indicators_df) >= 200:
                analyzer = ATRAnalyzer(indicators_df)
                result = analyzer.analyze_atr_metrics(stage=2, market_condition='neutral')
                
                print(f"  ATR Multiple (MA50): {result['atr_multiple_ma50']:.2f}")
                print(f"  状態: {result['status']}")
                print(f"  アクション: {result['action']}")
                print(f"  利確推奨: {result['profit_take_pct']}%")
                print(f"  ストップロス価格: ${result['suggested_stop_loss']:.2f}")
