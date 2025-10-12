"""
VCP(Volatility Contraction Pattern)検出モジュール
Mark Minerviniの理論に基づく段階的収縮パターンの検出
"""
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from typing import Tuple, List, Dict


class VCPDetector:
    """
    VCPパターン検出システム
    
    5つの主要特徴:
    1. 価格収縮 - 各連続する価格レンジがより狭くなる
    2. 出来高減少 - 各収縮フェーズで取引出来高も減少
    3. ピボットポイント - 明確なブレイクアウトレベル
    4. より高い安値 - テニスボール・アクション
    5. 上昇トレンド継続 - Stage 2中に現れる
    """
    
    def __init__(self, df: pd.DataFrame, lookback_weeks: int = 52):
        """
        Args:
            df: 指標計算済みのDataFrame
            lookback_weeks: VCP検出の探索期間(週)
        """
        self.df = df
        self.lookback_days = lookback_weeks * 5  # 週を営業日に変換
        
    def detect_peaks_and_troughs(self, window: int = 5) -> Tuple[List[int], List[int]]:
        """
        ローカルピーク(高値)とトラフ(安値)を検出
        
        Args:
            window: 検出に使用するウィンドウサイズ
            
        Returns:
            tuple: (ピークのインデックスリスト, トラフのインデックスリスト)
        """
        # 分析期間のデータ
        analysis_df = self.df.tail(self.lookback_days)
        
        # scipyを使用してローカル極値を検出
        # ピーク(極大値)
        peak_indices = argrelextrema(analysis_df['High'].values, np.greater, order=window)[0]
        peaks = [analysis_df.index[i] for i in peak_indices]
        
        # トラフ(極小値)
        trough_indices = argrelextrema(analysis_df['Low'].values, np.less, order=window)[0]
        troughs = [analysis_df.index[i] for i in trough_indices]
        
        return peaks, troughs
    
    def calculate_contractions(self, peaks: List, troughs: List) -> Tuple[List[float], List[float], List[int]]:
        """
        各収縮の深さ、出来高、期間を計算
        
        Returns:
            tuple: (収縮深さリスト%, 出来高比率リスト, 収縮期間リスト)
        """
        contractions = []
        volume_ratios = []
        contraction_periods = []
        
        # ピークとトラフのペアを作成
        for i in range(min(len(peaks), len(troughs))):
            peak_date = peaks[i]
            
            # このピークの後の最初のトラフを見つける
            subsequent_troughs = [t for t in troughs if t > peak_date]
            if not subsequent_troughs:
                continue
                
            trough_date = subsequent_troughs[0]
            
            # 収縮深さの計算
            peak_price = self.df.loc[peak_date, 'High']
            trough_price = self.df.loc[trough_date, 'Low']
            
            if peak_price > 0:
                contraction_depth = (peak_price - trough_price) / peak_price * 100
                contractions.append(contraction_depth)
            else:
                continue
            
            # 収縮期間の平均出来高比率
            period_data = self.df.loc[peak_date:trough_date]
            if len(period_data) > 0:
                avg_volume_in_period = period_data['Volume'].mean()
                avg_volume_50d = period_data['Volume_SMA_50'].mean()
                
                if avg_volume_50d > 0:
                    vol_ratio = avg_volume_in_period / avg_volume_50d
                    volume_ratios.append(vol_ratio)
                else:
                    volume_ratios.append(1.0)
                
                # 収縮期間(日数)
                contraction_periods.append(len(period_data))
            else:
                volume_ratios.append(1.0)
                contraction_periods.append(0)
        
        return contractions, volume_ratios, contraction_periods
    
    def evaluate_vcp_quality(self, contractions: List[float], volume_ratios: List[float]) -> Tuple[bool, float, Dict]:
        """
        VCPの品質を評価（段階的スコアリング）
        
        Args:
            contractions: 収縮深さのリスト
            volume_ratios: 出来高比率のリスト
            
        Returns:
            tuple: (VCP検出フラグ, スコア, 詳細情報)
        """
        details = {}
        
        if len(contractions) < 2:
            return False, 0, {'error': '収縮回数が不足（最低2回必要）'}
        
        # 1. 段階的縮小の評価
        contraction_quality = 0
        contraction_ratios = []
        
        for i in range(len(contractions) - 1):
            if contractions[i] > 0:
                ratio = contractions[i+1] / contractions[i]
                contraction_ratios.append(ratio)
                
                # スコアリング（理想的には各修正が前回の30-70%程度）
                if ratio < 0.60:
                    contraction_quality += 30  # 優れた縮小
                elif ratio < 0.75:
                    contraction_quality += 25  # 良好な縮小
                elif ratio < 0.90:
                    contraction_quality += 15  # 許容できる縮小
                else:
                    contraction_quality += 0   # 縮小不十分
        
        details['contraction_ratios'] = contraction_ratios
        details['contraction_quality'] = contraction_quality
        
        # 2. 出来高の段階的減少
        volume_decreasing = all(
            volume_ratios[i+1] < volume_ratios[i] 
            for i in range(len(volume_ratios)-1)
        ) if len(volume_ratios) > 1 else True
        
        details['volume_decreasing'] = volume_decreasing
        
        # 3. 最終収縮が10%以下
        final_contraction_small = contractions[-1] < 10 if contractions else False
        details['final_contraction'] = contractions[-1] if contractions else 0
        details['final_contraction_small'] = final_contraction_small
        
        # 4. 最終収縮時の出来高が極小（平均の30%以下）
        final_volume_low = volume_ratios[-1] < 0.30 if volume_ratios else False
        details['final_volume_ratio'] = volume_ratios[-1] if volume_ratios else 0
        details['final_volume_low'] = final_volume_low
        
        # 5. VCPスコアの計算
        max_quality = (len(contractions) - 1) * 30  # 理論上の最大スコア
        if max_quality > 0:
            normalized_quality = (contraction_quality / max_quality * 100)
        else:
            normalized_quality = 0
        
        details['normalized_quality'] = normalized_quality
        
        # 6. 総合判定
        if (contraction_quality >= 50 and 
            volume_decreasing and 
            final_contraction_small and 
            final_volume_low):
            is_valid_vcp = True
            vcp_score = min(100, normalized_quality)
        elif contraction_quality >= 30:
            is_valid_vcp = False  # 部分的パターン
            vcp_score = min(50, normalized_quality)
        else:
            is_valid_vcp = False
            vcp_score = 0
        
        details['score'] = vcp_score
        details['num_contractions'] = len(contractions)
        details['contractions'] = contractions
        
        return is_valid_vcp, vcp_score, details
    
    def check_higher_lows(self, troughs: List) -> bool:
        """
        より高い安値(テニスボール・アクション)をチェック
        
        Returns:
            bool: 各安値が前回より高い場合True
        """
        if len(troughs) < 2:
            return False
        
        trough_prices = [self.df.loc[t, 'Low'] for t in troughs]
        
        # 連続して高い安値を形成しているか
        for i in range(len(trough_prices) - 1):
            if trough_prices[i+1] <= trough_prices[i]:
                return False
        
        return True
    
    def detect_vcp(self) -> Dict:
        """
        VCPパターンを検出
        
        Returns:
            dict: VCP検出結果
        """
        # 1. ピークとトラフの検出
        peaks, troughs = self.detect_peaks_and_troughs()
        
        if len(peaks) < 2 or len(troughs) < 2:
            return {
                'detected': False,
                'score': 0,
                'reason': 'ピーク/トラフが不足'
            }
        
        # 2. 収縮の計算
        contractions, volume_ratios, periods = self.calculate_contractions(peaks, troughs)
        
        if len(contractions) < 2:
            return {
                'detected': False,
                'score': 0,
                'reason': '収縮回数が不足'
            }
        
        # 3. VCP品質評価
        is_valid, score, details = self.evaluate_vcp_quality(contractions, volume_ratios)
        
        # 4. より高い安値のチェック
        higher_lows = self.check_higher_lows(troughs)
        details['higher_lows'] = higher_lows
        
        # 5. ピボットポイントの特定（最後のピーク）
        pivot_point = self.df.loc[peaks[-1], 'High'] if peaks else None
        details['pivot_point'] = pivot_point
        
        result = {
            'detected': is_valid,
            'score': score,
            'num_contractions': len(contractions),
            'contractions': contractions,
            'volume_ratios': volume_ratios,
            'details': details,
            'peaks': peaks,
            'troughs': troughs
        }
        
        return result


if __name__ == '__main__':
    # テスト用
    from data_fetcher import fetch_stock_data
    from indicators import calculate_all_basic_indicators
    
    print("VCP検出のテストを開始...")
    
    test_tickers = ['AAPL', 'TSLA', 'NVDA']
    
    for ticker in test_tickers:
        print(f"\n{ticker} のVCP分析:")
        stock_df, _ = fetch_stock_data(ticker, period='2y')
        
        if stock_df is not None:
            indicators_df = calculate_all_basic_indicators(stock_df)
            indicators_df = indicators_df.dropna()
            
            if len(indicators_df) >= 200:
                detector = VCPDetector(indicators_df, lookback_weeks=26)
                result = detector.detect_vcp()
                
                print(f"  VCP検出: {result['detected']}")
                print(f"  スコア: {result['score']:.1f}")
                print(f"  収縮回数: {result['num_contractions']}")
                if result['num_contractions'] > 0:
                    print(f"  収縮深さ: {[f'{c:.1f}%' for c in result['contractions']]}")
