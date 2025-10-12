"""
出来高分析モジュール
Wyckoff理論 + O'Neil + Minerviniの統合
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple


class VolumeAnalyzer:
    """
    出来高分析システム
    
    理論的基盤:
    - Wyckoffの三大法則（供給と需要、原因と結果、努力と結果）
    - O'Neilのブレイクアウト出来高理論
    - MinerviniのDry Up概念
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Args:
            df: 指標計算済みのDataFrame
        """
        self.df = df
        self.latest = df.iloc[-1]
        
    def calculate_up_down_volume_ratio(self, period: int = 20) -> Dict:
        """
        上昇日と下落日の出来高比率を計算
        
        Args:
            period: 分析期間
            
        Returns:
            dict: 出来高比率の詳細
        """
        recent_data = self.df.tail(period)
        
        # 上昇日と下落日を分類
        up_days = recent_data[recent_data['Close'] > recent_data['Close'].shift(1)]
        down_days = recent_data[recent_data['Close'] < recent_data['Close'].shift(1)]
        
        up_volume = up_days['Volume'].sum()
        down_volume = down_days['Volume'].sum()
        
        # 比率計算
        if down_volume > 0:
            ratio = up_volume / down_volume
        else:
            ratio = 999 if up_volume > 0 else 1
        
        # パターン判定
        if ratio > 1.5:
            pattern = 'Accumulation'  # 蓄積
            interpretation = '機関投資家が蓄積中'
        elif ratio < 0.67:
            pattern = 'Distribution'  # 分配
            interpretation = '機関投資家が分配中'
        else:
            pattern = 'Neutral'
            interpretation = '均衡状態'
        
        return {
            'up_volume': up_volume,
            'down_volume': down_volume,
            'ratio': ratio,
            'pattern': pattern,
            'interpretation': interpretation,
            'up_days_count': len(up_days),
            'down_days_count': len(down_days)
        }
    
    def detect_pocket_pivot(self, lookback: int = 10) -> Dict:
        """
        Pocket Pivot (O'Neil/Minervini)を検出
        
        条件:
        - その日の出来高 > 過去10日間のすべての下落日の出来高
        - 価格が8EMAの上または付近
        
        Args:
            lookback: 確認期間
            
        Returns:
            dict: Pocket Pivot検出結果
        """
        recent_data = self.df.tail(lookback + 1)
        
        # 過去10日の下落日を取得
        down_days = recent_data.iloc[:-1][
            recent_data.iloc[:-1]['Close'] < recent_data.iloc[:-1]['Close'].shift(1)
        ]
        
        if len(down_days) == 0:
            return {
                'detected': False,
                'reason': '過去に下落日なし'
            }
        
        # 最大の下落日出来高
        max_down_volume = down_days['Volume'].max()
        
        # 今日の出来高
        current_volume = self.latest['Volume']
        
        # 8EMAの近く（±3%）
        if 'EMA_8' in self.df.columns:
            ema_8 = self.latest['EMA_8']
            current_price = self.latest['Close']
            near_ema = 0.97 <= current_price / ema_8 <= 1.03
        else:
            near_ema = False
        
        # Pocket Pivot判定
        if current_volume > max_down_volume and (near_ema or current_price > ema_8):
            return {
                'detected': True,
                'current_volume': current_volume,
                'max_down_volume': max_down_volume,
                'volume_ratio': current_volume / max_down_volume,
                'near_8ema': near_ema,
                'interpretation': 'ベース内での静かな蓄積の兆候'
            }
        
        return {
            'detected': False,
            'reason': '条件未達'
        }
    
    def count_pocket_pivots(self, period: int = 20) -> int:
        """
        指定期間内のPocket Pivot発生回数をカウント
        
        Args:
            period: 期間
            
        Returns:
            int: Pocket Pivot回数
        """
        count = 0
        
        for i in range(len(self.df) - period, len(self.df)):
            if i < 10:
                continue
            
            # その時点でのPocket Pivotチェック
            slice_data = self.df.iloc[:i+1]
            recent = slice_data.tail(11)
            
            down_days = recent.iloc[:-1][
                recent.iloc[:-1]['Close'] < recent.iloc[:-1]['Close'].shift(1)
            ]
            
            if len(down_days) == 0:
                continue
            
            max_down_volume = down_days['Volume'].max()
            current_volume = recent.iloc[-1]['Volume']
            
            if current_volume > max_down_volume:
                count += 1
        
        return count
    
    def detect_climax_volume(self, lookback: int = 60) -> Dict:
        """
        Climax Volume (Selling/Buying Climax)を検出
        
        Args:
            lookback: 確認期間
            
        Returns:
            dict: Climax検出結果
        """
        recent_data = self.df.tail(lookback)
        
        # 最高出来高の日を検出
        max_volume_idx = recent_data['Volume'].idxmax()
        max_volume_day = recent_data.loc[max_volume_idx]
        
        # その日の価格レンジ
        day_range = max_volume_day['High'] - max_volume_day['Low']
        avg_range = (recent_data['High'] - recent_data['Low']).mean()
        
        # 終値の位置（レンジ内）
        if day_range > 0:
            close_position = (max_volume_day['Close'] - max_volume_day['Low']) / day_range
        else:
            close_position = 0.5
        
        # Climax判定
        is_climax = max_volume_day['Volume'] > recent_data['Volume'].mean() * 2.5
        
        if is_climax:
            if close_position >= 0.5 and max_volume_day['Close'] < max_volume_day['Open']:
                # Selling Climax
                climax_type = 'Selling Climax'
                interpretation = 'パニック売りの頂点、弱気筋の降伏'
                implication = 'Stage 4末期 → Stage 1への移行可能性'
            elif close_position < 0.5 and max_volume_day['Close'] > max_volume_day['Open']:
                # Buying Climax
                climax_type = 'Buying Climax'
                interpretation = '強気筋の降伏(遅れた参入)、分配'
                implication = 'Stage 2末期 → Stage 3への移行可能性'
            else:
                climax_type = 'Neutral Climax'
                interpretation = '高出来高だが明確な方向性なし'
                implication = '継続監視'
            
            return {
                'detected': True,
                'type': climax_type,
                'date': max_volume_idx,
                'volume': max_volume_day['Volume'],
                'close_position': close_position,
                'interpretation': interpretation,
                'implication': implication
            }
        
        return {
            'detected': False,
            'reason': '出来高が基準未達'
        }
    
    def determine_wyckoff_phase(self) -> Dict:
        """
        Wyckoffの蓄積/分配フェーズを判定
        
        Returns:
            dict: フェーズ情報
        """
        # Selling Climax検出
        selling_climax = self.detect_climax_volume(60)
        
        # 出来高トレンド
        volume_trend = self.calculate_up_down_volume_ratio(20)
        
        # Pocket Pivot
        pocket_pivots = self.count_pocket_pivots(20)
        
        # フェーズ判定ロジック
        if selling_climax['detected'] and selling_climax['type'] == 'Selling Climax':
            # Phase A: Selling Climax検出
            phase = 'Phase A'
            description = 'Selling Climax検出、パニック売りの頂点'
            
        elif volume_trend['pattern'] == 'Accumulation' and pocket_pivots >= 2:
            # Phase C-D: 蓄積が進行中
            if pocket_pivots >= 3:
                phase = 'Phase D'
                description = 'Sign of Strength (SOS)、需要が供給を圧倒'
            else:
                phase = 'Phase C'
                description = 'Spring/Shakeout、最後の弱気筋の一掃'
                
        elif volume_trend['pattern'] == 'Accumulation':
            # Phase B: 横ばい取引、供給の枯渇
            phase = 'Phase B'
            description = '横ばい取引、供給の枯渇が進行中'
            
        elif volume_trend['pattern'] == 'Distribution':
            # 分配フェーズ
            phase = 'Distribution'
            description = '機関投資家が分配中、警戒が必要'
            
        else:
            phase = 'Undefined'
            description = '明確なフェーズ判定不可'
        
        return {
            'phase': phase,
            'description': description,
            'selling_climax': selling_climax['detected'],
            'volume_pattern': volume_trend['pattern'],
            'pocket_pivots_count': pocket_pivots
        }
    
    def calculate_volume_score(self) -> Dict:
        """
        出来高総合スコアを計算（100点満点）
        
        内訳:
        - Up/Down比率スコア (40点)
        - Pocket Pivot (20点)
        - Wyckoff蓄積確認 (20点)
        - OBV状態 (20点)
        """
        score = 0
        details = {}
        
        # 1. Up/Down比率スコア (40点)
        ratio_analysis = self.calculate_up_down_volume_ratio(20)
        ratio = ratio_analysis['ratio']
        
        if ratio >= 1.5:
            ratio_score = 40
            details['ratio_rating'] = 'A'
        elif ratio >= 1.2:
            ratio_score = 30
            details['ratio_rating'] = 'B'
        elif ratio >= 1.0:
            ratio_score = 20
            details['ratio_rating'] = 'C'
        else:
            ratio_score = 10
            details['ratio_rating'] = 'D'
        
        score += ratio_score
        details['ratio_score'] = ratio_score
        details['up_down_ratio'] = ratio
        
        # 2. Pocket Pivot (20点)
        pp_count = self.count_pocket_pivots(20)
        
        if pp_count >= 3:
            pp_score = 20
            details['pp_rating'] = 'A'
        elif pp_count >= 1:
            pp_score = 10
            details['pp_rating'] = 'B'
        else:
            pp_score = 0
            details['pp_rating'] = 'C'
        
        score += pp_score
        details['pp_score'] = pp_score
        details['pocket_pivots'] = pp_count
        
        # 3. Wyckoff蓄積確認 (20点)
        wyckoff = self.determine_wyckoff_phase()
        phase = wyckoff['phase']
        
        if phase in ['Phase D', 'Phase E']:
            wyckoff_score = 20
            details['wyckoff_rating'] = 'A'
        elif phase == 'Phase C':
            wyckoff_score = 15
            details['wyckoff_rating'] = 'B'
        elif phase == 'Phase B':
            wyckoff_score = 10
            details['wyckoff_rating'] = 'C'
        else:
            wyckoff_score = 5
            details['wyckoff_rating'] = 'D'
        
        score += wyckoff_score
        details['wyckoff_score'] = wyckoff_score
        details['wyckoff_phase'] = phase
        
        # 4. OBV状態 (20点)
        if 'OBV' in self.df.columns and len(self.df) >= 50:
            current_obv = self.df['OBV'].iloc[-1]
            obv_50d_ago = self.df['OBV'].iloc[-50] if len(self.df) >= 50 else self.df['OBV'].iloc[0]
            
            if current_obv > obv_50d_ago:
                obv_score = 20
                details['obv_rating'] = 'A'
                details['obv_trend'] = 'rising'
            elif current_obv == obv_50d_ago:
                obv_score = 10
                details['obv_rating'] = 'B'
                details['obv_trend'] = 'flat'
            else:
                obv_score = 0
                details['obv_rating'] = 'C'
                details['obv_trend'] = 'falling'
        else:
            obv_score = 10
            details['obv_rating'] = 'N/A'
            details['obv_trend'] = 'unknown'
        
        score += obv_score
        details['obv_score'] = obv_score
        
        details['total_score'] = score
        
        return details


if __name__ == '__main__':
    # テスト用
    from data_fetcher import fetch_stock_data
    from indicators import calculate_all_basic_indicators
    
    print("出来高分析のテストを開始...")
    
    test_tickers = ['AAPL', 'TSLA', 'NVDA']
    
    for ticker in test_tickers:
        print(f"\n{ticker} の出来高分析:")
        stock_df, _ = fetch_stock_data(ticker, period='2y')
        
        if stock_df is not None:
            indicators_df = calculate_all_basic_indicators(stock_df)
            indicators_df = indicators_df.dropna()
            
            if len(indicators_df) >= 60:
                analyzer = VolumeAnalyzer(indicators_df)
                
                # スコア計算
                score_result = analyzer.calculate_volume_score()
                print(f"  総合スコア: {score_result['total_score']}/100")
                print(f"  Up/Down比率: {score_result['up_down_ratio']:.2f}")
                print(f"  Wyckoffフェーズ: {score_result['wyckoff_phase']}")
                print(f"  Pocket Pivots: {score_result['pocket_pivots']}回")
