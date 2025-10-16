"""
出来高分析モジュール（Stage統合版）
Wyckoff理論 + O'Neil + Minervini + Stan Weinsteinの統合

【改善点】
1. Stage別の出来高判定基準を追加
2. Dry up（出来高減少）とSurge（出来高急増）の明確な検出
3. Stage移行時の出来高検証機能
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple


class VolumeAnalyzer:
    """
    出来高分析システム（Stage統合版）
    
    理論的基盤:
    - Wyckoffの三大法則（供給と需要、原因と結果、努力と結果）
    - O'Neilのブレイクアウト出来高理論
    - MinerviniのDry Up概念
    - Stan WeinsteinのStage別出来高パターン
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
    
    def detect_dry_up(self, lookback: int = 20) -> Dict:
        """
        Dry Up（出来高減少）を検出
        
        Stage 1後期やベース右側での出来高減少
        
        Args:
            lookback: 確認期間
            
        Returns:
            dict: Dry up検出結果
        """
        recent_data = self.df.tail(lookback)
        
        if 'Volume_SMA_50' not in recent_data.columns:
            return {
                'detected': False,
                'reason': 'Volume_SMA_50が存在しません'
            }
        
        # 最近の平均出来高と50日平均出来高の比較
        recent_avg_volume = recent_data['Volume'].mean()
        avg_volume_sma = recent_data['Volume_SMA_50'].mean()
        
        if avg_volume_sma > 0:
            dry_up_ratio = recent_avg_volume / avg_volume_sma
        else:
            return {
                'detected': False,
                'reason': '出来高データ不足'
            }
        
        # Dry up判定基準
        # - 最近の平均出来高が50日平均の50%以下
        # - トレンドが減少傾向
        is_dry_up = dry_up_ratio < 0.50
        
        # 減少トレンドの確認
        mid_point = len(recent_data) // 2
        first_half_avg = recent_data['Volume'].iloc[:mid_point].mean()
        second_half_avg = recent_data['Volume'].iloc[mid_point:].mean()
        
        decreasing_trend = second_half_avg < first_half_avg
        
        if is_dry_up and decreasing_trend:
            return {
                'detected': True,
                'dry_up_ratio': dry_up_ratio,
                'recent_avg': recent_avg_volume,
                'baseline_avg': avg_volume_sma,
                'interpretation': 'Stage 1後期、供給が枯渇、ブレイクアウト準備の兆候',
                'quality': 'excellent' if dry_up_ratio < 0.30 else 'good'
            }
        
        return {
            'detected': False,
            'dry_up_ratio': dry_up_ratio,
            'reason': '出来高減少が不十分'
        }
    
    def detect_volume_surge(self, threshold: float = 2.0) -> Dict:
        """
        出来高Surge（急増）を検出
        
        Stage 2ブレイクアウト時やStage 4 Selling Climax時
        
        Args:
            threshold: 出来高倍率の閾値（デフォルト2.0倍）
            
        Returns:
            dict: Surge検出結果
        """
        if 'Relative_Volume' not in self.df.columns:
            return {
                'detected': False,
                'reason': 'Relative_Volume列が存在しません'
            }
        
        current_rvol = self.latest['Relative_Volume']
        
        # Surge判定
        is_surge = current_rvol >= threshold
        
        if is_surge:
            # 価格変動を確認して分類
            current_close = self.latest['Close']
            prev_close = self.df['Close'].iloc[-2] if len(self.df) >= 2 else current_close
            
            price_change_pct = ((current_close - prev_close) / prev_close * 100) if prev_close > 0 else 0
            
            if price_change_pct > 5:
                surge_type = 'Breakout Surge'
                interpretation = 'Stage 2ブレイクアウト、強力な買い需要'
            elif price_change_pct < -5:
                surge_type = 'Selling Climax'
                interpretation = 'Stage 4パニック売り、底打ちの可能性'
            else:
                surge_type = 'Churning'
                interpretation = 'Stage 3天井形成、分配の可能性'
            
            return {
                'detected': True,
                'relative_volume': current_rvol,
                'surge_type': surge_type,
                'price_change_pct': price_change_pct,
                'interpretation': interpretation,
                'quality': 'excellent' if current_rvol >= 3.0 else 'good'
            }
        
        return {
            'detected': False,
            'relative_volume': current_rvol,
            'reason': f'出来高が閾値{threshold}倍未満'
        }
    
    def verify_stage_transition_volume(self, from_stage: int, to_stage: int) -> Dict:
        """
        Stage移行時の出来高を検証
        
        Args:
            from_stage: 移行元のステージ
            to_stage: 移行先のステージ
            
        Returns:
            dict: 検証結果
        """
        surge = self.detect_volume_surge()
        
        # Stage 1 → Stage 2: 高出来高ブレイクアウトが必要
        if from_stage == 1 and to_stage == 2:
            if surge['detected'] and surge['surge_type'] == 'Breakout Surge':
                return {
                    'valid': True,
                    'confidence': 'high',
                    'message': 'Stage 2ブレイクアウト確認、高出来高でサポート',
                    'volume_multiplier': surge['relative_volume']
                }
            else:
                return {
                    'valid': False,
                    'confidence': 'low',
                    'message': 'ブレイクアウトに必要な出来高が不足、偽ブレイクアウトの可能性',
                    'volume_multiplier': surge.get('relative_volume', 0)
                }
        
        # Stage 2 → Stage 3: 出来高増加とChurning
        elif from_stage == 2 and to_stage == 3:
            ratio_analysis = self.calculate_up_down_volume_ratio(20)
            if ratio_analysis['pattern'] == 'Distribution':
                return {
                    'valid': True,
                    'confidence': 'medium',
                    'message': 'Stage 3移行、分配の兆候あり',
                    'up_down_ratio': ratio_analysis['ratio']
                }
            else:
                return {
                    'valid': False,
                    'confidence': 'medium',
                    'message': 'Stage 3移行の出来高パターンが不明確',
                    'up_down_ratio': ratio_analysis['ratio']
                }
        
        # Stage 3 → Stage 4: 下抜け時の出来高
        elif from_stage == 3 and to_stage == 4:
            if surge['detected']:
                return {
                    'valid': True,
                    'confidence': 'high',
                    'message': 'Stage 4移行確認、高出来高でブレイクダウン',
                    'volume_multiplier': surge['relative_volume']
                }
            else:
                return {
                    'valid': True,
                    'confidence': 'medium',
                    'message': 'Stage 4移行、出来高は中程度',
                    'volume_multiplier': surge.get('relative_volume', 0)
                }
        
        # Stage 4 → Stage 1: Selling Climaxの有無
        elif from_stage == 4 and to_stage == 1:
            if surge['detected'] and surge['surge_type'] == 'Selling Climax':
                return {
                    'valid': True,
                    'confidence': 'high',
                    'message': 'Selling Climax検出、Stage 1入りの可能性',
                    'volume_multiplier': surge['relative_volume']
                }
            else:
                return {
                    'valid': True,
                    'confidence': 'low',
                    'message': 'Stage 1移行、Selling Climaxは未検出',
                    'volume_multiplier': surge.get('relative_volume', 0)
                }
        
        return {
            'valid': False,
            'confidence': 'low',
            'message': f'Stage {from_stage} → {to_stage}の移行は想定外',
        }
    
    def analyze_volume_for_stage(self, stage: int, substage: str) -> Dict:
        """
        Stage別の出来高分析
        
        Args:
            stage: 現在のステージ
            substage: サブステージ
            
        Returns:
            dict: Stage別の出来高評価
        """
        result = {
            'stage': stage,
            'substage': substage,
            'volume_assessment': '',
            'action': '',
            'quality_score': 0
        }
        
        # Stage 1の出来高分析
        if stage == 1:
            dry_up = self.detect_dry_up(20)
            ratio_analysis = self.calculate_up_down_volume_ratio(20)
            
            if substage == '1B':
                # ブレイクアウト準備中
                if dry_up['detected']:
                    result['volume_assessment'] = 'Excellent - Dry up確認、供給枯渇'
                    result['action'] = 'ブレイクアウト監視、高出来高での上抜けを待つ'
                    result['quality_score'] = 90
                else:
                    result['volume_assessment'] = 'Moderate - さらなる出来高減少を待つ'
                    result['action'] = '監視継続'
                    result['quality_score'] = 60
            
            elif substage == '1':
                # ベース形成中
                if ratio_analysis['pattern'] == 'Accumulation':
                    result['volume_assessment'] = 'Good - 蓄積の兆候'
                    result['action'] = 'ベース発展を監視'
                    result['quality_score'] = 70
                else:
                    result['volume_assessment'] = 'Neutral - 様子見'
                    result['action'] = '蓄積パターン待ち'
                    result['quality_score'] = 50
            
            else:  # 1A
                result['volume_assessment'] = 'Early - ベース形成初期'
                result['action'] = 'さらなる時間が必要'
                result['quality_score'] = 40
        
        # Stage 2の出来高分析
        elif stage == 2:
            surge = self.detect_volume_surge()
            ratio_analysis = self.calculate_up_down_volume_ratio(20)
            
            if substage == '2A':
                # 上昇初期
                if surge['detected'] and surge['surge_type'] == 'Breakout Surge':
                    result['volume_assessment'] = 'Excellent - 強力なブレイクアウト'
                    result['action'] = '積極的エントリー検討'
                    result['quality_score'] = 95
                elif ratio_analysis['pattern'] == 'Accumulation':
                    result['volume_assessment'] = 'Good - 健全な需要'
                    result['action'] = 'エントリー検討'
                    result['quality_score'] = 80
                else:
                    result['volume_assessment'] = 'Weak - 出来高不足'
                    result['action'] = '偽ブレイクアウトに注意'
                    result['quality_score'] = 40
            
            elif substage == '2':
                # 上昇中期
                if ratio_analysis['pattern'] == 'Accumulation':
                    result['volume_assessment'] = 'Good - 上昇継続の可能性'
                    result['action'] = 'ホールド、押し目買い検討'
                    result['quality_score'] = 75
                else:
                    result['volume_assessment'] = 'Warning - 需要減少の兆候'
                    result['action'] = '注意深く監視'
                    result['quality_score'] = 55
            
            else:  # 2B
                # 上昇後期
                if ratio_analysis['pattern'] == 'Distribution':
                    result['volume_assessment'] = 'Caution - 分配の兆候'
                    result['action'] = '利確検討、新規エントリー非推奨'
                    result['quality_score'] = 30
                else:
                    result['volume_assessment'] = 'Late Stage - 慎重に'
                    result['action'] = 'タイトなストップロス'
                    result['quality_score'] = 50
        
        # Stage 3の出来高分析
        elif stage == 3:
            ratio_analysis = self.calculate_up_down_volume_ratio(20)
            surge = self.detect_volume_surge()
            
            if ratio_analysis['pattern'] == 'Distribution':
                result['volume_assessment'] = 'Distribution Confirmed - 分配進行中'
                result['action'] = '速やかに利確、新規エントリー回避'
                result['quality_score'] = 20
            elif surge['detected'] and surge['surge_type'] == 'Churning':
                result['volume_assessment'] = 'Churning - 激しい変動'
                result['action'] = 'ポジション削減推奨'
                result['quality_score'] = 25
            else:
                result['volume_assessment'] = 'Topping - 天井形成の可能性'
                result['action'] = 'ポジション削減検討'
                result['quality_score'] = 35
        
        # Stage 4の出来高分析
        elif stage == 4:
            surge = self.detect_volume_surge()
            
            if surge['detected'] and surge['surge_type'] == 'Selling Climax':
                if substage == '4B-':
                    result['volume_assessment'] = 'Selling Climax - 底打ちの可能性'
                    result['action'] = 'Stage 1入り監視開始'
                    result['quality_score'] = 50
                else:
                    result['volume_assessment'] = 'Selling Climax - パニック売り'
                    result['action'] = 'ロング回避、底打ち待ち'
                    result['quality_score'] = 30
            else:
                result['volume_assessment'] = 'Declining - 下降継続'
                result['action'] = 'ロングポジション回避'
                result['quality_score'] = 10
        
        return result
    
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
            near_ema = 0.97 <= current_price / ema_8 <= 1.03 if ema_8 > 0 else False
        else:
            near_ema = False
            ema_8 = None
        
        # Pocket Pivot判定
        if current_volume > max_down_volume and (near_ema or (ema_8 and current_price > ema_8)):
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
    
    def calculate_volume_score(self) -> Dict:
        """
        出来高総合スコアを計算（100点満点）
        
        内訳:
        - Up/Down比率スコア (40点)
        - Pocket Pivot (20点)
        - Dry up/Surge (20点)
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
        
        # 3. Dry up/Surge (20点)
        dry_up = self.detect_dry_up(20)
        surge = self.detect_volume_surge()
        
        if dry_up['detected'] and dry_up.get('quality') == 'excellent':
            dry_surge_score = 20
            details['dry_surge_rating'] = 'A'
        elif surge['detected'] and surge.get('quality') == 'excellent':
            dry_surge_score = 20
            details['dry_surge_rating'] = 'A'
        elif dry_up['detected'] or surge['detected']:
            dry_surge_score = 15
            details['dry_surge_rating'] = 'B'
        else:
            dry_surge_score = 5
            details['dry_surge_rating'] = 'C'
        
        score += dry_surge_score
        details['dry_surge_score'] = dry_surge_score
        details['dry_up_detected'] = dry_up['detected']
        details['surge_detected'] = surge['detected']
        
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
    
    print("出来高分析（Stage統合版）のテストを開始...")
    
    test_tickers = ['AAPL', 'TSLA', 'NVDA']
    
    for ticker in test_tickers:
        print(f"\n{'='*60}")
        print(f"{ticker} の出来高分析:")
        print(f"{'='*60}")
        
        stock_df, _ = fetch_stock_data(ticker, period='2y')
        
        if stock_df is not None:
            indicators_df = calculate_all_basic_indicators(stock_df)
            indicators_df = indicators_df.dropna()
            
            if len(indicators_df) >= 60:
                analyzer = VolumeAnalyzer(indicators_df)
                
                # スコア計算
                score_result = analyzer.calculate_volume_score()
                print(f"総合スコア: {score_result['total_score']}/100")
                print(f"Up/Down比率: {score_result['up_down_ratio']:.2f}")
                print(f"Pocket Pivots: {score_result['pocket_pivots']}回")
                print(f"Dry up検出: {score_result['dry_up_detected']}")
                print(f"Surge検出: {score_result['surge_detected']}")