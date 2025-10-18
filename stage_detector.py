"""
Stan Weinstein Stage Analysis Detector (完全改訂版)
Stan Weinsteinの4ステージ理論に基づく正確なステージ判定

【重要な改善】
1. Volume_Analyzerとの完全統合
2. 4ステージが相補的な関係（必ずどれかに分類）
3. Stage 1とStage 3を正確に対称実装
4. 出来高を判定基準に統合
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional


class StageDetector:
    """
    Stan Weinstein Stage Analysis システム（完全改訂版）
    
    4つのステージの相補的関係:
    - Stage 1: Basing Area (下降後の横ばい、MA平坦、出来高減少)
    - Stage 2: Advancing Phase (上昇トレンド、MA上昇、高出来高)
    - Stage 3: Top Area (上昇後の横ばい、MA平坦、出来高増加)
    - Stage 4: Declining Phase (下降トレンド、MA下降)

    判定優先順位:
    1. Stage 2とStage 4（明確なトレンド）を先に判定
    2. その後、Stage 1とStage 3（横ばい）を判定
    """
    
    def __init__(self, df: pd.DataFrame, benchmark_df: pd.DataFrame = None):
        """
        Args:
            df: 指標計算済みのDataFrame（日足）
            benchmark_df: ベンチマークデータ（オプション）
        """
        self.df = df
        self.benchmark_df = benchmark_df
        self.latest = df.iloc[-1]
        
        # 主要な移動平均線（日足ベース）
        self.ma_30w = 'SMA_150'  # 30週 ≈ 150日
        self.ma_40w = 'SMA_200'  # 40週 ≈ 200日
        self.ma_10w = 'SMA_50'   # 10週 ≈ 50日
        
    def _check_ma_trend(self, ma_series: pd.Series, lookback: int = 20) -> str:
        """
        移動平均線のトレンドを判定
        
        Returns:
            str: 'rising', 'flat', 'declining'
        """
        if len(ma_series) < lookback:
            return 'unknown'
        
        recent = ma_series.tail(lookback)
        x = np.arange(len(recent))
        y = recent.values
        
        if np.linalg.norm(y) > 0:
            y_norm = y / np.linalg.norm(y)
        else:
            return 'flat'
        
        slope = np.polyfit(x, y_norm, 1)[0]
        
        # 閾値調整: より正確な判定
        if slope > 0.008:  # 上昇
            return 'rising'
        elif slope < -0.008:  # 下降
            return 'declining'
        else:  # 平坦
            return 'flat'
    
    def _calculate_price_position(self) -> Dict:
        """価格とMAの位置関係を計算"""
        current_price = self.latest['Close']
        ma_10w = self.latest[self.ma_10w]
        ma_30w = self.latest[self.ma_30w]
        ma_40w = self.latest[self.ma_40w]
        
        return {
            'price': current_price,
            'above_10w': current_price > ma_10w,
            'above_30w': current_price > ma_30w,
            'above_40w': current_price > ma_40w,
            'ma_10w_above_30w': ma_10w > ma_30w,
            'ma_30w_above_40w': ma_30w > ma_40w,
            'ma_10w_above_40w': ma_10w > ma_40w,
        }
    
    def _get_volume_characteristics(self) -> Dict:
        """
        出来高特性を取得
        Volume_Analyzerと連携
        """
        from volume_analyzer import VolumeAnalyzer

        characteristics = {
            'current_relative_volume': 1.0,
            'volume_trend': 'unknown',
            'dry_up': False,
            'surge': False,
            'churning': False
        }
        
        try:
            analyzer = VolumeAnalyzer(self.df)

            # 相対出来高
            if 'Relative_Volume' in self.df.columns:
                characteristics['current_relative_volume'] = self.latest['Relative_Volume']

            # Dry up検出（Stage 1特徴）
            dry_up_result = analyzer.detect_dry_up(20)
            characteristics['dry_up'] = dry_up_result.get('detected', False)
            
            # Surge検出（Stage 2ブレイクアウト特徴）
            surge_result = analyzer.detect_volume_surge(threshold=1.5)
            characteristics['surge'] = surge_result.get('detected', False)

            # Churning検出（Stage 3特徴）
            # 出来高の変動係数が高い
            recent_volume = self.df['Volume'].tail(20)
            vol_cv = recent_volume.std() / recent_volume.mean() if recent_volume.mean() > 0 else 0
            characteristics['churning'] = vol_cv > 0.5

            # 出来高トレンド
            recent_rvol = self.df['Relative_Volume'].tail(20) if 'Relative_Volume' in self.df.columns else None
            if recent_rvol is not None and len(recent_rvol) >= 2:
                if recent_rvol.iloc[-1] < recent_rvol.mean() * 0.7:
                    characteristics['volume_trend'] = 'decreasing'
                elif recent_rvol.iloc[-1] > recent_rvol.mean() * 1.5:
                    characteristics['volume_trend'] = 'increasing'
                else:
                    characteristics['volume_trend'] = 'stable'
        
        except Exception as e:
            pass  # Volume_Analyzerが利用できない場合は基本情報のみ
        
        return characteristics
    
    def _detect_stage_2(self) -> Tuple[bool, str, Dict]:
        """
        Stage 2（上昇期）を検出
        
        判定基準:
        1. 価格が30週MA・40週MAの上
        2. 30週MAが上昇トレンド
        3. 高値更新と高安値を形成
        4. 【重要】ブレイクアウト時に高出来高（1.5倍以上）
        
        Returns:
            tuple: (Stage 2判定, サブステージ, 詳細情報)
        """
        position = self._calculate_price_position()
        ma_30w_trend = self._check_ma_trend(self.df[self.ma_30w], 20)
        volume_chars = self._get_volume_characteristics()
        
        details = {
            'ma_30w_trend': ma_30w_trend,
            'volume_trend': volume_chars['volume_trend'],
            'relative_volume': volume_chars['current_relative_volume']
        }
        
        # Stage 2の基本条件
        price_above_mas = position['above_30w'] and position['above_40w']
        ma_rising = ma_30w_trend == 'rising'
        higher_highs_lows = self._check_higher_highs_lows(50)
        
        # 【重要追加】出来高確認
        # Stage 2初期（ブレイクアウト）には高出来高が必須
        high_volume = (
            volume_chars['current_relative_volume'] >= 1.5 or
            volume_chars['surge']
        )

        details['higher_highs_lows'] = higher_highs_lows
        details['high_volume'] = high_volume
        
        # Stage 2判定
        if price_above_mas and ma_rising and higher_highs_lows:
            # 出来高が不十分な場合は警告
            if not high_volume:
                details['volume_warning'] = 'Stage 2だが出来高不足 - 偽ブレイクアウトの可能性'

            # サブステージ判定
            substage = self._determine_stage2_substage(position, volume_chars, details)
            return True, substage, details
        
        return False, "", details
    
    def _check_higher_highs_lows(self, lookback: int = 50) -> bool:
        """価格が高値更新と高安値を形成しているかチェック"""
        recent = self.df.tail(lookback)
        
        if len(recent) < 20:
            return False
        
        mid_point = len(recent) // 2
        recent_high = recent['High'].iloc[mid_point:].max()
        past_high = recent['High'].iloc[:mid_point].max()
        
        higher_high = recent_high > past_high
        
        recent_low = recent['Low'].iloc[mid_point:].min()
        past_low = recent['Low'].iloc[:mid_point].min()
        
        higher_low = recent_low >= past_low * 0.95
        
        return higher_high and higher_low
    
    def _determine_stage2_substage(self, position: Dict, volume_chars: Dict,
                                   details: Dict) -> str:
        """Stage 2のサブステージを判定"""
        stage2_duration = self._estimate_stage2_duration()
        
        # 52週高値との距離
        if 'High_52W' in self.df.columns:
            high_52w = self.latest['High_52W']
            current_price = position['price']
            dist_from_high = (high_52w - current_price) / high_52w if high_52w > 0 else 0
        else:
            dist_from_high = 0
        
        above_10w = position['above_10w']
        
        # サブステージ判定ロジック
        if stage2_duration < 60 and above_10w:
            return "2A"  # Stage 2初期（ブレイクアウト後2-3ヶ月）
        elif dist_from_high < 0.20 and stage2_duration > 120:
            return "2B"  # Stage 2後期（高値に近く、長期間経過）
        else:
            return "2"   # Stage 2中期
    
    def _estimate_stage2_duration(self) -> int:
        """Stage 2開始からの経過日数を推定"""
        ma_10w_series = self.df[self.ma_10w]
        ma_30w_series = self.df[self.ma_30w]
        
        for i in range(len(self.df) - 1, max(0, len(self.df) - 250), -1):
            if i < 1:
                break
            if (ma_10w_series.iloc[i] > ma_30w_series.iloc[i] and
                ma_10w_series.iloc[i-1] <= ma_30w_series.iloc[i-1]):
                return len(self.df) - i
        
        return 200  # デフォルト値
    
    def _detect_stage_4(self) -> Tuple[bool, str, Dict]:
        """
        Stage 4（下降期）を検出
        
        判定基準:
        1. 価格が30週MA・40週MAの下
        2. 30週MAが下降トレンド
        3. 低い高値と低い安値を形成

        Returns:
            tuple: (Stage 4判定, サブステージ, 詳細情報)
        """
        position = self._calculate_price_position()
        ma_30w_trend = self._check_ma_trend(self.df[self.ma_30w], 20)
        volume_chars = self._get_volume_characteristics()

        details = {
            'ma_30w_trend': ma_30w_trend,
            'volume_trend': volume_chars['volume_trend']
        }

        # Stage 4の基本条件
        price_below_mas = not position['above_30w'] and not position['above_40w']
        ma_declining = ma_30w_trend == 'declining'
        lower_highs_lows = self._check_lower_highs_lows(50)

        details['lower_highs_lows'] = lower_highs_lows

        # Stage 4判定
        if price_below_mas and (ma_declining or lower_highs_lows):
            substage = self._determine_stage4_substage(position, volume_chars, details)
            return True, substage, details

        return False, "", details

    def _check_lower_highs_lows(self, lookback: int = 50) -> bool:
        """価格が低い高値と低い安値を形成しているかチェック"""
        recent = self.df.tail(lookback)

        if len(recent) < 20:
            return False

        mid_point = len(recent) // 2

        recent_high = recent['High'].iloc[mid_point:].max()
        past_high = recent['High'].iloc[:mid_point].max()
        lower_high = recent_high < past_high

        recent_low = recent['Low'].iloc[mid_point:].min()
        past_low = recent['Low'].iloc[:mid_point].min()
        lower_low = recent_low < past_low

        return lower_high and lower_low

    def _determine_stage4_substage(self, position: Dict, volume_chars: Dict,
                                   details: Dict) -> str:
        """Stage 4のサブステージを判定"""
        # 52週安値との距離
        if 'Low_52W' in self.df.columns:
            low_52w = self.latest['Low_52W']
            current_price = position['price']
            dist_from_low = (current_price - low_52w) / low_52w if low_52w > 0 else 1
        else:
            dist_from_low = 1

        # 下降の勢い
        recent_momentum = self._calculate_recent_momentum(20)

        # Selling Climax検出
        volume_surge = volume_chars.get('surge', False)

        # サブステージ判定ロジック
        if dist_from_low < 0.05 and recent_momentum > -0.02:
            return "4B-"  # 底打ち近い
        elif dist_from_low < 0.15 and not volume_surge:
            return "4B"   # Stage 4後期
        elif recent_momentum < -0.10 or volume_surge:
            return "4A"   # Stage 4初期、急落中
        else:
            return "4"    # Stage 4中期

    def _calculate_recent_momentum(self, lookback: int = 20) -> float:
        """最近の価格モメンタムを計算"""
        recent = self.df['Close'].tail(lookback)
        
        if len(recent) < 2:
            return 0

        return (recent.iloc[-1] - recent.iloc[0]) / recent.iloc[0]

    def _detect_stage_1(self) -> Tuple[bool, str, Dict]:
        """
        Stage 1（ベース形成期）を検出

        判定基準（理論通り）:
        1. 30週MA（150日MA）が平坦化
        2. 価格が30週MAの上下を行き来（oscillates）
        3. 出来高が減少（Dry up）
        4. 横ばいの価格動き（下降後）
        
        Returns:
            tuple: (Stage 1判定, サブステージ, 詳細情報)
        """
        position = self._calculate_price_position()
        ma_30w_trend = self._check_ma_trend(self.df[self.ma_30w], 20)
        ma_40w_trend = self._check_ma_trend(self.df[self.ma_40w], 30)
        volume_chars = self._get_volume_characteristics()
        
        details = {
            'ma_30w_trend': ma_30w_trend,
            'ma_40w_trend': ma_40w_trend,
            'volume_trend': volume_chars['volume_trend'],
            'dry_up': volume_chars['dry_up']
        }
        
        # Stage 1の基本条件（理論通り）
        ma_flat = ma_30w_trend == 'flat'
        
        # 価格が30週MAの周辺で推移（±8%以内）
        current_price = position['price']
        ma_30w = self.latest[self.ma_30w]
        
        if ma_30w > 0:
            price_ratio = current_price / ma_30w
            price_oscillating = 0.92 < price_ratio < 1.08  # ±8%（理論に基づく調整）
        else:
            price_oscillating = False
        
        # 価格の横ばいチェック
        price_range = self._calculate_recent_price_range(50)
        price_sideways = price_range < 0.25
        
        # 出来高の減少傾向（Dry up）
        volume_decreasing = (
            volume_chars['dry_up'] or
            volume_chars['volume_trend'] == 'decreasing'
        )

        details['price_range'] = price_range
        details['price_oscillating'] = price_oscillating
        details['volume_decreasing'] = volume_decreasing
        
        # Stage 1判定
        if ma_flat and price_oscillating and price_sideways:
            # 出来高減少は必須ではないが、あれば強力なシグナル
            if volume_decreasing:
                details['quality'] = 'High - Dry up confirmed'
            else:
                details['quality'] = 'Moderate - Volume not decreasing yet'

            # サブステージ判定
            substage = self._determine_stage1_substage(position, volume_chars, details)
            return True, substage, details
        
        return False, "", details
    
    def _calculate_recent_price_range(self, lookback: int = 50) -> float:
        """最近の価格変動幅を計算"""
        recent = self.df.tail(lookback)
        
        if len(recent) < 10:
            return 0
        
        high = recent['High'].max()
        low = recent['Low'].min()
        avg = recent['Close'].mean()
        
        if avg > 0:
            return (high - low) / avg
        else:
            return 0
    
    def _determine_stage1_substage(self, position: Dict, volume_chars: Dict,
                                   details: Dict) -> str:
        """Stage 1のサブステージを判定"""
        ma_40w_trend = details['ma_40w_trend']
        current_price = position['price']
        ma_40w = self.latest[self.ma_40w]
        
        # 価格が40週MAより上か
        above_40w = current_price > ma_40w if ma_40w > 0 else False
        
        # 最近の価格動向
        recent_prices = self.df['Close'].tail(20)
        price_improving = recent_prices.iloc[-1] > recent_prices.iloc[0]
        
        # 52週高値への接近度
        if 'High_52W' in self.df.columns:
            high_52w = self.latest['High_52W']
            dist_from_high = abs(high_52w - current_price) / high_52w if high_52w > 0 else 1
        else:
            dist_from_high = 1
        
        # 出来高のDry up
        volume_dry_up = volume_chars.get('dry_up', False)
        
        # サブステージ判定ロジック（理論通り）
        if (above_40w and ma_40w_trend == 'rising' and
            price_improving and dist_from_high < 0.10 and volume_dry_up):
            return "1B"  # ブレイクアウト準備完了
        elif above_40w or (price_improving and ma_40w_trend in ['flat', 'rising']):
            return "1"   # ベース形成中
        else:
            return "1A"  # ベース初期
    
    def _detect_stage_3(self) -> Tuple[bool, str, Dict]:
        """
        Stage 3（天井形成期）を検出
        
        判定基準（Stage 1の逆ロジック）:
        1. 30週MA（150日MA）が平坦化（上昇後）
        2. 価格が30週MAの上下を行き来
        3. 出来高が増加（Churning）
        4. 横ばいの価格動き（上昇後）
        
        Returns:
            tuple: (Stage 3判定, サブステージ, 詳細情報)
        """
        position = self._calculate_price_position()
        ma_30w_trend = self._check_ma_trend(self.df[self.ma_30w], 20)
        ma_40w_trend = self._check_ma_trend(self.df[self.ma_40w], 30)
        volume_chars = self._get_volume_characteristics()
        
        details = {
            'ma_30w_trend': ma_30w_trend,
            'ma_40w_trend': ma_40w_trend,
            'volume_trend': volume_chars['volume_trend'],
            'churning': volume_chars['churning']
        }
        
        # Stage 3の基本条件（Stage 1の逆）
        ma_flat = ma_30w_trend == 'flat'
        
        # 価格が30週MAの周辺で推移
        current_price = position['price']
        ma_30w = self.latest[self.ma_30w]
        
        if ma_30w > 0:
            price_ratio = current_price / ma_30w
            price_oscillating = 0.92 < price_ratio < 1.08
        else:
            price_oscillating = False
        
        # 価格の横ばいチェック
        price_range = self._calculate_recent_price_range(50)
        price_sideways = price_range < 0.25
        
        # 出来高の増加傾向（Churning - Stage 1の逆）
        volume_increasing = (
            volume_chars['churning'] or
            volume_chars['volume_trend'] == 'increasing'
        )
        
        details['price_range'] = price_range
        details['price_oscillating'] = price_oscillating
        details['volume_increasing'] = volume_increasing
        
        # Stage 3判定
        if ma_flat and price_oscillating and price_sideways:
            # 出来高増加（Churning）は重要なシグナル
            if volume_increasing:
                details['quality'] = 'High - Churning detected'
            else:
                details['quality'] = 'Moderate - Volume not churning yet'

            # サブステージ判定
            substage = self._determine_stage3_substage(position, volume_chars, details)
            return True, substage, details
        
        return False, "", details
    
    def _determine_stage3_substage(self, position: Dict, volume_chars: Dict,
                                   details: Dict) -> str:
        """Stage 3のサブステージを判定"""
        # 10週MAとの関係
        current_price = position['price']
        ma_10w = self.latest[self.ma_10w]
        below_10w = current_price < ma_10w
        
        # 最近の価格モメンタム
        recent_momentum = self._calculate_recent_momentum(20)
        
        # 出来高のChurning
        volume_churning = volume_chars['churning']
        
        # サブステージ判定ロジック
        if below_10w:
            return "3B"  # 天井形成の最終段階
        elif recent_momentum < 0 or volume_churning:
            return "3"   # 天井形成中
        else:
            return "3A"  # 天井形成の初期
    
    def determine_stage(self) -> Tuple[int, str]:
        """
        現在のステージを判定（相補的な4ステージ）

        判定優先順位:
        1. Stage 2（明確な上昇トレンド）
        2. Stage 4（明確な下降トレンド）
        3. Stage 1とStage 3（横ばい）を文脈で判定
        
        Returns:
            tuple: (ステージ番号, サブステージ)
        """
        # Stage 2の判定（最優先）
        is_stage2, stage2_substage, stage2_details = self._detect_stage_2()
        if is_stage2:
            return 2, stage2_substage
        
        # Stage 4の判定
        is_stage4, stage4_substage, stage4_details = self._detect_stage_4()
        if is_stage4:
            return 4, stage4_substage
        
        # Stage 1とStage 3の判定（両方とも横ばいパターン）
        is_stage1, stage1_substage, stage1_details = self._detect_stage_1()
        is_stage3, stage3_substage, stage3_details = self._detect_stage_3()

        # Stage 1とStage 3の両方が検出された場合、文脈で判断
        if is_stage1 and is_stage3:
            # 過去のトレンドを確認（より長期的な文脈）
            historical_trend = self._check_historical_trend(100)

            if historical_trend == 'prior_uptrend':
                # 上昇後の横ばい → Stage 3
                return 3, stage3_substage
            else:
                # 下降後の横ばい → Stage 1
                return 1, stage1_substage

        # Stage 1のみ検出
        if is_stage1:
            return 1, stage1_substage
        
        # Stage 3のみ検出
        if is_stage3:
            return 3, stage3_substage
        
        # どれにも当てはまらない場合、フォールバック判定
        return self._fallback_stage_detection()
    
    def _check_historical_trend(self, lookback: int = 100) -> str:
        """
        過去のトレンドを確認（Stage 1 vs Stage 3の文脈判定用）

        Returns:
            str: 'prior_uptrend', 'prior_downtrend', 'neutral'
        """
        if len(self.df) < lookback:
            lookback = len(self.df)

        historical_data = self.df.tail(lookback)

        # 期間の前半と後半で価格を比較
        mid_point = len(historical_data) // 2
        first_half_avg = historical_data['Close'].iloc[:mid_point].mean()
        second_half_avg = historical_data['Close'].iloc[mid_point:].mean()

        if second_half_avg > first_half_avg * 1.15:
            return 'prior_uptrend'
        elif second_half_avg < first_half_avg * 0.85:
            return 'prior_downtrend'
        else:
            return 'neutral'

    def _fallback_stage_detection(self) -> Tuple[int, str]:
        """
        フォールバック判定（MAの配置のみで判断）

        Returns:
            tuple: (ステージ番号, サブステージ)
        """
        position = self._calculate_price_position()
        
        current_price = position['price']
        ma_30w = self.latest[self.ma_30w]
        ma_40w = self.latest[self.ma_40w]
        
        if current_price > ma_30w > ma_40w:
            return 2, "2"  # 上昇トレンドの可能性
        elif current_price < ma_30w < ma_40w:
            return 4, "4"  # 下降トレンドの可能性
        else:
            # 横ばいだが、文脈が不明
            # 安全側に倒してStage 1と判定
            return 1, "1"
    
    def check_minervini_template(self) -> Dict:
        """
        Mark Minervini Trend Template（8基準）をチェック
        
        ※ Stage 2とStage 1B（準備段階）で使用可能
        
        Returns:
            dict: 各基準の結果とスコア
        """
        checks = {}
        
        # 現在のステージを確認
        current_stage, current_substage = self.determine_stage()
        
        # Stage 2またはStage 1Bで適用
        if current_stage == 2 or (current_stage == 1 and current_substage == '1B'):
            applicable = True
        else:
            return {
                'applicable': False,
                'reason': f'Minerviniテンプレートは Stage 2 または Stage 1B で適用（現在: Stage {current_stage} {current_substage}）',
                'criteria_met': 0,
                'total_criteria': 8,
                'score': 0
            }
        
        # 現在の価格とMA
        current_price = self.latest['Close']
        sma_50 = self.latest['SMA_50']
        sma_150 = self.latest['SMA_150']
        sma_200 = self.latest['SMA_200']
        
        # 基準1: 現在価格 > 150日MA and 200日MA
        checks['criterion_1'] = {
            'passed': (current_price > sma_150) and (current_price > sma_200),
            'description': '現在価格 > 150日MA & 200日MA',
            'values': {
                'price': current_price,
                'sma_150': sma_150,
                'sma_200': sma_200
            }
        }
        
        # 基準2: 150日MA > 200日MA
        checks['criterion_2'] = {
            'passed': sma_150 > sma_200,
            'description': '150日MA > 200日MA',
            'values': {
                'sma_150': sma_150,
                'sma_200': sma_200
            }
        }
        
        # 基準3: 200日MAが上昇トレンド
        if len(self.df) >= 21:
            sma_200_20d_ago = self.df['SMA_200'].iloc[-21]
            criterion_3_passed = sma_200 > sma_200_20d_ago
            
            if len(self.df) >= 101:
                sma_200_100d_ago = self.df['SMA_200'].iloc[-101]
                long_term_rising = sma_200 > sma_200_100d_ago
            else:
                long_term_rising = False
        else:
            criterion_3_passed = False
            long_term_rising = False
        
        checks['criterion_3'] = {
            'passed': criterion_3_passed,
            'description': '200日MAが最低1ヶ月上昇トレンド',
            'values': {
                'sma_200_current': sma_200,
                'sma_200_20d_ago': sma_200_20d_ago if len(self.df) >= 21 else None,
                'long_term_rising': long_term_rising
            }
        }
        
        # 基準4: 50日MA > 150日MA and 200日MA
        checks['criterion_4'] = {
            'passed': (sma_50 > sma_150) and (sma_50 > sma_200),
            'description': '50日MA > 150日MA & 200日MA',
            'values': {
                'sma_50': sma_50,
                'sma_150': sma_150,
                'sma_200': sma_200
            }
        }
        
        # 基準5: 現在価格 > 50日MA
        checks['criterion_5'] = {
            'passed': current_price > sma_50,
            'description': '現在価格 > 50日MA',
            'values': {
                'price': current_price,
                'sma_50': sma_50
            }
        }
        
        # 基準6: 52週安値から30%以上上
        low_52w = self.latest['Low_52W']
        if pd.notna(low_52w) and low_52w > 0:
            gain_from_low = (current_price - low_52w) / low_52w
            criterion_6_passed = gain_from_low > 0.30
        else:
            gain_from_low = 0
            criterion_6_passed = False
        
        checks['criterion_6'] = {
            'passed': criterion_6_passed,
            'description': '52週安値から30%以上上',
            'values': {
                'price': current_price,
                'low_52w': low_52w,
                'gain_pct': gain_from_low * 100 if pd.notna(gain_from_low) else 0
            }
        }
        
        # 基準7: 52週高値の25%以内
        high_52w = self.latest['High_52W']
        if pd.notna(high_52w) and high_52w > 0:
            dist_from_high = (high_52w - current_price) / high_52w
            criterion_7_passed = dist_from_high < 0.25
        else:
            dist_from_high = 1
            criterion_7_passed = False
        
        checks['criterion_7'] = {
            'passed': criterion_7_passed,
            'description': '52週高値の25%以内',
            'values': {
                'price': current_price,
                'high_52w': high_52w,
                'dist_pct': dist_from_high * 100 if pd.notna(dist_from_high) else 100
            }
        }
        
        # 基準8: RS Rating ≥ 70
        if 'RS_Rating' in self.df.columns:
            rs_rating = self.latest['RS_Rating']
            criterion_8_passed = rs_rating >= 70
        else:
            rs_rating = None
            criterion_8_passed = False
        
        checks['criterion_8'] = {
            'passed': criterion_8_passed,
            'description': 'RS Rating ≥ 70',
            'values': {
                'rs_rating': rs_rating
            }
        }
        
        # スコア計算
        criteria_met = sum(1 for c in checks.values() if c['passed'])
        template_score = (criteria_met / len(checks)) * 100
        all_pass = all(c['passed'] for c in checks.values())
        
        return {
            'applicable': True,
            'all_criteria_met': all_pass,
            'score': template_score,
            'checks': checks,
            'criteria_met': criteria_met,
            'total_criteria': len(checks),
            'stage': current_stage,
            'substage': current_substage
        }


if __name__ == '__main__':
    # テスト用
    from data_fetcher import fetch_stock_data
    from indicators import calculate_all_basic_indicators
    
    print("Stan Weinstein Stage Analysis（完全改訂版）のテストを開始...")
    
    test_tickers = ['AAPL', 'TSLA', 'NVDA']
    
    for ticker in test_tickers:
        print(f"\n{'='*60}")
        print(f"{ticker} のステージ分析:")
        print(f"{'='*60}")
        
        stock_df, benchmark_df = fetch_stock_data(ticker, period='2y')
        
        if stock_df is not None:
            indicators_df = calculate_all_basic_indicators(stock_df)
            indicators_df = indicators_df.dropna()
            
            if len(indicators_df) >= 200:
                detector = StageDetector(indicators_df, benchmark_df)
                stage, substage = detector.determine_stage()
                
                print(f"ステージ: Stage {stage} ({substage})")
                
                # Minerviniテンプレートチェック
                template_result = detector.check_minervini_template()
                if template_result['applicable']:
                    print(f"\nMinerviniテンプレート:")
                    print(f"  基準充足: {template_result['criteria_met']}/{template_result['total_criteria']}")
                    print(f"  スコア: {template_result['score']:.1f}/100")
                else:
                    print(f"\nMinerviniテンプレート: {template_result['reason']}")
                
                # 価格情報
                current_price = detector.latest['Close']
                ma_50 = detector.latest['SMA_50']
                ma_150 = detector.latest['SMA_150']
                ma_200 = detector.latest['SMA_200']
                
                print(f"\n価格情報:")
                print(f"  現在価格: ${current_price:.2f}")
                print(f"  50日MA: ${ma_50:.2f} ({'+' if current_price > ma_50 else '-'})")
                print(f"  150日MA: ${ma_150:.2f} ({'+' if current_price > ma_150 else '-'})")
                print(f"  200日MA: ${ma_200:.2f} ({'+' if current_price > ma_200 else '-'})")