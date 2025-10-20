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
    
    def __init__(self, df: pd.DataFrame, benchmark_df: pd.DataFrame = None, interval: str = '1d'):
        """
        Args:
            df: 指標計算済みのDataFrame
            benchmark_df: ベンチマークデータ（オプション）
            interval (str): '1d' or '1wk'
        """
        self.df = df
        self.benchmark_df = benchmark_df
        self.latest = df.iloc[-1]
        self.interval = interval

        # --- 時間軸に応じたパラメータ設定 ---
        if self.interval == '1wk':
            self.ma_10w = 'SMA_10'
            self.ma_30w = 'SMA_30'
            self.ma_40w = 'SMA_40'
            self.slope_lookback = 10
            self.price_range_lookback = 26 # 半年
        else: # 日足
            self.ma_10w = 'SMA_50'
            self.ma_30w = 'SMA_150'
            self.ma_40w = 'SMA_200'
            self.slope_lookback = 20
            self.price_range_lookback = 126 # 半年

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
    
    def _detect_stage_2(self, rs_rating: Optional[float] = None) -> Tuple[bool, Dict]:
        """
        Stage 2（上昇期）を検出
        
        判定基準:
        1. 価格が30週MA・40週MAの上
        2. 30週MAが上昇トレンド
        3. 高値更新と高安値を形成
        4. 【重要】ブレイクアウト時に高出来高（1.5倍以上）
        5. 【追加】RS Ratingが70以上
        
        Returns:
            tuple: (Stage 2判定, 詳細情報)
        """
        position = self._calculate_price_position()
        ma_30w_trend = self._check_ma_trend(self.df[self.ma_30w], self.slope_lookback)
        volume_chars = self._get_volume_characteristics()
        
        # RS Rating を取得
        if rs_rating is None and 'RS_Rating' in self.df.columns:
            rs_rating = self.latest.get('RS_Rating')

        details = {
            'ma_30w_trend': ma_30w_trend,
            'volume_trend': volume_chars['volume_trend'],
            'relative_volume': volume_chars['current_relative_volume'],
            'rs_rating': rs_rating
        }
        
        # Stage 2の基本条件
        price_above_mas = position['above_30w'] and position['above_40w']
        ma_rising = ma_30w_trend == 'rising'
        higher_highs_lows = self._check_higher_highs_lows(50)
        
        # 【重要追加】出来高確認
        high_volume = (
            volume_chars['current_relative_volume'] >= 1.5 or
            volume_chars['surge']
        )

        # 【重要追加】RS Rating確認
        strong_rs = rs_rating is not None and rs_rating >= 70

        details['higher_highs_lows'] = higher_highs_lows
        details['high_volume'] = high_volume
        details['strong_rs'] = strong_rs
        
        # Stage 2判定
        if price_above_mas and ma_rising and higher_highs_lows and strong_rs:
            # 出来高が不十分な場合は警告
            if not high_volume:
                details['volume_warning'] = 'Stage 2だが出来高不足 - 偽ブレイクアウトの可能性'

            return True, details
        
        return False, details
    
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
    
    def _detect_stage_4(self) -> Tuple[bool, Dict]:
        """
        Stage 4（下降期）を検出
        
        判定基準:
        1. 価格が30週MA・40週MAの下
        2. 30週MAが下降トレンド
        3. 低い高値と低い安値を形成

        Returns:
            tuple: (Stage 4判定, 詳細情報)
        """
        position = self._calculate_price_position()
        ma_30w_trend = self._check_ma_trend(self.df[self.ma_30w], self.slope_lookback)
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
            return True, details

        return False, details

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

    def _estimate_stage4_duration(self) -> int:
        """Stage 4開始からの経過日数を推定"""
        ma_10w_series = self.df[self.ma_10w]
        ma_30w_series = self.df[self.ma_30w]

        for i in range(len(self.df) - 1, max(0, len(self.df) - 250), -1):
            if i < 1:
                break
            if (ma_10w_series.iloc[i] < ma_30w_series.iloc[i] and
                ma_10w_series.iloc[i-1] >= ma_30w_series.iloc[i-1]):
                return len(self.df) - i

        return 200  # デフォルト値

    def _calculate_recent_momentum(self, lookback: int = 20) -> float:
        """最近の価格モメンタムを計算"""
        recent = self.df['Close'].tail(lookback)
        
        if len(recent) < 2:
            return 0

        return (recent.iloc[-1] - recent.iloc[0]) / recent.iloc[0]

    def _detect_stage_1(self) -> Tuple[bool, Dict]:
        """
        Stage 1（ベース形成期）を検出

        判定基準（理論通り）:
        1. 30週MA（150日MA）が平坦化
        2. 価格が30週MAの上下を行き来（oscillates）
        3. 出来高が減少（Dry up）
        4. 横ばいの価格動き（下降後）
        
        Returns:
            tuple: (Stage 1判定, 詳細情報)
        """
        # 【重要ルール】長期MAが明確に下降中はステージ1と判定しない
        ma_30w_trend = self._check_ma_trend(self.df[self.ma_30w], self.slope_lookback + 10)
        ma_40w_trend = self._check_ma_trend(self.df[self.ma_40w], self.slope_lookback + 20)
        if ma_30w_trend == 'declining' or ma_40w_trend == 'declining':
            return False, "", {'reason': f'30w MA is {ma_30w_trend}, 40w MA is {ma_40w_trend}'}

        position = self._calculate_price_position()
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
        price_range = self._calculate_recent_price_range(self.price_range_lookback)
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

            return True, details
        
        return False, details
    
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
    
    def _detect_stage_3(self) -> Tuple[bool, Dict]:
        """
        Stage 3（天井形成期）を検出
        
        判定基準（Stage 1の逆ロジック）:
        1. 30週MA（150日MA）が平坦化（上昇後）
        2. 価格が30週MAの上下を行き来
        3. 出来高が増加（Churning）
        4. 横ばいの価格動き（上昇後）
        
        Returns:
            tuple: (Stage 3判定, 詳細情報)
        """
        # 【重要ルール】長期MAが明確に上昇中はステージ3と判定しない
        ma_30w_trend = self._check_ma_trend(self.df[self.ma_30w], self.slope_lookback + 10)
        ma_40w_trend = self._check_ma_trend(self.df[self.ma_40w], self.slope_lookback + 20) # より長期で確認
        if ma_30w_trend == 'rising' or ma_40w_trend == 'rising':
            return False, "", {'reason': f'30w MA is {ma_30w_trend}, 40w MA is {ma_40w_trend}'}

        position = self._calculate_price_position()
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
        price_range = self._calculate_recent_price_range(self.price_range_lookback)
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

            return True, details
        
        return False, details

    def _detect_early_stage3_signals(self) -> Tuple[bool, Dict]:
        """
        Stage 3への早期移行シグナルを検出（先行指標）

        先行指標:
        1. 10週MA/50日MAを大量出来高で下回る
        2. Churning: 下落日に出来高増加、上昇日に出来高減少
        3. 激しい価格変動（erratic price action）

        Returns:
            tuple: (早期シグナル検出, 詳細情報)
        """
        details = {
            'signal_detected': False,
            'signal_strength': 0,
            'reasons': []
        }

        if len(self.df) < 20:
            return False, details

        recent_data = self.df.tail(20)

        # シグナル1: 50日MA（10週MA）を大量出来高で下回る
        ma_50_breaks = 0
        high_volume_breaks = 0

        for i in range(1, len(recent_data)):
            prev_close = recent_data['Close'].iloc[i-1]
            curr_close = recent_data['Close'].iloc[i]
            ma_50 = recent_data[self.ma_10w].iloc[i]

            if 'Relative_Volume' in recent_data.columns:
                rel_vol = recent_data['Relative_Volume'].iloc[i]
            else:
                rel_vol = 1.0

            # 50日MAを下回り、かつ高出来高
            if prev_close >= ma_50 and curr_close < ma_50:
                ma_50_breaks += 1
                if rel_vol >= 1.5:
                    high_volume_breaks += 1
                    details['reasons'].append(
                        f"50日MA下抜け（高出来高{rel_vol:.1f}x）"
                    )

        if high_volume_breaks >= 1:
            details['signal_strength'] += 30
            details['ma_50_break_detected'] = True

        # シグナル2: Churning検出（下落日に出来高増加、上昇日に出来高減少）
        up_days_volume = []
        down_days_volume = []

        for i in range(1, len(recent_data)):
            prev_close = recent_data['Close'].iloc[i-1]
            curr_close = recent_data['Close'].iloc[i]
            curr_volume = recent_data['Volume'].iloc[i]

            if curr_close > prev_close:
                up_days_volume.append(curr_volume)
            elif curr_close < prev_close:
                down_days_volume.append(curr_volume)

        if len(up_days_volume) > 0 and len(down_days_volume) > 0:
            avg_up_volume = np.mean(up_days_volume)
            avg_down_volume = np.mean(down_days_volume)

            # Churning: 下落日の出来高 > 上昇日の出来高 × 1.3
            if avg_down_volume > avg_up_volume * 1.3:
                details['signal_strength'] += 30
                details['churning_detected'] = True
                details['reasons'].append(
                    f"Churning検出（下落日出来高 > 上昇日出来高 × 1.3）"
                )
                details['volume_ratio'] = avg_down_volume / avg_up_volume

        # シグナル3: 激しい価格変動（erratic price action）
        price_range = recent_data['High'] - recent_data['Low']
        avg_price = recent_data['Close'].mean()

        if avg_price > 0:
            volatility = (price_range / avg_price * 100).mean()

            # 通常の2倍以上の変動
            if volatility > 4.0:  # 4%以上の平均変動幅
                details['signal_strength'] += 20
                details['high_volatility_detected'] = True
                details['reasons'].append(
                    f"高ボラティリティ（平均{volatility:.1f}%）"
                )
                details['volatility'] = volatility

        # シグナル4: 価格が30週MAより上だが勢いを失っている
        current_price = self.latest['Close']
        ma_30w = self.latest[self.ma_30w]

        if current_price > ma_30w:
            # 最近の価格モメンタムを確認
            momentum = self._calculate_recent_momentum(10)
            if -0.05 < momentum < 0.05:  # ±5%以内の横ばい
                details['signal_strength'] += 20
                details['sideways_action'] = True
                details['reasons'].append(
                    f"30週MA上だが横ばい（モメンタム{momentum*100:.1f}%）"
                )

        # 総合判定
        if details['signal_strength'] >= 50:
            details['signal_detected'] = True
            details['confidence'] = 'high' if details['signal_strength'] >= 70 else 'medium'
            return True, details

        return False, details


    def _detect_early_stage1_signals(self) -> Tuple[bool, Dict]:
        """
        Stage 1への早期移行シグナルを検出（先行指標）

        先行指標:
        1. Selling Climax（出来高急増を伴う急落後の反発）
        2. Change of Character（下降パターンからの明確な変化）
        3. 10週MAが30週MAを上抜ける（ゴールデンクロス）

        Returns:
            tuple: (早期シグナル検出, 詳細情報)
        """
        details = {
            'signal_detected': False,
            'signal_strength': 0,
            'reasons': []
        }

        if len(self.df) < 30:
            return False, details

        recent_data = self.df.tail(30)

        # シグナル1: Selling Climax検出
        selling_climax_detected = False

        for i in range(len(recent_data) - 10, len(recent_data) - 1):
            if i < 1:
                continue

            curr_bar = recent_data.iloc[i]
            prev_bar = recent_data.iloc[i-1]
            next_bar = recent_data.iloc[i+1]

            # Selling Climax条件:
            # 1. 大幅な下落（前日比-3%以上）
            # 2. 出来高が平均の2倍以上
            # 3. 翌日反発

            price_drop_pct = ((curr_bar['Close'] - prev_bar['Close']) / prev_bar['Close'] * 100) if prev_bar['Close'] > 0 else 0

            if 'Relative_Volume' in recent_data.columns:
                rel_vol = curr_bar['Relative_Volume']
            else:
                rel_vol = 1.0

            next_day_bounce = next_bar['Close'] > curr_bar['Close']

            if price_drop_pct <= -3.0 and rel_vol >= 2.0 and next_day_bounce:
                selling_climax_detected = True
                details['signal_strength'] += 40
                details['selling_climax_date'] = curr_bar.name.strftime('%Y-%m-%d')
                details['reasons'].append(
                    f"Selling Climax検出（{price_drop_pct:.1f}%下落、出来高{rel_vol:.1f}x）"
                )
                break

        details['selling_climax_detected'] = selling_climax_detected

        # シグナル2: Change of Character（下降パターンからの変化）
        # 最近10日間の安値が、その前の10日間の安値より高い
        last_10_days = recent_data.tail(10)
        prior_10_days = recent_data.iloc[-20:-10]

        if len(last_10_days) >= 10 and len(prior_10_days) >= 10:
            recent_low = last_10_days['Low'].min()
            prior_low = prior_10_days['Low'].min()

            if recent_low > prior_low * 1.05:  # 5%以上高い安値
                details['signal_strength'] += 30
                details['higher_lows'] = True
                details['reasons'].append(
                    f"より高い安値形成（Change of Character）"
                )

        # シグナル3: ゴールデンクロス（10週MAが30週MAを上抜ける）
        ma_10w_series = self.df[self.ma_10w].tail(5)
        ma_30w_series = self.df[self.ma_30w].tail(5)

        for i in range(1, len(ma_10w_series)):
            prev_10w = ma_10w_series.iloc[i-1]
            curr_10w = ma_10w_series.iloc[i]
            prev_30w = ma_30w_series.iloc[i-1]
            curr_30w = ma_30w_series.iloc[i]

            # ゴールデンクロス
            if prev_10w <= prev_30w and curr_10w > curr_30w:
                details['signal_strength'] += 30
                details['golden_cross'] = True
                details['reasons'].append(
                    f"ゴールデンクロス発生（10週MA > 30週MA）"
                )
                break

        # シグナル4: 価格が30週MAを上回る（40週MAはまだ下降中でも可）
        current_price = self.latest['Close']
        ma_30w = self.latest[self.ma_30w]
        ma_40w = self.latest[self.ma_40w]

        if current_price > ma_30w:
            details['signal_strength'] += 20
            details['above_30w_ma'] = True
            details['reasons'].append(
                f"価格が30週MAを上回る"
            )

            # 40週MAが下降中でもOK（早期検出）
            ma_40w_trend = self._check_ma_trend(self.df[self.ma_40w], self.slope_lookback)
            if ma_40w_trend == 'declining':
                details['early_stage1_signal'] = True
                details['reasons'].append(
                    f"40週MA下降中だが30週MA上抜け（Stage 1初期）"
                )

        # 総合判定
        if details['signal_strength'] >= 50:
            details['signal_detected'] = True
            details['confidence'] = 'high' if details['signal_strength'] >= 70 else 'medium'
            return True, details

        return False, details
    
    def determine_stage(self, rs_rating: Optional[float] = None, previous_stage: Optional[int] = None) -> int:
        """
        現在のステージを判定（早期検出強化版）

        判定優先順位:
        1. 早期シグナルチェック（Stage 3, Stage 1）
        2. Stage 2（明確な上昇トレンド）
        3. Stage 4（明確な下降トレンド）
        4. Stage 1とStage 3（横ばい）を文脈で判定
        
        Args:
            rs_rating (Optional[float]): RS Ratingスコア
            previous_stage (Optional[int]): 直前のステージ番号

        Returns:
            int: ステージ番号
        """
        # **【追加】早期シグナルチェック**

        # Stage 2 → Stage 3への早期移行シグナル
        if previous_stage == 2:
            early_stage3, stage3_details = self._detect_early_stage3_signals()
            if early_stage3 and stage3_details.get('confidence') == 'high':
                # 高信頼度の早期Stage 3シグナル
                return 3

        # Stage 4 → Stage 1への早期移行シグナル
        if previous_stage == 4:
            early_stage1, stage1_details = self._detect_early_stage1_signals()
            if early_stage1 and stage1_details.get('confidence') == 'high':
                # 高信頼度の早期Stage 1シグナル
                return 1

        # 既存のロジック（Stage 2の判定）
        is_stage2, stage2_details = self._detect_stage_2(rs_rating=rs_rating)
        
        # Stage 4の判定
        is_stage4, stage4_details = self._detect_stage_4()
        
        # Stage 1とStage 3の判定
        is_stage1, stage1_details = self._detect_stage_1()
        is_stage3, stage3_details = self._detect_stage_3()

        # デフォルトのステージを定義
        final_stage = 1

        if is_stage2:
            final_stage = 2
        elif is_stage4:
            final_stage = 4
        elif is_stage1 and is_stage3:
            # 過去のトレンドを確認
            historical_trend = self._check_historical_trend(100)
            if historical_trend == 'prior_uptrend':
                final_stage = 3
            else:
                final_stage = 1
        elif is_stage1:
            final_stage = 1
        elif is_stage3:
            final_stage = 3
        else:
            # フォールバック判定
            final_stage = self._fallback_stage_detection()

        # ステージ移行ルールを適用
        if previous_stage == 2 and final_stage == 1:
            # **【修正】早期Stage 3シグナルがない場合のみStage 2を維持**
            early_stage3, _ = self._detect_early_stage3_signals()
            if not early_stage3:
                return 2
            else:
                # 早期シグナルがある場合はStage 3へ移行を許可
                return 3

        if previous_stage == 3 and final_stage == 1:
            return 3

        return final_stage
    
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

    def _fallback_stage_detection(self) -> int:
        """
        フォールバック判定（MAの配置のみで判断）

        Returns:
            int: ステージ番号
        """
        position = self._calculate_price_position()
        
        current_price = position['price']
        ma_30w = self.latest[self.ma_30w]
        ma_40w = self.latest[self.ma_40w]
        
        if current_price > ma_30w > ma_40w:
            return 2  # 上昇トレンドの可能性
        elif current_price < ma_30w < ma_40w:
            return 4  # 下降トレンドの可能性
        else:
            # 横ばいだが、文脈が不明
            # 安全側に倒してStage 1と判定
            return 1
    
    def check_minervini_template(self) -> Dict:
        """
        Mark Minervini Trend Template（8基準）をチェック
        
        ※ Stage 2とStage 1B（準備段階）で使用可能
        
        Returns:
            dict: 各基準の結果とスコア
        """
        checks = {}
        
        # 現在のステージを確認
        current_stage = self.determine_stage()
        
        # Minerviniテンプレートは日足でのみ適用
        if self.interval != '1d':
            return {'applicable': False, 'reason': 'Minerviniテンプレートは日足データでのみ適用されます。'}

        # Stage 2またはStage 1で適用(簡易化)
        if current_stage not in [1, 2]:
            return {
                'applicable': False,
                'reason': f'Minerviniテンプレートは Stage 2 または Stage 1 で適用（現在: Stage {current_stage}）',
                'criteria_met': 0,
                'total_criteria': 8,
                'score': 0
            }
        
        # 現在の価格とMA
        current_price = self.latest['Close']
        sma_50 = self.latest.get('SMA_50')
        sma_150 = self.latest.get('SMA_150')
        sma_200 = self.latest.get('SMA_200')

        if any(v is None for v in [sma_50, sma_150, sma_200]):
             return {'applicable': False, 'reason': '必要なSMAデータが不足しています。'}

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
            'stage': current_stage
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
        
        # --- 日足分析 ---
        stock_df_daily, benchmark_df_daily = fetch_stock_data(ticker, interval='1d')
        if stock_df_daily is not None:
            indicators_df_daily = calculate_all_basic_indicators(stock_df_daily, '1d').dropna()
            if len(indicators_df_daily) >= 200:
                detector_daily = StageDetector(indicators_df_daily, benchmark_df_daily, '1d')
                stage_daily = detector_daily.determine_stage()
                print(f"日足ステージ: Stage {stage_daily}")

                template_result = detector_daily.check_minervini_template()
                if template_result['applicable']:
                    print(f"Minerviniテンプレート: {template_result['score']:.1f}/100")
                else:
                    print(f"Minerviniテンプレート: {template_result['reason']}")

        # --- 週足分析 ---
        stock_df_weekly, benchmark_df_weekly = fetch_stock_data(ticker, interval='1wk')
        if stock_df_weekly is not None:
            indicators_df_weekly = calculate_all_basic_indicators(stock_df_weekly, '1wk').dropna()
            if len(indicators_df_weekly) >= 40:
                detector_weekly = StageDetector(indicators_df_weekly, benchmark_df_weekly, '1wk')
                stage_weekly = detector_weekly.determine_stage()
                print(f"週足ステージ: Stage {stage_weekly}")