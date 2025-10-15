"""
Stan Weinstein Stage Analysis Detector
Stan Weinsteinの4ステージ理論に基づくステージ判定
日足データを使用し、出来高は参考情報として扱う
"""
import pandas as pd
import numpy as np
from typing import Tuple


class StageDetector:
    """
    Stan Weinstein Stage Analysis システム
    
    4つのステージ:
    - Stage 1: Basing Area (ベース形成期) - 横ばい、需給均衡
    - Stage 2: Advancing Phase (上昇期) - 上昇トレンド
    - Stage 3: Top Area (天井形成期) - 横ばい、分配期
    - Stage 4: Declining Phase (下降期) - 下降トレンド
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
        
        # 主要な移動平均線
        # 150日MA ≈ 30週MA、200日MA ≈ 40週MA
        self.ma_30w = 'SMA_150'  # 30週 ≈ 150日
        self.ma_40w = 'SMA_200'  # 40週 ≈ 200日
        self.ma_10w = 'SMA_50'   # 10週 ≈ 50日
        
    def _check_ma_flatness(self, ma_series: pd.Series, lookback: int = 20, 
                          threshold: float = 0.015) -> bool:
        """
        移動平均線が横ばい（フラット）かチェック
        
        Args:
            ma_series: 移動平均線のシリーズ
            lookback: 確認期間
            threshold: フラット判定の閾値
            
        Returns:
            bool: フラットの場合True
        """
        if len(ma_series) < lookback:
            return False
        
        recent = ma_series.tail(lookback)
        
        # 線形回帰の傾き
        x = np.arange(len(recent))
        y = recent.values
        
        if np.linalg.norm(y) > 0:
            y_norm = y / np.linalg.norm(y)
        else:
            return True
        
        slope = np.polyfit(x, y_norm, 1)[0]
        
        return abs(slope) < threshold
    
    def _check_ma_rising(self, ma_series: pd.Series, lookback: int = 20,
                        min_slope: float = 0.005) -> bool:
        """
        移動平均線が上昇トレンドかチェック
        
        Args:
            ma_series: 移動平均線のシリーズ
            lookback: 確認期間
            min_slope: 上昇判定の最小傾き
            
        Returns:
            bool: 上昇トレンドの場合True
        """
        if len(ma_series) < lookback:
            return False
        
        recent = ma_series.tail(lookback)
        
        x = np.arange(len(recent))
        y = recent.values
        
        if np.linalg.norm(y) > 0:
            y_norm = y / np.linalg.norm(y)
        else:
            return False
        
        slope = np.polyfit(x, y_norm, 1)[0]
        
        return slope > min_slope
    
    def _check_ma_declining(self, ma_series: pd.Series, lookback: int = 20,
                           max_slope: float = -0.005) -> bool:
        """
        移動平均線が下降トレンドかチェック
        
        Args:
            ma_series: 移動平均線のシリーズ
            lookback: 確認期間
            max_slope: 下降判定の最大傾き
            
        Returns:
            bool: 下降トレンドの場合True
        """
        if len(ma_series) < lookback:
            return False
        
        recent = ma_series.tail(lookback)
        
        x = np.arange(len(recent))
        y = recent.values
        
        if np.linalg.norm(y) > 0:
            y_norm = y / np.linalg.norm(y)
        else:
            return False
        
        slope = np.polyfit(x, y_norm, 1)[0]
        
        return slope < max_slope
    
    def _detect_stage_1(self) -> Tuple[bool, str]:
        """
        Stage 1（ベース形成期）を検出
        
        条件:
        - 30週MA（150日MA）が横ばい
        - 価格が30週MAの周辺で変動
        - Stage 4の下降トレンドの後
        
        Returns:
            tuple: (Stage 1判定, サブステージ)
        """
        current_price = self.latest['Close']
        ma_30w = self.latest[self.ma_30w]
        ma_40w = self.latest[self.ma_40w]
        
        # 30週MAが横ばい
        ma_30w_flat = self._check_ma_flatness(self.df[self.ma_30w], 20, 0.015)
        
        # 価格が30週MAの周辺（±15%）
        if ma_30w > 0:
            price_near_ma = 0.85 < current_price / ma_30w < 1.15
        else:
            price_near_ma = False
        
        # Stage 1の基本条件
        if ma_30w_flat and price_near_ma:
            # サブステージの判定
            substage = self._determine_stage1_substage()
            return True, substage
        
        return False, ""
    
    def _determine_stage1_substage(self) -> str:
        """
        Stage 1のサブステージを判定
        
        Returns:
            str: サブステージ (1A, 1, 1B)
        """
        current_price = self.latest['Close']
        ma_30w = self.latest[self.ma_30w]
        ma_40w = self.latest[self.ma_40w]
        
        # 価格と40週MAの関係
        price_above_40w = current_price > ma_40w if ma_40w > 0 else False
        
        # 40週MAの傾き
        ma_40w_rising = self._check_ma_rising(self.df[self.ma_40w], 30, 0.003)
        
        # 最近の価格動向
        recent_prices = self.df['Close'].tail(20)
        price_trend_up = recent_prices.iloc[-1] > recent_prices.iloc[0]
        
        # サブステージ判定
        if price_above_40w and ma_40w_rising:
            return "1B"  # ブレイクアウト準備中
        elif price_above_40w or price_trend_up:
            return "1"   # ベース形成中
        else:
            return "1A"  # ベースの初期
    
    def _detect_stage_2(self) -> Tuple[bool, str]:
        """
        Stage 2（上昇期）を検出
        
        条件:
        - 価格が30週MAと40週MAの上
        - 30週MAが上昇トレンド
        - 価格が高値更新と高安値を形成（低安値がない）
        - ブレイクアウト時に大きな出来高（参考）
        
        Returns:
            tuple: (Stage 2判定, サブステージ)
        """
        current_price = self.latest['Close']
        ma_10w = self.latest[self.ma_10w]
        ma_30w = self.latest[self.ma_30w]
        ma_40w = self.latest[self.ma_40w]
        
        # 価格がMAの上
        price_above_mas = (current_price > ma_30w) and (current_price > ma_40w)
        
        # 30週MAが上昇トレンド
        ma_30w_rising = self._check_ma_rising(self.df[self.ma_30w], 20, 0.005)
        
        # 高値更新と高安値の確認（過去50日）
        higher_highs_lows = self._check_higher_highs_lows(50)
        
        # Stage 2の基本条件
        if price_above_mas and ma_30w_rising and higher_highs_lows:
            # サブステージの判定
            substage = self._determine_stage2_substage()
            return True, substage
        
        return False, ""
    
    def _check_higher_highs_lows(self, lookback: int = 50) -> bool:
        """
        価格が高値更新と高安値を形成しているかチェック
        
        Args:
            lookback: 確認期間
            
        Returns:
            bool: 高値更新と高安値が確認される場合True
        """
        recent = self.df.tail(lookback)
        
        if len(recent) < 20:
            return False
        
        # 最近の高値が過去の高値より高い
        mid_point = len(recent) // 2
        recent_high = recent['High'].iloc[mid_point:].max()
        past_high = recent['High'].iloc[:mid_point].max()
        
        higher_high = recent_high > past_high
        
        # 最近の安値が過去の安値より高い（または同程度）
        recent_low = recent['Low'].iloc[mid_point:].min()
        past_low = recent['Low'].iloc[:mid_point].min()
        
        higher_low = recent_low >= past_low * 0.95  # 5%の許容
        
        return higher_high and higher_low
    
    def _determine_stage2_substage(self) -> str:
        """
        Stage 2のサブステージを判定
        
        Returns:
            str: サブステージ (2A, 2, 2B)
        """
        current_price = self.latest['Close']
        ma_10w = self.latest[self.ma_10w]
        
        # 52週高値との距離
        high_52w = self.latest['High_52W']
        if high_52w > 0:
            dist_from_high = (high_52w - current_price) / high_52w
        else:
            dist_from_high = 0
        
        # Stage 2初期からの経過期間を推定
        # 10週MAが30週MAを上抜けた時点からの日数
        stage2_duration = self._estimate_stage2_duration()
        
        # サブステージ判定
        if stage2_duration < 60 and current_price > ma_10w:
            return "2A"  # Stage 2初期
        elif dist_from_high < 0.20 and stage2_duration > 120:
            return "2B"  # Stage 2後期
        else:
            return "2"   # Stage 2中期
    
    def _estimate_stage2_duration(self) -> int:
        """
        Stage 2開始からの経過日数を推定
        
        Returns:
            int: 推定日数
        """
        ma_10w_series = self.df[self.ma_10w]
        ma_30w_series = self.df[self.ma_30w]
        
        # 10週MAが30週MAを上抜けた最新の時点を探す
        for i in range(len(self.df) - 1, max(0, len(self.df) - 250), -1):
            if (ma_10w_series.iloc[i] > ma_30w_series.iloc[i] and
                ma_10w_series.iloc[i-1] <= ma_30w_series.iloc[i-1]):
                return len(self.df) - i
        
        return 200  # デフォルト値
    
    def _detect_stage_3(self) -> Tuple[bool, str]:
        """
        Stage 3（天井形成期）を検出
        
        条件:
        - 価格が横ばい
        - 30週MAと40週MAが平坦化
        - 価格が10週MA（50日MA）を大きな出来高で下回る（警告）
        
        Returns:
            tuple: (Stage 3判定, サブステージ)
        """
        current_price = self.latest['Close']
        ma_10w = self.latest[self.ma_10w]
        ma_30w = self.latest[self.ma_30w]
        ma_40w = self.latest[self.ma_40w]
        
        # 30週MAと40週MAが平坦化
        ma_30w_flat = self._check_ma_flatness(self.df[self.ma_30w], 20, 0.015)
        ma_40w_flat = self._check_ma_flatness(self.df[self.ma_40w], 30, 0.010)
        
        # 価格が30週MAより上または周辺
        if ma_30w > 0:
            price_near_or_above = current_price > ma_30w * 0.95
        else:
            price_near_or_above = False
        
        # Stage 3の基本条件
        if (ma_30w_flat or ma_40w_flat) and price_near_or_above:
            # 価格が横ばいかチェック
            price_range = self._calculate_recent_price_range(50)
            
            if price_range < 0.20:  # 20%以内の変動
                substage = self._determine_stage3_substage()
                return True, substage
        
        return False, ""
    
    def _calculate_recent_price_range(self, lookback: int = 50) -> float:
        """
        最近の価格変動幅を計算
        
        Args:
            lookback: 確認期間
            
        Returns:
            float: 価格変動幅（高値-安値）/ 平均価格
        """
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
    
    def _determine_stage3_substage(self) -> str:
        """
        Stage 3のサブステージを判定
        
        Returns:
            str: サブステージ (3A, 3, 3B)
        """
        current_price = self.latest['Close']
        ma_10w = self.latest[self.ma_10w]
        
        # 10週MAとの関係
        below_10w_ma = current_price < ma_10w
        
        # 最近の価格の勢い
        recent_momentum = self._calculate_recent_momentum(20)
        
        # サブステージ判定
        if below_10w_ma:
            return "3B"  # 天井形成の最終段階
        elif recent_momentum < 0:
            return "3"   # 天井形成中
        else:
            return "3A"  # 天井形成の初期
    
    def _calculate_recent_momentum(self, lookback: int = 20) -> float:
        """
        最近の価格モメンタムを計算
        
        Args:
            lookback: 確認期間
            
        Returns:
            float: モメンタム（正=上昇、負=下降）
        """
        recent = self.df['Close'].tail(lookback)
        
        if len(recent) < 2:
            return 0
        
        return (recent.iloc[-1] - recent.iloc[0]) / recent.iloc[0]
    
    def _detect_stage_4(self) -> Tuple[bool, str]:
        """
        Stage 4（下降期）を検出
        
        条件:
        - 価格がStage 3のサポートを下抜け
        - 価格が30週MAと40週MAの下
        - 低い高値と低い安値を形成
        - 30週MAが下降トレンド
        
        Returns:
            tuple: (Stage 4判定, サブステージ)
        """
        current_price = self.latest['Close']
        ma_30w = self.latest[self.ma_30w]
        ma_40w = self.latest[self.ma_40w]
        
        # 価格がMAの下
        price_below_mas = (current_price < ma_30w) and (current_price < ma_40w)
        
        # 30週MAが下降トレンド
        ma_30w_declining = self._check_ma_declining(self.df[self.ma_30w], 20, -0.005)
        
        # 低い高値と低い安値の確認
        lower_highs_lows = self._check_lower_highs_lows(50)
        
        # Stage 4の基本条件
        if price_below_mas and (ma_30w_declining or lower_highs_lows):
            substage = self._determine_stage4_substage()
            return True, substage
        
        return False, ""
    
    def _check_lower_highs_lows(self, lookback: int = 50) -> bool:
        """
        価格が低い高値と低い安値を形成しているかチェック
        
        Args:
            lookback: 確認期間
            
        Returns:
            bool: 低い高値と低い安値が確認される場合True
        """
        recent = self.df.tail(lookback)
        
        if len(recent) < 20:
            return False
        
        mid_point = len(recent) // 2
        
        # 最近の高値が過去の高値より低い
        recent_high = recent['High'].iloc[mid_point:].max()
        past_high = recent['High'].iloc[:mid_point].max()
        
        lower_high = recent_high < past_high
        
        # 最近の安値が過去の安値より低い
        recent_low = recent['Low'].iloc[mid_point:].min()
        past_low = recent['Low'].iloc[:mid_point].min()
        
        lower_low = recent_low < past_low
        
        return lower_high and lower_low
    
    def _determine_stage4_substage(self) -> str:
        """
        Stage 4のサブステージを判定
        
        Returns:
            str: サブステージ (4A, 4, 4B, 4B-)
        """
        current_price = self.latest['Close']
        
        # 52週安値との距離
        low_52w = self.latest['Low_52W']
        if low_52w > 0:
            dist_from_low = (current_price - low_52w) / low_52w
        else:
            dist_from_low = 1
        
        # 下降の勢い
        recent_momentum = self._calculate_recent_momentum(20)
        
        # サブステージ判定
        if dist_from_low < 0.05 and recent_momentum > -0.02:
            return "4B-"  # Stage 4の最終段階、底打ち近い
        elif dist_from_low < 0.15:
            return "4B"   # Stage 4後期
        elif recent_momentum < -0.10:
            return "4A"   # Stage 4初期、急落中
        else:
            return "4"    # Stage 4中期
    
    def determine_stage(self) -> Tuple[int, str]:
        """
        現在のステージを判定（Stan Weinstein理論）
        
        Returns:
            tuple: (ステージ番号, サブステージ)
        """
        # Stage 2の判定（最も重要）
        is_stage2, stage2_substage = self._detect_stage_2()
        if is_stage2:
            return 2, stage2_substage
        
        # Stage 4の判定
        is_stage4, stage4_substage = self._detect_stage_4()
        if is_stage4:
            return 4, stage4_substage
        
        # Stage 1の判定
        is_stage1, stage1_substage = self._detect_stage_1()
        if is_stage1:
            return 1, stage1_substage
        
        # Stage 3の判定
        is_stage3, stage3_substage = self._detect_stage_3()
        if is_stage3:
            return 3, stage3_substage
        
        # どれにも当てはまらない場合、MAの配置で判断
        return self._fallback_stage_detection()
    
    def _fallback_stage_detection(self) -> Tuple[int, str]:
        """
        フォールバック判定（MAの配置のみで判断）
        
        Returns:
            tuple: (ステージ番号, サブステージ)
        """
        current_price = self.latest['Close']
        ma_30w = self.latest[self.ma_30w]
        ma_40w = self.latest[self.ma_40w]
        
        # 価格とMAの関係で大まかに判定
        if current_price > ma_30w > ma_40w:
            return 2, "2"  # 上昇トレンドの可能性
        elif current_price < ma_30w < ma_40w:
            return 4, "4"  # 下降トレンドの可能性
        else:
            return 1, "1"  # ベース形成の可能性


if __name__ == '__main__':
    # テスト用
    from data_fetcher import fetch_stock_data
    from indicators import calculate_all_basic_indicators
    
    print("Stan Weinstein Stage Analysis のテストを開始...")
    
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
                
                # 詳細情報
                current_price = detector.latest['Close']
                ma_50 = detector.latest['SMA_50']
                ma_150 = detector.latest['SMA_150']
                ma_200 = detector.latest['SMA_200']
                
                print(f"\n価格情報:")
                print(f"  現在価格: ${current_price:.2f}")
                print(f"  50日MA: ${ma_50:.2f} ({'+' if current_price > ma_50 else '-'})")
                print(f"  150日MA: ${ma_150:.2f} ({'+' if current_price > ma_150 else '-'})")
                print(f"  200日MA: ${ma_200:.2f} ({'+' if current_price > ma_200 else '-'})")
                
                print(f"\nMA配列:")
                print(f"  50日MA > 150日MA: {ma_50 > ma_150}")
                print(f"  150日MA > 200日MA: {ma_150 > ma_200}")