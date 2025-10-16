"""
ベースカウンティングとブレイクアウト検出モジュール
William O'NeilとMark Minerviniの理論に基づく

主要機能:
1. ベース期間の識別（横ばい統合期間）
2. Stage 1ベース検出とサブステージ判定（1A, 1, 1B）
3. Stage 2内のベースカウンティング（StageDetectorと連携）
4. ブレイクアウトの検出と検証
5. ベース品質の評価
6. Stage情報を含む包括的分析

StageDetectorとの連携:
- analyze_with_stage(): StageDetectorインスタンスを渡して詳細分析
- detect_stage1_bases(): Stage 1のベース検出とサブステージ判定
- count_bases_in_stage2(): Stage 2内のベース数をカウント
- detect_breakouts(): ブレイクアウト時のStage情報を記録
- get_stage2_breakouts(): Stage 2内のブレイクアウトのみを抽出
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.signal import argrelextrema


class BaseDetector:
    """
    ベースカウンティングとブレイクアウト検出システム
    
    オニール/ミネルヴィニの原則:
    - ベース最小期間: 柔軟に設定可能（デフォルト20日）
    - ブレイクアウト: 高出来高（平均の1.5倍以上）必須
    - ベースカウント: Stage 2内で1-2番目が最良
    - 深さ基準: 10-40%の調整（極端でなければ許容）
    
    StageDetectorとの連携:
    StageDetectorインスタンスをメソッドに渡すことで、
    Stage情報を含む詳細な分析が可能。特に以下のメソッドで活用:
    - analyze_with_stage(stage_detector): 最も包括的な分析
    - detect_stage1_bases(stage_detector): Stage 1のベース検出
    - count_bases_in_stage2(stage_detector): 正確なStage 2ベース数
    - detect_breakouts(stage_detector=...): ブレイクアウト時のStage記録
    """
    
    def __init__(self, df: pd.DataFrame, min_base_days: int = 20, max_base_days: int = 200):
        """
        Args:
            df: 指標計算済みのDataFrame
            min_base_days: ベースの最小期間（日数）、デフォルト20日≈4週間
            max_base_days: ベースの最大期間（日数）、デフォルト200日≈40週間
        """
        self.df = df.copy()
        self.min_base_days = min_base_days
        self.max_base_days = max_base_days
        
        # ベース検出結果を格納
        self.bases = []
        self.stage1_bases = []  # Stage 1専用ベースリスト
        self.breakouts = []
        
    def identify_bases(self) -> List[Dict]:
        """
        価格の横ばい統合期間（ベース）を識別
        
        ベース判定基準:
        1. 価格が狭いレンジ内で推移（変動係数が低い）
        2. 移動平均線が平坦化
        3. 最小期間以上継続
        
        Returns:
            List[Dict]: 検出されたベースのリスト
        """
        if len(self.df) < self.min_base_days:
            return []
        
        self.bases = []
        
        # ローリングウィンドウで統合期間を探索
        window_sizes = [self.min_base_days, self.min_base_days * 2, self.min_base_days * 3]
        
        for window in window_sizes:
            if window > len(self.df):
                continue
                
            for i in range(window, len(self.df)):
                window_data = self.df.iloc[i-window:i]
                
                # 変動係数（CV）を計算
                price_std = window_data['Close'].std()
                price_mean = window_data['Close'].mean()
                cv = price_std / price_mean if price_mean > 0 else 999
                
                # ベース判定基準
                if cv < 0.10:  # 変動係数が10%未満
                    # 移動平均の傾きをチェック
                    if 'SMA_50' in window_data.columns:
                        ma_slope = self._calculate_slope(window_data['SMA_50'])
                        ma_flat = abs(ma_slope) < 0.02
                    else:
                        ma_flat = True
                    
                    if ma_flat:
                        # 既存のベースと重複していないかチェック
                        base_start = window_data.index[0]
                        base_end = window_data.index[-1]
                        
                        if not self._overlaps_existing_base(base_start, base_end):
                            base_info = self._analyze_base(window_data, base_start, base_end)
                            self.bases.append(base_info)
        
        # 開始日でソート
        self.bases.sort(key=lambda x: x['start_date'])
        
        return self.bases
    
    def detect_stage1_bases(self, stage_detector=None) -> List[Dict]:
        """
        Stage 1のベースを検出（横ばい統合期間）
        
        Stage 1判定基準:
        - 30週MA（150日MA）が平坦化
        - 価格が30週MAの周辺で変動（±15%以内）
        - Stage 4の下降トレンドからの回復
        - 最低20日以上の横ばい期間
        
        Args:
            stage_detector: StageDetectorインスタンス（オプション）
            
        Returns:
            List[Dict]: 検出されたStage 1ベースのリスト
        """
        if len(self.df) < self.min_base_days * 2:
            return []
        
        self.stage1_bases = []
        
        # 分析期間（過去1年程度）
        lookback = min(252, len(self.df))
        analysis_data = self.df.tail(lookback)
        
        # ウィンドウサイズ（最低20日から検索）
        window_sizes = [self.min_base_days, self.min_base_days * 2, self.min_base_days * 3]
        
        for window in window_sizes:
            if window > len(analysis_data):
                continue
            
            for i in range(window, len(analysis_data)):
                window_data = analysis_data.iloc[i-window:i]
                
                # Stage 1の基本条件チェック
                is_stage1_candidate = self._check_stage1_conditions(window_data)
                
                if not is_stage1_candidate:
                    continue
                
                # 既存のStage 1ベースと重複していないかチェック
                base_start = window_data.index[0]
                base_end = window_data.index[-1]
                
                if self._overlaps_existing_stage1_base(base_start, base_end):
                    continue
                
                # ベース情報を分析
                base_info = self._analyze_stage1_base(
                    window_data, base_start, base_end, stage_detector
                )
                
                if base_info:
                    self.stage1_bases.append(base_info)
        
        # 開始日でソート
        self.stage1_bases.sort(key=lambda x: x['start_date'])
        
        return self.stage1_bases
    
    def _check_stage1_conditions(self, window_data: pd.DataFrame) -> bool:
        """
        Stage 1の基本条件をチェック
        
        Args:
            window_data: 分析対象のウィンドウデータ
            
        Returns:
            bool: Stage 1の条件を満たす場合True
        """
        # 1. 30週MA（150日MA）の存在確認
        if 'SMA_150' not in window_data.columns:
            return False
        
        # 2. 変動係数（CV）の計算
        price_std = window_data['Close'].std()
        price_mean = window_data['Close'].mean()
        cv = price_std / price_mean if price_mean > 0 else 999
        
        # Stage 1は横ばいなので、変動係数が小さい
        if cv >= 0.15:  # 15%以上の変動はStage 1ではない
            return False
        
        # 3. 30週MAの平坦化チェック
        ma_150 = window_data['SMA_150']
        if len(ma_150) < 2:
            return False
        
        ma_slope = abs((ma_150.iloc[-1] - ma_150.iloc[0]) / ma_150.iloc[0])
        if ma_slope > 0.03:  # 3%以上の傾きはStage 1ではない
            return False
        
        # 4. 価格が30週MAの周辺（±15%以内）
        price_mean = window_data['Close'].mean()
        ma_mean = ma_150.mean()
        
        if ma_mean > 0:
            deviation = abs(price_mean - ma_mean) / ma_mean
            if deviation > 0.15:  # 15%以上離れていればStage 1ではない
                return False
        
        return True
    
    def _overlaps_existing_stage1_base(self, start: pd.Timestamp, end: pd.Timestamp) -> bool:
        """既存のStage 1ベースと重複しているかチェック"""
        for base in self.stage1_bases:
            if (start <= base['end_date'] and end >= base['start_date']):
                return True
        return False
    
    def _analyze_stage1_base(self, window_data: pd.DataFrame, 
                            start_date: pd.Timestamp, end_date: pd.Timestamp,
                            stage_detector=None) -> Optional[Dict]:
        """
        Stage 1ベースの詳細情報を分析
        
        Args:
            window_data: ベース期間のデータ
            start_date: ベース開始日
            end_date: ベース終了日
            stage_detector: StageDetectorインスタンス（オプション）
            
        Returns:
            Optional[Dict]: ベース情報（条件を満たさない場合はNone）
        """
        high = window_data['High'].max()
        low = window_data['Low'].min()
        depth_pct = ((high - low) / high * 100) if high > 0 else 0
        
        duration_days = len(window_data)
        duration_weeks = duration_days / 5
        
        # Stage 1ベースの深さは通常10-40%程度
        if depth_pct < 5 or depth_pct > 50:
            return None
        
        # 平均出来高
        avg_volume = window_data['Volume'].mean()
        
        # 右側が左側より高いか（健全な蓄積の兆候）
        mid_point = len(window_data) // 2
        left_avg = window_data['Close'].iloc[:mid_point].mean()
        right_avg = window_data['Close'].iloc[mid_point:].mean()
        right_higher = right_avg > left_avg
        
        # 現在価格がベース上限に近いか
        current_price = self.df['Close'].iloc[-1] if end_date == self.df.index[-1] else window_data['Close'].iloc[-1]
        distance_from_high_pct = ((high - current_price) / high * 100) if high > 0 else 0
        
        # 30週MAの傾き（最近の動向）
        ma_150 = window_data['SMA_150']
        recent_ma_slope = (ma_150.iloc[-1] - ma_150.iloc[-10]) / ma_150.iloc[-10] if len(ma_150) >= 10 else 0
        
        # サブステージの判定
        substage = self._classify_stage1_substage(
            window_data, right_higher, recent_ma_slope, distance_from_high_pct
        )
        
        base_info = {
            'start_date': start_date,
            'end_date': end_date,
            'duration_days': duration_days,
            'duration_weeks': duration_weeks,
            'high': high,
            'low': low,
            'depth_pct': depth_pct,
            'avg_volume': avg_volume,
            'right_higher_than_left': right_higher,
            'distance_from_high_pct': distance_from_high_pct,
            'pivot_point': high,  # ブレイクアウトレベル
            'substage': substage,
            'ma_150_slope': recent_ma_slope,
            'stage': 1,
        }
        
        # Stage判定の追加検証（StageDetectorがある場合）
        if stage_detector is not None:
            try:
                # ベース終了時点でのStage判定
                end_idx = self.df.index.get_loc(end_date)
                temp_df = self.df.iloc[:end_idx+1].copy()
                temp_detector = stage_detector.__class__(temp_df)
                stage, detected_substage = temp_detector.determine_stage()
                
                # Stage 1でない場合は除外
                if stage != 1:
                    return None
                
                base_info['verified_stage'] = stage
                base_info['verified_substage'] = detected_substage
            except:
                pass
        
        return base_info
    
    def _classify_stage1_substage(self, window_data: pd.DataFrame, 
                                  right_higher: bool, ma_slope: float,
                                  distance_from_high_pct: float) -> str:
        """
        Stage 1のサブステージを分類
        
        Args:
            window_data: ベース期間のデータ
            right_higher: 右側が左側より高いか
            ma_slope: 30週MAの最近の傾き
            distance_from_high_pct: 高値からの距離（%）
            
        Returns:
            str: サブステージ（1A, 1, 1B）
        """
        # 1B: ブレイクアウト準備中
        # - 右側が左側より高い
        # - MAが上向き（正の傾き）
        # - 価格がベース上限に近い（5%以内）
        if (right_higher and ma_slope > 0.01 and distance_from_high_pct < 5):
            return "1B"
        
        # 1A: ベース初期
        # - 右側が左側より低い、または
        # - MAが依然として下向き
        if (not right_higher or ma_slope < -0.01):
            return "1A"
        
        # 1: ベース形成中（中間段階）
        return "1"
    
    def calculate_stage1_base_quality(self, base: Dict) -> Dict:
        """
        Stage 1ベースの品質スコアを計算（100点満点）
        
        評価項目:
        1. 期間要件 (25点) - 7週間以上が理想
        2. 深さ基準 (25点) - 15-35%が理想
        3. 形状 (25点) - 右側が高い、MAが平坦
        4. 出来高パターン (25点) - 右側で出来高減少
        
        Args:
            base: Stage 1ベース情報
            
        Returns:
            Dict: 品質評価の詳細
        """
        score = 0
        details = {}
        
        # 1. 期間スコア (25点)
        duration_weeks = base['duration_weeks']
        if 7 <= duration_weeks <= 15:
            period_score = 25
            details['period_rating'] = 'A'
        elif 5 <= duration_weeks < 7 or 15 < duration_weeks <= 20:
            period_score = 20
            details['period_rating'] = 'B'
        elif 4 <= duration_weeks < 5 or 20 < duration_weeks <= 30:
            period_score = 15
            details['period_rating'] = 'C'
        else:
            period_score = 5
            details['period_rating'] = 'D'
        
        score += period_score
        details['period_score'] = period_score
        details['duration_weeks'] = duration_weeks
        
        # 2. 深さスコア (25点)
        depth = base['depth_pct']
        if 15 <= depth <= 35:
            depth_score = 25
            details['depth_rating'] = 'A'
        elif 10 <= depth < 15 or 35 < depth <= 40:
            depth_score = 20
            details['depth_rating'] = 'B'
        elif 5 <= depth < 10 or 40 < depth <= 45:
            depth_score = 15
            details['depth_rating'] = 'C'
        else:
            depth_score = 5
            details['depth_rating'] = 'D'
        
        score += depth_score
        details['depth_score'] = depth_score
        details['depth_pct'] = depth
        
        # 3. 形状スコア (25点)
        shape_score = 0
        
        # 右側が左側より高い
        if base['right_higher_than_left']:
            shape_score += 15
            details['right_higher'] = True
        else:
            details['right_higher'] = False
        
        # MAの傾きが適切（平坦または軽度上向き）
        ma_slope = base.get('ma_150_slope', 0)
        if -0.01 <= ma_slope <= 0.03:
            shape_score += 10
            details['ma_slope_ideal'] = True
        else:
            details['ma_slope_ideal'] = False
        
        score += shape_score
        details['shape_score'] = shape_score
        details['shape_rating'] = 'A' if shape_score >= 20 else ('B' if shape_score >= 10 else 'C')
        
        # 4. 出来高パターンスコア (25点)
        # 実装は簡易版（データがある場合のみ）
        volume_score = 15  # デフォルト中程度
        details['volume_score'] = volume_score
        details['volume_rating'] = 'B'
        
        score += volume_score
        
        # サブステージボーナス
        substage = base.get('substage', '1')
        if substage == '1B':
            bonus = 10
            details['substage_bonus'] = bonus
            score = min(100, score + bonus)  # 最大100点
        
        details['total_score'] = score
        details['substage'] = substage
        
        return details
    
    def _overlaps_existing_base(self, start: pd.Timestamp, end: pd.Timestamp) -> bool:
        """既存のベースと重複しているかチェック"""
        for base in self.bases:
            if (start <= base['end_date'] and end >= base['start_date']):
                return True
        return False
    
    def _analyze_base(self, window_data: pd.DataFrame, 
                     start_date: pd.Timestamp, end_date: pd.Timestamp) -> Dict:
        """
        ベースの詳細情報を分析
        
        Returns:
            Dict: ベース情報
        """
        high = window_data['High'].max()
        low = window_data['Low'].min()
        depth_pct = ((high - low) / high * 100) if high > 0 else 0
        
        duration_days = len(window_data)
        duration_weeks = duration_days / 5
        
        # 平均出来高
        avg_volume = window_data['Volume'].mean()
        
        # 右側が左側より高いか（健全な蓄積の兆候）
        mid_point = len(window_data) // 2
        left_avg = window_data['Close'].iloc[:mid_point].mean()
        right_avg = window_data['Close'].iloc[mid_point:].mean()
        right_higher = right_avg > left_avg
        
        # 現在価格がベース上限に近いか
        current_price = self.df['Close'].iloc[-1] if end_date == self.df.index[-1] else window_data['Close'].iloc[-1]
        distance_from_high_pct = ((high - current_price) / high * 100) if high > 0 else 0
        
        return {
            'start_date': start_date,
            'end_date': end_date,
            'duration_days': duration_days,
            'duration_weeks': duration_weeks,
            'high': high,
            'low': low,
            'depth_pct': depth_pct,
            'avg_volume': avg_volume,
            'right_higher_than_left': right_higher,
            'distance_from_high_pct': distance_from_high_pct,
            'pivot_point': high,  # ブレイクアウトレベル
        }
    
    def _calculate_slope(self, series: pd.Series) -> float:
        """時系列データの傾きを計算"""
        if len(series) < 2:
            return 0.0
        
        x = np.arange(len(series))
        y = series.values
        
        # 正規化
        if np.linalg.norm(y) > 0:
            y = y / np.linalg.norm(y)
        
        slope = np.polyfit(x, y, 1)[0]
        return slope
    
    def count_bases_in_stage2(self, stage_detector=None) -> int:
        """
        Stage 2内のベース数をカウント
        
        Args:
            stage_detector: StageDetectorインスタンス（オプション）
            
        Returns:
            int: Stage 2内で検出されたベース数
        """
        if not self.bases:
            self.identify_bases()
        
        stage2_bases = 0
        
        for base in self.bases:
            is_stage2 = False
            
            if stage_detector is not None:
                # StageDetectorを使用して正確に判定
                # ベース期間の中間点でステージをチェック
                base_mid_idx = self.df.index.get_loc(base['start_date']) + \
                              (self.df.index.get_loc(base['end_date']) - self.df.index.get_loc(base['start_date'])) // 2
                
                if base_mid_idx < len(self.df):
                    # その時点までのデータでStage判定
                    temp_df = self.df.iloc[:base_mid_idx+1].copy()
                    temp_detector = stage_detector.__class__(temp_df)
                    stage, _ = temp_detector.determine_stage()
                    is_stage2 = (stage == 2)
            else:
                # StageDetectorがない場合は簡易判定
                base_data = self.df.loc[base['start_date']:base['end_date']]
                if len(base_data) > 0 and 'SMA_50' in base_data.columns:
                    above_ma = (base_data['Close'] > base_data['SMA_50']).mean() > 0.7
                    is_stage2 = above_ma
            
            if is_stage2:
                stage2_bases += 1
        
        return stage2_bases
    
    def detect_breakouts(self, volume_multiplier: float = 1.5, stage_detector=None) -> List[Dict]:
        """
        各ベースからのブレイクアウトを検出
        
        ブレイクアウト条件:
        1. 価格がベースの高値（ピボットポイント）を上抜ける
        2. 出来高が平均の1.5倍以上（パラメータ調整可能）
        3. 終値がブレイクアウトレベルの上
        4. （オプション）Stage 2内でのブレイクアウト
        
        Args:
            volume_multiplier: 出来高倍率の閾値（デフォルト1.5倍）
            stage_detector: StageDetectorインスタンス（オプション）
            
        Returns:
            List[Dict]: 検出されたブレイクアウトのリスト
        """
        if not self.bases:
            self.identify_bases()
        
        self.breakouts = []
        
        for base in self.bases:
            breakout = self._detect_breakout_from_base(base, volume_multiplier, stage_detector)
            if breakout:
                self.breakouts.append(breakout)
        
        return self.breakouts
    
    def _detect_breakout_from_base(self, base: Dict, volume_multiplier: float, 
                                   stage_detector=None) -> Optional[Dict]:
        """
        特定のベースからのブレイクアウトを検出
        
        Args:
            base: ベース情報
            volume_multiplier: 出来高倍率
            stage_detector: StageDetectorインスタンス（オプション）
            
        Returns:
            Optional[Dict]: ブレイクアウト情報（なければNone）
        """
        # ベース終了後30日間をチェック
        base_end_idx = self.df.index.get_loc(base['end_date'])
        search_end_idx = min(base_end_idx + 30, len(self.df))
        
        pivot_point = base['pivot_point']
        base_avg_volume = base['avg_volume']
        
        for i in range(base_end_idx, search_end_idx):
            current_bar = self.df.iloc[i]
            current_date = self.df.index[i]
            
            # ブレイクアウト条件チェック
            price_breakout = current_bar['Close'] > pivot_point
            high_breakout = current_bar['High'] > pivot_point
            volume_surge = current_bar['Volume'] > base_avg_volume * volume_multiplier
            
            if price_breakout and high_breakout and volume_surge:
                # Stage判定（StageDetectorがある場合）
                stage = None
                substage = None
                if stage_detector is not None:
                    temp_df = self.df.iloc[:i+1].copy()
                    temp_detector = stage_detector.__class__(temp_df)
                    stage, substage = temp_detector.determine_stage()
                
                # ブレイクアウト確認
                breakout_info = {
                    'base_start': base['start_date'],
                    'base_end': base['end_date'],
                    'base_duration_weeks': base['duration_weeks'],
                    'breakout_date': current_date,
                    'breakout_price': current_bar['Close'],
                    'pivot_point': pivot_point,
                    'breakout_volume': current_bar['Volume'],
                    'volume_ratio': current_bar['Volume'] / base_avg_volume,
                    'base_depth_pct': base['depth_pct'],
                    'quality_score': self._calculate_breakout_quality(base, current_bar, base_avg_volume),
                    'stage': stage,
                    'substage': substage,
                    'is_stage2': stage == 2 if stage is not None else None,
                }
                
                return breakout_info
        
        return None
    
    def _calculate_breakout_quality(self, base: Dict, breakout_bar: pd.Series, 
                                    base_avg_volume: float) -> float:
        """
        ブレイクアウトの品質スコアを計算（0-100点）
        
        評価項目:
        - ベース期間（7週間以上が理想）
        - ベースの深さ（15-35%が理想）
        - 出来高倍率（2倍以上が理想）
        - ベース右側の強さ
        
        Returns:
            float: 品質スコア
        """
        score = 0
        
        # 1. 期間スコア（30点）
        weeks = base['duration_weeks']
        if weeks >= 7:
            score += 30
        elif weeks >= 5:
            score += 25
        elif weeks >= 4:
            score += 20
        else:
            score += 10
        
        # 2. 深さスコア（25点）
        depth = base['depth_pct']
        if 15 <= depth <= 35:
            score += 25
        elif 10 <= depth < 15 or 35 < depth <= 40:
            score += 20
        elif 5 <= depth < 10 or 40 < depth <= 50:
            score += 15
        else:
            score += 5
        
        # 3. 出来高スコア（30点）
        volume_ratio = breakout_bar['Volume'] / base_avg_volume
        if volume_ratio >= 2.5:
            score += 30
        elif volume_ratio >= 2.0:
            score += 25
        elif volume_ratio >= 1.5:
            score += 20
        else:
            score += 10
        
        # 4. ベース構造スコア（15点）
        if base['right_higher_than_left']:
            score += 15
        else:
            score += 5
        
        return score
    
    def find_latest_breakout(self) -> Optional[Dict]:
        """
        最新のブレイクアウトを取得
        
        Returns:
            Optional[Dict]: 最新のブレイクアウト情報
        """
        if not self.breakouts:
            self.detect_breakouts()
        
        if not self.breakouts:
            return None
        
        # ブレイクアウト日でソート
        sorted_breakouts = sorted(self.breakouts, key=lambda x: x['breakout_date'], reverse=True)
        return sorted_breakouts[0]
    
    def get_current_base_count(self, lookback_days: int = 252) -> int:
        """
        直近の期間内でのベース数を取得
        
        Args:
            lookback_days: 確認期間（デフォルト252日=約1年）
            
        Returns:
            int: ベース数
        """
        if not self.bases:
            self.identify_bases()
        
        if len(self.df) < lookback_days:
            lookback_start = self.df.index[0]
        else:
            lookback_start = self.df.index[-lookback_days]
        
        recent_bases = [b for b in self.bases if b['end_date'] >= lookback_start]
        return len(recent_bases)
    
    def is_near_breakout(self, threshold_pct: float = 5.0) -> Tuple[bool, Optional[Dict]]:
        """
        現在価格がブレイクアウトに近いかチェック
        
        Args:
            threshold_pct: ピボットポイントからの距離閾値（%）
            
        Returns:
            Tuple[bool, Optional[Dict]]: (ブレイクアウト近いか, ベース情報)
        """
        if not self.bases:
            self.identify_bases()
        
        if not self.bases:
            return False, None
        
        # 最新のベースを確認
        latest_base = self.bases[-1]
        current_price = self.df['Close'].iloc[-1]
        pivot_point = latest_base['pivot_point']
        
        distance_pct = ((pivot_point - current_price) / pivot_point * 100)
        
        if 0 <= distance_pct <= threshold_pct:
            return True, latest_base
        
        return False, None
    
    def get_stage2_breakouts(self) -> List[Dict]:
        """
        Stage 2内で発生したブレイクアウトのみを返す
        
        Returns:
            List[Dict]: Stage 2ブレイクアウトのリスト
        """
        if not self.breakouts:
            return []
        
        stage2_breakouts = [bo for bo in self.breakouts if bo.get('is_stage2') is True]
        return stage2_breakouts
    
    def analyze_with_stage(self, stage_detector) -> Dict:
        """
        StageDetectorと連携した包括的分析
        
        Args:
            stage_detector: StageDetectorインスタンス
            
        Returns:
            Dict: Stage情報を含む詳細な分析結果
        """
        # 現在のステージ
        current_stage, current_substage = stage_detector.determine_stage()
        
        # Stage別の分析
        if current_stage == 1:
            # Stage 1: ベース検出とサブステージ分析
            stage1_bases = self.detect_stage1_bases(stage_detector)
            
            # 基本レポート（Stage 2用のメソッドは使わない）
            basic_report = {
                'total_bases_detected': len(stage1_bases),
                'stage1_bases': stage1_bases,
            }
            
            # 最新のStage 1ベース
            latest_stage1_base = stage1_bases[-1] if stage1_bases else None
            
            if latest_stage1_base:
                # ベース品質評価
                quality = self.calculate_stage1_base_quality(latest_stage1_base)
                latest_stage1_base['quality_score'] = quality['total_score']
                latest_stage1_base['quality_details'] = quality
            
            # Stage 1解釈
            if latest_stage1_base:
                substage = latest_stage1_base['substage']
                
                if substage == '1B':
                    interpretation = 'Stage 1後期: ブレイクアウト準備中、高出来高での上抜けを監視'
                    action = 'ウォッチリスト追加、ブレイクアウト待ち'
                    priority = 'High'
                elif substage == '1':
                    interpretation = 'Stage 1中期: ベース形成中、蓄積フェーズ'
                    action = '監視継続、ベース発展を待つ'
                    priority = 'Medium'
                else:  # 1A
                    interpretation = 'Stage 1初期: ベース形成開始、まだ時間が必要'
                    action = '監視のみ、エントリーは時期尚早'
                    priority = 'Low'
            else:
                interpretation = 'Stage 1だがベース未検出'
                action = 'ベース形成を待つ'
                priority = 'Low'
            
            enhanced_report = {
                **basic_report,
                'current_stage': current_stage,
                'current_substage': current_substage,
                'latest_stage1_base': latest_stage1_base,
                'stage_interpretation': interpretation,
                'stage_action': action,
                'priority': priority,
            }
            
            return enhanced_report
        
        else:
            # Stage 2以降: 既存の処理
            # ベースとブレイクアウトを検出（StageDetector連携）
            if not self.bases:
                self.identify_bases()
            
            self.detect_breakouts(volume_multiplier=1.5, stage_detector=stage_detector)
            
            # Stage 2内のベース数
            stage2_base_count = self.count_bases_in_stage2(stage_detector)
            
            # Stage 2内のブレイクアウト
            stage2_breakouts = self.get_stage2_breakouts()
            
            # 基本レポート
            basic_report = self.generate_report()
            
            # Stage情報を追加
            enhanced_report = {
                **basic_report,
                'current_stage': current_stage,
                'current_substage': current_substage,
                'stage2_base_count': stage2_base_count,
                'stage2_breakout_count': len(stage2_breakouts),
                'stage2_breakouts': stage2_breakouts,
            }
            
            # Stage別の解釈を追加
            if current_stage == 2:
                if stage2_base_count <= 2:
                    enhanced_report['stage_interpretation'] = f'Stage 2: 上昇期、ベース{stage2_base_count}個目（最良の機会）'
                    enhanced_report['stage_action'] = 'エントリー検討、特に1-2番目のベース後が理想的'
                elif stage2_base_count == 3:
                    enhanced_report['stage_interpretation'] = f'Stage 2: 上昇期、ベース{stage2_base_count}個目（注意が必要）'
                    enhanced_report['stage_action'] = '慎重にエントリー検討、利確も視野に'
                else:
                    enhanced_report['stage_interpretation'] = f'Stage 2: 上昇期、ベース{stage2_base_count}個目（後期の可能性）'
                    enhanced_report['stage_action'] = '新規エントリー非推奨、既存ポジションは利確検討'
            elif current_stage == 3:
                enhanced_report['stage_interpretation'] = 'Stage 3: 天井形成期、分配フェーズ'
                enhanced_report['stage_action'] = '新規エントリー回避、既存ポジション撤退'
            elif current_stage == 4:
                enhanced_report['stage_interpretation'] = 'Stage 4: 下降期'
                enhanced_report['stage_action'] = 'ロングポジション回避、Stage 1入り待ち'
            
            return enhanced_report
    
    def generate_report(self) -> Dict:
        """
        包括的な分析レポートを生成
        
        Returns:
            Dict: 分析結果の要約
        """
        if not self.bases:
            self.identify_bases()
        
        if not self.breakouts:
            self.detect_breakouts()
        
        # 最新ベース情報
        latest_base = self.bases[-1] if self.bases else None
        
        # 最新ブレイクアウト情報
        latest_breakout = self.find_latest_breakout()
        
        # ベース数（直近1年）
        recent_base_count = self.get_current_base_count(252)
        
        # ブレイクアウト接近チェック
        near_breakout, near_base = self.is_near_breakout(5.0)
        
        report = {
            'total_bases_detected': len(self.bases),
            'total_breakouts_detected': len(self.breakouts),
            'recent_base_count_1yr': recent_base_count,
            'latest_base': latest_base,
            'latest_breakout': latest_breakout,
            'near_breakout': near_breakout,
            'near_breakout_base': near_base,
        }
        
        # 解釈とアクション
        if near_breakout and near_base:
            report['interpretation'] = 'ブレイクアウト接近中'
            report['action'] = 'ブレイクアウト監視、出来高確認必須'
        elif latest_breakout:
            days_since_breakout = (self.df.index[-1] - latest_breakout['breakout_date']).days
            if days_since_breakout <= 10:
                report['interpretation'] = '最近ブレイクアウト発生'
                report['action'] = 'エントリー検討、ただし過熱に注意'
            else:
                report['interpretation'] = 'ブレイクアウト後'
                report['action'] = 'プルバック待ちまたはトレンドフォロー'
        elif latest_base:
            report['interpretation'] = 'ベース形成中'
            report['action'] = 'ブレイクアウト待ち、監視継続'
        else:
            report['interpretation'] = 'ベース未検出'
            report['action'] = 'ベース形成待ち'
        
        return report


if __name__ == '__main__':
    # テスト用
    from data_fetcher import fetch_stock_data
    from indicators import calculate_all_basic_indicators
    from stage_detector import StageDetector
    
    print("ベース検出のテストを開始...")
    print("Stage 1ベース検出機能を含むStageDetector連携機能をテスト\n")
    
    test_tickers = ['AAPL', 'NVDA', 'TSLA']
    
    for ticker in test_tickers:
        print(f"\n{'='*60}")
        print(f"{ticker} のベース分析（StageDetector連携 + Stage 1対応）:")
        print(f"{'='*60}")
        
        stock_df, _ = fetch_stock_data(ticker, period='2y')
        
        if stock_df is not None:
            indicators_df = calculate_all_basic_indicators(stock_df)
            indicators_df = indicators_df.dropna()
            
            if len(indicators_df) >= 100:
                # BaseDetectorとStageDetectorを初期化
                base_detector = BaseDetector(indicators_df, min_base_days=20)
                stage_detector = StageDetector(indicators_df)
                
                # StageDetectorと連携した包括的分析
                report = base_detector.analyze_with_stage(stage_detector)
                
                print(f"現在のステージ: Stage {report['current_stage']} ({report['current_substage']})")
                
                # Stage 1の場合
                if report['current_stage'] == 1:
                    print(f"Stage 1ベース検出数: {report['total_bases_detected']}")
                    
                    if report.get('latest_stage1_base'):
                        base = report['latest_stage1_base']
                        print(f"\n最新Stage 1ベース:")
                        print(f"  期間: {base['start_date'].strftime('%Y-%m-%d')} - {base['end_date'].strftime('%Y-%m-%d')}")
                        print(f"  継続期間: {base['duration_weeks']:.1f}週")
                        print(f"  深さ: {base['depth_pct']:.1f}%")
                        print(f"  サブステージ: {base['substage']}")
                        print(f"  ピボットポイント: ${base['pivot_point']:.2f}")
                        
                        if 'quality_score' in base:
                            print(f"  品質スコア: {base['quality_score']:.1f}/100")
                            quality = base['quality_details']
                            print(f"    期間評価: {quality['period_rating']} ({quality['period_score']}点)")
                            print(f"    深さ評価: {quality['depth_rating']} ({quality['depth_score']}点)")
                            print(f"    形状評価: {quality['shape_rating']} ({quality['shape_score']}点)")
                    
                    print(f"\nStage解釈: {report.get('stage_interpretation', 'N/A')}")
                    print(f"推奨アクション: {report.get('stage_action', 'N/A')}")
                    print(f"優先度: {report.get('priority', 'N/A')}")
                
                # Stage 2以降の場合
                else:
                    if report['current_stage'] == 2:
                        print(f"検出されたベース数（全体）: {report['total_bases_detected']}")
                        print(f"Stage 2内のベース数: {report['stage2_base_count']}")
                        print(f"検出されたブレイクアウト数（全体）: {report['total_breakouts_detected']}")
                        print(f"Stage 2内のブレイクアウト数: {report['stage2_breakout_count']}")
                    
                    if report.get('latest_base'):
                        base = report['latest_base']
                        print(f"\n最新ベース:")
                        print(f"  期間: {base['start_date'].strftime('%Y-%m-%d')} - {base['end_date'].strftime('%Y-%m-%d')}")
                        print(f"  継続期間: {base['duration_weeks']:.1f}週")
                        print(f"  深さ: {base['depth_pct']:.1f}%")
                        print(f"  ピボットポイント: ${base['pivot_point']:.2f}")
                    
                    if report.get('latest_breakout'):
                        bo = report['latest_breakout']
                        print(f"\n最新ブレイクアウト:")
                        print(f"  日付: {bo['breakout_date'].strftime('%Y-%m-%d')}")
                        print(f"  価格: ${bo['breakout_price']:.2f}")
                        print(f"  出来高倍率: {bo['volume_ratio']:.2f}x")
                        print(f"  品質スコア: {bo['quality_score']:.1f}/100")
                        if bo['stage'] is not None:
                            print(f"  ブレイクアウト時のステージ: Stage {bo['stage']} ({bo['substage']})")
                            print(f"  Stage 2ブレイクアウト: {'✓ Yes' if bo['is_stage2'] else '✗ No'}")
                    
                    print(f"\nStage解釈: {report.get('stage_interpretation', 'N/A')}")
                    print(f"Stage別アクション: {report.get('stage_action', 'N/A')}")
                    print(f"\n基本解釈: {report['interpretation']}")
                    print(f"基本アクション: {report['action']}")