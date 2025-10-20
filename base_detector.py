"""
ベースパターン分析モジュール（完全版 - StageDetector連携 + 30%ルール必須条件）
William O'NeilとMark Minerviniのベース理論を完全実装

【責任分担】
✓ ステージ判定 → stage_detector.py（StageDetector）が担当
✓ ベースカウンティング → base_detector.py（このファイル）が担当

【重要な実装】
1. **30%ルール（必須条件）**：ベース形成前に30%以上の価格上昇が必須
   - 30%未満の上昇後の統合期間はベースとして認識しない
2. 20%ルールによるベースカウントリセット
3. ベースパターンの検出と品質評価
4. ブレイクアウト検出と検証
5. ベースカウンティング（早期/後期の判定）
6. StageDetectorとの完全連携

【30%ルール - William O'Neil - 必須条件】
- ベース形成前に、価格が前回の中間安値から最低30%上昇している必要がある
- 30%未満の上昇後に形成された統合期間は「ベース」ではない
- フラットベース、カップウィズハンドル、その他のベースパターンすべてに適用
- これは品質の問題ではなく、ベース定義の必須条件

【20%ルール - MarketSmith Indiaより】
- ピボットポイントから現在のベースの左側高値まで20%以上上昇
  → Stage数値的に増加（Stage 1 → Stage 2）
- 20%未満の上昇
  → Stageアルファベット的に増加（Stage 1a → Stage 1b）
- 日中安値が前のベースの安値を下回る
  → ベースステージとカウントは1にリセット
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional, TYPE_CHECKING

# 循環importを避けるための型チェック
if TYPE_CHECKING:
    from stage_detector import StageDetector


class BaseDetector:
    """
    ベースパターン分析システム（完全版 - 30%ルール必須条件）
    
    【このクラスの責任】
    - ベース期間の識別（横ばい統合期間 + 30%ルール必須）
    - 20%ルールによるベースカウントリセット
    - ベース品質の評価
    - ブレイクアウトの検出と検証
    - ベースカウンティング（1st, 2nd, 3rd, 4th+）
    
    【StageDetectorの責任】
    - Stage判定（Stage 1, 2, 3, 4）
    - サブステージ判定（1A, 1, 1B等）
    - Minerviniテンプレート判定
    
    【連携方法】
    detector = BaseDetector(df)
    stage_detector = StageDetector(df)
    report = detector.analyze_with_stage(stage_detector)
    """
    
    def __init__(self, df: pd.DataFrame, min_base_days: int = 35, max_base_days: int = 325):
        """
        Args:
            df: 指標計算済みのDataFrame
            min_base_days: ベースの最小期間（日数）、デフォルト35日≈7週間
            max_base_days: ベースの最大期間（日数）、デフォルト325日≈65週間
        """
        self.df = df.copy()
        self.min_base_days = min_base_days
        self.max_base_days = max_base_days
        
        # ベース検出結果を格納
        self.bases = []
        self.breakouts = []
        self.base_sequence = []  # ベースのシーケンス（カウント付き）
        
    def check_30_percent_rule(self, base_start_date: pd.Timestamp, 
                              lookback_days: int = 120) -> Dict:
        """
        【重要】30%ルールをチェック（必須条件）
        
        William O'Neilの原則：
        - ベース形成前に、価格が前回の中間安値から最低30%上昇している必要がある
        - これはカップウィズハンドル、フラットベース等すべてのベースパターンに適用
        - これは品質評価ではなく、ベース定義の必須条件
        
        Args:
            base_start_date: ベース開始日
            lookback_days: ベース開始前に遡って確認する期間（デフォルト120日≈6ヶ月）
            
        Returns:
            dict: 30%ルールの判定結果
        """
        try:
            base_start_idx = self.df.index.get_loc(base_start_date)
        except KeyError:
            return {
                'valid': False,
                'reason': 'ベース開始日が見つかりません',
                'prior_gain_pct': 0
            }
        
        # ベース開始前のデータを取得
        if base_start_idx < lookback_days:
            lookback_days = base_start_idx
        
        if lookback_days < 20:  # 最低1ヶ月のデータが必要
            return {
                'valid': False,
                'reason': 'ベース開始前のデータが不足（最低20日必要）',
                'prior_gain_pct': 0
            }
        
        # ベース開始前のデータ
        prior_data = self.df.iloc[max(0, base_start_idx - lookback_days):base_start_idx]
        
        if len(prior_data) < 20:
            return {
                'valid': False,
                'reason': 'ベース開始前のデータが不足',
                'prior_gain_pct': 0
            }
        
        # 前回の中間安値（intermediate low）を特定
        # これは、ベース開始前の期間での最安値
        intermediate_low = prior_data['Low'].min()
        intermediate_low_date = prior_data['Low'].idxmin()
        
        # ベース開始時の価格（またはベース直前の高値）
        # より正確には、ベース開始前の高値を使用
        base_left_high = prior_data['High'].max()
        base_left_high_date = prior_data['High'].idxmax()
        
        # 中間安値からベース開始前の高値までの上昇率を計算
        if intermediate_low > 0:
            prior_gain_pct = ((base_left_high - intermediate_low) / intermediate_low) * 100
        else:
            return {
                'valid': False,
                'reason': '中間安値が0以下',
                'prior_gain_pct': 0
            }
        
        # 30%ルールの判定（必須条件）
        is_valid = prior_gain_pct >= 30.0
        
        # 追加の品質チェック
        # 1. 中間安値がベース開始前の十分前に発生しているか
        low_idx = self.df.index.get_loc(intermediate_low_date)
        days_from_low_to_base = base_start_idx - low_idx
        
        # 2. 上昇トレンドの強さ（継続的な上昇か）
        rise_is_sustained = days_from_low_to_base >= 20  # 最低4週間
        
        result = {
            'valid': is_valid,
            'prior_gain_pct': prior_gain_pct,
            'intermediate_low': intermediate_low,
            'intermediate_low_date': intermediate_low_date.strftime('%Y-%m-%d'),
            'base_left_high': base_left_high,
            'base_left_high_date': base_left_high_date.strftime('%Y-%m-%d'),
            'days_from_low_to_base': days_from_low_to_base,
            'rise_is_sustained': rise_is_sustained,
        }
        
        # 判定理由の追加
        if is_valid:
            if prior_gain_pct >= 50:
                result['quality'] = 'Excellent'
                result['interpretation'] = f'強力な先行上昇（{prior_gain_pct:.1f}%）、理想的なベース前提条件'
            elif prior_gain_pct >= 40:
                result['quality'] = 'Very Good'
                result['interpretation'] = f'良好な先行上昇（{prior_gain_pct:.1f}%）、ベース形成に適している'
            else:
                result['quality'] = 'Good'
                result['interpretation'] = f'十分な先行上昇（{prior_gain_pct:.1f}%）、30%ルール合格'
        else:
            result['quality'] = 'Invalid'
            result['reason'] = f'30%ルール未達成（{prior_gain_pct:.1f}% < 30%）- これはベースではない'
            result['interpretation'] = '30%未満の上昇後の統合期間はベースとして定義されない'
        
        return result
    
    def identify_bases(self) -> List[Dict]:
        """
        価格の横ばい統合期間（ベース）を識別
        
        ベース判定基準（すべて必須）:
        1. **30%ルール: ベース形成前に30%以上の価格上昇（必須条件）**
        2. 価格が狭いレンジ内で推移（変動係数が低い）
        3. 移動平均線が平坦化
        4. 最小期間以上継続
        
        ※ 30%ルールを満たさない統合期間はベースとして認識しない
        ※ ステージ情報は含まない（純粋にベースパターンのみ）
        
        Returns:
            List[Dict]: 検出されたベースのリスト（すべて30%ルール合格済み）
        """
        if len(self.df) < self.min_base_days:
            return []
        
        self.bases = []
        rejected_consolidations = 0  # 30%ルール不合格でリジェクトされた数
        
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
                
                # 横ばい判定基準
                if cv < 0.12:  # 変動係数が12%未満
                    # 移動平均の傾きをチェック
                    if 'SMA_50' in window_data.columns:
                        ma_slope = self._calculate_slope(window_data['SMA_50'])
                        ma_flat = abs(ma_slope) < 0.025
                    else:
                        ma_flat = True
                    
                    if ma_flat:
                        # 既存のベースと重複していないかチェック
                        base_start = window_data.index[0]
                        base_end = window_data.index[-1]
                        
                        if not self._overlaps_existing_base(base_start, base_end):
                            # **30%ルールをチェック（必須条件）**
                            rule_30_result = self.check_30_percent_rule(base_start, lookback_days=120)
                            
                            # 30%ルールを満たす場合のみベースとして認定
                            if rule_30_result['valid']:
                                base_info = self._analyze_base(window_data, base_start, base_end)
                                base_info['rule_30_percent'] = rule_30_result
                                self.bases.append(base_info)
                            else:
                                # 30%ルール不合格は統計としてカウントするが、ベースには含めない
                                rejected_consolidations += 1
        
        # 開始日でソート
        self.bases.sort(key=lambda x: x['start_date'])
        
        # デバッグ情報（必要に応じて）
        if rejected_consolidations > 0:
            print(f"  注意: {rejected_consolidations}個の統合期間が30%ルール不合格によりリジェクト")
        
        return self.bases
    
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
        
        注意: このメソッドが呼ばれる時点で、30%ルールは既に合格している
        
        Returns:
            Dict: ベース情報（ステージ情報なし）
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
        
        # 出来高のDry up（右側での減少）
        if len(window_data) > 10:
            right_volume = window_data['Volume'].iloc[mid_point:].mean()
            if 'Volume_SMA_50' in window_data.columns:
                avg_volume_sma = window_data['Volume_SMA_50'].mean()
                dry_up_ratio = right_volume / avg_volume_sma if avg_volume_sma > 0 else 1
            else:
                dry_up_ratio = 1
        else:
            dry_up_ratio = 1
        
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
            'dry_up_ratio': dry_up_ratio,
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
    
    def calculate_base_quality_score(self, base: Dict) -> Dict:
        """
        ベースの品質を100点満点で評価
        
        注意: 30%ルールは必須条件なので評価項目には含まれない
        （すべてのベースは自動的に30%ルール合格済み）
        
        評価項目:
        1. 期間要件 (25点) - 7-12週が理想
        2. 深さ基準 (25点) - 15-30%が理想
        3. 出来高パターン (25点) - 右側でDry up
        4. 形状 (25点) - 右側が左側より高い
        
        Args:
            base: ベース情報
            
        Returns:
            Dict: 品質評価の詳細
        """
        score = 0
        details = {}
        
        # 30%ルール情報（参考情報として記録）
        rule_30 = base.get('rule_30_percent', {})
        details['prior_gain_pct'] = rule_30.get('prior_gain_pct', 0)
        details['prior_gain_quality'] = rule_30.get('quality', 'Unknown')
        
        # 1. 期間スコア (25点)
        duration_weeks = base['duration_weeks']
        
        if 7 <= duration_weeks <= 12:
            period_score = 25
            details['period_rating'] = 'A'
        elif (5 <= duration_weeks < 7) or (12 < duration_weeks <= 20):
            period_score = 20
            details['period_rating'] = 'B'
        elif 20 < duration_weeks <= 30:
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
        
        if 15 <= depth <= 30:
            depth_score = 25
            details['depth_rating'] = 'A'
        elif (12 <= depth < 15) or (30 < depth <= 35):
            depth_score = 20
            details['depth_rating'] = 'B'
        elif (10 <= depth < 12) or (35 < depth <= 40):
            depth_score = 15
            details['depth_rating'] = 'C'
        elif depth > 60:
            depth_score = 0
            details['depth_rating'] = 'F (Too Deep - Failure Prone)'
        else:
            depth_score = 5
            details['depth_rating'] = 'D'
        
        score += depth_score
        details['depth_score'] = depth_score
        details['depth_pct'] = depth
        
        # 3. 出来高パターンスコア (25点)
        dry_up_ratio = base.get('dry_up_ratio', 1)
        
        if dry_up_ratio < 0.30:
            volume_score = 25
            details['volume_rating'] = 'A'
        elif dry_up_ratio < 0.50:
            volume_score = 20
            details['volume_rating'] = 'B'
        elif dry_up_ratio < 0.70:
            volume_score = 15
            details['volume_rating'] = 'C'
        else:
            volume_score = 5
            details['volume_rating'] = 'D'
        
        score += volume_score
        details['volume_score'] = volume_score
        details['dry_up_ratio'] = dry_up_ratio
        
        # 4. 形状スコア (25点)
        shape_score = 0
        
        # 右側が左側より高い（健全な蓄積）
        if base['right_higher_than_left']:
            shape_score += 25
            details['right_higher_than_left'] = True
        else:
            shape_score += 10
            details['right_higher_than_left'] = False
        
        score += shape_score
        details['shape_score'] = shape_score
        details['shape_rating'] = 'A' if shape_score >= 20 else ('B' if shape_score >= 10 else 'C')
        
        details['total_score'] = score
        
        # 総合評価
        if score >= 90:
            details['overall_grade'] = 'A+ (Excellent)'
        elif score >= 80:
            details['overall_grade'] = 'A (Very Good)'
        elif score >= 70:
            details['overall_grade'] = 'B (Good)'
        elif score >= 60:
            details['overall_grade'] = 'C (Fair)'
        else:
            details['overall_grade'] = 'D (Poor)'
        
        # 先行上昇の質を追加情報として記載
        if details['prior_gain_quality'] == 'Excellent':
            details['bonus_note'] = '⭐ 先行上昇が非常に強力（50%+）'
        elif details['prior_gain_quality'] == 'Very Good':
            details['bonus_note'] = '✓ 先行上昇が良好（40%+）'
        
        return details
    
    def detect_breakouts(self, volume_multiplier: float = 1.5) -> List[Dict]:
        """
        各ベースからのブレイクアウトを検出
        
        ブレイクアウト条件:
        1. 価格がベースの高値（ピボットポイント）を上抜ける
        2. 出来高が平均の1.5倍以上（パラメータ調整可能）
        3. 終値がブレイクアウトレベルの上
        
        注意: すべてのベースは自動的に30%ルール合格済み
        
        ※ Stage情報は含まない（純粋にブレイクアウトのみ）
        
        Args:
            volume_multiplier: 出来高倍率の閾値（デフォルト1.5倍）
            
        Returns:
            List[Dict]: 検出されたブレイクアウトのリスト
        """
        if not self.bases:
            self.identify_bases()
        
        self.breakouts = []
        
        for base in self.bases:
            breakout = self._detect_breakout_from_base(base, volume_multiplier)
            if breakout:
                self.breakouts.append(breakout)
        
        return self.breakouts
    
    def _detect_breakout_from_base(self, base: Dict, volume_multiplier: float) -> Optional[Dict]:
        """
        特定のベースからのブレイクアウトを検出
        
        Args:
            base: ベース情報
            volume_multiplier: 出来高倍率
            
        Returns:
            Optional[Dict]: ブレイクアウト情報（なければNone）
        """
        # ベース終了後30日間をチェック
        try:
            base_end_idx = self.df.index.get_loc(base['end_date'])
        except KeyError:
            return None
        
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
                    'prior_gain_pct': base.get('rule_30_percent', {}).get('prior_gain_pct', 0),
                    'quality_score': self._calculate_breakout_quality(base, current_bar, base_avg_volume),
                }
                
                return breakout_info
        
        return None
    
    def _calculate_breakout_quality(self, base: Dict, breakout_bar: pd.Series, 
                                    base_avg_volume: float) -> float:
        """
        ブレイクアウトの品質スコアを計算（0-100点）
        
        評価項目:
        - ベース期間（7週間以上が理想）(30点)
        - ベースの深さ（15-35%が理想）(25点)
        - 出来高倍率（2倍以上が理想）(30点)
        - ベース右側の強さ (15点)
        
        注意: 30%ルールは必須条件なので評価項目には含まれない
        
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
    
    def apply_20_percent_rule(self) -> List[Dict]:
        """
        【20%ルール適用】MarketSmith Indiaの基準に基づく
        
        ルール:
        1. ピボットポイントから現在のベースの左側高値まで20%以上上昇
           → Stage数値的に増加（Stage 1 → Stage 2）
        2. 20%未満の上昇
           → Stageアルファベット的に増加（Stage 1a → Stage 1b）
        3. 日中安値が前のベースの安値を下回る
           → ベースステージとカウントは1にリセット
        
        注意: すべてのベースは自動的に30%ルール合格済み
        
        ※ このメソッドはベースカウンティングのみを行う
        ※ Stage判定（Stage 1, 2, 3, 4）はStageDetectorが担当
        
        Returns:
            List[Dict]: ベースシーケンス（カウント付き）
        """
        if not self.bases:
            self.identify_bases()
        
        if not self.breakouts:
            self.detect_breakouts()
        
        self.base_sequence = []
        
        if not self.bases:
            return self.base_sequence
        
        # 初期状態
        current_count = 1
        current_stage_letter = ''
        
        for i, base in enumerate(self.bases):
            base_info = {
                'base_index': i,
                'start_date': base['start_date'],
                'end_date': base['end_date'],
                'pivot_point': base['pivot_point'],
                'base_low': base['low'],
                'base_count': current_count,
                'stage_letter': current_stage_letter,
                'display_stage': f"{current_count}{current_stage_letter}",
                'reset_reason': None,
                'gain_from_prior_pivot_pct': 0,
                'prior_gain_pct': base.get('rule_30_percent', {}).get('prior_gain_pct', 0),
            }
            
            # 前のベースが存在する場合
            if i > 0:
                prior_base = self.bases[i - 1]
                prior_pivot = prior_base['pivot_point']
                current_base_left_high = base['high']  # 現在のベースの左側高値
                
                # ピボットから現在ベースの左側高値までの上昇率
                gain_pct = ((current_base_left_high - prior_pivot) / prior_pivot * 100) if prior_pivot > 0 else 0
                base_info['gain_from_prior_pivot_pct'] = gain_pct
                
                # 【リセットチェック1】前のベースの安値を下回ったか
                prior_base_low = prior_base['low']
                
                # ベース期間中の最安値
                base_data = self.df.loc[base['start_date']:base['end_date']]
                intraday_low = base_data['Low'].min()
                
                if intraday_low < prior_base_low:
                    # リセット条件満たす
                    current_count = 1
                    current_stage_letter = ''
                    base_info['reset_reason'] = f'Undercut prior base low (${prior_base_low:.2f} vs ${intraday_low:.2f})'
                    base_info['base_count'] = current_count
                    base_info['stage_letter'] = current_stage_letter
                    base_info['display_stage'] = f"{current_count}{current_stage_letter}"
                
                # 【20%ルール適用】
                elif gain_pct >= 20.0:
                    # 数値的にインクリメント
                    current_count += 1
                    current_stage_letter = ''
                    base_info['base_count'] = current_count
                    base_info['stage_letter'] = current_stage_letter
                    base_info['display_stage'] = f"{current_count}{current_stage_letter}"
                    base_info['reset_reason'] = f'20%+ gain from prior pivot ({gain_pct:.1f}%)'
                
                elif gain_pct > 0:
                    # アルファベット的にインクリメント
                    if current_stage_letter == '':
                        current_stage_letter = 'a'
                    elif current_stage_letter == 'a':
                        current_stage_letter = 'b'
                    elif current_stage_letter == 'b':
                        current_stage_letter = 'c'
                    else:
                        # c以降はそのまま
                        pass
                    
                    base_info['base_count'] = current_count
                    base_info['stage_letter'] = current_stage_letter
                    base_info['display_stage'] = f"{current_count}{current_stage_letter}"
                    base_info['reset_reason'] = f'Sub-20% gain ({gain_pct:.1f}%)'
            
            self.base_sequence.append(base_info)
        
        return self.base_sequence
    
    def check_early_vs_late_stage(self) -> Dict:
        """
        早期ステージ vs 後期ステージの判定（Minerviniの基準）
        
        Minerviniの基準:
        - 早期（1st, 2nd base）: 最良の機会
        - 中期（3rd base）: まだ許容可能
        - 後期（4th base以上）: 深い調整に陥りやすい、クライマックストップの可能性
        
        注意: すべてのベースは自動的に30%ルール合格済み
        
        ※ これはベースカウントに基づく評価
        ※ Stage判定（Stage 1, 2, 3, 4）とは別の概念
        
        Returns:
            Dict: 早期/後期の評価
        """
        if not self.base_sequence:
            self.apply_20_percent_rule()
        
        if not self.base_sequence:
            return {
                'total_bases': 0,
                'latest_base_count': 0,
                'stage_category': 'Unknown',
                'recommendation': 'ベース未検出',
                'risk_level': 'N/A',
            }
        
        latest_base = self.base_sequence[-1]
        base_count = latest_base['base_count']
        
        result = {
            'total_bases': len(self.base_sequence),
            'latest_base_count': base_count,
            'latest_display_stage': latest_base['display_stage'],
            'stage_category': '',
            'recommendation': '',
            'risk_level': '',
        }
        
        if base_count == 1:
            result['stage_category'] = 'Early Stage (1st Base)'
            result['recommendation'] = '最良の機会、積極的エントリー検討'
            result['risk_level'] = 'Low'
        elif base_count == 2:
            result['stage_category'] = 'Early Stage (2nd Base)'
            result['recommendation'] = '優れた機会、まだパブリックのレーダーに載っていない'
            result['risk_level'] = 'Low'
        elif base_count == 3:
            result['stage_category'] = 'Mid Stage (3rd Base)'
            result['recommendation'] = 'まだ許容可能だが、慎重に。一般に認知され始めている'
            result['risk_level'] = 'Medium'
        elif base_count >= 4:
            result['stage_category'] = f'Late Stage ({base_count}th Base)'
            result['recommendation'] = '深い調整に陥りやすい。新規エントリー非推奨、クライマックストップに警戒'
            result['risk_level'] = 'High'
        
        return result
    
    def check_too_fast_surge(self, breakout_info: Dict) -> Dict:
        """
        早すぎる急騰の検出
        
        3週間（15営業日）以内に20%以上の上昇は「買われすぎ」のサイン
        
        Args:
            breakout_info: ブレイクアウト情報
            
        Returns:
            Dict: 急騰判定結果
        """
        pivot_point = breakout_info['pivot_point']
        breakout_date = breakout_info['breakout_date']
        
        # ブレイクアウト後のデータ
        try:
            breakout_idx = self.df.index.get_loc(breakout_date)
        except KeyError:
            return {'too_fast': False, 'reason': 'データ不足'}
        
        # 3週間 = 15営業日をチェック
        check_days = min(15, len(self.df) - breakout_idx - 1)
        
        if check_days < 5:
            return {'too_fast': False, 'reason': 'データ不足'}
        
        data_after_breakout = self.df.iloc[breakout_idx:breakout_idx + check_days + 1]
        
        max_price = data_after_breakout['High'].max()
        max_price_date = data_after_breakout['High'].idxmax()
        days_to_max = (max_price_date - breakout_date).days
        
        gain_pct = ((max_price - pivot_point) / pivot_point) * 100
        
        result = {
            'too_fast': False,
            'gain_pct': gain_pct,
            'days_to_max': days_to_max,
            'max_price': max_price,
            'max_price_date': max_price_date,
        }
        
        # 15営業日以内に20%以上の上昇
        if days_to_max <= 15 and gain_pct >= 20.0:
            result['too_fast'] = True
            result['warning'] = f'買われすぎ！{days_to_max}日で{gain_pct:.1f}%上昇。追いかけ買い危険、調整待ち推奨'
        else:
            result['too_fast'] = False
            result['warning'] = None
        
        return result
    
    def analyze_with_stage(self, stage_detector: 'StageDetector') -> Dict:
        """
        【重要】StageDetectorと連携した包括的分析
        
        このメソッドは両者の責任を明確に分離:
        - StageDetector: Stage判定（Stage 1/2/3/4, サブステージ）
        - BaseDetector: ベースカウンティング（1st/2nd/3rd/4th base）
        
        注意: すべてのベースは自動的に30%ルール合格済み
        
        Args:
            stage_detector: StageDetectorインスタンス（必須）
            
        Returns:
            Dict: Stage情報とベースカウント情報を統合した詳細な分析結果
        """
        # ベースとブレイクアウトを検出
        if not self.bases:
            self.identify_bases()
        
        if not self.breakouts:
            self.detect_breakouts()
        
        if not self.base_sequence:
            self.apply_20_percent_rule()
        
        # StageDetectorから現在のステージ情報を取得
        current_stage, current_substage = stage_detector.determine_stage()
        
        # 早期/後期評価（ベースカウントベース）
        early_late_assessment = self.check_early_vs_late_stage()
        
        # 基本レポート
        report = {
            # StageDetectorからの情報
            'weinstein_stage': current_stage,
            'weinstein_substage': current_substage,
            
            # BaseDetectorからの情報
            'base_count_stage': early_late_assessment['latest_display_stage'] if self.base_sequence else 'N/A',
            'base_count': early_late_assessment['latest_base_count'] if self.base_sequence else 0,
            'base_count_category': early_late_assessment['stage_category'],
            'base_count_risk_level': early_late_assessment['risk_level'],
            
            # 統計情報
            'total_bases_detected': len(self.bases),
            'total_breakouts_detected': len(self.breakouts),
            
            # 重要: すべてのベースは30%ルール合格済み
            'note_30_percent_rule': 'すべての検出ベースは30%ルール合格済み（必須条件）',
        }
        
        # ベースシーケンス情報
        report['base_sequence'] = self.base_sequence
        
        # 最新ベース情報
        if self.base_sequence:
            latest_base_seq = self.base_sequence[-1]
            corresponding_base = self.bases[latest_base_seq['base_index']]
            
            # ベース品質評価
            quality = self.calculate_base_quality_score(corresponding_base)
            
            report['latest_base'] = {
                'base_count_display': latest_base_seq['display_stage'],
                'start_date': latest_base_seq['start_date'].strftime('%Y-%m-%d'),
                'end_date': latest_base_seq['end_date'].strftime('%Y-%m-%d'),
                'duration_weeks': corresponding_base['duration_weeks'],
                'depth_pct': corresponding_base['depth_pct'],
                'pivot_point': corresponding_base['pivot_point'],
                'prior_gain_pct': latest_base_seq['prior_gain_pct'],
                'quality_score': quality['total_score'],
                'quality_details': quality,
            }
        
        # 最新ブレイクアウト情報
        if self.breakouts:
            latest_breakout = self.breakouts[-1]
            
            # 早すぎる急騰チェック
            surge_check = self.check_too_fast_surge(latest_breakout)
            
            report['latest_breakout'] = {
                'breakout_date': latest_breakout['breakout_date'].strftime('%Y-%m-%d'),
                'breakout_price': latest_breakout['breakout_price'],
                'volume_ratio': latest_breakout['volume_ratio'],
                'quality_score': latest_breakout['quality_score'],
                'prior_gain_pct': latest_breakout['prior_gain_pct'],
                'too_fast_surge': surge_check,
            }
        
        # 統合解釈とアクション（Weinstein StageとBase Countの両方を考慮）
        report['integrated_analysis'] = self._integrate_stage_and_base_count(
            current_stage, 
            current_substage,
            early_late_assessment,
            report
        )
        
        return report
    
    def _integrate_stage_and_base_count(self, weinstein_stage: int, weinstein_substage: str,
                                       base_count_assessment: Dict, report: Dict) -> Dict:
        """
        Weinstein StageとBase Countを統合して最終判定
        
        注意: すべてのベースは自動的に30%ルール合格済み
        
        Args:
            weinstein_stage: Weinstein Stage (1, 2, 3, 4)
            weinstein_substage: Weinstein Substage (1A, 1, 1B, etc.)
            base_count_assessment: ベースカウント評価
            report: 完全レポート
            
        Returns:
            Dict: 統合分析結果
        """
        analysis = {
            'priority': '',
            'action': '',
            'risk_assessment': '',
            'detailed_interpretation': '',
            'key_factors': []
        }
        
        base_count = base_count_assessment.get('latest_base_count', 0)
        
        # Stage 1: ベース形成期
        if weinstein_stage == 1:
            if weinstein_substage == '1B':
                if base_count <= 2:
                    analysis['priority'] = '最優先エントリー候補'
                    analysis['action'] = f'Stage 1B（ブレイクアウト直前）+ {base_count_assessment["latest_display_stage"]} Base - 理想的なセットアップ！'
                    analysis['confidence'] = 'Very High'
                    analysis['key_factors'] = [
                        'Stage 1B - ブレイクアウト直前',
                        f'{base_count_assessment["latest_display_stage"]} Base - 早期ベース',
                        '✓ 30%ルール合格済み（すべてのベース必須条件）'
                    ]
                else:
                    analysis['priority'] = '監視継続'
                    analysis['action'] = f'Stage 1B + {base_count}th Base - やや後期だが監視価値あり'
                    analysis['confidence'] = 'Medium'
            
            elif weinstein_substage in ['1', '1A']:
                if base_count <= 2:
                    analysis['priority'] = '継続監視'
                    analysis['action'] = f'Stage {weinstein_substage} - ベース発展を待つ'
                    analysis['confidence'] = 'Medium'
                else:
                    analysis['priority'] = 'さらなる時間が必要'
                    analysis['action'] = f'Stage {weinstein_substage} - まだ早期段階'
                    analysis['confidence'] = 'Low'
        
        # Stage 2: 上昇期
        elif weinstein_stage == 2:
            if weinstein_substage == '2A':
                if base_count <= 2:
                    analysis['priority'] = '積極的エントリー'
                    analysis['action'] = f'Stage 2A（上昇初期）+ {base_count_assessment["latest_display_stage"]} Base - 優れた機会'
                    analysis['confidence'] = 'Very High'
                    analysis['key_factors'] = [
                        'Stage 2A - 上昇初期',
                        f'{base_count_assessment["latest_display_stage"]} Base - 早期ベース',
                        '✓ 30%ルール合格済み - 質の高いブレイクアウト'
                    ]
                elif base_count == 3:
                    analysis['priority'] = 'エントリー検討（慎重）'
                    analysis['action'] = f'Stage 2A + 3rd Base - まだ許容可能'
                    analysis['confidence'] = 'Medium'
                else:
                    analysis['priority'] = '新規エントリー非推奨'
                    analysis['action'] = f'Stage 2A + {base_count}th Base - 後期、警戒'
                    analysis['confidence'] = 'Low'
            
            elif weinstein_substage == '2':
                if base_count <= 2:
                    analysis['priority'] = 'エントリー検討'
                    analysis['action'] = f'Stage 2中期 + {base_count_assessment["latest_display_stage"]} Base - 良好'
                    analysis['confidence'] = 'High'
                elif base_count == 3:
                    analysis['priority'] = '慎重にエントリー検討'
                    analysis['action'] = f'Stage 2中期 + 3rd Base - 注意が必要'
                    analysis['confidence'] = 'Medium'
                else:
                    analysis['priority'] = '利確検討'
                    analysis['action'] = f'Stage 2中期 + {base_count}th Base - 後期、利確優先'
                    analysis['confidence'] = 'Low'
            
            elif weinstein_substage == '2B':
                analysis['priority'] = '利確準備'
                analysis['action'] = f'Stage 2B（上昇後期）- 天井近い可能性'
                analysis['confidence'] = 'High (Sell Signal)'
        
        # Stage 3: 天井形成期
        elif weinstein_stage == 3:
            analysis['priority'] = '撤退'
            analysis['action'] = 'Stage 3（天井形成）- 分配フェーズ、積極的利確'
            analysis['confidence'] = 'Very High (Avoid)'
            analysis['key_factors'] = [
                'Stage 3 - 分配フェーズ',
                '新規エントリー絶対回避',
                '既存ポジション速やかに撤退'
            ]
        
        # Stage 4: 下降期
        elif weinstein_stage == 4:
            analysis['priority'] = 'ロング回避'
            analysis['action'] = 'Stage 4（下降期）- ロングポジション回避'
            analysis['confidence'] = 'Very High (Avoid)'
            analysis['key_factors'] = [
                'Stage 4下降期',
                'ロングポジション完全回避',
                'Stage 1入り待ち'
            ]
        
        return analysis
    
    def generate_standalone_report(self) -> Dict:
        """
        StageDetectorなしでの基本レポート生成
        
        注意: すべてのベースは自動的に30%ルール合格済み
        
        ※ ベースカウンティングのみ実施（Stage判定なし）
        ※ 完全な分析にはanalyze_with_stage()を推奨
        
        Returns:
            Dict: 基本的な分析結果
        """
        # ベース検出
        if not self.bases:
            self.identify_bases()
        
        # ブレイクアウト検出
        if not self.breakouts:
            self.detect_breakouts()
        
        # 20%ルール適用
        if not self.base_sequence:
            self.apply_20_percent_rule()
        
        # 早期/後期判定
        stage_assessment = self.check_early_vs_late_stage()
        
        report = {
            'note': 'StageDetectorなしの基本レポート。完全な分析にはanalyze_with_stage()を使用してください。',
            'note_30_percent_rule': 'すべての検出ベースは30%ルール合格済み（必須条件）',
            'total_bases_detected': len(self.bases),
            'total_breakouts_detected': len(self.breakouts),
            'base_sequence': self.base_sequence,
            'base_count_assessment': stage_assessment,
        }
        
        # 最新ベース情報
        if self.base_sequence:
            latest_base_seq = self.base_sequence[-1]
            corresponding_base = self.bases[latest_base_seq['base_index']]
            
            quality = self.calculate_base_quality_score(corresponding_base)
            
            report['latest_base'] = {
                'base_count_display': latest_base_seq['display_stage'],
                'start_date': latest_base_seq['start_date'].strftime('%Y-%m-%d'),
                'end_date': latest_base_seq['end_date'].strftime('%Y-%m-%d'),
                'duration_weeks': corresponding_base['duration_weeks'],
                'depth_pct': corresponding_base['depth_pct'],
                'pivot_point': corresponding_base['pivot_point'],
                'prior_gain_pct': latest_base_seq['prior_gain_pct'],
                'quality_score': quality['total_score'],
            }
        
        # 最新ブレイクアウト情報
        if self.breakouts:
            latest_breakout = self.breakouts[-1]
            surge_check = self.check_too_fast_surge(latest_breakout)
            
            report['latest_breakout'] = {
                'breakout_date': latest_breakout['breakout_date'].strftime('%Y-%m-%d'),
                'breakout_price': latest_breakout['breakout_price'],
                'volume_ratio': latest_breakout['volume_ratio'],
                'quality_score': latest_breakout['quality_score'],
                'prior_gain_pct': latest_breakout['prior_gain_pct'],
                'too_fast_surge': surge_check,
            }
        
        return report


if __name__ == '__main__':
    # テスト用
    from data_fetcher import fetch_stock_data
    from indicators import calculate_all_basic_indicators
    from stage_detector import StageDetector
    
    print("ベース検出（完全版 - 30%ルール必須条件）のテストを開始...")
    print("="*80)
    print("注意: 30%ルールは必須条件なので、30%未満の統合期間はベースとして検出されません")
    print("="*80)
    
    test_tickers = ['AAPL', 'NVDA', 'TSLA']
    
    for ticker in test_tickers:
        print(f"\n{'='*80}")
        print(f"{ticker} のベース分析（30%ルール必須条件適用）:")
        print(f"{'='*80}")
        
        stock_df, _ = fetch_stock_data(ticker, period='2y')
        
        if stock_df is not None:
            indicators_df = calculate_all_basic_indicators(stock_df)
            indicators_df = indicators_df.dropna()
            
            if len(indicators_df) >= 200:
                # BaseDetectorとStageDetectorを初期化
                base_detector = BaseDetector(indicators_df, min_base_days=35)
                stage_detector = StageDetector(indicators_df)
                
                # StageDetectorと連携した包括的分析
                report = base_detector.analyze_with_stage(stage_detector)
                
                print(f"\n【Weinstein Stage Analysis】")
                print(f"  Stage: {report['weinstein_stage']} ({report['weinstein_substage']})")
                
                print(f"\n【Base Counting (O'Neil/Minervini)】")
                print(f"  Base Count: {report['base_count_stage']}")
                print(f"  Category: {report['base_count_category']}")
                print(f"  Risk Level: {report['base_count_risk_level']}")
                
                print(f"\n【30%ルール（必須条件）】")
                print(f"  ✓ すべての検出ベースは30%ルール合格済み")
                
                if 'latest_base' in report:
                    print(f"  最新ベース先行上昇: {report['latest_base']['prior_gain_pct']:.1f}%")
                
                print(f"\n【統計】")
                print(f"  検出ベース数: {report['total_bases_detected']}")
                print(f"  （30%ルール合格のみ）")
                print(f"  ブレイクアウト数: {report['total_breakouts_detected']}")
                
                # ベースシーケンス
                if report['base_sequence']:
                    print(f"\n【ベースシーケンス（20%ルール適用）】")
                    for seq in report['base_sequence'][-3:]:  # 最新3つ
                        print(f"  {seq['display_stage']} Base (先行上昇: {seq.get('prior_gain_pct', 0):.1f}%): "
                              f"{seq['start_date'].strftime('%Y-%m-%d')} - "
                              f"{seq['end_date'].strftime('%Y-%m-%d')}")
                        if seq['reset_reason']:
                            print(f"    理由: {seq['reset_reason']}")
                
                # 統合分析
                integrated = report['integrated_analysis']
                print(f"\n【統合判定】")
                print(f"  優先度: {integrated['priority']}")
                print(f"  アクション: {integrated['action']}")
                print(f"  リスク評価: {integrated['risk_assessment']}")
                if integrated.get('key_factors'):
                    print(f"  重要要素:")
                    for factor in integrated['key_factors']:
                        print(f"    • {factor}")
            else:
                print(f"  データ不足（252日以上必要）")
        else:
            print(f"  データ取得失敗")