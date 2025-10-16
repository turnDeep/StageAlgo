"""
ベースパターン分析モジュール（リファクタリング版）
William O'Neilのベース理論を実装

【重要な設計変更】
1. ステージ判定はStageDetectorに完全委譲
2. ベースパターンの検出と品質評価に特化
3. StageDetectorと連携してStage情報を活用
4. ベースカウンティングとブレイクアウト検出が主機能
5. ベースタイプの分類機能は削除（シンプル化）
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional


class BaseDetector:
    """
    ベースパターン分析システム（リファクタリング版）
    
    主な機能:
    - ベース期間の識別（横ばい統合期間）
    - ベース品質の評価
    - ブレイクアウトの検出と検証
    - Stage 2内のベースカウンティング
    
    ※ ステージ判定はStageDetectorが担当
    ※ ベースタイプの分類は行わない（シンプル化）
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
        self.breakouts = []
        
    def identify_bases(self) -> List[Dict]:
        """
        価格の横ばい統合期間（ベース）を識別
        
        ベース判定基準:
        1. 価格が狭いレンジ内で推移（変動係数が低い）
        2. 移動平均線が平坦化
        3. 最小期間以上継続
        
        ※ ステージ情報は含めず、純粋にベースパターンのみ検出
        
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
        
        duration_weeks = base['duration_weeks']
        depth = base['depth_pct']
        
        # 1. 期間スコア (25点)
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
        if 15 <= depth <= 30:
            depth_score = 25
            details['depth_rating'] = 'A'
        elif (12 <= depth < 15) or (30 < depth <= 35):
            depth_score = 20
            details['depth_rating'] = 'B'
        elif (10 <= depth < 12) or (35 < depth <= 40):
            depth_score = 15
            details['depth_rating'] = 'C'
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
        
        return details
    
    def count_bases_in_stage(self, stage: int, stage_detector=None) -> int:
        """
        指定されたStage内のベース数をカウント
        
        Args:
            stage: カウント対象のステージ（1, 2, 3, 4）
            stage_detector: StageDetectorインスタンス（必須）
            
        Returns:
            int: 指定Stage内で検出されたベース数
        """
        if not self.bases:
            self.identify_bases()
        
        if stage_detector is None:
            # StageDetectorがない場合は簡易判定（非推奨）
            return len(self.bases)
        
        stage_bases = 0
        
        for base in self.bases:
            # ベース期間の中間点でステージをチェック
            try:
                base_mid_idx = self.df.index.get_loc(base['start_date']) + \
                              (self.df.index.get_loc(base['end_date']) - self.df.index.get_loc(base['start_date'])) // 2
            except KeyError:
                continue
            
            if base_mid_idx < len(self.df):
                # その時点までのデータでStage判定
                temp_df = self.df.iloc[:base_mid_idx+1].copy()
                temp_detector = stage_detector.__class__(temp_df)
                detected_stage, _ = temp_detector.determine_stage()
                
                if detected_stage == stage:
                    stage_bases += 1
        
        return stage_bases
    
    def detect_breakouts(self, volume_multiplier: float = 1.5, stage_detector=None) -> List[Dict]:
        """
        各ベースからのブレイクアウトを検出
        
        ブレイクアウト条件:
        1. 価格がベースの高値（ピボットポイント）を上抜ける
        2. 出来高が平均の1.5倍以上（パラメータ調整可能）
        3. 終値がブレイクアウトレベルの上
        4. （オプション）Stage情報を記録
        
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
    
    def analyze_with_stage(self, stage_detector) -> Dict:
        """
        StageDetectorと連携した包括的分析
        
        Args:
            stage_detector: StageDetectorインスタンス（必須）
            
        Returns:
            Dict: Stage情報を含む詳細な分析結果
        """
        # 現在のステージをStageDetectorから取得
        current_stage, current_substage = stage_detector.determine_stage()
        
        # ベースとブレイクアウトを検出（StageDetector連携）
        if not self.bases:
            self.identify_bases()
        
        self.detect_breakouts(volume_multiplier=1.5, stage_detector=stage_detector)
        
        # 基本レポート
        report = {
            'current_stage': current_stage,
            'current_substage': current_substage,
            'total_bases_detected': len(self.bases),
            'total_breakouts_detected': len(self.breakouts),
        }
        
        # Stage別の分析
        if current_stage == 1:
            # Stage 1: ベース分析に焦点
            latest_base = self.bases[-1] if self.bases else None
            
            if latest_base:
                # ベース品質評価
                quality = self.calculate_base_quality_score(latest_base)
                latest_base['quality_score'] = quality['total_score']
                latest_base['quality_details'] = quality
                
                report['latest_base'] = latest_base
                
                # Stage 1での解釈
                if current_substage == '1B':
                    interpretation = 'Stage 1後期: ブレイクアウト準備中、高出来高での上抜けを監視'
                    action = 'ウォッチリスト追加、ブレイクアウト待ち'
                    priority = 'High'
                elif current_substage == '1':
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
            
            report['stage_interpretation'] = interpretation
            report['stage_action'] = action
            report['priority'] = priority
        
        elif current_stage == 2:
            # Stage 2: ベースカウントとブレイクアウト分析
            stage2_base_count = self.count_bases_in_stage(2, stage_detector)
            stage2_breakouts = [bo for bo in self.breakouts if bo.get('is_stage2') is True]
            
            report['stage2_base_count'] = stage2_base_count
            report['stage2_breakout_count'] = len(stage2_breakouts)
            report['stage2_breakouts'] = stage2_breakouts
            
            # 最新ブレイクアウト
            latest_breakout = self.breakouts[-1] if self.breakouts else None
            if latest_breakout:
                report['latest_breakout'] = latest_breakout
            
            # Stage 2での解釈（ベースカウント重視）
            if stage2_base_count <= 2:
                interpretation = f'Stage 2: 上昇期、ベース{stage2_base_count}個目（最良の機会）'
                action = 'エントリー検討、特に1-2番目のベース後が理想的'
            elif stage2_base_count == 3:
                interpretation = f'Stage 2: 上昇期、ベース{stage2_base_count}個目（注意が必要）'
                action = '慎重にエントリー検討、利確も視野に'
            else:
                interpretation = f'Stage 2: 上昇期、ベース{stage2_base_count}個目（後期の可能性）'
                action = '新規エントリー非推奨、既存ポジションは利確検討'
            
            report['stage_interpretation'] = interpretation
            report['stage_action'] = action
        
        elif current_stage == 3:
            report['stage_interpretation'] = 'Stage 3: 天井形成期、分配フェーズ'
            report['stage_action'] = '新規エントリー回避、既存ポジション撤退'
        
        elif current_stage == 4:
            report['stage_interpretation'] = 'Stage 4: 下降期'
            report['stage_action'] = 'ロングポジション回避、Stage 1入り待ち'
        
        return report
    
    def generate_report(self) -> Dict:
        """
        包括的な分析レポートを生成（Stage情報なし版）
        
        ※ Stage情報が必要な場合は analyze_with_stage() を使用
        
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
        latest_breakout = self.breakouts[-1] if self.breakouts else None
        
        # ベース数（直近1年）
        one_year_ago = self.df.index[-1] - pd.Timedelta(days=252)
        recent_bases = [b for b in self.bases if b['end_date'] >= one_year_ago]
        
        report = {
            'total_bases_detected': len(self.bases),
            'total_breakouts_detected': len(self.breakouts),
            'recent_base_count_1yr': len(recent_bases),
            'latest_base': latest_base,
            'latest_breakout': latest_breakout,
        }
        
        # 解釈とアクション（Stage情報なし）
        if latest_base and latest_base['distance_from_high_pct'] < 5:
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
    
    print("ベース検出（リファクタリング版）のテストを開始...")
    print("StageDetectorとの連携機能をテスト\n")
    
    test_tickers = ['AAPL', 'NVDA', 'TSLA']
    
    for ticker in test_tickers:
        print(f"\n{'='*60}")
        print(f"{ticker} のベース分析（StageDetector連携）:")
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
                print(f"検出されたベース数: {report['total_bases_detected']}")
                
                # Stage 1の場合
                if report['current_stage'] == 1:
                    if report.get('latest_base'):
                        base = report['latest_base']
                        print(f"\n最新ベース:")
                        print(f"  期間: {base['start_date'].strftime('%Y-%m-%d')} - {base['end_date'].strftime('%Y-%m-%d')}")
                        print(f"  継続期間: {base['duration_weeks']:.1f}週")
                        print(f"  深さ: {base['depth_pct']:.1f}%")
                        print(f"  ピボットポイント: ${base['pivot_point']:.2f}")
                        
                        if 'quality_score' in base:
                            print(f"  品質スコア: {base['quality_score']:.1f}/100")
                            quality = base['quality_details']
                            print(f"    期間: {quality['period_rating']} ({quality['period_score']}点)")
                            print(f"    深さ: {quality['depth_rating']} ({quality['depth_score']}点)")
                            print(f"    出来高: {quality['volume_rating']} ({quality['volume_score']}点)")
                            print(f"    形状: {quality['shape_rating']} ({quality['shape_score']}点)")
                    
                    print(f"\nStage解釈: {report.get('stage_interpretation', 'N/A')}")
                    print(f"推奨アクション: {report.get('stage_action', 'N/A')}")
                    print(f"優先度: {report.get('priority', 'N/A')}")
                
                # Stage 2の場合
                elif report['current_stage'] == 2:
                    print(f"Stage 2内のベース数: {report.get('stage2_base_count', 0)}")
                    print(f"Stage 2内のブレイクアウト数: {report.get('stage2_breakout_count', 0)}")
                    
                    if report.get('latest_breakout'):
                        bo = report['latest_breakout']
                        print(f"\n最新ブレイクアウト:")
                        print(f"  日付: {bo['breakout_date'].strftime('%Y-%m-%d')}")
                        print(f"  価格: ${bo['breakout_price']:.2f}")
                        print(f"  出来高倍率: {bo['volume_ratio']:.2f}x")
                        print(f"  品質スコア: {bo['quality_score']:.1f}/100")
                    
                    print(f"\nStage解釈: {report.get('stage_interpretation', 'N/A')}")
                    print(f"推奨アクション: {report.get('stage_action', 'N/A')}")