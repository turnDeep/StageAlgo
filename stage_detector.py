"""
ステージ検出モジュール（Stage 1判定強化版）
Minerviniテンプレート充足度によるStage 1サブステージ判定
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple


class StageDetector:
    """
    ステージ判定システム（Stage 1強化版）
    """
    
    def __init__(self, df: pd.DataFrame, benchmark_df: pd.DataFrame = None):
        self.df = df
        self.benchmark_df = benchmark_df
        self.latest = df.iloc[-1]

    def check_minervini_template(self) -> Dict:
        """
        Mark MinerviniのTrend Template 8基準をチェック
        
        Returns:
            dict: 各基準の結果とスコア
        """
        checks = {}
        
        # 現在の価格とMA
        current_price = self.latest['Close']
        sma_50 = self.latest['SMA_50']
        sma_150 = self.latest['SMA_150']
        sma_200 = self.latest['SMA_200']
        
        # 基準1: 現在価格 > 150日MA and 200日MA
        checks['criterion_1'] = (current_price > sma_150) and (current_price > sma_200)
        
        # 基準2: 150日MA > 200日MA
        checks['criterion_2'] = sma_150 > sma_200
        
        # 基準3: 200日MAが上昇トレンド (最低1ヶ月 = 20営業日)
        if len(self.df) >= 21:
            sma_200_20d_ago = self.df['SMA_200'].iloc[-21]
            checks['criterion_3'] = sma_200 > sma_200_20d_ago
        else:
            checks['criterion_3'] = False
        
        # 基準4: 50日MA > 150日MA and 200日MA
        checks['criterion_4'] = (sma_50 > sma_150) and (sma_50 > sma_200)
        
        # 基準5: 現在価格 > 50日MA
        checks['criterion_5'] = current_price > sma_50
        
        # 基準6: 52週安値から30%以上上
        low_52w = self.latest['Low_52W']
        if pd.notna(low_52w) and low_52w > 0:
            gain_from_low = (current_price - low_52w) / low_52w
            checks['criterion_6'] = gain_from_low > 0.30
        else:
            checks['criterion_6'] = False
        
        # 基準7: 52週高値の25%以内
        high_52w = self.latest['High_52W']
        if pd.notna(high_52w) and high_52w > 0:
            dist_from_high = (high_52w - current_price) / high_52w
            checks['criterion_7'] = dist_from_high < 0.25
        else:
            checks['criterion_7'] = False
        
        # 基準8: RS Rating ≥ 70
        if 'RS_Rating' in self.df.columns:
            rs_rating = self.latest['RS_Rating']
            checks['criterion_8'] = rs_rating >= 70
        else:
            checks['criterion_8'] = False
        
        # スコア計算
        criteria_met = sum(checks.values())
        template_score = (criteria_met / len(checks)) * 100
        all_pass = all(checks.values())
        
        return {
            'all_criteria_met': all_pass,
            'score': template_score,
            'checks': checks,
            'criteria_met': criteria_met,
            'total_criteria': len(checks)
        }
    
    def determine_stage(self) -> Tuple[int, str]:
        """
        現在のステージを判定（Stage 1強化版）
        
        Returns:
            tuple: (ステージ番号, サブステージ)
        """
        # Minerviniトレンドテンプレートをチェック
        template_result = self.check_minervini_template()
        
        # 全8基準を満たす場合はStage 2確定
        if template_result['all_criteria_met']:
            substage = self._determine_stage2_substage()
            return 2, substage
        
        # MAの配置と傾きを確認
        current_price = self.latest['Close']
        sma_150 = self.latest['SMA_150']
        sma_200 = self.latest['SMA_200']
        sma_150_slope = self.latest['SMA_150_Slope']
        sma_200_slope = self.latest['SMA_200_Slope']
        
        # ===== Stage 4判定: 明確な下降トレンド =====
        if sma_150_slope < -0.002 and current_price < sma_150:
            substage = self._determine_stage4_substage()
            return 4, substage
        
        # ===== Stage 1判定: ベース形成（横ばい） =====
        # MA横ばい条件を緩和：Stage 1Fは上向きでもよい
        if abs(sma_200_slope) < 0.02:  # 200日MAが比較的平坦
            # 価格が両MA の±25%以内（従来より広く）
            if 0.75 < current_price/sma_150 < 1.25:
                # Minerviniテンプレート充足度で詳細サブステージを判定
                substage = self._determine_stage1_substage_enhanced(template_result)
                return 1, substage
        
        # ===== Stage 3判定: 天井圏（MAは横ばい、価格は上） =====
        if abs(sma_150_slope) < 0.02 and current_price > sma_150:
            substage = self._determine_stage3_substage()
            return 3, substage
        
        # どれにも当てはまらない場合
        return 0, "Undefined"
    
    def _determine_stage1_substage_enhanced(self, template_result: Dict) -> str:
        """
        Stage 1のサブステージをMinerviniテンプレート充足度で判定（強化版）

        Args:
            template_result: check_minervini_template()の結果

        Returns:
            str: サブステージ（1A, 1C, 1D, 1E, 1F）
        """
        criteria_met = template_result['criteria_met']  # 満たした基準数
        checks = template_result['checks']

        # 個別基準の確認
        price_above_150_200 = checks['criterion_1']  # 価格 > 150日MA & 200日MA
        ma_150_above_200 = checks['criterion_2']      # 150日MA > 200日MA
        ma_200_trending_up = checks['criterion_3']    # 200日MA上昇
        ma_50_above_150_200 = checks['criterion_4']   # 50日MA > 150日MA & 200日MA
        price_above_50 = checks['criterion_5']        # 価格 > 50日MA
        above_52w_low = checks['criterion_6']         # 52週安値から30%以上
        near_52w_high = checks['criterion_7']         # 52週高値の25%以内
        rs_rating_70 = checks['criterion_8']          # RS Rating ≥ 70

        # 補助的指標
        rs_rating = self.latest.get('RS_Rating', 50)
        relative_vol = self.latest['Relative_Volume']

        # ベース期間の推定
        base_weeks = self._estimate_base_weeks()

        # ベース内位置（0.0-1.0）
        base_position = self._calculate_base_position()

        # =====================================================
        # Stage 1F: ブレイク直前（8基準中7個満たす）
        # =====================================================
        if criteria_met >= 7:
            # 重要な4つの核心基準が満たされているか
            critical_criteria = (
                price_above_150_200 and  # 価格位置が良い
                ma_150_above_200 and     # MA配列が整っている
                ma_200_trending_up and   # トレンドが上向き
                price_above_50           # 短期MAの上
            )
            if critical_criteria:
                # 出来高Dry UpとRS Ratingの最終確認
                if relative_vol < 0.30 and rs_rating >= 85:
                    return "1F"  # 完璧な準備完了
                elif relative_vol < 0.40 or rs_rating >= 80:
                    return "1F"  # ほぼ準備完了

        # =====================================================
        # Stage 1E: ベース後期（8基準中6個満たす）
        # =====================================================
        if criteria_met >= 6:
            # MA配列の核心部分が整っているか
            ma_alignment_core = (
                ma_150_above_200 and
                ma_200_trending_up and
                price_above_50
            )
            if ma_alignment_core:
                # RS RatingまたはDry Upのいずれかが良好
                if rs_rating >= 70:
                    return "1E"  # RS Ratingが基準達成
                elif relative_vol < 0.40 and rs_rating >= 65:
                    return "1E"  # 出来高が縮小中
                elif base_position > 0.75:
                    return "1E"  # ベース上部に位置

        # 5基準満たす場合も、重要な基準が揃っていれば1E
        if criteria_met == 5:
            if ma_150_above_200 and ma_200_trending_up and above_52w_low:
                if rs_rating >= 75 or (relative_vol < 0.40 and base_position > 0.7):
                    return "1E"

        # =====================================================
        # Stage 1D: ベース中期（8基準中4-5個満たす）
        # =====================================================
        if criteria_met >= 4:
            # 基本的な回復が見られるか
            basic_recovery = (
                above_52w_low and  # 底から30%以上回復
                (ma_150_above_200 or ma_200_trending_up)  # MAが整い始める
            )
            if basic_recovery:
                # RS Ratingが上昇中、またはベース期間が適切
                if rs_rating >= 60:
                    return "1D"  # RS改善中
                elif base_weeks >= 10 and base_weeks <= 25:
                    return "1D"  # 適切な期間
                elif relative_vol < 0.50:
                    return "1D"  # 出来高が減少傾向

        # =====================================================
        # Stage 1C: 初期安定化（8基準中2-3個満たす）
        # =====================================================
        if criteria_met >= 2:
            # ベース形成が始まったばかり
            if base_weeks < 10:
                return "1C"  # 初期安定化
            # または、期間は長いが基準が少ない
            elif base_weeks >= 10 and criteria_met == 3:
                if above_52w_low:  # 底からの回復は確認
                    return "1C"  # まだ初期段階

        # =====================================================
        # Stage 1A: ベースの開始（8基準中0-1個満たす）
        # =====================================================
        return "1A"  # ベースの最初期

    def _estimate_base_weeks(self) -> float:
        """
        ベース期間を推定（週数）
        150日MAが横ばいになった時点から現在まで
        """
        slope_threshold = 0.02
        base_start_idx = None
        
        # 最大42週（210日）前まで探索
        lookback = min(210, len(self.df) - 1)

        for i in range(len(self.df)-1, max(0, len(self.df)-lookback), -1):
            if abs(self.df['SMA_150_Slope'].iloc[i]) >= slope_threshold:
                base_start_idx = i + 1
                break
        
        if base_start_idx is None:
            return 30.0  # デフォルト30週
        
        base_days = len(self.df) - base_start_idx
        base_weeks = base_days / 5
        
        return base_weeks

    def _calculate_base_position(self) -> float:
        """
        ベース内での現在価格の相対位置を計算（0.0-1.0）
        
        Returns:
            float: 0.0（ベース最安値）〜 1.0（ベース最高値）
        """
        # ベース期間を推定
        base_weeks = self._estimate_base_weeks()
        base_days = int(base_weeks * 5)
        
        if base_days > len(self.df):
            base_days = len(self.df)
        
        base_data = self.df.tail(base_days)
        
        if len(base_data) < 10:
            return 0.5  # データ不足
        
        base_high = base_data['High'].max()
        base_low = base_data['Low'].min()
        current_price = self.latest['Close']
        
        if base_high - base_low < 1e-6:
            return 0.5
        
        position = (current_price - base_low) / (base_high - base_low)

        return max(0.0, min(1.0, position))

    def _determine_stage2_substage(self) -> str:
        """Stage 2のサブステージを判定"""
        # 既存のロジックを使用
        # （元のコードと同じ）
        # ... 省略 ...
        return "2A"  # 簡略化
    
    def _determine_stage3_substage(self) -> str:
        """Stage 3のサブステージを判定"""
        return "3"
    
    def _determine_stage4_substage(self) -> str:
        """Stage 4のサブステージを判定"""
        return "4"


if __name__ == '__main__':
    # テスト用
    from data_fetcher import fetch_stock_data
    from indicators import calculate_all_basic_indicators
    
    print("ステージ検出（Stage 1強化版）のテストを開始...")
    
    test_tickers = ['AAPL', 'TSLA', 'NVDA']
    
    for ticker in test_tickers:
        print(f"\n{'='*60}")
        print(f"{ticker} の分析:")
        print(f"{'='*60}")

        stock_df, benchmark_df = fetch_stock_data(ticker, period='2y')
        
        if stock_df is not None:
            indicators_df = calculate_all_basic_indicators(stock_df)
            indicators_df = indicators_df.dropna()
            
            if len(indicators_df) >= 200:
                detector = StageDetector(indicators_df, benchmark_df)
                
                # Minerviniテンプレートチェック
                template = detector.check_minervini_template()
                print(f"トレンドテンプレート: {template['criteria_met']}/8基準 満たす")
                print(f"  スコア: {template['score']:.1f}点")

                # 各基準の詳細
                print(f"\n基準の詳細:")
                for i, (key, value) in enumerate(template['checks'].items(), 1):
                    status = "✓" if value else "✗"
                    print(f"  基準{i}: {status}")
                
                # ステージ判定
                stage, substage = detector.determine_stage()
                print(f"\nステージ: {stage} ({substage})")

                # Stage 1の場合、詳細情報
                if stage == 1:
                    base_weeks = detector._estimate_base_weeks()
                    base_position = detector._calculate_base_position()
                    rs_rating = indicators_df.iloc[-1].get('RS_Rating', 50)
                    relative_vol = indicators_df.iloc[-1]['Relative_Volume']

                    print(f"\nStage 1詳細:")
                    print(f"  ベース期間: {base_weeks:.1f}週")
                    print(f"  ベース内位置: {base_position:.1%}")
                    print(f"  RS Rating: {rs_rating:.0f}")
                    print(f"  相対出来高: {relative_vol:.2f}")