"""
総合スコアリングシステム（Stage 1強化版）
Minerviniテンプレート充足度を重視
"""
import pandas as pd
from typing import Dict
from stage_detector import StageDetector
from base_analyzer import BaseAnalyzer
from rs_calculator import analyze_rs_metrics
from atr_analyzer import ATRAnalyzer
from volume_analyzer import VolumeAnalyzer
from vwap_analyzer import VWAPAnalyzer


class ScoringSystem:
    """
    統合スコアリングシステム（Stage 1強化版）
    """
    
    def __init__(self, df: pd.DataFrame, ticker: str, benchmark_df: pd.DataFrame):
        """
        Args:
            df: 指標計算済みのDataFrame
            ticker: ティッカーシンボル
            benchmark_df: ベンチマークデータ
        """
        self.df = df
        self.ticker = ticker
        self.benchmark_df = benchmark_df
        
        # 各分析器の初期化
        self.stage_detector = StageDetector(df, benchmark_df)
        self.base_analyzer = BaseAnalyzer(df)
        self.atr_analyzer = ATRAnalyzer(df)
        self.volume_analyzer = VolumeAnalyzer(df)
        self.vwap_analyzer = VWAPAnalyzer(df)

    def score_stage1_candidate(self) -> Dict:
        """
        Stage 1候補のスコアリング（100点満点）- テンプレート充足度重視版
        
        配分:
        - テンプレート充足度: 30点（VCPの25点を統合）
        - ベース品質: 25点
        - RS Rating: 25点
        - 出来高: 15点
        - ATR位置: 5点（縮小）
        """
        result = {
            'stage': 1,
            'total_score': 0,
            'breakdown': {},
            'details': {}
        }
        
        # 1. Minerviniテンプレート充足度（30点）
        template_result = self.stage_detector.check_minervini_template()
        criteria_met = template_result['criteria_met']

        # 充足度スコアリング（段階的）
        if criteria_met >= 7:
            template_score = 30  # Stage 1F相当
        elif criteria_met >= 6:
            template_score = 25  # Stage 1E相当
        elif criteria_met >= 5:
            template_score = 20  # Stage 1E寄り
        elif criteria_met >= 4:
            template_score = 15  # Stage 1D相当
        elif criteria_met >= 3:
            template_score = 12  # Stage 1D初期
        elif criteria_met >= 2:
            template_score = 8   # Stage 1C相当
        else:
            template_score = 5   # Stage 1A相当

        result['breakdown']['template_fulfillment'] = template_score
        result['details']['template'] = template_result
        result['details']['criteria_met'] = criteria_met

        # 2. ベース品質（25点）
        base_result = self.base_analyzer.calculate_base_quality_score()
        if base_result['base_detected']:
            base_score = base_result['total_score'] * 0.25
            result['details']['base'] = base_result['details']
        else:
            base_score = 0
            result['details']['base'] = {'error': 'ベース未検出'}
        
        result['breakdown']['base_quality'] = base_score
        
        # 3. RS Rating（25点）
        rs_result = analyze_rs_metrics(self.df, self.benchmark_df)
        rs_rating = rs_result['rs_rating']
        
        # Stage 1では、RS Ratingの基準を段階的に
        if rs_rating >= 85:
            rs_score = 25
        elif rs_rating >= 80:
            rs_score = 23
        elif rs_rating >= 75:
            rs_score = 20
        elif rs_rating >= 70:
            rs_score = 18
        elif rs_rating >= 65:
            rs_score = 15
        elif rs_rating >= 60:
            rs_score = 12
        else:
            rs_score = 8
        
        result['breakdown']['rs_rating'] = rs_score
        result['details']['rs'] = rs_result
        
        # 4. 出来高（15点）
        volume_result = self.volume_analyzer.calculate_volume_score()
        volume_score = volume_result['total_score'] * 0.15
        
        result['breakdown']['volume'] = volume_score
        result['details']['volume'] = volume_result
        
        # 5. ATR位置（5点）- 重要度低下
        atr_result = self.atr_analyzer.analyze_atr_metrics(stage=1)
        atr_multiple = atr_result['atr_multiple_ma50']
        
        if -0.5 <= atr_multiple <= 0.5:
            atr_score = 5
        elif -1.0 <= atr_multiple <= 1.0:
            atr_score = 4
        elif -2.0 <= atr_multiple <= 2.0:
            atr_score = 3
        else:
            atr_score = 1
        
        result['breakdown']['atr_position'] = atr_score
        result['details']['atr'] = atr_result
        
        # 総合スコア
        result['total_score'] = sum(result['breakdown'].values())
        
        # 判定（Stage 1サブステージも考慮）
        stage_info = self.stage_detector.determine_stage()
        substage = stage_info[1]

        if result['total_score'] >= 90:
            result['grade'] = 'A+'
            result['priority'] = '最優先監視'
            if substage in ['1F', '1E']:
                result['action'] = f'即座に買い準備（{substage}）、ブレイクアウト監視'
            else:
                result['action'] = '優先監視リスト、条件改善を待つ'
        elif result['total_score'] >= 80:
            result['grade'] = 'A'
            result['priority'] = '優先監視'
            if substage == '1F':
                result['action'] = f'エントリー準備（{substage}）、最終確認'
            elif substage == '1E':
                result['action'] = f'エントリー準備（{substage}）、条件確認'
            else:
                result['action'] = '発展を監視'
        elif result['total_score'] >= 70:
            result['grade'] = 'B'
            result['priority'] = '監視継続'
            result['action'] = f'発展を監視（{substage}）'
        else:
            result['grade'] = 'C'
            result['priority'] = '低優先度'
            result['action'] = f'条件が揃うまで待機（{substage}）'
        
        return result
    
    def score_stage2_candidate(self, base_count: int = 1) -> Dict:
        """
        Stage 2候補のスコアリング（100点満点）
        """
        # (This method remains unchanged, but needs to be included in the file)
        result = {
            'stage': 2,
            'total_score': 0,
            'breakdown': {},
            'details': {}
        }
        
        # 1. トレンド強度 (20点) - Minerviniテンプレート
        template_result = self.stage_detector.check_minervini_template()
        trend_score = template_result['score'] * 0.20
        
        result['breakdown']['trend_strength'] = trend_score
        result['details']['template'] = template_result
        
        # 2. ベース品質 + ベースカウント (20点)
        base_result = self.base_analyzer.calculate_base_quality_score()
        
        if base_result['base_detected']:
            base_quality_component = base_result['total_score'] * 0.10
        else:
            base_quality_component = 5  # 最低点
        
        # ベースカウント評価
        if base_count <= 2:
            base_count_component = 10
        elif base_count == 3:
            base_count_component = 5
        else:
            base_count_component = 0
        
        base_score = base_quality_component + base_count_component
        
        result['breakdown']['base_quality'] = base_score
        result['details']['base'] = base_result
        result['details']['base_count'] = base_count
        
        # 3. RS Rating (20点)
        rs_result = analyze_rs_metrics(self.df, self.benchmark_df)
        rs_rating = rs_result['rs_rating']
        
        if rs_rating >= 90:
            rs_score = 20
        elif rs_rating >= 85:
            rs_score = 18
        elif rs_rating >= 80:
            rs_score = 16
        elif rs_rating >= 70:
            rs_score = 12
        else:
            rs_score = 8
        
        result['breakdown']['rs_rating'] = rs_score
        result['details']['rs'] = rs_result
        
        # 4. 出来高 (20点)
        volume_result = self.volume_analyzer.calculate_volume_score()
        volume_score = volume_result['total_score'] * 0.20
        
        result['breakdown']['volume'] = volume_score
        result['details']['volume'] = volume_result
        
        # 5. ATR位置 (10点)
        atr_result = self.atr_analyzer.analyze_atr_metrics(stage=2)
        atr_multiple = atr_result['atr_multiple_ma50']
        
        if 0 <= atr_multiple < 3.0:
            atr_score = 10
        elif 3.0 <= atr_multiple < 5.0:
            atr_score = 8
        elif 5.0 <= atr_multiple < 7.0:
            atr_score = 5
        elif 7.0 <= atr_multiple < 10.0:
            atr_score = 3  # 利確検討
        else:
            atr_score = 0  # 即座利確
        
        result['breakdown']['atr_position'] = atr_score
        result['details']['atr'] = atr_result
        
        # 6. MA配列 (10点)
        latest = self.df.iloc[-1]
        sma_50 = latest['SMA_50']
        sma_150 = latest['SMA_150']
        sma_200 = latest['SMA_200']
        
        slope_50 = latest['SMA_50_Slope']
        slope_150 = latest['SMA_150_Slope']
        slope_200 = latest['SMA_200_Slope']
        
        if (sma_50 > sma_150 > sma_200 and 
            slope_50 > 0 and slope_150 > 0 and slope_200 > 0):
            ma_score = 10
            ma_rating = 'A'
        elif sma_50 > sma_150 > sma_200:
            ma_score = 7
            ma_rating = 'B'
        else:
            ma_score = 3
            ma_rating = 'C'
        
        result['breakdown']['ma_alignment'] = ma_score
        result['details']['ma_rating'] = ma_rating
        
        # 総合スコア
        result['total_score'] = sum(result['breakdown'].values())
        
        # 判定
        if result['total_score'] >= 90:
            result['grade'] = 'A+'
            result['priority'] = '即座に買い'
            result['action'] = '今すぐエントリー検討'
        elif result['total_score'] >= 85:
            result['grade'] = 'A'
            result['priority'] = '押し目待ち'
            result['action'] = '健全な押し目でエントリー'
        elif result['total_score'] >= 75:
            result['grade'] = 'B'
            result['priority'] = '監視継続'
            result['action'] = '様子見、条件改善を待つ'
        else:
            result['grade'] = 'C'
            result['priority'] = '見送り'
            result['action'] = '新規エントリー非推奨'
        
        return result

    def comprehensive_analysis(self) -> Dict:
        """
        包括的な分析を実行
        
        Returns:
            dict: 完全な分析結果
        """
        result = {
            'ticker': self.ticker,
            'analysis_date': self.df.index[-1].strftime('%Y-%m-%d')
        }
        
        # ステージ判定
        stage, substage = self.stage_detector.determine_stage()
        result['stage'] = stage
        result['substage'] = substage
        
        # ステージ別スコアリング
        if stage == 1:
            scoring_result = self.score_stage1_candidate()
            result.update(scoring_result)
        elif stage == 2:
            # ベースカウントの推定（簡易版）
            base_count = 1  # TODO: より正確な実装
            scoring_result = self.score_stage2_candidate(base_count)
            result.update(scoring_result)
        else:
            result['total_score'] = 0
            result['grade'] = 'N/A'
            result['priority'] = 'Not applicable'
            result['action'] = f'Stage {stage} - スクリーニング対象外'
        
        # VWAP分析（参考情報）
        vwap_result = self.vwap_analyzer.analyze_vwap()
        result['vwap'] = vwap_result
        
        return result

if __name__ == '__main__':
    # テスト用
    from data_fetcher import fetch_stock_data
    from indicators import calculate_all_basic_indicators
    
    print("総合スコアリング（Stage 1強化版）のテストを開始...")
    
    test_tickers = ['AAPL', 'TSLA', 'NVDA']
    
    # ベンチマーク取得
    _, benchmark_df = fetch_stock_data('SPY', period='2y')
    benchmark_df = calculate_all_basic_indicators(benchmark_df)
    
    for ticker in test_tickers:
        print(f"\n{'='*60}")
        print(f"{ticker} の包括的分析:")
        print(f"{'='*60}")
        
        stock_df, _ = fetch_stock_data(ticker, period='2y')
        
        if stock_df is not None and benchmark_df is not None:
            indicators_df = calculate_all_basic_indicators(stock_df)
            
            # RS Ratingを追加
            rs_result = analyze_rs_metrics(indicators_df, benchmark_df)
            if 'rs_rating' in rs_result:
                indicators_df['RS_Rating'] = rs_result['rs_rating']
            else:
                indicators_df['RS_Rating'] = 50 # Default value
            
            indicators_df = indicators_df.dropna()
            
            if len(indicators_df) >= 252:
                scorer = ScoringSystem(indicators_df, ticker, benchmark_df)
                result = scorer.comprehensive_analysis()
                
                print(f"ステージ: {result['stage']} ({result['substage']})")
                print(f"総合スコア: {result.get('total_score', 0):.1f}/100")
                print(f"評価: {result.get('grade', 'N/A')}")
                print(f"優先度: {result.get('priority', 'N/A')}")
                print(f"アクション: {result.get('action', 'N/A')}")
                
                if 'breakdown' in result:
                    print(f"\nスコア内訳:")
                    for key, value in result['breakdown'].items():
                        print(f"  {key}: {value:.1f}")