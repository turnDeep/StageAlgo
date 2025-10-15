"""
Mark Minervini Trend Template Detector
Minerviniの8基準トレンドテンプレート判定
Stage 2上昇トレンド内の強い銘柄を特定
"""
import pandas as pd
import numpy as np
from typing import Dict


class MinerviniTemplateDetector:
    """
    Mark Minervini Trend Template システム
    
    8つの基準:
    1. 現在価格 > 150日MA & 200日MA
    2. 150日MA > 200日MA
    3. 200日MAが最低1ヶ月（好ましくは4-5ヶ月）上昇トレンド
    4. 50日MA > 150日MA & 200日MA
    5. 現在価格 > 50日MA
    6. 現在価格が52週安値から30%以上上
    7. 現在価格が52週高値の25%以内
    8. RS Rating ≥ 70（理想的には80-90）
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Args:
            df: 指標計算済みのDataFrame（日足）
        """
        self.df = df
        self.latest = df.iloc[-1]
        
    def check_template(self) -> Dict:
        """
        Minerviniのトレンドテンプレート8基準をチェック
        
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
        
        # 基準3: 200日MAが上昇トレンド（最低1ヶ月=20営業日）
        if len(self.df) >= 21:
            sma_200_20d_ago = self.df['SMA_200'].iloc[-21]
            criterion_3_passed = sma_200 > sma_200_20d_ago
            
            # 4-5ヶ月（100日）の上昇トレンドもチェック
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
            'all_criteria_met': all_pass,
            'score': template_score,
            'checks': checks,
            'criteria_met': criteria_met,
            'total_criteria': len(checks),
            'interpretation': self._interpret_results(criteria_met, checks)
        }
    
    def _interpret_results(self, criteria_met: int, checks: Dict) -> Dict:
        """
        結果の解釈とアクション推奨
        
        Args:
            criteria_met: 満たした基準数
            checks: 各基準の結果
            
        Returns:
            dict: 解釈とアクション
        """
        interpretation = {
            'status': '',
            'action': '',
            'strength': '',
            'warnings': []
        }
        
        # 全8基準を満たす場合
        if criteria_met == 8:
            interpretation['status'] = '完璧なStage 2上昇トレンド'
            interpretation['action'] = '強力な買い候補、エントリー検討'
            interpretation['strength'] = 'Excellent'
            
            # RS Ratingのレベルでさらに評価
            rs_rating = checks['criterion_8']['values'].get('rs_rating')
            if rs_rating and rs_rating >= 90:
                interpretation['strength'] = 'Exceptional'
                interpretation['action'] = '最優先買い候補、即座にエントリー検討'
        
        # 7基準を満たす場合
        elif criteria_met == 7:
            interpretation['status'] = '非常に強いStage 2トレンド'
            interpretation['action'] = '買い候補、押し目でエントリー'
            interpretation['strength'] = 'Very Strong'
            
            # どの基準を満たしていないか確認
            failed_criteria = [k for k, v in checks.items() if not v['passed']]
            if 'criterion_8' in failed_criteria:
                interpretation['warnings'].append('RS Rating不足、他の基準は完璧')
            elif 'criterion_3' in failed_criteria:
                interpretation['warnings'].append('200日MAの上昇期間が短い')
        
        # 6基準を満たす場合
        elif criteria_met == 6:
            interpretation['status'] = '強いトレンド'
            interpretation['action'] = '条件付き買い候補、リスク管理重視'
            interpretation['strength'] = 'Strong'
            
            # 重要な基準が満たされているか確認
            critical_met = (
                checks['criterion_1']['passed'] and
                checks['criterion_2']['passed'] and
                checks['criterion_4']['passed'] and
                checks['criterion_5']['passed']
            )
            
            if not critical_met:
                interpretation['warnings'].append('重要なMA配列基準が未達')
        
        # 5基準以下
        else:
            interpretation['status'] = 'トレンドが不明確'
            interpretation['action'] = '見送り、条件改善を待つ'
            interpretation['strength'] = 'Weak'
            
            if criteria_met >= 4:
                interpretation['status'] = 'トレンド形成中の可能性'
                interpretation['action'] = '監視継続、Stage 2移行を待つ'
                interpretation['strength'] = 'Moderate'
        
        return interpretation
    
    def get_detailed_report(self) -> str:
        """
        詳細なレポートを生成
        
        Returns:
            str: 詳細レポート
        """
        result = self.check_template()
        
        report = []
        report.append("="*60)
        report.append("Mark Minervini Trend Template 分析")
        report.append("="*60)
        report.append(f"\n総合判定: {result['criteria_met']}/8 基準 満たす")
        report.append(f"スコア: {result['score']:.1f}/100")
        report.append(f"ステータス: {result['interpretation']['status']}")
        report.append(f"強度: {result['interpretation']['strength']}")
        report.append(f"アクション: {result['interpretation']['action']}")
        
        if result['interpretation']['warnings']:
            report.append(f"\n注意事項:")
            for warning in result['interpretation']['warnings']:
                report.append(f"  ⚠ {warning}")
        
        report.append(f"\n基準の詳細:")
        for i, (key, check) in enumerate(result['checks'].items(), 1):
            status = "✓" if check['passed'] else "✗"
            report.append(f"  {i}. {status} {check['description']}")
            
            # 値の詳細表示
            for value_key, value in check['values'].items():
                if value is not None:
                    if isinstance(value, float):
                        report.append(f"      {value_key}: {value:.2f}")
                    else:
                        report.append(f"      {value_key}: {value}")
        
        return "\n".join(report)


if __name__ == '__main__':
    # テスト用
    from data_fetcher import fetch_stock_data
    from indicators import calculate_all_basic_indicators
    from rs_calculator import analyze_rs_metrics
    
    print("Mark Minervini Trend Template のテストを開始...")
    
    test_tickers = ['AAPL', 'TSLA', 'NVDA']
    
    # ベンチマーク取得
    _, benchmark_df = fetch_stock_data('SPY', period='2y')
    if benchmark_df is not None:
        benchmark_df = calculate_all_basic_indicators(benchmark_df)
    
    for ticker in test_tickers:
        print(f"\n{'='*60}")
        print(f"{ticker} の分析:")
        print(f"{'='*60}")
        
        stock_df, _ = fetch_stock_data(ticker, period='2y')
        
        if stock_df is not None and benchmark_df is not None:
            indicators_df = calculate_all_basic_indicators(stock_df)
            
            # RS Ratingを追加
            rs_result = analyze_rs_metrics(indicators_df, benchmark_df)
            if 'rs_rating' in rs_result:
                indicators_df['RS_Rating'] = rs_result['rs_rating']
            
            indicators_df = indicators_df.dropna()
            
            if len(indicators_df) >= 200:
                detector = MinerviniTemplateDetector(indicators_df)
                
                # 詳細レポート出力
                print(detector.get_detailed_report())
