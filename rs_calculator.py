"""
RS Rating (Relative Strength Rating) Calculation Module (Improved)
Based on William O'Neil's IBD Methodology

【Theoretical Basis】
- IBD RS Rating: Weighted average of 12-month price performance
- RS Line: Relative strength against the benchmark
- Multi-timeframe analysis: 1M, 3M, 6M, 9M, 12M

【Key Features】
1. Accurate IBD-style weighted calculation (40-20-20-20)
2. RS Line new high detection (Blue Sky setup)
3. Multi-timeframe consistency check
4. Integration with Stage Analysis
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional


class RSCalculator:
    """
    RS Rating Calculation System (IBD/O'Neil Style)
    
    Formula (IBD Style):
    RS Score = 40% * ROC(63d) + 20% * ROC(126d) + 20% * ROC(189d) + 20% * ROC(252d)
    
    Where:
    - ROC = Rate of Change
    - 63d approx 1 Quarter (3 months)
    """
    
    def __init__(self, df: pd.DataFrame, benchmark_df: pd.DataFrame):
        """
        Args:
            df: Stock DataFrame with 'Close' column
            benchmark_df: Benchmark DataFrame (e.g. SPY) with 'Close' column
        """
        self.df = df.copy()
        self.benchmark_df = benchmark_df.copy()
        self.latest = df.iloc[-1]
        
    def calculate_roc(self, series: pd.Series, period: int) -> pd.Series:
        """
        Calculate Rate of Change
        """
        if len(series) < period + 1:
            return pd.Series([0] * len(series), index=series.index)
        
        roc = (series / series.shift(period) - 1) * 100
        return roc.fillna(0)
    
    def calculate_ibd_rs_score(self) -> pd.Series:
        """
        Calculate IBD-style RS Score (Time Series)
        
        Weighted Average:
        - 40% * 3 months (63 days)
        - 20% * 6 months (126 days)
        - 20% * 9 months (189 days)
        - 20% * 12 months (252 days)
        
        Returns:
            pd.Series: RS Score Time Series
        """
        close = self.df['Close']
        
        # Calculate ROC for each period
        roc_63 = self.calculate_roc(close, 63)    # 3m
        roc_126 = self.calculate_roc(close, 126)  # 6m
        roc_189 = self.calculate_roc(close, 189)  # 9m
        roc_252 = self.calculate_roc(close, 252)  # 12m
        
        # Weighted Average
        rs_score = (
            0.40 * roc_63 +
            0.20 * roc_126 +
            0.20 * roc_189 +
            0.20 * roc_252
        )
        
        return rs_score
    
    def calculate_percentile_rating(self, rs_score: float, window: int = 252) -> float:
        """
        Convert RS Score to Percentile Rating (1-99) based on self-history.
        Note: True IBD RS is vs Market. This is a proxy using self-history range
        or relative rank if used in a screener loop.
        """
        if len(self.df) < window:
            window = len(self.df)
        
        rs_score_series = self.calculate_ibd_rs_score()
        recent_scores = rs_score_series.tail(window)
        
        valid_scores = recent_scores[recent_scores != 0].dropna()
        
        if len(valid_scores) == 0:
            return 50
        
        rank = (valid_scores < rs_score).sum()
        percentile = (rank / len(valid_scores)) * 98 + 1
        
        return min(99, max(1, percentile))
    
    def calculate_rs_line(self) -> pd.Series:
        """
        Calculate RS Line (Relative Strength Line)
        RS Line = (Stock Price / Benchmark Price) * 100
        """
        common_index = self.df.index.intersection(self.benchmark_df.index)
        
        if len(common_index) == 0:
            return pd.Series([100] * len(self.df), index=self.df.index)
        
        stock_close = self.df.loc[common_index, 'Close']
        benchmark_close = self.benchmark_df.loc[common_index, 'Close']
        
        rs_line = (stock_close / benchmark_close) * 100
        
        rs_line_full = pd.Series(index=self.df.index, dtype=float)
        rs_line_full.loc[common_index] = rs_line
        rs_line_full = rs_line_full.ffill().bfill().fillna(100)
        
        return rs_line_full
    
    def check_rs_line_new_high(self, rs_line: pd.Series, lookback_days: int = 252) -> Dict:
        """
        Check if RS Line is at a new high
        """
        if len(rs_line) < lookback_days + 1:
            return {
                'is_new_high': False,
                'reason': 'Insufficient Data',
                'days_since_high': None,
                'percent_from_high': None
            }
        
        current_rs = rs_line.iloc[-1]
        historical_data = rs_line.iloc[-lookback_days:-1]
        historical_max = historical_data.max()
        
        is_new_high = current_rs > historical_max
        
        if historical_max > 0:
            days_since_high = len(rs_line) - historical_data.idxmax() - 1
            percent_from_high = ((current_rs - historical_max) / historical_max) * 100
        else:
            days_since_high = None
            percent_from_high = None
        
        return {
            'is_new_high': is_new_high,
            'current_rs_line': current_rs,
            'historical_max': historical_max,
            'days_since_high': days_since_high if not is_new_high else 0,
            'percent_from_high': percent_from_high,
            'strength': self._interpret_rs_line_strength(percent_from_high) if percent_from_high is not None else 'Unknown'
        }

    def detect_blue_sky_setup(self, window: int = 50) -> Dict:
        """
        Detect 'Blue Sky' setup where RS Line makes a new high but Price is still in a base.
        (Price Lagging, RS Leading)

        Args:
            window: Lookback window for highs (default 50 days)

        Returns:
            dict: Detection result
        """
        if len(self.df) < window + 1:
            return {'is_blue_sky': False, 'details': 'Insufficient data'}

        rs_line = self.calculate_rs_line()
        close = self.df['Close']

        # Rolling max of the *previous* 'window' days (as of yesterday)
        rs_high_prev = rs_line.shift(1).rolling(window).max().iloc[-1]
        price_high_prev = close.shift(1).rolling(window).max().iloc[-1]

        current_rs = rs_line.iloc[-1]
        current_price = close.iloc[-1]

        # Check RS Breakout (Current RS >= Previous 50-day High)
        rs_breakout = current_rs >= rs_high_prev

        # Check Price in Base (Price < Previous 50-day High AND Price > 85% of High)
        price_in_base = (current_price < price_high_prev) and (current_price > price_high_prev * 0.85)

        is_blue_sky = bool(rs_breakout and price_in_base)

        return {
            'is_blue_sky': is_blue_sky,
            'rs_breakout': rs_breakout,
            'price_in_base': price_in_base,
            'current_rs': current_rs,
            'prev_rs_high': rs_high_prev,
            'current_price': current_price,
            'prev_price_high': price_high_prev
        }
    
    def _interpret_rs_line_strength(self, percent_from_high: float) -> str:
        """Interpret RS Line Strength"""
        if percent_from_high > 5:
            return 'Excellent - Strong Breakout'
        elif percent_from_high > 2:
            return 'Very Strong - New High'
        elif percent_from_high > 0:
            return 'Strong - At New High'
        elif percent_from_high > -5:
            return 'Good - Near High'
        elif percent_from_high > -10:
            return 'Moderate - Some Weakness'
        else:
            return 'Weak - Significant Pullback'
    
    def calculate_multi_timeframe_rs(self) -> Dict:
        """
        Multi-timeframe RS Analysis
        """
        close = self.df['Close']
        
        timeframes = {
            '1M': 21,
            '3M': 63,
            '6M': 126,
            '9M': 189,
            '12M': 252
        }
        
        results = {}
        
        for name, period in timeframes.items():
            if len(close) >= period + 1:
                roc = self.calculate_roc(close, period).iloc[-1]
                
                # Compare with Benchmark
                if len(self.benchmark_df) >= period + 1:
                    benchmark_roc = self.calculate_roc(
                        self.benchmark_df['Close'], period
                    ).iloc[-1] if len(self.benchmark_df['Close']) > period else 0
                    
                    outperformance = roc - benchmark_roc
                else:
                    benchmark_roc = 0
                    outperformance = roc
                
                results[name] = {
                    'roc': roc,
                    'benchmark_roc': benchmark_roc,
                    'outperformance': outperformance,
                    'rating': self._rate_performance(roc, outperformance)
                }
            else:
                results[name] = {
                    'roc': 0,
                    'benchmark_roc': 0,
                    'outperformance': 0,
                    'rating': 'N/A'
                }
        
        # Consistency Check
        all_positive = all(
            results[tf]['outperformance'] > 0 
            for tf in timeframes.keys() 
            if results[tf]['rating'] != 'N/A'
        )
        
        results['consistency'] = {
            'all_timeframes_positive': all_positive,
            'strength': 'Excellent' if all_positive else 'Mixed'
        }
        
        return results
    
    def _rate_performance(self, roc: float, outperformance: float) -> str:
        if outperformance > 20 and roc > 20:
            return 'A+'
        elif outperformance > 15 and roc > 15:
            return 'A'
        elif outperformance > 10 and roc > 10:
            return 'B+'
        elif outperformance > 5 and roc > 5:
            return 'B'
        elif outperformance > 0:
            return 'C'
        else:
            return 'D'
    
    def analyze_rs_with_stage(self, current_stage: int, current_substage: str) -> Dict:
        """
        Comprehensive RS Analysis integrated with Stage Analysis
        """
        rs_score_series = self.calculate_ibd_rs_score()
        current_rs_score = rs_score_series.iloc[-1]
        rs_rating = self.calculate_percentile_rating(current_rs_score)
        
        rs_line = self.calculate_rs_line()
        rs_line_analysis = self.check_rs_line_new_high(rs_line)
        
        multi_tf = self.calculate_multi_timeframe_rs()
        
        result = {
            'rs_score': current_rs_score,
            'rs_rating': rs_rating,
            'rs_line_current': rs_line_analysis['current_rs_line'],
            'rs_line_new_high': rs_line_analysis['is_new_high'],
            'rs_line_strength': rs_line_analysis['strength'],
            'multi_timeframe': multi_tf,
            'stage': current_stage,
            'substage': current_substage
        }
        
        result['rs_grade'] = self._grade_rs_rating(rs_rating)
        result['rs_category'] = self._categorize_rs_rating(rs_rating)
        
        result['integrated_analysis'] = self._integrate_with_stage(
            rs_rating, 
            rs_line_analysis, 
            multi_tf,
            current_stage, 
            current_substage
        )
        
        return result
    
    def _grade_rs_rating(self, rs_rating: float) -> str:
        if rs_rating >= 95:
            return 'A++'
        elif rs_rating >= 90:
            return 'A+'
        elif rs_rating >= 85:
            return 'A'
        elif rs_rating >= 80:
            return 'B+'
        elif rs_rating >= 70:
            return 'B'
        elif rs_rating >= 60:
            return 'C'
        else:
            return 'D'
    
    def _categorize_rs_rating(self, rs_rating: float) -> str:
        if rs_rating >= 90:
            return 'Top 10% - Market Leader'
        elif rs_rating >= 85:
            return 'Top 15% - Strong Leader'
        elif rs_rating >= 80:
            return 'Top 20% - Above Average'
        elif rs_rating >= 70:
            return 'Top 30% - Average+'
        else:
            return 'Below Average'
    
    def _integrate_with_stage(self, rs_rating: float, rs_line_analysis: Dict,
                             multi_tf: Dict, stage: int, substage: str) -> Dict:
        """
        Generate textual analysis based on RS and Stage
        """
        analysis = {
            'assessment': '',
            'action': '',
            'confidence': '',
            'key_factors': []
        }
        
        # Simplified Logic for report generation
        # (Full logic is in previous iterations, keeping core logic here)

        if stage == 2:
            if rs_rating >= 80:
                analysis['assessment'] = 'Strong Stage 2 Leader'
                analysis['action'] = 'Buy / Hold'
                analysis['confidence'] = 'High'
            else:
                analysis['assessment'] = 'Weak Stage 2'
                analysis['action'] = 'Watch / Sell'
                analysis['confidence'] = 'Medium'
        else:
            analysis['assessment'] = f'Stage {stage}'
            analysis['action'] = 'Monitor'
            analysis['confidence'] = 'Low'

        return analysis
    
    def generate_comprehensive_report(self, current_stage: int, current_substage: str) -> str:
        """
        Generate text report
        """
        analysis = self.analyze_rs_with_stage(current_stage, current_substage)
        return f"RS Rating: {analysis['rs_rating']:.0f}, Assessment: {analysis['integrated_analysis']['assessment']}"

if __name__ == '__main__':
    print("RSCalculator Module Loaded.")
