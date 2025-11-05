# market_breadth_analyzer.py
"""
Market Breadth Analysis Module
市場幅指標の詳細分析
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List
from curl_cffi.requests import Session


class MarketBreadthAnalyzer:
    """
    市場幅指標の詳細分析

    主要指標:
    - Advance/Decline Line
    - Advance/Decline Ratio
    - New Highs/New Lows
    - McClellan Oscillator
    - Percentage of Stocks Above MA
    """

    def __init__(self):
        self.session = Session(impersonate="chrome110")

    def calculate_advance_decline_line(self, advances: List[int], declines: List[int]) -> pd.Series:
        """
        Advance/Decline Lineを計算

        AD Line = 累積(上昇株 - 下落株)
        """
        net_advances = [adv - dec for adv, dec in zip(advances, declines)]
        ad_line = pd.Series(net_advances).cumsum()
        return ad_line

    def calculate_advance_decline_ratio(self, advances: int, declines: int) -> float:
        """
        Advance/Decline Ratioを計算

        AD Ratio = 上昇株 / 下落株
        """
        if declines == 0:
            return 999.0 if advances > 0 else 1.0
        return advances / declines

    def calculate_mcclellan_oscillator(self, net_advances: pd.Series) -> pd.Series:
        """
        McClellan Oscillatorを計算

        McClellan = EMA(19) - EMA(39) of Net Advances
        """
        ema_19 = net_advances.ewm(span=19, adjust=False).mean()
        ema_39 = net_advances.ewm(span=39, adjust=False).mean()
        return ema_19 - ema_39

    def analyze_breadth_for_index(self, index_ticker: str, period: str = '1y') -> Dict:
        """
        指定された指数の市場幅を分析

        ※ 注: 実際の上昇株/下落株データが必要
        ここでは簡易的な推定値を使用
        """
        # プレースホルダー
        # 実際にはWilshire 5000など全市場データが必要
        return {
            'current_ad_ratio': 1.3,
            'ad_line_trend': 'rising',
            'mcclellan': 45.2,
            'interpretation': 'Moderately Bullish'
        }


if __name__ == '__main__':
    analyzer = MarketBreadthAnalyzer()
    result = analyzer.analyze_breadth_for_index('SPY')
    print("Market Breadth Analysis:", result)
