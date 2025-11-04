# market_dashboard.py
"""
Market Dashboard Generator
マーケットダッシュボードを再現
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from curl_cffi.requests import Session
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# StageAlgoの既存モジュールをインポート
from data_fetcher import fetch_stock_data
from indicators import calculate_all_basic_indicators
from rs_calculator import RSCalculator
from stage_detector import StageDetector


class MarketDashboard:
    """
    マーケットダッシュボードジェネレーター

    主要機能:
    - Market Exposure (市場エクスポージャー)
    - Market Performance Overview
    - VIX Analysis
    - Broad Market Overview
    - Sector Analysis
    - Power Law Indicators
    - RS Rating Lists
    """

    def __init__(self):
        self.session = Session(impersonate="chrome110")
        self.current_date = datetime.now()

        # 主要指数のティッカー
        self.major_indices = {
            'SPY': 'S&P 500',
            'QQQ': 'Nasdaq 100',
            'IWM': 'Russell 2000',
            'DIA': 'Dow Jones',
        }

        # セクターETF
        self.sectors = {
            'XLK': 'Technology',
            'XLF': 'Financials',
            'XLV': 'Healthcare',
            'XLE': 'Energy',
            'XLI': 'Industrials',
            'XLY': 'Consumer Discretionary',
            'XLP': 'Consumer Staples',
            'XLB': 'Materials',
            'XLU': 'Utilities',
            'XLRE': 'Real Estate',
            'XLC': 'Communication Services',
        }

        # VIX
        self.vix_ticker = '^VIX'

        # データキャッシュ
        self.data_cache = {}

    def fetch_ticker_data(self, ticker: str, period: str = '2y', interval: str = '1d') -> pd.DataFrame:
        """
        ティッカーデータを取得（キャッシュ付き）
        """
        cache_key = f"{ticker}_{period}_{interval}"

        if cache_key in self.data_cache:
            return self.data_cache[cache_key]

        try:
            data = yf.download(
                ticker,
                period=period,
                interval=interval,
                session=self.session,
                progress=False
            )

            if not data.empty:
                self.data_cache[cache_key] = data
                return data
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")

        return pd.DataFrame()

    def calculate_market_exposure(self) -> Dict:
        """
        Market Exposure (市場エクスポージャー) を計算

        ルール:
        - 100%: Bullish (強気)
        - 60-100%: Positive (ポジティブ)
        - 20-60%: Neutral (中立)
        - -20-20%: Negative (ネガティブ)
        - -60-(-20)%: Bearish (弱気)
        - -60%以下: Extreme Bearish (超弱気)

        計算方法:
        - SPY, QQQ, IWMのステージを判定
        - Stage 2 = +30%, Stage 1 = +10%, Stage 3 = -10%, Stage 4 = -30%
        - VIXレベルで調整
        - Market Breadthで調整
        """
        exposure_score = 0
        stage_weights = {}

        # 主要指数のステージ判定
        for ticker in ['SPY', 'QQQ', 'IWM']:
            try:
                df, benchmark_df = fetch_stock_data(ticker, period='2y')
                if df is None or len(df) < 252:
                    continue

                indicators_df = calculate_all_basic_indicators(df)

                # ベンチマークがNoneの場合はSPYを使用
                if benchmark_df is None:
                    benchmark_df, _ = fetch_stock_data('SPY', period='2y')
                    if benchmark_df is not None:
                        benchmark_df = calculate_all_basic_indicators(benchmark_df)
                else:
                    benchmark_df = calculate_all_basic_indicators(benchmark_df)

                if benchmark_df is None:
                    continue

                detector = StageDetector(indicators_df, benchmark_df)
                stage = detector.determine_stage()

                stage_weights[ticker] = stage

                # ステージに応じたスコア
                if stage == 2:
                    exposure_score += 30
                elif stage == 1:
                    exposure_score += 10
                elif stage == 3:
                    exposure_score -= 10
                elif stage == 4:
                    exposure_score -= 30
            except Exception as e:
                print(f"Error processing {ticker}: {e}")
                continue

        # VIXで調整
        vix_level = None
        try:
            vix_data = self.fetch_ticker_data(self.vix_ticker, period='1mo', interval='1d')
            if not vix_data.empty:
                vix_level = vix_data['Close'].iloc[-1]

                if vix_level < 15:
                    exposure_score += 10  # 低VIX = 安定
                elif vix_level > 30:
                    exposure_score -= 20  # 高VIX = 恐怖
        except Exception as e:
            print(f"Error fetching VIX: {e}")

        # Market Breadthで調整（簡易版：SPYの上昇株/下落株比率）
        try:
            spy_components = self.calculate_market_breadth('SPY')
            if spy_components['advance_decline_ratio'] > 1.5:
                exposure_score += 10
            elif spy_components['advance_decline_ratio'] < 0.67:
                exposure_score -= 10
        except Exception as e:
            print(f"Error calculating market breadth: {e}")

        # スコアを-60〜100に正規化
        exposure_score = max(-60, min(100, exposure_score))

        # レベル判定
        if exposure_score >= 80:
            level = 'Bullish'
        elif exposure_score >= 60:
            level = 'Positive'
        elif exposure_score >= 20:
            level = 'Neutral'
        elif exposure_score >= -20:
            level = 'Negative'
        elif exposure_score >= -60:
            level = 'Bearish'
        else:
            level = 'Extreme Bearish'

        return {
            'score': exposure_score,
            'level': level,
            'stage_weights': stage_weights,
            'vix_level': vix_level
        }

    def calculate_market_breadth(self, index_ticker: str = 'SPY') -> Dict:
        """
        Market Breadth (市場幅指標) を計算

        Returns:
            - advance_decline_ratio: 上昇株/下落株比率
            - new_highs_lows: 新高値/新安値比率
            - stocks_above_ma: 移動平均線上の株式割合
        """
        # 簡易版：主要指数のデータから推定
        # 実際にはすべての構成銘柄を分析する必要がある

        # ここではプレースホルダーとして簡易的な値を返す
        return {
            'advance_decline_ratio': 1.2,  # 仮の値
            'new_highs_lows': 2.5,  # 仮の値
            'stocks_above_50ma_pct': 65.0,  # 仮の値
            'stocks_above_200ma_pct': 58.0,  # 仮の値
        }

    def calculate_market_performance(self) -> pd.DataFrame:
        """
        Market Performance Overview を計算

        各指数の:
        - % YTD (年初来)
        - % 1W (1週間)
        - % 1M (1ヶ月)
        - % 1Y (1年)
        - % From 52W High (52週高値からの距離)
        """
        results = []

        for ticker, name in self.major_indices.items():
            try:
                df = self.fetch_ticker_data(ticker, period='2y', interval='1d')

                if df.empty:
                    continue

                current_price = df['Close'].iloc[-1]

                # YTD
                ytd_start = df.loc[df.index >= f"{self.current_date.year}-01-01"]
                if len(ytd_start) > 0:
                    ytd_price = ytd_start['Close'].iloc[0]
                    ytd_pct = ((current_price - ytd_price) / ytd_price) * 100
                else:
                    ytd_pct = 0

                # 1週間
                one_week_ago = df.iloc[-5] if len(df) >= 5 else df.iloc[0]
                week_pct = ((current_price - one_week_ago['Close']) / one_week_ago['Close']) * 100

                # 1ヶ月
                one_month_ago = df.iloc[-21] if len(df) >= 21 else df.iloc[0]
                month_pct = ((current_price - one_month_ago['Close']) / one_month_ago['Close']) * 100

                # 1年
                one_year_ago = df.iloc[-252] if len(df) >= 252 else df.iloc[0]
                year_pct = ((current_price - one_year_ago['Close']) / one_year_ago['Close']) * 100

                # 52週高値
                high_52w = df['High'].tail(252).max()
                from_high_pct = ((current_price - high_52w) / high_52w) * 100

                results.append({
                    'Index': name,
                    'Ticker': ticker,
                    'YTD %': ytd_pct,
                    '1W %': week_pct,
                    '1M %': month_pct,
                    '1Y %': year_pct,
                    'From 52W High %': from_high_pct,
                    'Current Price': current_price
                })
            except Exception as e:
                print(f"Error processing {ticker}: {e}")
                continue

        return pd.DataFrame(results)

    def get_vix_analysis(self) -> Dict:
        """
        VIX分析

        VIXレベルの解釈:
        - 0-12: 低ボラティリティ (安定)
        - 13-19: 通常
        - 20-30: やや高い (注意)
        - 30+: 高い (恐怖)
        - 40+: 極度の恐怖
        """
        try:
            vix_data = self.fetch_ticker_data(self.vix_ticker, period='6mo', interval='1d')

            if vix_data.empty:
                return {}

            current_vix = vix_data['Close'].iloc[-1]

            # VIXレベルの解釈
            if current_vix < 12:
                interpretation = 'Very Low - Market Complacency'
            elif current_vix < 20:
                interpretation = 'Low - Stable Market'
            elif current_vix < 30:
                interpretation = 'Moderate - Elevated Uncertainty'
            elif current_vix < 40:
                interpretation = 'High - Significant Fear'
            else:
                interpretation = 'Extreme - Market Panic'

            # 52週高値/安値
            high_52w = vix_data['High'].tail(252).max()
            low_52w = vix_data['Low'].tail(252).min()

            return {
                'current': current_vix,
                'interpretation': interpretation,
                '52w_high': high_52w,
                '52w_low': low_52w,
                'from_high_pct': ((current_vix - high_52w) / high_52w) * 100,
                'from_low_pct': ((current_vix - low_52w) / low_52w) * 100
            }
        except Exception as e:
            print(f"Error analyzing VIX: {e}")
            return {}

    def calculate_power_law_indicators(self, tickers: List[str]) -> Dict:
        """
        Power Law Indicators を計算

        - % of stocks with 5-day above 20MA
        - % of stocks with 20MA above 50MA
        - % of stocks with 50MA above 200MA
        """
        results = {
            '5d_above_20ma': 0,
            '20ma_above_50ma': 0,
            '50ma_above_200ma': 0,
            'total': 0
        }

        for ticker in tickers:
            try:
                df, _ = fetch_stock_data(ticker, period='2y')
                if df is None or len(df) < 252:
                    continue

                indicators_df = calculate_all_basic_indicators(df)
                if len(indicators_df) < 252:
                    continue

                latest = indicators_df.iloc[-1]
                last_5 = indicators_df.tail(5)

                results['total'] += 1

                # 5日間50MA以上
                if all(last_5['Close'] > last_5['SMA_50']):
                    results['5d_above_20ma'] += 1

                # 50MA > 150MA
                if latest['SMA_50'] > latest['SMA_150']:
                    results['20ma_above_50ma'] += 1

                # 150MA > 200MA
                if latest['SMA_150'] > latest['SMA_200']:
                    results['50ma_above_200ma'] += 1
            except Exception as e:
                print(f"Error processing {ticker}: {e}")
                continue

        # パーセンテージに変換
        if results['total'] > 0:
            total = results['total']
            results['5d_above_20ma_pct'] = (results['5d_above_20ma'] / total) * 100
            results['20ma_above_50ma_pct'] = (results['20ma_above_50ma'] / total) * 100
            results['50ma_above_200ma_pct'] = (results['50ma_above_200ma'] / total) * 100
        else:
            results['5d_above_20ma_pct'] = 0
            results['20ma_above_50ma_pct'] = 0
            results['50ma_above_200ma_pct'] = 0

        return results

    def calculate_sector_performance(self) -> pd.DataFrame:
        """
        セクターパフォーマンスを計算
        """
        results = []

        # ベンチマーク (SPY)
        try:
            spy_df, _ = fetch_stock_data('SPY', period='2y')
            if spy_df is None:
                return pd.DataFrame()

            spy_indicators = calculate_all_basic_indicators(spy_df)
        except Exception as e:
            print(f"Error fetching SPY benchmark: {e}")
            return pd.DataFrame()

        for ticker, name in self.sectors.items():
            try:
                df, _ = fetch_stock_data(ticker, period='2y')

                if df is None or len(df) < 252:
                    continue

                indicators_df = calculate_all_basic_indicators(df)

                if len(indicators_df) < 252:
                    continue

                latest = indicators_df.iloc[-1]
                current_price = latest['Close']

                # 1日のパフォーマンス
                prev_close = indicators_df['Close'].iloc[-2]
                day_pct = ((current_price - prev_close) / prev_close) * 100

                # RS Rating計算
                rs_calc = RSCalculator(indicators_df, spy_indicators)
                rs_score_series = rs_calc.calculate_ibd_rs_score()

                # 最新のRSスコアを取得
                if len(rs_score_series) > 0:
                    rs_score = rs_score_series.iloc[-1]

                    # パーセンタイルレーティングを計算（0-100のスケール）
                    # 簡易版: rsスコアを0-100にマッピング
                    rs_rating = min(100, max(0, rs_score))
                else:
                    rs_rating = 50.0

                # Relative Strength (簡易版)
                spy_current = spy_indicators['Close'].iloc[-1]
                spy_prev = spy_indicators['Close'].iloc[-252]
                ticker_prev = indicators_df['Close'].iloc[-252]

                spy_perf = ((spy_current - spy_prev) / spy_prev) * 100
                ticker_perf = ((current_price - ticker_prev) / ticker_prev) * 100
                relative_strength = ticker_perf - spy_perf

                results.append({
                    'Sector': name,
                    'Ticker': ticker,
                    'Price': current_price,
                    '1D %': day_pct,
                    'Relative Strength': relative_strength,
                    'RS Rating': rs_rating
                })
            except Exception as e:
                print(f"Error processing sector {ticker}: {e}")
                continue

        return pd.DataFrame(results)

    def generate_dashboard(self, output_file: str = 'market_dashboard_output.txt'):
        """
        ダッシュボードを生成してファイルに出力
        """
        print("=" * 80)
        print("MARKET DASHBOARD")
        print(f"Generated: {self.current_date.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

        # 1. Market Exposure
        print("\n### MARKET EXPOSURE ###")
        exposure = self.calculate_market_exposure()
        print(f"Score: {exposure['score']}%")
        print(f"Level: {exposure['level']}")
        print(f"VIX: {exposure.get('vix_level', 'N/A')}")
        print(f"Stage Weights: {exposure.get('stage_weights', {})}")

        # 2. Market Performance Overview
        print("\n### MARKET PERFORMANCE OVERVIEW ###")
        performance = self.calculate_market_performance()
        if not performance.empty:
            print(performance.to_string(index=False))
        else:
            print("No performance data available")

        # 3. VIX Analysis
        print("\n### VIX ANALYSIS ###")
        vix = self.get_vix_analysis()
        if vix:
            print(f"Current VIX: {vix['current']:.2f}")
            print(f"Interpretation: {vix['interpretation']}")
            print(f"52W High: {vix['52w_high']:.2f}")
            print(f"52W Low: {vix['52w_low']:.2f}")
        else:
            print("VIX data not available")

        # 4. Sector Performance
        print("\n### SECTOR PERFORMANCE ###")
        sectors = self.calculate_sector_performance()
        if not sectors.empty:
            print(sectors.to_string(index=False))
        else:
            print("No sector data available")

        # 5. Power Law Indicators (サンプル銘柄で計算)
        print("\n### POWER LAW INDICATORS ###")
        sample_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'NFLX']
        power_law = self.calculate_power_law_indicators(sample_tickers)
        print(f"5 Days Above 50MA: {power_law.get('5d_above_20ma_pct', 0):.1f}%")
        print(f"50MA Above 150MA: {power_law.get('20ma_above_50ma_pct', 0):.1f}%")
        print(f"150MA Above 200MA: {power_law.get('50ma_above_200ma_pct', 0):.1f}%")
        print(f"Total stocks analyzed: {power_law.get('total', 0)}")

        print("\n" + "=" * 80)
        print("Dashboard generation complete!")
        print("=" * 80)

        return exposure, performance, vix, sectors, power_law


def main():
    """
    メイン実行関数
    """
    dashboard = MarketDashboard()
    dashboard.generate_dashboard()


if __name__ == '__main__':
    main()
