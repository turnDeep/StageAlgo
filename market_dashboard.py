# market_dashboard.py
"""
Market Dashboard Generator
マーケットダッシュボードを再現

yfinanceからFinancialModelingPrep APIに移行しました。
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# StageAlgoの既存モジュールをインポート
from data_fetcher import fetch_stock_data
from fmp_data_fetcher import FMPDataFetcher
from indicators import calculate_all_basic_indicators
from rs_calculator import RSCalculator
from stage_detector import StageDetector
from oratnek_screeners import OratnekScreener, get_default_tickers


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

    def __init__(self, enable_screeners: bool = True):
        self.fmp_fetcher = FMPDataFetcher()
        self.current_date = datetime.now()
        self.enable_screeners = enable_screeners

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

        # スクリーニング対象銘柄
        self.screening_tickers = get_default_tickers()

    def fetch_ticker_data(self, ticker: str, period: str = '2y', interval: str = '1d') -> pd.DataFrame:
        """
        ティッカーデータを取得（キャッシュ付き）
        FMP APIを使用
        """
        cache_key = f"{ticker}_{period}_{interval}"

        if cache_key in self.data_cache:
            return self.data_cache[cache_key]

        try:
            # 期間を日数に変換
            period_days = {
                '1mo': 30,
                '3mo': 90,
                '6mo': 180,
                '1y': 365,
                '2y': 730,
                '5y': 1825,
            }
            days = period_days.get(period, 730)  # デフォルト2年
            from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            to_date = datetime.now().strftime('%Y-%m-%d')

            data = self.fmp_fetcher.get_historical_price(ticker, from_date, to_date)

            if not data.empty:
                self.data_cache[cache_key] = data
                return data
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")

        return pd.DataFrame()

    def calculate_market_exposure(self) -> Dict:
        """
        Market Exposure (市場エクスポージャー) を計算

        12要因評価方式:
        1. パフォーマンス評価（4要因）:
           - YTD > 0のindex/sector数
           - 1W > 0のindex/sector数
           - 1M > 0のindex/sector数
           - 1Y > 0のindex/sector数
        2. 52週高値からの位置（1要因）:
           - 52週高値の90%以上にあるindex/sector数
        3. VIX状態（1要因）:
           - VIX < 20
        4. 移動平均線の状態（6要因）:
           - 10MA以上のindex/sector数
           - 20MA以上のindex/sector数
           - 50MA以上のindex/sector数
           - 200MA以上のindex/sector数
           - MA順序が正しい数(10>20>50>200)
           - 全MA上昇トレンドの数

        スコアリング:
        - Positive_Ratio = (Positive_Count / 12) * 100
        - 80%以上: Bullish
        - 60-80%: Positive
        - 40-60%: Neutral
        - 20-40%: Negative
        - 0-20%: Bearish
        """
        # 評価対象：主要指数 + セクターETF
        all_tickers = list(self.major_indices.keys()) + list(self.sectors.keys())

        # 各要因のカウンター
        factors = {
            'ytd_positive': 0,
            '1w_positive': 0,
            '1m_positive': 0,
            '1y_positive': 0,
            'above_90pct_52w': 0,
            'above_10ma': 0,
            'above_20ma': 0,
            'above_50ma': 0,
            'above_200ma': 0,
            'ma_alignment': 0,
            'ma_uptrend': 0,
        }

        total_tickers = len(all_tickers)
        ticker_details = {}

        # 各ティッカーを評価
        for ticker in all_tickers:
            try:
                df = self.fetch_ticker_data(ticker, period='2y', interval='1d')
                if df.empty or len(df) < 252:
                    continue

                # 基本指標を計算
                indicators_df = calculate_all_basic_indicators(df)
                if len(indicators_df) < 252:
                    continue

                latest = indicators_df.iloc[-1]
                current_price = latest['Close']

                # 追加のMA計算（10MA, 20MA）
                sma_10 = indicators_df['Close'].rolling(window=10, min_periods=10).mean().iloc[-1]
                sma_20 = indicators_df['Close'].rolling(window=20, min_periods=20).mean().iloc[-1]
                sma_50 = latest['SMA_50']
                sma_200 = latest['SMA_200']

                ticker_factors = {}

                # 1. パフォーマンス評価（4要因）
                # YTD
                ytd_start = indicators_df.loc[indicators_df.index >= f"{self.current_date.year}-01-01"]
                if len(ytd_start) > 0:
                    ytd_price = ytd_start['Close'].iloc[0]
                    ytd_pct = ((current_price - ytd_price) / ytd_price) * 100
                    if ytd_pct > 0:
                        factors['ytd_positive'] += 1
                        ticker_factors['ytd'] = True
                    else:
                        ticker_factors['ytd'] = False
                else:
                    ticker_factors['ytd'] = False

                # 1W
                if len(indicators_df) >= 5:
                    week_ago_price = indicators_df['Close'].iloc[-5]
                    week_pct = ((current_price - week_ago_price) / week_ago_price) * 100
                    if week_pct > 0:
                        factors['1w_positive'] += 1
                        ticker_factors['1w'] = True
                    else:
                        ticker_factors['1w'] = False
                else:
                    ticker_factors['1w'] = False

                # 1M
                if len(indicators_df) >= 21:
                    month_ago_price = indicators_df['Close'].iloc[-21]
                    month_pct = ((current_price - month_ago_price) / month_ago_price) * 100
                    if month_pct > 0:
                        factors['1m_positive'] += 1
                        ticker_factors['1m'] = True
                    else:
                        ticker_factors['1m'] = False
                else:
                    ticker_factors['1m'] = False

                # 1Y
                if len(indicators_df) >= 252:
                    year_ago_price = indicators_df['Close'].iloc[-252]
                    year_pct = ((current_price - year_ago_price) / year_ago_price) * 100
                    if year_pct > 0:
                        factors['1y_positive'] += 1
                        ticker_factors['1y'] = True
                    else:
                        ticker_factors['1y'] = False
                else:
                    ticker_factors['1y'] = False

                # 2. 52週高値からの位置（1要因）
                high_52w = latest['High_52W']
                if not pd.isna(high_52w) and current_price >= high_52w * 0.90:
                    factors['above_90pct_52w'] += 1
                    ticker_factors['above_90pct_52w'] = True
                else:
                    ticker_factors['above_90pct_52w'] = False

                # 4. 移動平均線の状態（6要因）
                # 10MA以上
                if not pd.isna(sma_10) and current_price >= sma_10:
                    factors['above_10ma'] += 1
                    ticker_factors['above_10ma'] = True
                else:
                    ticker_factors['above_10ma'] = False

                # 20MA以上
                if not pd.isna(sma_20) and current_price >= sma_20:
                    factors['above_20ma'] += 1
                    ticker_factors['above_20ma'] = True
                else:
                    ticker_factors['above_20ma'] = False

                # 50MA以上
                if not pd.isna(sma_50) and current_price >= sma_50:
                    factors['above_50ma'] += 1
                    ticker_factors['above_50ma'] = True
                else:
                    ticker_factors['above_50ma'] = False

                # 200MA以上
                if not pd.isna(sma_200) and current_price >= sma_200:
                    factors['above_200ma'] += 1
                    ticker_factors['above_200ma'] = True
                else:
                    ticker_factors['above_200ma'] = False

                # MA順序が正しい（10>20>50>200）
                if (not pd.isna(sma_10) and not pd.isna(sma_20) and
                    not pd.isna(sma_50) and not pd.isna(sma_200) and
                    sma_10 > sma_20 > sma_50 > sma_200):
                    factors['ma_alignment'] += 1
                    ticker_factors['ma_alignment'] = True
                else:
                    ticker_factors['ma_alignment'] = False

                # 全MA上昇トレンド
                # 各MAの傾きを確認（簡易版：最新5日の平均と5日前の平均を比較）
                if len(indicators_df) >= 10:
                    sma_10_series = indicators_df['Close'].rolling(window=10, min_periods=10).mean()
                    sma_20_series = indicators_df['Close'].rolling(window=20, min_periods=20).mean()
                    sma_50_series = indicators_df['SMA_50']
                    sma_200_series = indicators_df['SMA_200']

                    ma_uptrend = True
                    for ma_series in [sma_10_series, sma_20_series, sma_50_series, sma_200_series]:
                        if pd.isna(ma_series.iloc[-1]) or pd.isna(ma_series.iloc[-5]):
                            ma_uptrend = False
                            break
                        if ma_series.iloc[-1] <= ma_series.iloc[-5]:
                            ma_uptrend = False
                            break

                    if ma_uptrend:
                        factors['ma_uptrend'] += 1
                        ticker_factors['ma_uptrend'] = True
                    else:
                        ticker_factors['ma_uptrend'] = False
                else:
                    ticker_factors['ma_uptrend'] = False

                ticker_details[ticker] = ticker_factors

            except Exception as e:
                print(f"Error processing {ticker}: {e}")
                continue

        # 3. VIX状態（1要因）
        vix_level = None
        vix_positive = False
        try:
            vix_data = self.fetch_ticker_data(self.vix_ticker, period='1mo', interval='1d')
            if not vix_data.empty:
                vix_level = vix_data['Close'].iloc[-1].item()
                vix_positive = vix_level < 20
        except Exception as e:
            print(f"Error fetching VIX: {e}")

        # 各要因のポジティブ率を計算（total_tickersで正規化）
        if total_tickers > 0:
            factor_positive_counts = [
                factors['ytd_positive'],
                factors['1w_positive'],
                factors['1m_positive'],
                factors['1y_positive'],
                factors['above_90pct_52w'],
                1 if vix_positive else 0,  # VIXは単一要因
                factors['above_10ma'],
                factors['above_20ma'],
                factors['above_50ma'],
                factors['above_200ma'],
                factors['ma_alignment'],
                factors['ma_uptrend'],
            ]

            # ポジティブな要因の数を合計
            # 最初の5要因とMA6要因は比率で、VIXは絶対値
            # ポジティブ率 = (各要因の達成度の合計) / 12
            # 各要因の達成度 = (ポジティブなティッカー数 / 総ティッカー数) ただしVIXは0or1

            positive_count = 0
            # パフォーマンス要因（4つ）
            positive_count += (factors['ytd_positive'] / total_tickers)
            positive_count += (factors['1w_positive'] / total_tickers)
            positive_count += (factors['1m_positive'] / total_tickers)
            positive_count += (factors['1y_positive'] / total_tickers)
            # 52週高値要因（1つ）
            positive_count += (factors['above_90pct_52w'] / total_tickers)
            # VIX要因（1つ）
            positive_count += (1 if vix_positive else 0)
            # MA要因（6つ）
            positive_count += (factors['above_10ma'] / total_tickers)
            positive_count += (factors['above_20ma'] / total_tickers)
            positive_count += (factors['above_50ma'] / total_tickers)
            positive_count += (factors['above_200ma'] / total_tickers)
            positive_count += (factors['ma_alignment'] / total_tickers)
            positive_count += (factors['ma_uptrend'] / total_tickers)

            # ポジティブ率（0-100%）
            positive_ratio = (positive_count / 12) * 100
        else:
            positive_ratio = 0

        # レベル判定
        if positive_ratio >= 80:
            level = 'Bullish'
        elif positive_ratio >= 60:
            level = 'Positive'
        elif positive_ratio >= 40:
            level = 'Neutral'
        elif positive_ratio >= 20:
            level = 'Negative'
        else:
            level = 'Bearish'

        return {
            'score': positive_ratio,
            'level': level,
            'factors': factors,
            'vix_level': vix_level,
            'vix_positive': vix_positive,
            'total_tickers': total_tickers,
            'ticker_details': ticker_details,
            'factor_breakdown': {
                'performance': {
                    'ytd_positive': f"{factors['ytd_positive']}/{total_tickers}",
                    '1w_positive': f"{factors['1w_positive']}/{total_tickers}",
                    '1m_positive': f"{factors['1m_positive']}/{total_tickers}",
                    '1y_positive': f"{factors['1y_positive']}/{total_tickers}",
                },
                '52w_high': {
                    'above_90pct': f"{factors['above_90pct_52w']}/{total_tickers}",
                },
                'vix': {
                    'below_20': vix_positive,
                    'level': vix_level,
                },
                'moving_averages': {
                    'above_10ma': f"{factors['above_10ma']}/{total_tickers}",
                    'above_20ma': f"{factors['above_20ma']}/{total_tickers}",
                    'above_50ma': f"{factors['above_50ma']}/{total_tickers}",
                    'above_200ma': f"{factors['above_200ma']}/{total_tickers}",
                    'ma_alignment': f"{factors['ma_alignment']}/{total_tickers}",
                    'ma_uptrend': f"{factors['ma_uptrend']}/{total_tickers}",
                }
            }
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

                # 重複列を削除
                if df.columns.has_duplicates:
                    df = df.loc[:, ~df.columns.duplicated()]

                current_price = df['Close'].iloc[-1].item()

                # YTD
                ytd_start = df.loc[df.index >= f"{self.current_date.year}-01-01"]
                if len(ytd_start) > 0:
                    ytd_price = ytd_start['Close'].iloc[0].item()
                    ytd_pct = ((current_price - ytd_price) / ytd_price) * 100
                else:
                    ytd_pct = 0

                # 1週間
                one_week_ago = df.iloc[-5] if len(df) >= 5 else df.iloc[0]
                week_pct = ((current_price - one_week_ago['Close'].item()) / one_week_ago['Close'].item()) * 100

                # 1ヶ月
                one_month_ago = df.iloc[-21] if len(df) >= 21 else df.iloc[0]
                month_pct = ((current_price - one_month_ago['Close'].item()) / one_month_ago['Close'].item()) * 100

                # 1年
                one_year_ago = df.iloc[-252] if len(df) >= 252 else df.iloc[0]
                year_pct = ((current_price - one_year_ago['Close'].item()) / one_year_ago['Close'].item()) * 100

                # 52週高値
                high_52w = df['High'].tail(252).max().item()
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

            # 重複列を削除
            if vix_data.columns.has_duplicates:
                vix_data = vix_data.loc[:, ~vix_data.columns.duplicated()]

            current_vix = vix_data['Close'].iloc[-1].item()

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
            high_52w = vix_data['High'].tail(252).max().item()
            low_52w = vix_data['Low'].tail(252).min().item()

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

    def run_oratnek_screeners(self) -> Dict:
        """
        Oratnek式スクリーニングを実行

        Returns:
            スクリーニング結果の辞書
        """
        if not self.enable_screeners:
            print("\n[INFO] Screeners are disabled. Skipping...")
            return {}

        print("\n### ORATNEK SCREENERS ###")
        print("Running 6 screening lists...")

        try:
            screener = OratnekScreener(self.screening_tickers)
            results = screener.run_all_screens()
            return results
        except Exception as e:
            print(f"Error running screeners: {e}")
            return {}

    def generate_dashboard(self, output_file: str = 'market_dashboard_output.txt'):
        """
        ダッシュボードを生成してファイルに出力
        """
        print("=" * 80)
        print("MARKET DASHBOARD")
        print(f"Generated: {self.current_date.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

        # 1. Market Exposure
        print("\n### MARKET EXPOSURE (12要因評価) ###")
        exposure = self.calculate_market_exposure()
        print(f"Score: {exposure['score']:.2f}%")
        print(f"Level: {exposure['level']}")
        print(f"Total Tickers Evaluated: {exposure.get('total_tickers', 0)}")
        print(f"\n要因内訳:")
        print(f"  パフォーマンス評価:")
        for key, val in exposure['factor_breakdown']['performance'].items():
            print(f"    - {key}: {val}")
        print(f"  52週高値:")
        for key, val in exposure['factor_breakdown']['52w_high'].items():
            print(f"    - {key}: {val}")
        print(f"  VIX:")
        vix_info = exposure['factor_breakdown']['vix']
        print(f"    - below_20: {vix_info['below_20']}")
        print(f"    - current_level: {vix_info['level']:.2f}" if vix_info['level'] else "    - current_level: N/A")
        print(f"  移動平均線:")
        for key, val in exposure['factor_breakdown']['moving_averages'].items():
            print(f"    - {key}: {val}")

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
        power_law = self.calculate_power_law_indicators(self.screening_tickers)
        print(f"5 Days Above 50MA: {power_law.get('5d_above_20ma_pct', 0):.1f}%")
        print(f"50MA Above 150MA: {power_law.get('20ma_above_50ma_pct', 0):.1f}%")
        print(f"150MA Above 200MA: {power_law.get('50ma_above_200ma_pct', 0):.1f}%")
        print(f"Total stocks analyzed: {power_law.get('total', 0)}")

        # 6. Oratnek Screeners（追加）
        screener_results = self.run_oratnek_screeners()

        # スクリーニング結果を表示
        if screener_results:
            print("\n### SCREENING RESULTS ###")
            for name, df in screener_results.items():
                print(f"\n{name.upper().replace('_', ' ')}:")
                if not df.empty:
                    # 上位5銘柄を表示
                    display_df = df.head(5)
                    print(display_df.to_string(index=False))
                    print(f"  ... and {max(0, len(df) - 5)} more stocks")
                else:
                    print("  No stocks found")

        print("\n" + "=" * 80)
        print("Dashboard generation complete!")
        print("=" * 80)

        return exposure, performance, vix, sectors, power_law, screener_results


def main():
    """
    メイン実行関数
    """
    dashboard = MarketDashboard()
    dashboard.generate_dashboard()


if __name__ == '__main__':
    main()
