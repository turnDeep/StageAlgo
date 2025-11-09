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

        # Market
        self.market_tickers = {
            'SPY': 'S&P 500',
            'QQQ': 'NASDAQ 100',
            'MAGS': 'Magnificent Seven',
            'RSP': 'Eql Wght S&P 500',
            'QQEW': 'Eql Wght NASDAQ 100',
            'IWM': 'Russell 2000'
        }

        # Sectors
        self.sectors_tickers = {
            'EPOL': 'Poland', 'EWG': 'Germany', 'GLD': 'Gold', 'KWEB': 'China', 'IEV': 'Europe',
            'ITA': 'Aerospace/Defense', 'CIBR': 'Cybersecurity', 'IBIT': 'Bitcoin', 'BLOK': 'Blockchain',
            'IAI': 'Broker', 'NLR': 'Uranium/Nuclear', 'XLF': 'Finance', 'XLU': 'Utilities',
            'TAN': 'Solar', 'UFO': 'Space', 'XLP': 'Consumer Staples', 'FFTY': 'IBD 50',
            'INDA': 'India', 'ARKW': 'ARKW', 'XLK': 'Technology', 'XLE': 'Energy', 'IPO': 'IPO',
            'SOXX': 'Semiconductor', 'MDY': 'MidCap 400', 'SCHD': 'Dividend', 'DIA': 'Dow Jones',
            'ITB': 'Home Construction', 'USO': 'Oil', 'IBB': 'Biotechnology'
        }

        # Macro
        self.macro_tickers = {
            'NYICDX': 'U.S. Dollar',
            '^VIX': 'VIX',
            'TLT': 'Bond 20+ Year'
        }

        self.vix_ticker = '^VIX'

        # データキャッシュ
        self.data_cache = {}

        # スクリーニング対象銘柄
        self.screening_tickers = self._load_all_tickers()

    def _load_all_tickers(self) -> List[str]:
        """stock.csvからすべてのティッカーを読み込む"""
        try:
            df = pd.read_csv('stock.csv')
            return df['Ticker'].dropna().unique().tolist()
        except FileNotFoundError:
            print("Warning: stock.csv not found. Screener will be empty.")
            return []

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
        all_tickers = list(self.market_tickers.keys()) + list(self.sectors_tickers.keys())

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
        # 各要因を二値（positive/negative）として判定
        if total_tickers > 0:
            # 閾値：過半数（50%以上）でpositive
            threshold = 0.5

            # 各要因がpositiveかどうかを判定（0 or 1）
            factor_results = {
                'ytd_positive': 1 if (factors['ytd_positive'] / total_tickers) >= threshold else 0,
                '1w_positive': 1 if (factors['1w_positive'] / total_tickers) >= threshold else 0,
                '1m_positive': 1 if (factors['1m_positive'] / total_tickers) >= threshold else 0,
                '1y_positive': 1 if (factors['1y_positive'] / total_tickers) >= threshold else 0,
                'above_90pct_52w': 1 if (factors['above_90pct_52w'] / total_tickers) >= threshold else 0,
                'vix_positive': 1 if vix_positive else 0,
                'above_10ma': 1 if (factors['above_10ma'] / total_tickers) >= threshold else 0,
                'above_20ma': 1 if (factors['above_20ma'] / total_tickers) >= threshold else 0,
                'above_50ma': 1 if (factors['above_50ma'] / total_tickers) >= threshold else 0,
                'above_200ma': 1 if (factors['above_200ma'] / total_tickers) >= threshold else 0,
                'ma_alignment': 1 if (factors['ma_alignment'] / total_tickers) >= threshold else 0,
                'ma_uptrend': 1 if (factors['ma_uptrend'] / total_tickers) >= threshold else 0,
            }

            # ポジティブな要因の数を合計（12要因中何個がpositive）
            positive_count = sum(factor_results.values())

            # ポジティブ率（0-100%）
            positive_ratio = (positive_count / 12) * 100
        else:
            positive_ratio = 0
            positive_count = 0
            factor_results = {}

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
            'positive_count': int(positive_count) if total_tickers > 0 else 0,
            'total_factors': 12,
            'factors': factors,
            'factor_results': factor_results if total_tickers > 0 else {},
            'vix_level': vix_level,
            'vix_positive': vix_positive,
            'total_tickers': total_tickers,
            'ticker_details': ticker_details,
            'factor_breakdown': {
                'performance': {
                    'ytd_positive': {
                        'count': f"{factors['ytd_positive']}/{total_tickers}",
                        'ratio': f"{(factors['ytd_positive'] / total_tickers * 100):.1f}%",
                        'is_positive': bool(factor_results.get('ytd_positive', 0)) if total_tickers > 0 else False
                    },
                    '1w_positive': {
                        'count': f"{factors['1w_positive']}/{total_tickers}",
                        'ratio': f"{(factors['1w_positive'] / total_tickers * 100):.1f}%",
                        'is_positive': bool(factor_results.get('1w_positive', 0)) if total_tickers > 0 else False
                    },
                    '1m_positive': {
                        'count': f"{factors['1m_positive']}/{total_tickers}",
                        'ratio': f"{(factors['1m_positive'] / total_tickers * 100):.1f}%",
                        'is_positive': bool(factor_results.get('1m_positive', 0)) if total_tickers > 0 else False
                    },
                    '1y_positive': {
                        'count': f"{factors['1y_positive']}/{total_tickers}",
                        'ratio': f"{(factors['1y_positive'] / total_tickers * 100):.1f}%",
                        'is_positive': bool(factor_results.get('1y_positive', 0)) if total_tickers > 0 else False
                    },
                },
                '52w_high': {
                    'above_90pct': {
                        'count': f"{factors['above_90pct_52w']}/{total_tickers}",
                        'ratio': f"{(factors['above_90pct_52w'] / total_tickers * 100):.1f}%",
                        'is_positive': bool(factor_results.get('above_90pct_52w', 0)) if total_tickers > 0 else False
                    },
                },
                'vix': {
                    'below_20': vix_positive,
                    'level': vix_level,
                    'is_positive': bool(factor_results.get('vix_positive', 0)) if total_tickers > 0 else False
                },
                'moving_averages': {
                    'above_10ma': {
                        'count': f"{factors['above_10ma']}/{total_tickers}",
                        'ratio': f"{(factors['above_10ma'] / total_tickers * 100):.1f}%",
                        'is_positive': bool(factor_results.get('above_10ma', 0)) if total_tickers > 0 else False
                    },
                    'above_20ma': {
                        'count': f"{factors['above_20ma']}/{total_tickers}",
                        'ratio': f"{(factors['above_20ma'] / total_tickers * 100):.1f}%",
                        'is_positive': bool(factor_results.get('above_20ma', 0)) if total_tickers > 0 else False
                    },
                    'above_50ma': {
                        'count': f"{factors['above_50ma']}/{total_tickers}",
                        'ratio': f"{(factors['above_50ma'] / total_tickers * 100):.1f}%",
                        'is_positive': bool(factor_results.get('above_50ma', 0)) if total_tickers > 0 else False
                    },
                    'above_200ma': {
                        'count': f"{factors['above_200ma']}/{total_tickers}",
                        'ratio': f"{(factors['above_200ma'] / total_tickers * 100):.1f}%",
                        'is_positive': bool(factor_results.get('above_200ma', 0)) if total_tickers > 0 else False
                    },
                    'ma_alignment': {
                        'count': f"{factors['ma_alignment']}/{total_tickers}",
                        'ratio': f"{(factors['ma_alignment'] / total_tickers * 100):.1f}%",
                        'is_positive': bool(factor_results.get('ma_alignment', 0)) if total_tickers > 0 else False
                    },
                    'ma_uptrend': {
                        'count': f"{factors['ma_uptrend']}/{total_tickers}",
                        'ratio': f"{(factors['ma_uptrend'] / total_tickers * 100):.1f}%",
                        'is_positive': bool(factor_results.get('ma_uptrend', 0)) if total_tickers > 0 else False
                    },
                }
            }
        }

    def update_market_exposure_history(self, current_score: float):
        """Market Exposureの履歴を更新する"""
        history_file = 'market_exposure_history.csv'
        today = self.current_date.strftime('%Y-%m-%d')

        try:
            if pd.io.common.file_exists(history_file):
                history_df = pd.read_csv(history_file, index_col='date')
                history_df.loc[today] = current_score
            else:
                history_df = pd.DataFrame({'score': [current_score]}, index=[today])
                history_df.index.name = 'date'

            history_df.to_csv(history_file)
            print(f"Market exposure history updated in {history_file}")

        except Exception as e:
            print(f"Error updating market exposure history: {e}")

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

    def calculate_ticker_performance(self, tickers: Dict[str, str]) -> pd.DataFrame:
        """
        指定されたティッカーリストのパフォーマンス指標をすべて計算する
        """
        results = []

        # RS RatingのためにSPYデータを事前に取得
        spy_df = self.fetch_ticker_data('SPY', period='2y', interval='1d')
        if spy_df.empty:
            print("Error: Could not fetch SPY data for RS Rating calculation.")
            return pd.DataFrame()
        spy_indicators = calculate_all_basic_indicators(spy_df.copy())

        for ticker, name in tickers.items():
            try:
                df = self.fetch_ticker_data(ticker, period='2y', interval='1d')
                if df.empty or len(df) < 252:
                    print(f"Skipping {ticker} due to insufficient data.")
                    continue

                # インジケーター計算
                indicators_df = calculate_all_basic_indicators(df.copy())
                # 10MA, 20MAを追加
                indicators_df['SMA_10'] = indicators_df['Close'].rolling(window=10).mean()
                indicators_df['SMA_20'] = indicators_df['Close'].rolling(window=20).mean()

                latest = indicators_df.iloc[-1]
                current_price = latest['Close']

                # --- 基本情報 ---
                price = current_price
                day_pct = ((current_price - indicators_df['Close'].iloc[-2]) / indicators_df['Close'].iloc[-2]) * 100 if len(indicators_df) > 1 else 0

                # --- RS Rating ---
                rs_calc = RSCalculator(indicators_df, spy_indicators)
                rs_score_series = rs_calc.calculate_ibd_rs_score()
                current_rs_score = rs_score_series.iloc[-1] if not rs_score_series.empty else 0
                rs_rating = rs_calc.calculate_percentile_rating(current_rs_score)

                # --- Relative Strength (vs SPY, 過去20日分) ---
                rs_history = []
                if len(indicators_df) >= 20 and len(spy_indicators) >= 20:
                    for i in range(-20, 0):
                        try:
                            ticker_price = indicators_df['Close'].iloc[i]
                            ticker_price_base = indicators_df['Close'].iloc[-21]  # 20日前を基準
                            spy_price = spy_indicators['Close'].iloc[i]
                            spy_price_base = spy_indicators['Close'].iloc[-21]

                            if ticker_price_base > 0 and spy_price_base > 0:
                                ticker_return = (ticker_price / ticker_price_base - 1) * 100
                                spy_return = (spy_price / spy_price_base - 1) * 100
                                relative_strength = ticker_return - spy_return
                                rs_history.append(relative_strength)
                            else:
                                rs_history.append(0)
                        except:
                            rs_history.append(0)
                else:
                    rs_history = [0] * 20

                # 最新のRelative Strength
                spy_perf_1y = ((spy_indicators['Close'].iloc[-1] - spy_indicators['Close'].iloc[-252]) / spy_indicators['Close'].iloc[-252]) * 100 if len(spy_indicators) >= 252 else 0
                ticker_perf_1y = ((current_price - indicators_df['Close'].iloc[-252]) / indicators_df['Close'].iloc[-252]) * 100 if len(indicators_df) >= 252 else 0
                relative_strength = ticker_perf_1y - spy_perf_1y

                # --- Performance ---
                ytd_start_price = indicators_df.loc[indicators_df.index >= f"{self.current_date.year}-01-01"]['Close'].iloc[0] if not indicators_df.loc[indicators_df.index >= f"{self.current_date.year}-01-01"].empty else np.nan
                ytd_pct = ((current_price - ytd_start_price) / ytd_start_price) * 100 if not pd.isna(ytd_start_price) else 0

                week_ago_price = indicators_df['Close'].iloc[-6] if len(indicators_df) >= 6 else np.nan
                week_pct = ((current_price - week_ago_price) / week_ago_price) * 100 if not pd.isna(week_ago_price) else 0

                month_ago_price = indicators_df['Close'].iloc[-22] if len(indicators_df) >= 22 else np.nan
                month_pct = ((current_price - month_ago_price) / month_ago_price) * 100 if not pd.isna(month_ago_price) else 0

                year_ago_price = indicators_df['Close'].iloc[-252] if len(indicators_df) >= 252 else np.nan
                year_pct = ((current_price - year_ago_price) / year_ago_price) * 100 if not pd.isna(year_ago_price) else 0

                # --- Highs ---
                high_52w = indicators_df['High'].tail(252).max() if len(indicators_df) >= 252 else indicators_df['High'].max()
                from_high_pct = ((current_price - high_52w) / high_52w) * 100 if high_52w > 0 else 0

                # --- Trend Indicators (MAs) ---
                sma_10 = latest.get('SMA_10', np.nan)
                sma_20 = latest.get('SMA_20', np.nan)
                sma_50 = latest.get('SMA_50', np.nan)
                sma_200 = latest.get('SMA_200', np.nan)

                trends = {
                    'above_10ma': current_price > sma_10 if not pd.isna(sma_10) else False,
                    'above_20ma': current_price > sma_20 if not pd.isna(sma_20) else False,
                    'above_50ma': current_price > sma_50 if not pd.isna(sma_50) else False,
                    'above_200ma': current_price > sma_200 if not pd.isna(sma_200) else False,
                    '20ma_above_50ma': sma_20 > sma_50 if not pd.isna(sma_20) and not pd.isna(sma_50) else False,
                    '50ma_above_200ma': sma_50 > sma_200 if not pd.isna(sma_50) and not pd.isna(sma_200) else False,
                }

                results.append({
                    'ticker': ticker,
                    'index': name,
                    'price': price,
                    '% 1D': day_pct,
                    'Relative Strength': relative_strength,
                    'RS History': rs_history,  # 20日分のRS履歴
                    'RS STS %': rs_rating,
                    '% YTD': ytd_pct,
                    '% 1W': week_pct,
                    '% 1M': month_pct,
                    '% 1Y': year_pct,
                    '% From 52W High': from_high_pct,
                    '10MA': trends['above_10ma'],
                    '20MA': trends['above_20ma'],
                    '50MA': trends['above_50ma'],
                    '200MA': trends['above_200ma'],
                    '20>50MA': trends['20ma_above_50ma'],
                    '50>200MA': trends['50ma_above_200ma'],
                })
            except Exception as e:
                print(f"Error processing ticker {ticker}: {e}")
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

    def calculate_factors_vs_sp500(self) -> Dict[str, float]:
        """
        Factors vs SP500 (Yesterday) performance差分を計算
        Growth, Value, Large-Cap, Small-Cap の相対パフォーマンス
        """
        try:
            spy_df = self.fetch_ticker_data('SPY', period='5d')
            if len(spy_df) < 2:
                return {}

            spy_chg = ((spy_df['Close'].iloc[-1] - spy_df['Close'].iloc[-2]) / spy_df['Close'].iloc[-2]) * 100

            factors = {
                'Growth': 'IVW',
                'Value': 'IVE',
                'Large-Cap': 'SPY',
                'Small-Cap': 'IWM'
            }

            results = {}
            for name, ticker in factors.items():
                df = self.fetch_ticker_data(ticker, period='5d')
                if len(df) >= 2:
                    chg = ((df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100
                    results[name] = chg - spy_chg
            return results
        except Exception as e:
            print(f"Error calculating factors vs SP500: {e}")
            return {}

    def get_bond_yields(self) -> Dict[str, float]:
        """
        債券利回り情報を取得
        TLTとIEFから推定、または固定値を返す
        """
        try:
            # TLT (20+ Year Treasury) から長期金利を推定
            tlt_df = self.fetch_ticker_data('TLT', period='5d')
            # IEF (7-10 Year Treasury) から中期金利を推定
            ief_df = self.fetch_ticker_data('IEF', period='5d')

            results = {}

            # 簡易的な利回り推定 (実際のAPIが必要な場合は別途実装)
            if not tlt_df.empty:
                # TLTの価格から簡易推定 (逆相関)
                tlt_price = tlt_df['Close'].iloc[-1]
                # 簡易計算: 基準価格100からの乖離で推定
                estimated_10y = 4.5 - ((tlt_price - 85) / 85) * 2
                results['US 10Y'] = estimated_10y
            else:
                results['US 10Y'] = 4.25  # デフォルト値

            if not ief_df.empty:
                ief_price = ief_df['Close'].iloc[-1]
                estimated_2y = 4.7 - ((ief_price - 95) / 95) * 2
                results['US 02Y'] = estimated_2y
            else:
                results['US 02Y'] = 4.40  # デフォルト値

            return results
        except Exception as e:
            print(f"Error getting bond yields: {e}")
            return {'US 10Y': 4.25, 'US 02Y': 4.40}

    def calculate_power_trend(self) -> Dict:
        """
        Power Trend indicators calculation
        市場のモメンタムと勢いを示す指標
        """
        try:
            # SPYのモメンタムを計算
            spy_df = self.fetch_ticker_data('SPY', period='6mo')
            if spy_df.empty or len(spy_df) < 50:
                return {}

            # RSI計算
            delta = spy_df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1]

            # MACD計算
            ema_12 = spy_df['Close'].ewm(span=12, adjust=False).mean()
            ema_26 = spy_df['Close'].ewm(span=26, adjust=False).mean()
            macd = ema_12 - ema_26
            signal = macd.ewm(span=9, adjust=False).mean()
            macd_histogram = macd - signal
            current_macd_hist = macd_histogram.iloc[-1]

            return {
                'rsi': float(current_rsi),
                'macd_histogram': float(current_macd_hist),
                'trend': 'Bullish' if current_macd_hist > 0 and current_rsi > 50 else 'Bearish'
            }
        except Exception as e:
            print(f"Error calculating power trend: {e}")
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
        self.update_market_exposure_history(exposure['score'])
        print(f"Score: {exposure['score']:.2f}%")
        print(f"Level: {exposure['level']}")
        print(f"Positive Factors: {exposure.get('positive_count', 0)}/{exposure.get('total_factors', 12)}")
        print(f"Total Tickers Evaluated: {exposure.get('total_tickers', 0)}")
        print(f"\n要因内訳 (✓=Positive, ✗=Negative):")
        print(f"  パフォーマンス評価:")
        for key, val in exposure['factor_breakdown']['performance'].items():
            status = "✓" if val['is_positive'] else "✗"
            print(f"    {status} {key}: {val['count']} ({val['ratio']})")
        print(f"  52週高値:")
        for key, val in exposure['factor_breakdown']['52w_high'].items():
            status = "✓" if val['is_positive'] else "✗"
            print(f"    {status} {key}: {val['count']} ({val['ratio']})")
        print(f"  VIX:")
        vix_info = exposure['factor_breakdown']['vix']
        vix_status = "✓" if vix_info['is_positive'] else "✗"
        print(f"    {vix_status} below_20: {vix_info['below_20']}")
        print(f"    - current_level: {vix_info['level']:.2f}" if vix_info['level'] else "    - current_level: N/A")
        print(f"  移動平均線:")
        for key, val in exposure['factor_breakdown']['moving_averages'].items():
            status = "✓" if val['is_positive'] else "✗"
            print(f"    {status} {key}: {val['count']} ({val['ratio']})")

        # 2. Market, Sectors, Macro Performance
        print("\n### Calculating Market Performance ###")
        market_performance = self.calculate_ticker_performance(self.market_tickers)
        print(market_performance.to_string(index=False))

        print("\n### Calculating Sectors Performance ###")
        sectors_performance = self.calculate_ticker_performance(self.sectors_tickers)
        print(sectors_performance.to_string(index=False))

        print("\n### Calculating Macro Performance ###")
        macro_performance = self.calculate_ticker_performance(self.macro_tickers)
        print(macro_performance.to_string(index=False))

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

        return (exposure, market_performance, sectors_performance,
                macro_performance, screener_results)


def main():
    """
    メイン実行関数
    """
    dashboard = MarketDashboard()
    dashboard.generate_dashboard()


if __name__ == '__main__':
    main()
