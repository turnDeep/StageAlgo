"""
Oratnek式スクリーニングシステム
IBD (Investor's Business Daily) 手法に基づく6つのスクリーニングリスト

実装するスクリーナー:
1. Momentum 97 - 1M/3M/6Mすべてで上位3%
2. Explosive EPS Growth - 今四半期EPS予想が100%以上成長
3. Up on Volume - 出来高を伴って上昇している機関投資家注目銘柄
4. Top 2% RS Rating - RS Rating上位2%かつトレンドが完璧
5. 4% Bullish Yesterday - 昨日4%以上上昇
6. Healthy Chart Watch List - 健全なチャート形状を持つ高品質銘柄

yfinanceからFinancialModelingPrep APIに移行しました。
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

from data_fetcher import fetch_stock_data
from fmp_data_fetcher import FMPDataFetcher
from indicators import calculate_all_basic_indicators
from rs_calculator import RSCalculator


class IBDIndicators:
    """
    IBD (Investor's Business Daily) 式指標計算クラス

    計算する指標:
    - RS Rating: 相対的強さレーティング (1-99)
    - A/D Rating: 機関投資家の蓄積/分散評価 (A-E)
    - Comp Rating: 総合レーティング (1-99)
    - EPS Rating: EPS成長レーティング
    """

    @staticmethod
    def calculate_rs_rating(df: pd.DataFrame, benchmark_df: pd.DataFrame) -> float:
        """
        IBD式 RS Rating計算

        加重平均:
        - 40% × 直近3ヶ月（63日）
        - 20% × 直近6ヶ月（126日）
        - 20% × 直近9ヶ月（189日）
        - 20% × 直近12ヶ月（252日）

        Returns:
            RS Rating (0-100スケール)
        """
        if len(df) < 252:
            return 50.0  # デフォルト値

        try:
            rs_calc = RSCalculator(df, benchmark_df)
            rs_score_series = rs_calc.calculate_ibd_rs_score()

            if len(rs_score_series) > 0:
                # 最新のRSスコアを取得し、0-100にスケーリング
                rs_score = rs_score_series.iloc[-1]
                # パーセンタイルランキングに変換（簡易版）
                return min(100, max(0, rs_score + 50))  # 中心を50に調整

            return 50.0
        except Exception as e:
            print(f"RS Rating calculation error: {e}")
            return 50.0

    @staticmethod
    def calculate_ad_rating(df: pd.DataFrame, lookback: int = 13) -> str:
        """
        A/D Rating (Accumulation/Distribution) 計算

        機関投資家の蓄積/分散を評価

        Args:
            df: 株価データ
            lookback: 評価期間（週）

        Returns:
            A/D Rating: 'A' (強い蓄積) ~ 'E' (強い分散)
        """
        if len(df) < lookback:
            return 'C'  # デフォルト: 中立

        try:
            # 直近lookback週のデータ
            recent_data = df.tail(lookback * 5)  # 週 → 営業日変換（概算）

            ad_value = 0

            for i in range(1, len(recent_data)):
                price_change = recent_data['Close'].iloc[i] - recent_data['Close'].iloc[i-1]
                volume = recent_data['Volume'].iloc[i]

                if price_change > 0:
                    # 上昇日: 出来高をプラス
                    ad_value += volume
                elif price_change < 0:
                    # 下落日: 出来高をマイナス
                    ad_value -= volume

            # 平均出来高で正規化
            avg_volume = recent_data['Volume'].mean()
            if avg_volume > 0:
                normalized_ad = ad_value / (avg_volume * len(recent_data))
            else:
                normalized_ad = 0

            # A/D Ratingに変換
            if normalized_ad > 0.5:
                return 'A'  # 非常に強い蓄積
            elif normalized_ad > 0.2:
                return 'B'  # 蓄積
            elif normalized_ad > -0.2:
                return 'C'  # 中立
            elif normalized_ad > -0.5:
                return 'D'  # 分散
            else:
                return 'E'  # 非常に強い分散

        except Exception as e:
            print(f"A/D Rating calculation error: {e}")
            return 'C'

    @staticmethod
    def calculate_comp_rating(rs_rating: float, eps_rating: float = 50.0) -> float:
        """
        Comp Rating (Composite Rating) 計算

        Args:
            rs_rating: RS Rating (0-100)
            eps_rating: EPS Rating (0-100) - デフォルト50

        Returns:
            Composite Rating (0-100)
        """
        # EPSとRSの加重平均（RSをやや重視）
        comp_rating = (rs_rating * 0.6 + eps_rating * 0.4)
        return min(100, max(0, comp_rating))

    @staticmethod
    def calculate_relative_volume(df: pd.DataFrame, days: int = 50) -> float:
        """
        相対出来高を計算

        Args:
            df: 株価データ
            days: 平均出来高の計算期間

        Returns:
            相対出来高 (current_volume / avg_volume)
        """
        if len(df) < days:
            return 1.0

        try:
            current_volume = df['Volume'].iloc[-1]
            avg_volume = df['Volume'].tail(days).mean()

            if avg_volume > 0:
                return current_volume / avg_volume
            return 1.0
        except:
            return 1.0


class OratnekScreener:
    """
    Oratnekダッシュボード式スクリーニングシステム

    6つのスクリーニングリストを提供:
    1. Momentum 97
    2. Explosive EPS Growth
    3. Up on Volume
    4. Top 2% RS Rating
    5. 4% Bullish Yesterday
    6. Healthy Chart Watch List
    """

    def __init__(self, tickers: List[str]):
        """
        Args:
            tickers: スクリーニング対象の銘柄リスト
        """
        self.tickers = tickers
        self.data_cache = {}
        self.benchmark_data = None

        # ベンチマーク（SPY）を取得
        self._load_benchmark()

    def _load_benchmark(self):
        """ベンチマーク（SPY）データを読み込む"""
        try:
            spy_df, _ = fetch_stock_data('SPY', period='2y')
            if spy_df is not None:
                self.benchmark_data = calculate_all_basic_indicators(spy_df)
            else:
                print("Warning: Could not load SPY benchmark data")
        except Exception as e:
            print(f"Error loading benchmark: {e}")

    def _get_stock_data(self, ticker: str) -> Optional[Tuple[pd.DataFrame, Dict]]:
        """
        株価データと基本指標を取得

        Returns:
            (indicators_df, metrics_dict) or None
        """
        if ticker in self.data_cache:
            return self.data_cache[ticker]

        try:
            df, _ = fetch_stock_data(ticker, period='2y')
            if df is None or len(df) < 100:
                return None

            indicators_df = calculate_all_basic_indicators(df)

            if len(indicators_df) < 100:
                return None

            latest = indicators_df.iloc[-1]

            # 基本メトリクス計算
            metrics = {
                'ticker': ticker,
                'price': latest['Close'],
                'volume': latest['Volume'],
                'avg_volume_50d': indicators_df['Volume'].tail(50).mean(),
                'avg_volume_90d': indicators_df['Volume'].tail(90).mean(),
                'sma_10': latest.get('SMA_10', 0),
                'sma_21': latest.get('SMA_21', 0),
                'sma_50': latest.get('SMA_50', 0),
                'sma_150': latest.get('SMA_150', 0),
                'sma_200': latest.get('SMA_200', 0),
            }

            # RS Rating計算
            if self.benchmark_data is not None:
                metrics['rs_rating'] = IBDIndicators.calculate_rs_rating(
                    indicators_df, self.benchmark_data
                )
            else:
                metrics['rs_rating'] = 50.0

            # A/D Rating計算
            metrics['ad_rating'] = IBDIndicators.calculate_ad_rating(indicators_df)

            # Comp Rating計算
            metrics['comp_rating'] = IBDIndicators.calculate_comp_rating(
                metrics['rs_rating']
            )

            # 相対出来高
            metrics['rel_volume'] = IBDIndicators.calculate_relative_volume(indicators_df)

            # パフォーマンス計算
            if len(indicators_df) >= 252:
                metrics['returns_1m'] = ((latest['Close'] / indicators_df['Close'].iloc[-21] - 1) * 100)
                metrics['returns_3m'] = ((latest['Close'] / indicators_df['Close'].iloc[-63] - 1) * 100)
                metrics['returns_6m'] = ((latest['Close'] / indicators_df['Close'].iloc[-126] - 1) * 100)
                metrics['returns_1y'] = ((latest['Close'] / indicators_df['Close'].iloc[-252] - 1) * 100)
            else:
                metrics['returns_1m'] = 0
                metrics['returns_3m'] = 0
                metrics['returns_6m'] = 0
                metrics['returns_1y'] = 0

            # 日次変化率
            if len(indicators_df) >= 2:
                prev_close = indicators_df['Close'].iloc[-2]
                metrics['price_change_pct'] = ((latest['Close'] - prev_close) / prev_close) * 100

                # 寄り高からの変化
                if 'Open' in indicators_df.columns:
                    today_open = latest.get('Open', latest['Close'])
                    metrics['change_from_open_pct'] = ((latest['Close'] - today_open) / today_open) * 100
                else:
                    metrics['change_from_open_pct'] = 0
            else:
                metrics['price_change_pct'] = 0
                metrics['change_from_open_pct'] = 0

            # 出来高変化率
            if metrics['avg_volume_50d'] > 0:
                metrics['vol_change_pct'] = ((metrics['volume'] / metrics['avg_volume_50d'] - 1) * 100)
            else:
                metrics['vol_change_pct'] = 0

            # RS Line新高値チェック（簡易版）
            metrics['rs_line_new_high'] = (metrics['rs_rating'] >= 90)

            result = (indicators_df, metrics)
            self.data_cache[ticker] = result

            return result

        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            return None

    def screen_momentum_97(self) -> pd.DataFrame:
        """
        Momentum 97スクリーニング

        条件:
        - 1M, 3M, 6M すべてで上位3% (≥97パーセンタイル)

        Returns:
            該当銘柄のDataFrame
        """
        results = []

        print("\n[Momentum 97] Screening...")

        for ticker in self.tickers:
            data = self._get_stock_data(ticker)
            if data is None:
                continue

            _, metrics = data

            results.append({
                'ticker': ticker,
                'returns_1m': metrics['returns_1m'],
                'returns_3m': metrics['returns_3m'],
                'returns_6m': metrics['returns_6m'],
            })

        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results)

        # パーセンタイルランキング計算
        df['rank_1m_pct'] = df['returns_1m'].rank(pct=True) * 100
        df['rank_3m_pct'] = df['returns_3m'].rank(pct=True) * 100
        df['rank_6m_pct'] = df['returns_6m'].rank(pct=True) * 100

        # すべての期間で97%以上
        momentum_97 = df[
            (df['rank_1m_pct'] >= 97) &
            (df['rank_3m_pct'] >= 97) &
            (df['rank_6m_pct'] >= 97)
        ].copy()

        # ソート
        momentum_97 = momentum_97.sort_values('returns_1m', ascending=False)

        print(f"  → Found {len(momentum_97)} stocks")

        return momentum_97

    def screen_explosive_eps_growth(self) -> pd.DataFrame:
        """
        Explosive EPS Growth スクリーニング

        条件:
        - RS Rating ≥ 80
        - EPS成長予想 ≥ 100% (※データ制約により、RS Ratingで代用)
        - 50日平均出来高 ≥ 100,000
        - 価格 ≥ 50日移動平均

        Returns:
            該当銘柄のDataFrame
        """
        results = []

        print("\n[Explosive EPS Growth] Screening...")

        for ticker in self.tickers:
            data = self._get_stock_data(ticker)
            if data is None:
                continue

            _, metrics = data

            # スクリーニング条件
            if (metrics['rs_rating'] >= 80 and
                metrics['avg_volume_50d'] >= 100_000 and
                metrics['price'] >= metrics['sma_50']):

                results.append({
                    'ticker': ticker,
                    'price': metrics['price'],
                    'rs_rating': metrics['rs_rating'],
                    'avg_volume_50d': metrics['avg_volume_50d'],
                    'price_vs_sma50_pct': ((metrics['price'] / metrics['sma_50'] - 1) * 100) if metrics['sma_50'] > 0 else 0,
                })

        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results)
        df = df.sort_values('rs_rating', ascending=False)

        print(f"  → Found {len(df)} stocks")

        return df

    def screen_up_on_volume(self) -> pd.DataFrame:
        """
        Up on Volume スクリーニング

        条件:
        - 当日上昇 (≥ 0%)
        - 出来高が50日平均の120%以上
        - 価格 ≥ $10
        - 50日平均出来高 ≥ 100,000
        - RS Rating ≥ 80
        - A/D Rating: A, B, or C

        Returns:
            該当銘柄のDataFrame
        """
        results = []

        print("\n[Up on Volume] Screening...")

        for ticker in self.tickers:
            data = self._get_stock_data(ticker)
            if data is None:
                continue

            _, metrics = data

            # スクリーニング条件
            if (metrics['price_change_pct'] >= 0 and
                metrics['vol_change_pct'] >= 20 and  # 120%以上
                metrics['price'] >= 10 and
                metrics['avg_volume_50d'] >= 100_000 and
                metrics['rs_rating'] >= 80 and
                metrics['ad_rating'] in ['A', 'B', 'C']):

                results.append({
                    'ticker': ticker,
                    'price': metrics['price'],
                    'price_change_pct': metrics['price_change_pct'],
                    'vol_change_pct': metrics['vol_change_pct'],
                    'rs_rating': metrics['rs_rating'],
                    'ad_rating': metrics['ad_rating'],
                    'avg_volume_50d': metrics['avg_volume_50d'],
                })

        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results)
        df = df.sort_values('vol_change_pct', ascending=False)

        print(f"  → Found {len(df)} stocks")

        return df

    def screen_top_2_percent_rs(self) -> pd.DataFrame:
        """
        Top 2% RS Rating スクリーニング

        条件:
        - RS Rating ≥ 98 (上位2%)
        - MA順序: 10日 > 21日 > 50日
        - 50日平均出来高 ≥ 100,000
        - 当日出来高 ≥ 100,000

        Returns:
            該当銘柄のDataFrame
        """
        results = []

        print("\n[Top 2% RS Rating] Screening...")

        for ticker in self.tickers:
            data = self._get_stock_data(ticker)
            if data is None:
                continue

            _, metrics = data

            # スクリーニング条件
            if (metrics['rs_rating'] >= 98 and
                metrics['sma_10'] > metrics['sma_21'] and
                metrics['sma_21'] > metrics['sma_50'] and
                metrics['avg_volume_50d'] >= 100_000 and
                metrics['volume'] >= 100_000):

                results.append({
                    'ticker': ticker,
                    'price': metrics['price'],
                    'rs_rating': metrics['rs_rating'],
                    'sma_10': metrics['sma_10'],
                    'sma_21': metrics['sma_21'],
                    'sma_50': metrics['sma_50'],
                    'avg_volume_50d': metrics['avg_volume_50d'],
                })

        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results)
        df = df.sort_values('rs_rating', ascending=False)

        print(f"  → Found {len(df)} stocks")

        return df

    def screen_4_percent_bullish_yesterday(self) -> pd.DataFrame:
        """
        4% Bullish Yesterday スクリーニング

        条件:
        - 昨日4%以上上昇
        - 価格 ≥ $1
        - 相対出来高 > 1.0
        - 寄り高から更に上昇
        - 90日平均出来高 > 100,000

        Returns:
            該当銘柄のDataFrame
        """
        results = []

        print("\n[4% Bullish Yesterday] Screening...")

        for ticker in self.tickers:
            data = self._get_stock_data(ticker)
            if data is None:
                continue

            indicators_df, metrics = data

            # 昨日のデータを取得
            if len(indicators_df) < 3:
                continue

            yesterday = indicators_df.iloc[-2]
            day_before = indicators_df.iloc[-3]

            yesterday_change = ((yesterday['Close'] - day_before['Close']) / day_before['Close']) * 100

            # スクリーニング条件
            if (yesterday_change > 4.0 and
                metrics['price'] >= 1.0 and
                metrics['rel_volume'] > 1.0 and
                metrics['change_from_open_pct'] > 0 and
                metrics['avg_volume_90d'] > 100_000):

                results.append({
                    'ticker': ticker,
                    'price': metrics['price'],
                    'yesterday_change_pct': yesterday_change,
                    'rel_volume': metrics['rel_volume'],
                    'change_from_open_pct': metrics['change_from_open_pct'],
                    'avg_volume_90d': metrics['avg_volume_90d'],
                })

        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results)
        df = df.sort_values('yesterday_change_pct', ascending=False)

        print(f"  → Found {len(df)} stocks")

        return df

    def screen_healthy_chart_watchlist(self) -> pd.DataFrame:
        """
        Healthy Chart Watch List スクリーニング

        条件:
        - 短期MA順序: 10日 > 21日 > 50日
        - 長期MA順序: 50日 > 150日 > 200日 (Stage 2確認)
        - RS Line新高値
        - RS Rating ≥ 90 (上位10%)
        - A/D Rating: A or B
        - Comp Rating ≥ 80
        - 50日平均出来高 ≥ 100,000

        Returns:
            該当銘柄のDataFrame
        """
        results = []

        print("\n[Healthy Chart Watch List] Screening...")

        for ticker in self.tickers:
            data = self._get_stock_data(ticker)
            if data is None:
                continue

            _, metrics = data

            # スクリーニング条件
            if (metrics['sma_10'] > metrics['sma_21'] and
                metrics['sma_21'] > metrics['sma_50'] and
                metrics['sma_50'] > metrics['sma_150'] and
                metrics['sma_150'] > metrics['sma_200'] and
                metrics['rs_line_new_high'] and
                metrics['rs_rating'] >= 90 and
                metrics['ad_rating'] in ['A', 'B'] and
                metrics['comp_rating'] >= 80 and
                metrics['avg_volume_50d'] >= 100_000):

                results.append({
                    'ticker': ticker,
                    'price': metrics['price'],
                    'rs_rating': metrics['rs_rating'],
                    'ad_rating': metrics['ad_rating'],
                    'comp_rating': metrics['comp_rating'],
                    'sma_50': metrics['sma_50'],
                    'sma_150': metrics['sma_150'],
                    'sma_200': metrics['sma_200'],
                    'avg_volume_50d': metrics['avg_volume_50d'],
                })

        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results)
        df = df.sort_values('comp_rating', ascending=False)

        print(f"  → Found {len(df)} stocks")

        return df

    def run_all_screens(self) -> Dict[str, pd.DataFrame]:
        """
        全スクリーニングを実行

        Returns:
            各スクリーニング結果の辞書
        """
        print("\n" + "="*80)
        print("ORATNEK SCREENER - Running All Screens")
        print("="*80)

        results = {
            'momentum_97': self.screen_momentum_97(),
            'explosive_eps': self.screen_explosive_eps_growth(),
            'up_on_volume': self.screen_up_on_volume(),
            'top_2_rs': self.screen_top_2_percent_rs(),
            'bullish_4pct': self.screen_4_percent_bullish_yesterday(),
            'healthy_chart': self.screen_healthy_chart_watchlist(),
        }

        print("\n" + "="*80)
        print("SCREENING SUMMARY")
        print("="*80)
        for name, df in results.items():
            print(f"{name:25s}: {len(df):3d} stocks")
        print("="*80)

        return results


def get_default_tickers() -> List[str]:
    """
    デフォルトのスクリーニング対象銘柄リストを取得

    Returns:
        銘柄リスト
    """
    # S&P 100主要銘柄（サンプル）
    return [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B',
        'UNH', 'XOM', 'JNJ', 'JPM', 'V', 'PG', 'MA', 'HD', 'CVX', 'MRK',
        'ABBV', 'PEP', 'KO', 'AVGO', 'COST', 'MCD', 'TMO', 'WMT', 'CSCO',
        'ABT', 'DHR', 'ACN', 'VZ', 'CMCSA', 'ADBE', 'NKE', 'TXN', 'NEE',
        'PM', 'UNP', 'CRM', 'RTX', 'LOW', 'HON', 'ORCL', 'QCOM', 'UPS',
        'AMD', 'INTC', 'IBM', 'NFLX', 'BA', 'GE', 'CAT', 'SBUX', 'AMGN',
        'NOW', 'SPGI', 'GS', 'AXP', 'BLK', 'MDT', 'ISRG', 'DE', 'TGT',
        'CI', 'GILD', 'CVS', 'SYK', 'MO', 'MMM', 'ZTS', 'DUK', 'PLD',
        'SO', 'ITW', 'CB', 'BSX', 'EQIX', 'APD', 'SHW', 'CL', 'CME',
        'USB', 'PNC', 'SCHW', 'EOG', 'NOC', 'MMC', 'AON', 'TJX', 'ICE',
    ]


if __name__ == '__main__':
    """
    スタンドアロンテスト実行
    """
    tickers = get_default_tickers()

    screener = OratnekScreener(tickers)
    results = screener.run_all_screens()

    # 結果をCSVに保存
    for name, df in results.items():
        if not df.empty:
            filename = f"screener_{name}.csv"
            df.to_csv(filename, index=False)
            print(f"\nSaved {name} results to {filename}")
