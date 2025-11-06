"""
FinancialModelingPrep API Data Fetcher

yfinanceの代替として、FinancialModelingPrep APIを使用してデータを取得します。

環境変数 FMP_API_KEY に APIキーを設定する必要があります。
FMP Starter Planの制限:
- 20GB/月の帯域制限
- 一部のエンドポイントでティッカー数制限あり

公式ドキュメント: https://site.financialmodelingprep.com/developer/docs
"""

import os
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import time
from curl_cffi.requests import Session, errors
from fmp_sqlite_manager import FMPSQLiteManager


class FMPDataFetcher:
    """FinancialModelingPrep API Data Fetcher with SQLite caching"""

    BASE_URL = "https://financialmodelingprep.com/api/v3"
    STABLE_URL = "https://financialmodelingprep.com/stable"

    # 非個別銘柄リスト（SQLiteでキャッシュするティッカー）
    NON_STOCK_TICKERS = {
        # Market
        'SPY', 'QQQ', 'MAGS', 'RSP', 'QQEW', 'IWM',
        # Sectors
        'EPOL', 'EWG', 'GLD', 'KWEB', 'IEV', 'ITA', 'CIBR', 'IBIT', 'BLOK',
        'IAI', 'NLR', 'XLF', 'XLU', 'TAN', 'UFO', 'XLP', 'FFTY', 'INDA',
        'ARKW', 'XLK', 'XLE', 'IPO', 'SOXX', 'MDY', 'SCHD', 'DIA', 'ITB',
        'USO', 'IBB',
        # Macro
        'NYICDX', '^VIX', 'TLT'
    }

    def __init__(self, api_key: Optional[str] = None, rate_limit: Optional[int] = None,
                 use_sqlite: bool = True, db_path: str = 'data/fmp_data.db'):
        """
        Initialize FMP Data Fetcher

        Args:
            api_key: FMP API Key (環境変数 FMP_API_KEY から自動取得可能)
            rate_limit: API rate limit per minute (環境変数 FMP_RATE_LIMIT から自動取得可能, デフォルト: 750)
            use_sqlite: Use SQLite for caching (default: True)
            db_path: SQLite database path
        """
        self.api_key = api_key or os.getenv('FMP_API_KEY')
        if not self.api_key:
            raise ValueError(
                "FMP API Key is required. Set FMP_API_KEY environment variable "
                "or pass api_key parameter."
            )

        # レート制限の設定（環境変数から取得、デフォルトは750 req/min - Premium Plan）
        self.rate_limit = rate_limit or int(os.getenv('FMP_RATE_LIMIT', '750'))

        # データキャッシュ（メモリ）
        self.cache = {}
        self.session = Session(impersonate="chrome110")
        self.request_timestamps = []

        # SQLite Manager
        self.use_sqlite = use_sqlite
        self.sqlite_manager = FMPSQLiteManager(db_path) if use_sqlite else None

    def _enforce_rate_limit(self):
        """Enforce the configured API rate limit per minute."""
        current_time = time.time()
        # Remove timestamps older than 60 seconds
        self.request_timestamps = [t for t in self.request_timestamps if current_time - t < 60]
        if len(self.request_timestamps) >= self.rate_limit:
            # Sleep until the oldest request is older than 60 seconds
            time.sleep(60 - (current_time - self.request_timestamps[0]))
            # Trim the list again after sleeping
            current_time = time.time()
            self.request_timestamps = [t for t in self.request_timestamps if current_time - t < 60]
        self.request_timestamps.append(current_time)

    def _make_request(self, url: str, params: Optional[Dict] = None) -> Dict:
        """
        Make API request with error handling and rate limiting.

        Args:
            url: API endpoint URL
            params: Query parameters

        Returns:
            JSON response as dict
        """
        self._enforce_rate_limit()
        if params is None:
            params = {}

        params['apikey'] = self.api_key

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except errors.RequestsError as e:
            print(f"API request failed: {e}")
            return {}

    def get_historical_price(
        self,
        ticker: str,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get historical price data with SQLite caching

        Args:
            ticker: Stock ticker symbol
            from_date: Start date (YYYY-MM-DD format)
            to_date: End date (YYYY-MM-DD format)

        Returns:
            DataFrame with OHLCV data
        """
        # SQLiteキャッシュを確認（非個別銘柄のみ）
        if self.use_sqlite and ticker in self.NON_STOCK_TICKERS and self.sqlite_manager:
            cached_df = self.sqlite_manager.get_historical_prices(
                ticker, from_date, to_date, max_age_days=1
            )
            if cached_df is not None:
                return cached_df

        # メモリキャッシュを確認
        cache_key = f"hist_{ticker}_{from_date}_{to_date}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        # APIからデータ取得
        url = f"{self.BASE_URL}/historical-price-full/{ticker}"

        params = {}
        if from_date:
            params['from'] = from_date
        if to_date:
            params['to'] = to_date

        data = self._make_request(url, params)

        if not data or 'historical' not in data:
            return pd.DataFrame()

        df = pd.DataFrame(data['historical'])

        if df.empty:
            return pd.DataFrame()

        # データ整形
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)

        # 列名を大文字に変換（yfinance互換）
        df.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume',
            'adjClose': 'Adj Close'
        }, inplace=True)

        # 必要な列のみ保持
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if 'Adj Close' in df.columns:
            required_cols.append('Adj Close')

        df = df[required_cols]

        # SQLiteに保存（非個別銘柄のみ）
        if self.use_sqlite and ticker in self.NON_STOCK_TICKERS and self.sqlite_manager:
            self.sqlite_manager.save_historical_prices(ticker, df)

        # メモリキャッシュに保存
        self.cache[cache_key] = df
        return df

    def get_quote(self, ticker: str) -> Dict:
        """
        Get real-time quote data with SQLite caching

        Args:
            ticker: Stock ticker symbol

        Returns:
            Quote data as dict
        """
        # SQLiteキャッシュを確認（非個別銘柄のみ）
        if self.use_sqlite and ticker in self.NON_STOCK_TICKERS and self.sqlite_manager:
            cached_quote = self.sqlite_manager.get_quote(ticker, max_age_minutes=15)
            if cached_quote is not None:
                return cached_quote

        # APIからデータ取得
        url = f"{self.BASE_URL}/quote/{ticker}"
        data = self._make_request(url)

        if data and isinstance(data, list) and len(data) > 0:
            quote = data[0]
            # SQLiteに保存（非個別銘柄のみ）
            if self.use_sqlite and ticker in self.NON_STOCK_TICKERS and self.sqlite_manager:
                self.sqlite_manager.save_quote(ticker, quote)
            return quote
        return {}

    def get_profile(self, ticker: str) -> Dict:
        """
        Get company profile with SQLite caching

        Args:
            ticker: Stock ticker symbol

        Returns:
            Company profile data
        """
        # SQLiteキャッシュを確認（非個別銘柄のみ）
        if self.use_sqlite and ticker in self.NON_STOCK_TICKERS and self.sqlite_manager:
            cached_profile = self.sqlite_manager.get_profile(ticker, max_age_days=30)
            if cached_profile is not None:
                return cached_profile

        # メモリキャッシュを確認
        cache_key = f"profile_{ticker}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        # APIからデータ取得
        url = f"{self.BASE_URL}/profile/{ticker}"
        data = self._make_request(url)

        if data and isinstance(data, list) and len(data) > 0:
            profile = data[0]
            # SQLiteに保存（非個別銘柄のみ）
            if self.use_sqlite and ticker in self.NON_STOCK_TICKERS and self.sqlite_manager:
                self.sqlite_manager.save_profile(ticker, profile)
            # メモリキャッシュに保存
            self.cache[cache_key] = profile
            return profile
        return {}

    def get_key_metrics(self, ticker: str, period: str = 'annual', limit: int = 10) -> List[Dict]:
        """
        Get key financial metrics (EPS, P/E, etc.)

        Args:
            ticker: Stock ticker symbol
            period: 'annual' or 'quarter'
            limit: Number of periods to retrieve

        Returns:
            List of key metrics
        """
        url = f"{self.BASE_URL}/key-metrics/{ticker}"
        params = {'period': period, 'limit': limit}
        data = self._make_request(url, params)

        return data if isinstance(data, list) else []

    def get_income_statement(self, ticker: str, period: str = 'quarter', limit: int = 4) -> List[Dict]:
        """
        Get income statement (for EPS data)

        Args:
            ticker: Stock ticker symbol
            period: 'annual' or 'quarter'
            limit: Number of periods to retrieve

        Returns:
            List of income statements
        """
        url = f"{self.BASE_URL}/income-statement/{ticker}"
        params = {'period': period, 'limit': limit}
        data = self._make_request(url, params)

        return data if isinstance(data, list) else []

    def get_earnings_surprises(self, ticker: str) -> List[Dict]:
        """
        Get earnings surprises (actual vs estimated EPS)

        Args:
            ticker: Stock ticker symbol

        Returns:
            List of earnings surprises
        """
        url = f"{self.BASE_URL}/earnings-surprises/{ticker}"
        data = self._make_request(url)

        return data if isinstance(data, list) else []


def fetch_stock_data(
    ticker: str,
    benchmark_ticker: str = "SPY",
    period: str = "5y",
    interval: str = "1d",
    fetch_benchmark: bool = True
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    FMP APIを使用して株価データを取得します（yfinance互換インターフェース）

    Args:
        ticker: 株式ティッカーシンボル
        benchmark_ticker: ベンチマークのティッカー（デフォルト: "SPY"）
        period: データ期間（"1y", "2y", "5y" など）
        interval: データ間隔（現在は "1d" のみサポート）
        fetch_benchmark: ベンチマークデータも同時に取得するかどうか

    Returns:
        tuple[pd.DataFrame | None, pd.DataFrame | None]: (stock_data, benchmark_data)
    """
    try:
        fetcher = FMPDataFetcher()

        # 期間を日数に変換
        period_days = {
            '1mo': 30,
            '3mo': 90,
            '6mo': 180,
            '1y': 365,
            '2y': 730,
            '5y': 1825,
            '10y': 3650,
        }

        days = period_days.get(period, 1825)  # デフォルト5年
        from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        to_date = datetime.now().strftime('%Y-%m-%d')

        # 株価データ取得
        stock_data = fetcher.get_historical_price(ticker, from_date, to_date)

        if stock_data.empty:
            return None, None

        stock_data.dropna(inplace=True)

        # ベンチマークデータ取得
        benchmark_data = None
        if fetch_benchmark:
            if ticker.upper() == benchmark_ticker.upper():
                # 自身がベンチマーク
                benchmark_data = stock_data.copy()
            else:
                benchmark_data = fetcher.get_historical_price(benchmark_ticker, from_date, to_date)
                if benchmark_data is not None and not benchmark_data.empty:
                    benchmark_data.dropna(inplace=True)

        return stock_data, benchmark_data

    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None, None


if __name__ == '__main__':
    # テストコード
    print("Testing FMP Data Fetcher...")

    # APIキーの確認
    api_key = os.getenv('FMP_API_KEY')
    if not api_key:
        print("ERROR: FMP_API_KEY environment variable is not set")
        print("Please set your API key: export FMP_API_KEY='your_api_key_here'")
        exit(1)

    print(f"API Key found: {api_key[:10]}...")

    # テスト用ティッカー
    test_ticker = 'AAPL'
    print(f"\n1. Testing historical price data for {test_ticker}...")
    stock_df, benchmark_df = fetch_stock_data(test_ticker, period='1y')

    if stock_df is not None:
        print(f"\n{test_ticker} Historical Data (last 5 rows):")
        print(stock_df.tail())
        print(f"\nShape: {stock_df.shape}")
    else:
        print(f"Failed to fetch data for {test_ticker}")

    if benchmark_df is not None:
        print(f"\nSPY Benchmark Data (last 5 rows):")
        print(benchmark_df.tail())

    # Quote データのテスト
    print(f"\n2. Testing quote data for {test_ticker}...")
    fetcher = FMPDataFetcher()
    quote = fetcher.get_quote(test_ticker)
    if quote:
        print(f"Current Price: ${quote.get('price', 'N/A')}")
        print(f"Volume: {quote.get('volume', 'N/A'):,}")
        print(f"Market Cap: ${quote.get('marketCap', 0):,}")

    # Profile データのテスト
    print(f"\n3. Testing profile data for {test_ticker}...")
    profile = fetcher.get_profile(test_ticker)
    if profile:
        print(f"Company: {profile.get('companyName', 'N/A')}")
        print(f"Sector: {profile.get('sector', 'N/A')}")
        print(f"Industry: {profile.get('industry', 'N/A')}")
        print(f"Market Cap: ${profile.get('mktCap', 0):,}")

    print("\nTest completed!")
