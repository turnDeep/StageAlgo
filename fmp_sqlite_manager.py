"""
FMP Data SQLite Manager

FinancialModelingPrep APIから取得したデータをSQLiteで管理します。
主にMarket Ticker、Sectors Ticker、Macro Tickerなどの非個別銘柄データを対象とします。

機能:
- 価格データの保存・取得
- Quote データの保存・取得
- Profile データの保存・取得
- 自動キャッシュ管理
"""

import sqlite3
import pandas as pd
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path


class FMPSQLiteManager:
    """FMP Data SQLite Manager"""

    def __init__(self, db_path: str = 'data/fmp_data.db'):
        """
        Initialize SQLite Manager

        Args:
            db_path: SQLite database file path
        """
        self.db_path = db_path
        # データディレクトリを作成
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._create_tables()

    def _get_connection(self):
        """Get database connection"""
        return sqlite3.connect(self.db_path)

    def _create_tables(self):
        """Create database tables if they don't exist"""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Historical price data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS historical_prices (
                ticker TEXT NOT NULL,
                date TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                adj_close REAL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (ticker, date)
            )
        ''')

        # Quote data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS quotes (
                ticker TEXT PRIMARY KEY,
                price REAL,
                change_percentage REAL,
                change REAL,
                day_low REAL,
                day_high REAL,
                year_high REAL,
                year_low REAL,
                market_cap INTEGER,
                volume INTEGER,
                avg_volume INTEGER,
                open REAL,
                previous_close REAL,
                eps REAL,
                pe REAL,
                data TEXT,
                updated_at TEXT NOT NULL
            )
        ''')

        # Company profile table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS profiles (
                ticker TEXT PRIMARY KEY,
                company_name TEXT,
                sector TEXT,
                industry TEXT,
                market_cap INTEGER,
                description TEXT,
                ceo TEXT,
                website TEXT,
                data TEXT,
                updated_at TEXT NOT NULL
            )
        ''')

        # Create indexes for faster queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_hist_ticker ON historical_prices(ticker)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_hist_date ON historical_prices(date)')

        conn.commit()
        conn.close()

    def save_historical_prices(self, ticker: str, df: pd.DataFrame):
        """
        Save historical price data to SQLite

        Args:
            ticker: Stock ticker symbol
            df: DataFrame with OHLCV data (index: date)
        """
        if df.empty:
            return

        conn = self._get_connection()

        # DataFrameをリセットしてdateを列にする
        df_copy = df.copy()
        df_copy.reset_index(inplace=True)

        # 列名を標準化
        df_copy.rename(columns={
            'date': 'date',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            'Adj Close': 'adj_close'
        }, inplace=True)

        # updated_atを追加
        df_copy['updated_at'] = datetime.now().isoformat()
        df_copy['ticker'] = ticker

        # dateをstring形式に変換
        df_copy['date'] = pd.to_datetime(df_copy['date']).dt.strftime('%Y-%m-%d')

        # 必要な列のみ選択
        columns = ['ticker', 'date', 'open', 'high', 'low', 'close', 'volume', 'updated_at']
        if 'adj_close' in df_copy.columns:
            columns.insert(-1, 'adj_close')

        df_to_save = df_copy[columns]

        # データベースに保存（REPLACE: 既存データを上書き）
        df_to_save.to_sql('historical_prices', conn, if_exists='append', index=False, method='multi')

        # 重複を削除（最新のupdated_atを保持）
        cursor = conn.cursor()
        cursor.execute('''
            DELETE FROM historical_prices
            WHERE rowid NOT IN (
                SELECT MAX(rowid)
                FROM historical_prices
                GROUP BY ticker, date
            )
        ''')

        conn.commit()
        conn.close()

    def get_historical_prices(
        self,
        ticker: str,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        max_age_days: int = 1
    ) -> Optional[pd.DataFrame]:
        """
        Get historical price data from SQLite

        Args:
            ticker: Stock ticker symbol
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            max_age_days: Maximum age of cached data in days (default: 1)

        Returns:
            DataFrame with OHLCV data or None if not found/expired
        """
        conn = self._get_connection()

        # 基本クエリ
        query = 'SELECT * FROM historical_prices WHERE ticker = ?'
        params = [ticker]

        # 日付フィルター
        if from_date:
            query += ' AND date >= ?'
            params.append(from_date)
        if to_date:
            query += ' AND date <= ?'
            params.append(to_date)

        query += ' ORDER BY date ASC'

        df = pd.read_sql_query(query, conn, params=params)
        conn.close()

        if df.empty:
            return None

        # キャッシュの有効期限チェック
        latest_update = pd.to_datetime(df['updated_at'].iloc[-1])
        if datetime.now() - latest_update > timedelta(days=max_age_days):
            return None  # 期限切れ

        # DataFrameを整形
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

        # 列名を大文字に変換（yfinance互換）
        df.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume',
            'adj_close': 'Adj Close'
        }, inplace=True)

        # 必要な列のみ保持
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if 'Adj Close' in df.columns:
            required_cols.append('Adj Close')

        return df[required_cols]

    def save_quote(self, ticker: str, quote_data: Dict):
        """
        Save quote data to SQLite

        Args:
            ticker: Stock ticker symbol
            quote_data: Quote data dictionary
        """
        if not quote_data:
            return

        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            INSERT OR REPLACE INTO quotes (
                ticker, price, change_percentage, change, day_low, day_high,
                year_high, year_low, market_cap, volume, avg_volume, open,
                previous_close, eps, pe, data, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            ticker,
            quote_data.get('price'),
            quote_data.get('changesPercentage'),
            quote_data.get('change'),
            quote_data.get('dayLow'),
            quote_data.get('dayHigh'),
            quote_data.get('yearHigh'),
            quote_data.get('yearLow'),
            quote_data.get('marketCap'),
            quote_data.get('volume'),
            quote_data.get('avgVolume'),
            quote_data.get('open'),
            quote_data.get('previousClose'),
            quote_data.get('eps'),
            quote_data.get('pe'),
            json.dumps(quote_data),
            datetime.now().isoformat()
        ))

        conn.commit()
        conn.close()

    def get_quote(self, ticker: str, max_age_minutes: int = 15) -> Optional[Dict]:
        """
        Get quote data from SQLite

        Args:
            ticker: Stock ticker symbol
            max_age_minutes: Maximum age of cached data in minutes (default: 15)

        Returns:
            Quote data dictionary or None if not found/expired
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('SELECT * FROM quotes WHERE ticker = ?', (ticker,))
        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        # キャッシュの有効期限チェック
        updated_at = datetime.fromisoformat(row[16])  # updated_at column
        if datetime.now() - updated_at > timedelta(minutes=max_age_minutes):
            return None  # 期限切れ

        # JSON文字列からデータを復元
        return json.loads(row[15])  # data column

    def save_profile(self, ticker: str, profile_data: Dict):
        """
        Save company profile data to SQLite

        Args:
            ticker: Stock ticker symbol
            profile_data: Profile data dictionary
        """
        if not profile_data:
            return

        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            INSERT OR REPLACE INTO profiles (
                ticker, company_name, sector, industry, market_cap,
                description, ceo, website, data, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            ticker,
            profile_data.get('companyName'),
            profile_data.get('sector'),
            profile_data.get('industry'),
            profile_data.get('mktCap'),
            profile_data.get('description'),
            profile_data.get('ceo'),
            profile_data.get('website'),
            json.dumps(profile_data),
            datetime.now().isoformat()
        ))

        conn.commit()
        conn.close()

    def get_profile(self, ticker: str, max_age_days: int = 30) -> Optional[Dict]:
        """
        Get company profile data from SQLite

        Args:
            ticker: Stock ticker symbol
            max_age_days: Maximum age of cached data in days (default: 30)

        Returns:
            Profile data dictionary or None if not found/expired
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('SELECT * FROM profiles WHERE ticker = ?', (ticker,))
        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        # キャッシュの有効期限チェック
        updated_at = datetime.fromisoformat(row[9])  # updated_at column
        if datetime.now() - updated_at > timedelta(days=max_age_days):
            return None  # 期限切れ

        # JSON文字列からデータを復元
        return json.loads(row[8])  # data column

    def clear_old_data(self, days: int = 30):
        """
        Clear old data from database

        Args:
            days: Delete data older than this many days
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()

        cursor.execute('DELETE FROM historical_prices WHERE updated_at < ?', (cutoff_date,))
        cursor.execute('DELETE FROM quotes WHERE updated_at < ?', (cutoff_date,))
        cursor.execute('DELETE FROM profiles WHERE updated_at < ?', (cutoff_date,))

        conn.commit()
        conn.close()

    def get_cached_tickers(self) -> List[str]:
        """
        Get list of tickers with cached data

        Returns:
            List of ticker symbols
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('SELECT DISTINCT ticker FROM historical_prices')
        tickers = [row[0] for row in cursor.fetchall()]

        conn.close()
        return tickers


if __name__ == '__main__':
    # テストコード
    print("Testing FMP SQLite Manager...")

    manager = FMPSQLiteManager()
    print(f"Database created at: {manager.db_path}")

    # テストデータ
    test_ticker = 'SPY'

    # 1. 価格データのテスト
    print(f"\n1. Testing historical price data for {test_ticker}...")
    test_df = pd.DataFrame({
        'Open': [100.0, 101.0, 102.0],
        'High': [101.0, 102.0, 103.0],
        'Low': [99.0, 100.0, 101.0],
        'Close': [100.5, 101.5, 102.5],
        'Volume': [1000000, 1100000, 1200000]
    }, index=pd.date_range('2025-01-01', periods=3))
    test_df.index.name = 'date'

    manager.save_historical_prices(test_ticker, test_df)
    print("✓ Price data saved")

    retrieved_df = manager.get_historical_prices(test_ticker)
    if retrieved_df is not None:
        print("✓ Price data retrieved")
        print(retrieved_df)
    else:
        print("✗ Failed to retrieve price data")

    # 2. Quoteデータのテスト
    print(f"\n2. Testing quote data for {test_ticker}...")
    test_quote = {
        'price': 102.5,
        'changesPercentage': 1.5,
        'change': 1.5,
        'dayLow': 101.0,
        'dayHigh': 103.0,
        'marketCap': 5000000000
    }

    manager.save_quote(test_ticker, test_quote)
    print("✓ Quote data saved")

    retrieved_quote = manager.get_quote(test_ticker)
    if retrieved_quote:
        print("✓ Quote data retrieved")
        print(retrieved_quote)
    else:
        print("✗ Failed to retrieve quote data")

    # 3. Profileデータのテスト
    print(f"\n3. Testing profile data for {test_ticker}...")
    test_profile = {
        'companyName': 'SPDR S&P 500 ETF Trust',
        'sector': 'ETF',
        'industry': 'Index ETF',
        'mktCap': 5000000000
    }

    manager.save_profile(test_ticker, test_profile)
    print("✓ Profile data saved")

    retrieved_profile = manager.get_profile(test_ticker)
    if retrieved_profile:
        print("✓ Profile data retrieved")
        print(retrieved_profile)
    else:
        print("✗ Failed to retrieve profile data")

    # 4. キャッシュされたティッカーのリスト
    print("\n4. Testing cached tickers...")
    cached_tickers = manager.get_cached_tickers()
    print(f"Cached tickers: {cached_tickers}")

    print("\nTest completed!")
