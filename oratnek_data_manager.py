"""
Oratnek Screener Data Manager

SQLiteを使用してFinancialModelingPrepからのデータを管理します。
HanaView2/hwb_data_manager.pyの設計を参考にしています。
"""

import sqlite3
import os
import json
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import logging
import threading
from typing import Tuple, Dict, Optional, List
from fmp_data_fetcher import FMPDataFetcher

logger = logging.getLogger(__name__)


class CustomJSONEncoder(json.JSONEncoder):
    """カスタムJSONエンコーダー（numpy/pandasオブジェクト対応）"""
    def default(self, obj):
        if isinstance(obj, (datetime, date, pd.Timestamp)):
            return obj.isoformat()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super(CustomJSONEncoder, self).default(obj)


class OratnekDataManager:
    """
    Oratnekスクリーナー用データマネージャー

    機能:
    - SQLiteで株価データをキャッシュ (daily_prices, weekly_prices)
    - FinancialModelingPrepからのデータ取得
    - ファンダメンタルデータの管理 (EPS, etc.)
    - スレッドセーフなデータアクセス
    """

    def __init__(self, base_data_path='data/oratnek'):
        """
        初期化

        Args:
            base_data_path: データベースとファイルの保存先パス
        """
        self.base_dir = Path(base_data_path)
        self.db_path = self.base_dir / 'oratnek_cache.db'
        self.results_dir = self.base_dir / 'results'

        # ディレクトリ作成
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)

        # スレッドロック
        self.db_lock = threading.Lock()

        # FMP Data Fetcher
        self.fmp_fetcher = FMPDataFetcher()

        logger.info(f"OratnekDataManager initialized. DB path: {self.db_path}")
        self._init_database()

    def _init_database(self):
        """データベーススキーマを初期化"""
        logger.info("Initializing database schema...")
        try:
            with self.db_lock:
                with sqlite3.connect(self.db_path, timeout=30) as conn:
                    cursor = conn.cursor()

                    # 日次株価テーブル
                    cursor.execute("""
                    CREATE TABLE IF NOT EXISTS daily_prices (
                        symbol TEXT NOT NULL,
                        date DATE NOT NULL,
                        open REAL NOT NULL,
                        high REAL NOT NULL,
                        low REAL NOT NULL,
                        close REAL NOT NULL,
                        volume INTEGER NOT NULL,
                        adj_close REAL,
                        sma_10 REAL,
                        sma_21 REAL,
                        sma_50 REAL,
                        sma_150 REAL,
                        sma_200 REAL,
                        ema_200 REAL,
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (symbol, date)
                    );
                    """)

                    # 週次株価テーブル
                    cursor.execute("""
                    CREATE TABLE IF NOT EXISTS weekly_prices (
                        symbol TEXT NOT NULL,
                        week_start_date DATE NOT NULL,
                        open REAL NOT NULL,
                        high REAL NOT NULL,
                        low REAL NOT NULL,
                        close REAL NOT NULL,
                        volume INTEGER NOT NULL,
                        sma_200 REAL,
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (symbol, week_start_date)
                    );
                    """)

                    # ファンダメンタルデータテーブル
                    cursor.execute("""
                    CREATE TABLE IF NOT EXISTS fundamental_data (
                        symbol TEXT PRIMARY KEY,
                        market_cap REAL,
                        sector TEXT,
                        industry TEXT,
                        eps_ttm REAL,
                        pe_ratio REAL,
                        revenue_growth_yoy REAL,
                        earnings_growth_yoy REAL,
                        eps_growth_last_qtr REAL,
                        eps_est_cur_qtr_growth REAL,
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                    """)

                    # メタデータテーブル
                    cursor.execute("""
                    CREATE TABLE IF NOT EXISTS data_metadata (
                        symbol TEXT PRIMARY KEY,
                        first_date DATE,
                        last_date DATE,
                        last_updated TIMESTAMP,
                        daily_count INTEGER,
                        weekly_count INTEGER
                    );
                    """)

                    # インデックス作成
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_daily_symbol_date ON daily_prices(symbol, date DESC);")
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_weekly_symbol_date ON weekly_prices(symbol, week_start_date DESC);")
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_metadata_last_date ON data_metadata(last_date);")

                    conn.commit()
                    logger.info("Database schema initialized successfully.")
        except sqlite3.Error as e:
            logger.error(f"Database initialization failed: {e}", exc_info=True)
            raise

    def get_stock_data_with_cache(
        self,
        symbol: str,
        lookback_years: int = 10
    ) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        株価データをキャッシュ付きで取得

        Args:
            symbol: 銘柄シンボル
            lookback_years: 取得する過去データの年数

        Returns:
            (daily_df, weekly_df) または None
        """
        try:
            # メタデータチェック
            needs_update = False
            start_date = None

            with self.db_lock:
                with sqlite3.connect(self.db_path, timeout=30) as conn:
                    metadata = self._get_metadata(symbol, conn)
                    today = datetime.now().date()

                    if not metadata:
                        logger.info(f"'{symbol}': First time fetch. Getting full history.")
                        needs_update = True
                        start_date = today - timedelta(days=365 * lookback_years)
                    elif metadata['last_date'] < today:
                        logger.info(f"'{symbol}': Cache is outdated (last: {metadata['last_date']}). Fetching delta.")
                        needs_update = True
                        start_date = metadata['last_date'] + timedelta(days=1)
                    else:
                        logger.info(f"'{symbol}': Cache is up-to-date.")

            # 新しいデータを取得
            if needs_update:
                df_new_daily, df_new_weekly = self._fetch_from_fmp(symbol, start_date, datetime.now().date())

                # データを保存
                if (df_new_daily is not None and not df_new_daily.empty) or \
                   (df_new_weekly is not None and not df_new_weekly.empty):
                    with self.db_lock:
                        with sqlite3.connect(self.db_path, timeout=30) as conn:
                            df_old_daily = self._load_daily_from_db(symbol, conn, lookback_days=365*lookback_years)
                            df_old_weekly = self._load_weekly_from_db(symbol, conn, lookback_weeks=52*lookback_years)

                            df_full_daily = self._calculate_full_daily_ma(df_old_daily, df_new_daily)
                            df_full_weekly = self._calculate_full_weekly_ma(df_old_weekly, df_new_weekly)

                            self._save_to_db(symbol, conn, df_full_daily, df_full_weekly)
                            self._update_metadata(symbol, conn)
                else:
                    logger.info(f"'{symbol}': No new data returned from FMP.")

            # 最終データをDBから読み込み
            with self.db_lock:
                with sqlite3.connect(self.db_path, timeout=30) as conn:
                    final_df_daily = self._load_daily_from_db(symbol, conn, lookback_days=365 * lookback_years)
                    final_df_weekly = self._load_weekly_from_db(symbol, conn, lookback_weeks=52 * lookback_years)

            if final_df_daily.empty:
                logger.warning(f"'{symbol}': No data available after fetch/load process.")
                return None

            return final_df_daily, final_df_weekly

        except Exception as e:
            logger.error(f"Error in get_stock_data_with_cache for '{symbol}': {e}", exc_info=True)
            return None

    def _get_metadata(self, symbol: str, conn) -> Optional[Dict]:
        """メタデータを取得"""
        query = "SELECT symbol, first_date, last_date, last_updated, daily_count, weekly_count FROM data_metadata WHERE symbol = ?"
        try:
            cursor = conn.cursor()
            row = cursor.execute(query, (symbol,)).fetchone()
            if row:
                row_dict = dict(zip([d[0] for d in cursor.description], row))
                # Handle date parsing - split on space to handle both 'YYYY-MM-DD' and 'YYYY-MM-DD HH:MM:SS' formats
                if row_dict['first_date']:
                    date_str = str(row_dict['first_date']).split()[0]
                    row_dict['first_date'] = datetime.strptime(date_str, '%Y-%m-%d').date()
                else:
                    row_dict['first_date'] = None
                if row_dict['last_date']:
                    date_str = str(row_dict['last_date']).split()[0]
                    row_dict['last_date'] = datetime.strptime(date_str, '%Y-%m-%d').date()
                else:
                    row_dict['last_date'] = None
                return row_dict
            return None
        except Exception as e:
            logger.error(f"Failed to get metadata for {symbol}: {e}", exc_info=True)
            return None

    def _fetch_from_fmp(
        self,
        symbol: str,
        start_date: date,
        end_date: date
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """FMPからデータを取得"""
        logger.info(f"Fetching FMP data for '{symbol}' from {start_date} to {end_date}")
        try:
            from_date_str = start_date.strftime('%Y-%m-%d')
            to_date_str = end_date.strftime('%Y-%m-%d')

            # 日次データ取得
            df_daily = self.fmp_fetcher.get_historical_price(symbol, from_date_str, to_date_str)

            if df_daily is None or df_daily.empty:
                return None, None

            # 週次データを生成（日次データから）
            df_weekly = df_daily.resample('W-FRI').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()

            # 列名を小文字に統一し、スペースをアンダースコアに変換
            df_daily.rename(columns=str.lower, inplace=True)
            df_daily.rename(columns={'adj close': 'adj_close'}, inplace=True)
            df_weekly.rename(columns=str.lower, inplace=True)

            logger.info(f"'{symbol}': Fetched {len(df_daily)} daily and {len(df_weekly)} weekly records.")
            return df_daily, df_weekly

        except Exception as e:
            logger.error(f"FMP fetch error for '{symbol}': {e}", exc_info=True)
            return None, None

    def _calculate_full_daily_ma(self, df_old: pd.DataFrame, df_new: Optional[pd.DataFrame]) -> pd.DataFrame:
        """日次データに移動平均を追加"""
        if df_new is None or df_new.empty:
            return df_old

        df_full = pd.concat([df_old, df_new])
        df_full = df_full[~df_full.index.duplicated(keep='last')].sort_index()

        # 移動平均計算
        df_full['sma_10'] = df_full['close'].rolling(window=10, min_periods=5).mean()
        df_full['sma_21'] = df_full['close'].rolling(window=21, min_periods=10).mean()
        df_full['sma_50'] = df_full['close'].rolling(window=50, min_periods=25).mean()
        df_full['sma_150'] = df_full['close'].rolling(window=150, min_periods=75).mean()
        df_full['sma_200'] = df_full['close'].rolling(window=200, min_periods=100).mean()
        df_full['ema_200'] = df_full['close'].ewm(span=200, min_periods=100, adjust=False).mean()

        return df_full

    def _calculate_full_weekly_ma(self, df_old: pd.DataFrame, df_new: Optional[pd.DataFrame]) -> pd.DataFrame:
        """週次データに移動平均を追加"""
        if df_new is None or df_new.empty:
            return df_old

        df_full = pd.concat([df_old, df_new])
        df_full = df_full[~df_full.index.duplicated(keep='last')].sort_index()

        df_full['sma_200'] = df_full['close'].rolling(window=200, min_periods=100).mean()

        return df_full

    def _save_to_db(self, symbol: str, conn, df_daily: pd.DataFrame, df_weekly: pd.DataFrame):
        """データをデータベースに保存"""
        cursor = conn.cursor()
        try:
            cursor.execute("BEGIN;")

            # 既存データを削除
            cursor.execute("DELETE FROM daily_prices WHERE symbol = ?", (symbol,))
            cursor.execute("DELETE FROM weekly_prices WHERE symbol = ?", (symbol,))

            # 日次データ保存
            if df_daily is not None and not df_daily.empty:
                df_daily = df_daily.dropna(subset=['open', 'high', 'low', 'close'], how='any')
                df_daily = df_daily[df_daily.index.notna()]

                if not df_daily.empty:
                    cols = ['open', 'high', 'low', 'close', 'volume']
                    if 'adj_close' in df_daily.columns:
                        cols.append('adj_close')
                    cols.extend(['sma_10', 'sma_21', 'sma_50', 'sma_150', 'sma_200', 'ema_200'])

                    df_to_save = df_daily[cols].copy()
                    df_to_save['symbol'] = symbol
                    df_to_save.index.name = 'date'
                    df_to_save.reset_index(inplace=True)
                    df_to_save.to_sql('daily_prices', conn, if_exists='append', index=False)

            # 週次データ保存
            if df_weekly is not None and not df_weekly.empty:
                df_weekly = df_weekly.dropna(subset=['open', 'high', 'low', 'close'], how='any')
                df_weekly = df_weekly[df_weekly.index.notna()]

                if not df_weekly.empty:
                    df_to_save = df_weekly[['open', 'high', 'low', 'close', 'volume', 'sma_200']].copy()
                    df_to_save['symbol'] = symbol
                    df_to_save.index.name = 'week_start_date'
                    df_to_save.reset_index(inplace=True)
                    df_to_save.to_sql('weekly_prices', conn, if_exists='append', index=False)

            conn.commit()
            logger.info(f"Successfully saved data for '{symbol}' in DB.")

        except Exception as e:
            logger.error(f"Failed to save data for '{symbol}', rolling back. Error: {e}", exc_info=True)
            conn.rollback()
            raise

    def _update_metadata(self, symbol: str, conn):
        """メタデータを更新"""
        logger.info(f"Updating metadata for '{symbol}'...")
        try:
            cursor = conn.cursor()
            daily_stats_q = "SELECT COUNT(*), MIN(date), MAX(date) FROM daily_prices WHERE symbol = ?"
            weekly_stats_q = "SELECT COUNT(*), MIN(week_start_date), MAX(week_start_date) FROM weekly_prices WHERE symbol = ?"

            daily_count, first_daily, last_daily = cursor.execute(daily_stats_q, (symbol,)).fetchone()
            weekly_count, _, _ = cursor.execute(weekly_stats_q, (symbol,)).fetchone()

            metadata_values = {
                'symbol': symbol,
                'first_date': first_daily,
                'last_date': last_daily,
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'daily_count': daily_count,
                'weekly_count': weekly_count,
            }

            cols = ', '.join(metadata_values.keys())
            placeholders = ', '.join('?' for _ in metadata_values)
            sql = f"INSERT OR REPLACE INTO data_metadata ({cols}) VALUES ({placeholders})"
            cursor.execute(sql, tuple(metadata_values.values()))
            conn.commit()
            logger.info(f"Metadata for '{symbol}' updated successfully.")
        except Exception as e:
            logger.error(f"Failed to update metadata for '{symbol}': {e}", exc_info=True)
            raise

    def _load_daily_from_db(self, symbol: str, conn, lookback_days: int) -> pd.DataFrame:
        """DBから日次データを読み込み"""
        query = """
        SELECT date, open, high, low, close, volume, adj_close,
               sma_10, sma_21, sma_50, sma_150, sma_200, ema_200
        FROM daily_prices
        WHERE symbol = ?
        ORDER BY date DESC
        LIMIT ?
        """
        try:
            df = pd.read_sql_query(query, conn, params=(symbol, lookback_days),
                                  index_col='date', parse_dates=['date'])
            return df.sort_index() if not df.empty else pd.DataFrame()
        except Exception as e:
            logger.error(f"Failed to load daily data for '{symbol}': {e}", exc_info=True)
            return pd.DataFrame()

    def _load_weekly_from_db(self, symbol: str, conn, lookback_weeks: int) -> pd.DataFrame:
        """DBから週次データを読み込み"""
        query = """
        SELECT week_start_date, open, high, low, close, volume, sma_200
        FROM weekly_prices
        WHERE symbol = ?
        ORDER BY week_start_date DESC
        LIMIT ?
        """
        try:
            df = pd.read_sql_query(query, conn, params=(symbol, lookback_weeks),
                                  index_col='week_start_date', parse_dates=['week_start_date'])
            return df.sort_index() if not df.empty else pd.DataFrame()
        except Exception as e:
            logger.error(f"Failed to load weekly data for '{symbol}': {e}", exc_info=True)
            return pd.DataFrame()

    def get_fundamental_data(self, symbol: str, force_update: bool = False) -> Optional[Dict]:
        """
        ファンダメンタルデータを取得

        Args:
            symbol: 銘柄シンボル
            force_update: 強制的に最新データを取得するか

        Returns:
            ファンダメンタルデータの辞書
        """
        try:
            # キャッシュをチェック（1日以内なら再利用）
            if not force_update:
                with self.db_lock:
                    with sqlite3.connect(self.db_path, timeout=30) as conn:
                        cursor = conn.cursor()
                        query = "SELECT * FROM fundamental_data WHERE symbol = ?"
                        row = cursor.execute(query, (symbol,)).fetchone()

                        if row:
                            data = dict(zip([d[0] for d in cursor.description], row))
                            last_updated = datetime.strptime(data['last_updated'], '%Y-%m-%d %H:%M:%S')
                            if datetime.now() - last_updated < timedelta(days=1):
                                logger.info(f"'{symbol}': Using cached fundamental data.")
                                return data

            # FMPから取得
            logger.info(f"'{symbol}': Fetching fundamental data from FMP...")
            profile = self.fmp_fetcher.get_profile(symbol)
            quote = self.fmp_fetcher.get_quote(symbol)

            if not profile and not quote:
                return None

            # EPS成長率を計算
            eps_growth_data = self.get_eps_growth_rate(symbol)

            fundamental_data = {
                'symbol': symbol,
                'market_cap': profile.get('mktCap') or quote.get('marketCap'),
                'sector': profile.get('sector'),
                'industry': profile.get('industry'),
                'eps_ttm': quote.get('eps'),
                'pe_ratio': quote.get('pe'),
                'revenue_growth_yoy': None,  # 必要に応じて income_statement から計算
                'earnings_growth_yoy': None,
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            # EPS成長率データを追加
            if eps_growth_data:
                fundamental_data.update(eps_growth_data)

            # DBに保存
            with self.db_lock:
                with sqlite3.connect(self.db_path, timeout=30) as conn:
                    cursor = conn.cursor()
                    cols = ', '.join(fundamental_data.keys())
                    placeholders = ', '.join('?' for _ in fundamental_data)
                    sql = f"INSERT OR REPLACE INTO fundamental_data ({cols}) VALUES ({placeholders})"
                    cursor.execute(sql, tuple(fundamental_data.values()))
                    conn.commit()

            return fundamental_data

        except Exception as e:
            logger.error(f"Error getting fundamental data for '{symbol}': {e}", exc_info=True)
            return None

    def get_eps_growth_rate(self, symbol: str) -> Dict:
        """
        四半期EPS成長率と予想EPS成長率を取得

        Args:
            symbol: 銘柄シンボル

        Returns:
            EPS成長率データの辞書
            - eps_growth_last_qtr: 前四半期比EPS成長率 (%)
            - eps_est_cur_qtr_growth: 今四半期予想EPS成長率 (%)
        """
        try:
            # 四半期Income Statementから実績EPS成長率を計算
            income_statements = self.fmp_fetcher.get_income_statement(symbol, period='quarter', limit=4)

            eps_growth_last_qtr = None
            if income_statements and len(income_statements) >= 2:
                latest_eps = income_statements[0].get('eps', 0) or income_statements[0].get('epsdiluted', 0)
                prev_eps = income_statements[1].get('eps', 0) or income_statements[1].get('epsdiluted', 0)

                if prev_eps and prev_eps != 0:
                    eps_growth_last_qtr = ((latest_eps - prev_eps) / abs(prev_eps)) * 100
                    logger.info(f"'{symbol}': EPS growth last quarter: {eps_growth_last_qtr:.2f}%")

            # Earnings Surprisesから予想EPS成長率を取得
            eps_est_cur_qtr_growth = None
            try:
                earnings_data = self.fmp_fetcher.get_earnings_surprises(symbol)
                if earnings_data and len(earnings_data) > 0:
                    # 最新の予想EPSを取得
                    latest_earnings = earnings_data[0]
                    estimated_eps = latest_earnings.get('estimatedEarning')

                    # 前年同期の実績EPSと比較
                    if estimated_eps and income_statements and len(income_statements) >= 4:
                        # 4四半期前（前年同期）のEPS
                        year_ago_eps = income_statements[3].get('eps', 0) or income_statements[3].get('epsdiluted', 0)

                        if year_ago_eps and year_ago_eps != 0:
                            eps_est_cur_qtr_growth = ((estimated_eps - year_ago_eps) / abs(year_ago_eps)) * 100
                            logger.info(f"'{symbol}': Estimated EPS growth (YoY): {eps_est_cur_qtr_growth:.2f}%")
            except Exception as e:
                logger.warning(f"'{symbol}': Could not fetch earnings estimates: {e}")

            return {
                'eps_growth_last_qtr': eps_growth_last_qtr,
                'eps_est_cur_qtr_growth': eps_est_cur_qtr_growth
            }

        except Exception as e:
            logger.error(f"Error calculating EPS growth for '{symbol}': {e}", exc_info=True)
            return {
                'eps_growth_last_qtr': None,
                'eps_est_cur_qtr_growth': None
            }

    def save_screening_results(self, results: Dict, filename: str = None):
        """
        スクリーニング結果を保存

        Args:
            results: スクリーニング結果の辞書
            filename: 保存ファイル名（Noneの場合は日時から自動生成）
        """
        try:
            if filename is None:
                filename = f"screening_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            filepath = self.results_dir / filename

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, cls=CustomJSONEncoder)

            logger.info(f"Saved screening results to {filepath}")

            # latest.jsonも更新
            latest_path = self.results_dir / "latest.json"
            with open(latest_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, cls=CustomJSONEncoder)

        except Exception as e:
            logger.error(f"Failed to save screening results: {e}", exc_info=True)
