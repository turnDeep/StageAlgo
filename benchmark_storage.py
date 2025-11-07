"""
SQLite vs JSON ファイルのパフォーマンス比較ベンチマーク
"""
import sqlite3
import json
import time
import os
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil


def generate_sample_data(num_tickers=100, days=252):
    """サンプルデータを生成"""
    data = {}
    for i in range(num_tickers):
        ticker = f"TICK{i:04d}"
        dates = pd.date_range('2024-01-01', periods=days, freq='D')

        df = pd.DataFrame({
            'date': dates.strftime('%Y-%m-%d'),
            'open': np.random.uniform(100, 200, days),
            'high': np.random.uniform(100, 200, days),
            'low': np.random.uniform(100, 200, days),
            'close': np.random.uniform(100, 200, days),
            'volume': np.random.randint(1000000, 10000000, days),
            'sma_50': np.random.uniform(100, 200, days),
            'sma_200': np.random.uniform(100, 200, days),
        })
        data[ticker] = df
    return data


def setup_sqlite(data, db_path):
    """SQLiteにデータを保存"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS daily_prices (
            symbol TEXT NOT NULL,
            date DATE NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            sma_50 REAL,
            sma_200 REAL,
            PRIMARY KEY (symbol, date)
        )
    """)

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_symbol_date ON daily_prices(symbol, date)")

    for ticker, df in data.items():
        df['symbol'] = ticker
        df.to_sql('daily_prices', conn, if_exists='append', index=False)

    conn.commit()
    conn.close()


def setup_json(data, json_dir):
    """JSONファイルとしてデータを保存"""
    json_dir.mkdir(exist_ok=True)
    for ticker, df in data.items():
        file_path = json_dir / f"{ticker}.json"
        df.to_json(file_path, orient='records', date_format='iso')


# ========== ベンチマーク関数 ==========

def benchmark_sqlite_single_read(db_path, ticker):
    """SQLite: 単一銘柄読み取り"""
    conn = sqlite3.connect(db_path)
    query = "SELECT * FROM daily_prices WHERE symbol = ?"
    df = pd.read_sql_query(query, conn, params=(ticker,))
    conn.close()
    return df


def benchmark_json_single_read(json_dir, ticker):
    """JSON: 単一銘柄読み取り"""
    file_path = json_dir / f"{ticker}.json"
    with open(file_path, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    return df


def benchmark_sqlite_multi_read(db_path, tickers):
    """SQLite: 複数銘柄読み取り"""
    conn = sqlite3.connect(db_path)
    placeholders = ','.join('?' * len(tickers))
    query = f"SELECT * FROM daily_prices WHERE symbol IN ({placeholders})"
    df = pd.read_sql_query(query, conn, params=tickers)
    conn.close()
    return df


def benchmark_json_multi_read(json_dir, tickers):
    """JSON: 複数銘柄読み取り"""
    dfs = []
    for ticker in tickers:
        file_path = json_dir / f"{ticker}.json"
        with open(file_path, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        df['symbol'] = ticker
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def benchmark_sqlite_filtered_read(db_path):
    """SQLite: 条件付き読み取り"""
    conn = sqlite3.connect(db_path)
    query = """
        SELECT * FROM daily_prices
        WHERE close > sma_200
        AND volume > 5000000
        AND date >= '2024-06-01'
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def benchmark_json_filtered_read(json_dir, tickers):
    """JSON: 条件付き読み取り"""
    results = []
    for ticker in tickers:
        file_path = json_dir / f"{ticker}.json"
        with open(file_path, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        df['symbol'] = ticker

        # フィルタリング
        filtered = df[
            (df['close'] > df['sma_200']) &
            (df['volume'] > 5000000) &
            (df['date'] >= '2024-06-01')
        ]
        results.append(filtered)

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def run_benchmark():
    """ベンチマークを実行"""
    print("=" * 80)
    print("SQLite vs JSON パフォーマンスベンチマーク")
    print("=" * 80)

    # 一時ディレクトリ作成
    temp_dir = Path(tempfile.mkdtemp())
    db_path = temp_dir / "benchmark.db"
    json_dir = temp_dir / "json_files"

    try:
        # データ生成
        print("\n[1] データ生成中...")
        num_tickers = 100
        data = generate_sample_data(num_tickers=num_tickers, days=252)
        print(f"    生成完了: {num_tickers}銘柄 x 252日")

        # セットアップ
        print("\n[2] データ保存中...")

        start = time.time()
        setup_sqlite(data, db_path)
        sqlite_setup_time = time.time() - start
        print(f"    SQLite保存時間: {sqlite_setup_time:.3f}秒")

        start = time.time()
        setup_json(data, json_dir)
        json_setup_time = time.time() - start
        print(f"    JSON保存時間: {json_setup_time:.3f}秒")

        # ファイルサイズ比較
        sqlite_size = os.path.getsize(db_path) / (1024 * 1024)
        json_size = sum(f.stat().st_size for f in json_dir.glob('*.json')) / (1024 * 1024)
        print(f"\n[3] ストレージサイズ:")
        print(f"    SQLite: {sqlite_size:.2f} MB")
        print(f"    JSON: {json_size:.2f} MB")

        # ベンチマーク実行
        print("\n[4] 読み取りベンチマーク:")

        # Test 1: 単一銘柄読み取り
        test_ticker = "TICK0050"
        iterations = 50

        start = time.time()
        for _ in range(iterations):
            benchmark_sqlite_single_read(db_path, test_ticker)
        sqlite_single_time = (time.time() - start) / iterations

        start = time.time()
        for _ in range(iterations):
            benchmark_json_single_read(json_dir, test_ticker)
        json_single_time = (time.time() - start) / iterations

        print(f"\n    単一銘柄読み取り（平均 {iterations}回）:")
        print(f"      SQLite: {sqlite_single_time*1000:.2f} ms")
        print(f"      JSON:   {json_single_time*1000:.2f} ms")
        print(f"      勝者: {'SQLite' if sqlite_single_time < json_single_time else 'JSON'} "
              f"({max(sqlite_single_time, json_single_time)/min(sqlite_single_time, json_single_time):.2f}x faster)")

        # Test 2: 複数銘柄読み取り（10銘柄）
        test_tickers = [f"TICK{i:04d}" for i in range(10)]
        iterations = 20

        start = time.time()
        for _ in range(iterations):
            benchmark_sqlite_multi_read(db_path, test_tickers)
        sqlite_multi_time = (time.time() - start) / iterations

        start = time.time()
        for _ in range(iterations):
            benchmark_json_multi_read(json_dir, test_tickers)
        json_multi_time = (time.time() - start) / iterations

        print(f"\n    複数銘柄読み取り（10銘柄、平均 {iterations}回）:")
        print(f"      SQLite: {sqlite_multi_time*1000:.2f} ms")
        print(f"      JSON:   {json_multi_time*1000:.2f} ms")
        print(f"      勝者: {'SQLite' if sqlite_multi_time < json_multi_time else 'JSON'} "
              f"({max(sqlite_multi_time, json_multi_time)/min(sqlite_multi_time, json_multi_time):.2f}x faster)")

        # Test 3: 条件付き読み取り（全銘柄）
        all_tickers = [f"TICK{i:04d}" for i in range(num_tickers)]
        iterations = 5

        start = time.time()
        for _ in range(iterations):
            benchmark_sqlite_filtered_read(db_path)
        sqlite_filter_time = (time.time() - start) / iterations

        start = time.time()
        for _ in range(iterations):
            benchmark_json_filtered_read(json_dir, all_tickers)
        json_filter_time = (time.time() - start) / iterations

        print(f"\n    条件付き読み取り（全100銘柄をフィルタ、平均 {iterations}回）:")
        print(f"      SQLite: {sqlite_filter_time*1000:.2f} ms")
        print(f"      JSON:   {json_filter_time*1000:.2f} ms")
        print(f"      勝者: {'SQLite' if sqlite_filter_time < json_filter_time else 'JSON'} "
              f"({max(sqlite_filter_time, json_filter_time)/min(sqlite_filter_time, json_filter_time):.2f}x faster)")

        # サマリー
        print("\n" + "=" * 80)
        print("結果サマリー:")
        print("=" * 80)

        sqlite_wins = sum([
            sqlite_single_time < json_single_time,
            sqlite_multi_time < json_multi_time,
            sqlite_filter_time < json_filter_time
        ])

        print(f"SQLite勝利: {sqlite_wins}/3 テスト")
        print(f"JSON勝利: {3-sqlite_wins}/3 テスト")

        print("\n推奨:")
        if sqlite_wins >= 2:
            print("✓ SQLite を推奨")
            print("  理由: 複数銘柄アクセスと条件付き検索で圧倒的に高速")
        else:
            print("✓ JSON を推奨")
            print("  理由: 単一銘柄アクセスが主な用途の場合に有利")

    finally:
        # クリーンアップ
        shutil.rmtree(temp_dir)
        print(f"\n[クリーンアップ完了]")


if __name__ == '__main__':
    run_benchmark()
