"""
Data Fetcher - FinancialModelingPrep API版

yfinanceからFinancialModelingPrep APIに移行しました。
既存のコードとの互換性を保つため、関数インターフェースは維持しています。
"""

# FMPデータフェッチャーを使用
from fmp_data_fetcher import fetch_stock_data

# 後方互換性のため、元のインポートも残す
import pandas as pd

if __name__ == '__main__':
    # (テストコードは変更なし)
    test_ticker = 'AAPL'
    print(f"テスト用に {test_ticker} の日足データを取得中...")
    stock_df_daily, benchmark_df_daily = fetch_stock_data(test_ticker, interval="1d")
    if stock_df_daily is not None:
        print(f"\n{test_ticker} の日足データ:")
        print(stock_df_daily.head())
    if benchmark_df_daily is not None:
        print(f"\nSPY (ベンチマーク) の日足データ:")
        print(benchmark_df_daily.head())

    print(f"\nSPY自体のデータを取得（ベンチマークなし）...")
    spy_df, _ = fetch_stock_data('SPY', interval="1d", fetch_benchmark=False)
    if spy_df is not None:
        print(f"\nSPY の日足データ:")
        print(spy_df.head())

    print(f"\nSPY自体のデータを取得（ベンチマークあり）...")
    spy_df, spy_benchmark_df = fetch_stock_data('SPY', interval="1d", fetch_benchmark=True)
    if spy_df is not None and spy_benchmark_df is not None:
        print(f"\nSPYのデータと、SPYのベンチマークデータが両方取得できていることを確認")
        print(spy_df.head())
