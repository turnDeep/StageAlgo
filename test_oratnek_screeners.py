"""
Oratnek Screenersの簡易テスト
"""

from oratnek_screeners import OratnekScreener, IBDIndicators, get_default_tickers

def test_ibd_indicators():
    """IBD指標計算のテスト"""
    print("=" * 80)
    print("Testing IBD Indicators...")
    print("=" * 80)

    from data_fetcher import fetch_stock_data
    from indicators import calculate_all_basic_indicators

    # SPYベンチマークを取得
    print("\nFetching SPY benchmark...")
    spy_df, _ = fetch_stock_data('SPY', period='2y')
    if spy_df is None:
        print("Error: Could not fetch SPY data")
        return

    spy_indicators = calculate_all_basic_indicators(spy_df)
    print(f"✓ SPY data loaded: {len(spy_indicators)} records")

    # AAPLでテスト
    print("\nTesting with AAPL...")
    aapl_df, _ = fetch_stock_data('AAPL', period='2y')
    if aapl_df is None:
        print("Error: Could not fetch AAPL data")
        return

    aapl_indicators = calculate_all_basic_indicators(aapl_df)
    print(f"✓ AAPL data loaded: {len(aapl_indicators)} records")

    # RS Rating計算
    rs_rating = IBDIndicators.calculate_rs_rating(aapl_indicators, spy_indicators)
    print(f"  RS Rating: {rs_rating:.2f}")

    # A/D Rating計算
    ad_rating = IBDIndicators.calculate_ad_rating(aapl_indicators)
    print(f"  A/D Rating: {ad_rating}")

    # Comp Rating計算
    comp_rating = IBDIndicators.calculate_comp_rating(rs_rating)
    print(f"  Comp Rating: {comp_rating:.2f}")

    # Relative Volume計算
    rel_vol = IBDIndicators.calculate_relative_volume(aapl_indicators)
    print(f"  Relative Volume: {rel_vol:.2f}")

    print("\n✅ IBD Indicators test completed!")


def test_screeners():
    """スクリーナーのテスト（小規模）"""
    print("\n" + "=" * 80)
    print("Testing Screeners (Small Sample)...")
    print("=" * 80)

    # テスト用に少数の銘柄のみ使用
    test_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA']

    print(f"\nScreening {len(test_tickers)} tickers: {', '.join(test_tickers)}")

    screener = OratnekScreener(test_tickers)

    # 各スクリーニングを実行
    results = screener.run_all_screens()

    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    for name, df in results.items():
        print(f"\n{name.upper().replace('_', ' ')}:")
        print(f"  Found: {len(df)} stocks")
        if not df.empty:
            print(f"  Top stock: {df.iloc[0]['ticker']}")

    print("\n✅ Screeners test completed!")


if __name__ == '__main__':
    print("\n🚀 Starting Oratnek Screeners Test\n")

    # IBD指標のテスト
    test_ibd_indicators()

    # スクリーナーのテスト
    test_screeners()

    print("\n" + "=" * 80)
    print("🎉 All tests completed!")
    print("=" * 80)
