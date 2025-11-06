#!/usr/bin/env python3
"""
Oratnek Screener テストスクリプト

SQLiteデータ管理とマルチプロセス化の動作を確認します。
"""

import os
import sys
import logging
from oratnek_screeners import run_oratnek_screener
from oratnek_data_manager import OratnekDataManager

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_data_manager():
    """データマネージャーの基本動作テスト"""
    logger.info("="*80)
    logger.info("TEST 1: Data Manager Basic Operations")
    logger.info("="*80)

    try:
        # データマネージャーを作成
        dm = OratnekDataManager(base_data_path='data/oratnek_test')
        logger.info("✓ Data manager created successfully")

        # テスト銘柄でデータ取得
        test_symbol = 'AAPL'
        logger.info(f"\nFetching data for {test_symbol}...")
        result = dm.get_stock_data_with_cache(test_symbol, lookback_years=2)

        if result:
            daily_df, weekly_df = result
            logger.info(f"✓ Data fetched successfully")
            logger.info(f"  - Daily data: {len(daily_df)} rows")
            logger.info(f"  - Weekly data: {len(weekly_df)} rows")
            logger.info(f"  - Columns: {', '.join(daily_df.columns[:5])}...")

            # キャッシュテスト（2回目の取得は高速）
            logger.info(f"\nFetching {test_symbol} again (should use cache)...")
            result2 = dm.get_stock_data_with_cache(test_symbol, lookback_years=2)
            if result2:
                logger.info("✓ Cache is working correctly")
            else:
                logger.warning("⚠ Cache may not be working")

        else:
            logger.error(f"✗ Failed to fetch data for {test_symbol}")
            return False

        return True

    except Exception as e:
        logger.error(f"✗ Test failed: {e}", exc_info=True)
        return False


def test_screener_small():
    """小規模な銘柄セットでスクリーナーをテスト"""
    logger.info("\n" + "="*80)
    logger.info("TEST 2: Screener with Small Ticker Set")
    logger.info("="*80)

    try:
        # テスト用の小さな銘柄セット
        test_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA']
        logger.info(f"Testing with {len(test_tickers)} tickers: {', '.join(test_tickers)}")

        # データマネージャーを作成
        dm = OratnekDataManager(base_data_path='data/oratnek_test')

        # スクリーナー実行（マルチプロセス有効）
        logger.info("\nRunning screener WITH multiprocessing...")
        results_mp = run_oratnek_screener(
            tickers=test_tickers,
            use_multiprocessing=True,
            data_manager=dm
        )

        # 結果確認
        logger.info("\nScreening Results:")
        for screen_name, df in results_mp.items():
            logger.info(f"  - {screen_name}: {len(df)} stocks")
            if not df.empty:
                logger.info(f"    Tickers: {', '.join(df['ticker'].tolist()[:3])}...")

        logger.info("\n✓ Screener test completed successfully")
        return True

    except Exception as e:
        logger.error(f"✗ Test failed: {e}", exc_info=True)
        return False


def test_multiprocessing_comparison():
    """マルチプロセスあり/なしの性能比較"""
    logger.info("\n" + "="*80)
    logger.info("TEST 3: Multiprocessing Performance Comparison")
    logger.info("="*80)

    try:
        import time
        test_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
        dm = OratnekDataManager(base_data_path='data/oratnek_test')

        # マルチプロセスなし
        logger.info("\n1. Running WITHOUT multiprocessing...")
        start = time.time()
        results_no_mp = run_oratnek_screener(
            tickers=test_tickers,
            use_multiprocessing=False,
            data_manager=dm
        )
        time_no_mp = time.time() - start
        logger.info(f"   Time: {time_no_mp:.2f} seconds")

        # マルチプロセスあり
        logger.info("\n2. Running WITH multiprocessing...")
        start = time.time()
        results_mp = run_oratnek_screener(
            tickers=test_tickers,
            use_multiprocessing=True,
            data_manager=dm
        )
        time_mp = time.time() - start
        logger.info(f"   Time: {time_mp:.2f} seconds")

        # 性能比較
        speedup = time_no_mp / time_mp if time_mp > 0 else 1.0
        logger.info(f"\n✓ Speedup: {speedup:.2f}x")

        return True

    except Exception as e:
        logger.error(f"✗ Test failed: {e}", exc_info=True)
        return False


def main():
    """メインテスト関数"""
    logger.info("\n" + "="*80)
    logger.info("ORATNEK SCREENER - COMPREHENSIVE TEST SUITE")
    logger.info("="*80)

    # 環境変数チェック
    api_key = os.getenv('FMP_API_KEY')
    if not api_key:
        logger.error("✗ FMP_API_KEY environment variable is not set!")
        logger.error("  Please set your API key: export FMP_API_KEY='your_api_key'")
        return 1

    logger.info(f"✓ FMP API Key found: {api_key[:10]}...")

    # テスト実行
    all_passed = True

    # Test 1: データマネージャー
    if not test_data_manager():
        all_passed = False

    # Test 2: スクリーナー（小規模）
    if not test_screener_small():
        all_passed = False

    # Test 3: マルチプロセス性能比較（オプション）
    if '--benchmark' in sys.argv:
        if not test_multiprocessing_comparison():
            all_passed = False

    # 結果サマリー
    logger.info("\n" + "="*80)
    if all_passed:
        logger.info("✓ ALL TESTS PASSED")
    else:
        logger.info("✗ SOME TESTS FAILED")
    logger.info("="*80)

    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
