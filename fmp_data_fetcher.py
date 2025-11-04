"""
Financial Modeling Prep API を使用したデータ取得モジュール
Yahoo Finance APIの代替として使用
"""
import pandas as pd
import requests
from datetime import datetime, timedelta
import time

# API Key
FMP_API_KEY = "iEeBOyZFh7E6lX5qMPDvZNIFLr1kUNwS"
BASE_URL = "https://financialmodelingprep.com/api/v3"


def get_historical_data(ticker: str, years: int = 2) -> pd.DataFrame:
    """
    Financial Modeling Prepから株価データを取得

    Args:
        ticker: ティッカーシンボル
        years: 取得する年数

    Returns:
        pd.DataFrame: OHLCVデータ
    """
    try:
        # 日付範囲を計算
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years * 365)

        url = f"{BASE_URL}/historical-price-full/{ticker}"
        params = {
            'apikey': FMP_API_KEY,
            'from': start_date.strftime('%Y-%m-%d'),
            'to': end_date.strftime('%Y-%m-%d')
        }

        response = requests.get(url, params=params, timeout=30)

        if response.status_code != 200:
            print(f"  API Error {response.status_code} for {ticker}")
            return None

        data = response.json()

        if 'historical' not in data or not data['historical']:
            print(f"  No historical data for {ticker}")
            return None

        # DataFrameに変換
        df = pd.DataFrame(data['historical'])

        # 日付をインデックスに
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        df = df.sort_index()

        # カラム名を標準形式に変更
        df = df.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        })

        # 必要なカラムのみ保持
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

        return df

    except Exception as e:
        print(f"  Error fetching {ticker}: {e}")
        return None


def get_market_cap(ticker: str) -> float:
    """
    Financial Modeling Prepから時価総額を取得

    Args:
        ticker: ティッカーシンボル

    Returns:
        float: 時価総額（ドル）
    """
    try:
        url = f"{BASE_URL}/market-capitalization/{ticker}"
        params = {'apikey': FMP_API_KEY}

        response = requests.get(url, params=params, timeout=30)

        if response.status_code != 200:
            return None

        data = response.json()

        if not data or len(data) == 0:
            return None

        # 最新の時価総額を取得
        market_cap = data[0].get('marketCap')
        return market_cap

    except Exception as e:
        print(f"  Error fetching market cap for {ticker}: {e}")
        return None


def get_company_profile(ticker: str) -> dict:
    """
    企業プロファイル情報を取得

    Args:
        ticker: ティッカーシンボル

    Returns:
        dict: 企業情報
    """
    try:
        url = f"{BASE_URL}/profile/{ticker}"
        params = {'apikey': FMP_API_KEY}

        response = requests.get(url, params=params, timeout=30)

        if response.status_code != 200:
            return None

        data = response.json()

        if not data or len(data) == 0:
            return None

        return data[0]

    except Exception as e:
        print(f"  Error fetching profile for {ticker}: {e}")
        return None


def fetch_stock_data_fmp(ticker: str, benchmark_ticker: str = "SPY",
                         years: int = 2) -> tuple:
    """
    FMP APIを使用して株価データを取得（data_fetcherと同じインターフェース）

    Args:
        ticker: ティッカーシンボル
        benchmark_ticker: ベンチマークティッカー
        years: 取得する年数

    Returns:
        tuple: (stock_df, benchmark_df)
    """
    # 銘柄データ取得
    stock_df = get_historical_data(ticker, years=years)

    # ベンチマークデータ取得
    if ticker.upper() == benchmark_ticker.upper():
        benchmark_df = stock_df.copy() if stock_df is not None else None
    else:
        benchmark_df = get_historical_data(benchmark_ticker, years=years)

    return stock_df, benchmark_df


if __name__ == '__main__':
    # テスト
    print("Financial Modeling Prep API Test")
    print("=" * 60)

    test_ticker = 'AAPL'

    # 株価データ取得
    print(f"\nFetching historical data for {test_ticker}...")
    stock_df, benchmark_df = fetch_stock_data_fmp(test_ticker, years=2)

    if stock_df is not None:
        print(f"✓ Got {len(stock_df)} days of data")
        print(stock_df.tail())
    else:
        print("✗ Failed to fetch data")

    # 時価総額取得
    print(f"\nFetching market cap for {test_ticker}...")
    market_cap = get_market_cap(test_ticker)

    if market_cap:
        print(f"✓ Market Cap: ${market_cap:,.0f} (${market_cap/1e9:.2f}B)")
    else:
        print("✗ Failed to fetch market cap")

    # 企業プロファイル取得
    print(f"\nFetching company profile for {test_ticker}...")
    profile = get_company_profile(test_ticker)

    if profile:
        print(f"✓ Company: {profile.get('companyName')}")
        print(f"  Sector: {profile.get('sector')}")
        print(f"  Industry: {profile.get('industry')}")
        print(f"  Market Cap: ${profile.get('mktCap', 0)/1e9:.2f}B")
    else:
        print("✗ Failed to fetch profile")
