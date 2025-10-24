import yfinance as yf
import pandas as pd
from curl_cffi.requests import Session

def fetch_stock_data(ticker: str, benchmark_ticker: str = "SPY", period: str = "5y", interval: str = "1d", fetch_benchmark: bool = True):
    """
    yfinanceを使用して株価データを取得します。yfinanceの出力形式のばらつきに対応。

    Args:
        ticker (str): 株式ティッカーシンボル。
        benchmark_ticker (str, optional): ベンチマークのティッカー。デフォルトは "SPY"。
        period (str, optional): データ期間。デフォルトは "5y"。
        interval (str, optional): データ間隔。デフォルトは "1d"。
        fetch_benchmark (bool, optional): ベンチマークデータも同時に取得するかどうか。デフォルトは True。

    Returns:
        tuple[pd.DataFrame | None, pd.DataFrame | None]: (stock_data, benchmark_data)。
    """
    session = Session(impersonate="chrome110")

    tickers_to_download = [ticker]
    # tickerとbenchmark_tickerが同じ場合（SPY自体の取得時など）に重複させない
    if fetch_benchmark and ticker.upper() != benchmark_ticker.upper():
        tickers_to_download.append(benchmark_ticker)

    try:
        data = yf.download(
            tickers=tickers_to_download,
            period=period,
            interval=interval,
            session=session,
            progress=False,
            group_by='ticker' # 常にtickerでグループ化し、MultiIndex出力を強制
        )

        if data.empty:
            return None, None

        # MultiIndexでない場合はエラーとして扱う
        if not isinstance(data.columns, pd.MultiIndex):
            # yfinanceが何らかの理由で期待通りに動作しなかった場合
            if len(tickers_to_download) == 1:
                stock_data = data.copy()
                stock_data.dropna(inplace=True)
                return stock_data, None
            return None, None


        # 目的のティッカーデータが存在するか確認
        if ticker not in data.columns.get_level_values(0):
            return None, None

        stock_data = data[ticker].copy()
        stock_data.dropna(inplace=True)
        if stock_data.empty:
            return None, None

        benchmark_data = None
        if fetch_benchmark:
            if ticker.upper() == benchmark_ticker.upper():
                # 自身がベンチマークなので、データをコピー
                benchmark_data = stock_data.copy()
            elif benchmark_ticker in data.columns.get_level_values(0):
                benchmark_data = data[benchmark_ticker].copy()
                benchmark_data.dropna(inplace=True)

        return stock_data, benchmark_data

    except Exception as e:
        return None, None

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
