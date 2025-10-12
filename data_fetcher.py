import yfinance as yf
import pandas as pd
from curl_cffi.requests import Session

def fetch_stock_data(ticker: str, benchmark_ticker: str = "SPY", period: str = "2y"):
    """
    yfinanceを使用して、指定されたティッカーとベンチマークの過去の株価データを取得します。
    ブロックを回避するためにcurl-cffiセッションを使用します。

    Args:
        ticker (str): 株式ティッカーシンボル (例: "AAPL")。
        benchmark_ticker (str, optional): ベンチマークのティッカーシンボル。デフォルトは "SPY"。
        period (str, optional): データをダウンロードする期間 (例: "1y", "2y", "max")。デフォルトは "2y"。

    Returns:
        tuple[pd.DataFrame | None, pd.DataFrame | None]: 2つのpandas DataFrameを含むタプル:
        (stock_data, benchmark_data)。メインチッカーのデータが取得できない場合は(None, None)を返します。
    """
    # HTTP 403エラーを回避するために、ブラウザのようなユーザーエージェントを持つセッションを作成します。
    session = Session(impersonate="chrome110")

    try:
        # 株式とベンチマークの両方のデータを1回の呼び出しでダウンロードします
        data = yf.download(
            tickers=[ticker, benchmark_ticker],
            period=period,
            session=session,
            progress=False
        )

        if data.empty or ticker not in data.columns.get_level_values(1):
            return None, None

        # データフレームを分離し、警告を回避するためにコピーします
        stock_data = data.xs(ticker, level=1, axis=1).copy()
        benchmark_data = data.xs(benchmark_ticker, level=1, axis=1).copy()

        # 初めに発生する可能性のあるNaN値を持つ行を削除します
        stock_data.dropna(inplace=True)
        benchmark_data.dropna(inplace=True)

        if stock_data.empty:
             return None, None

        return stock_data, benchmark_data

    except Exception as e:
        return None, None

if __name__ == '__main__':
    # モジュールのテスト用
    test_ticker = 'AAPL'
    print(f"テスト用に {test_ticker} のデータを取得中...")
    stock_df, benchmark_df = fetch_stock_data(test_ticker)

    if stock_df is not None and benchmark_df is not None:
        print(f"\n{test_ticker} のデータ:")
        print(stock_df.head())
        print(f"\nSPY (ベンチマーク) のデータ:")
        print(benchmark_df.head())
    else:
        print(f"{test_ticker} のデータ取得に失敗しました。")
