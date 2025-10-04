import pandas as pd
import numpy as np
import pandas_ta as ta
from data_fetcher import fetch_stock_data

def calculate_ma_slope(series: pd.Series, period: int = 10) -> float:
    """
    指定された期間における系列（移動平均線など）の傾きを計算します。
    numpy.polyfitを使用して線形回帰を行い、その傾きを返します。

    Args:
        series (pd.Series): 傾きを計算するデータ系列。
        period (int): 傾きを計算するために使用する直近のデータポイント数。

    Returns:
        float: 計算された傾き。データが不足している場合は0.0。
    """
    if len(series) < period:
        return 0.0

    # y軸の値を正規化して、異なる価格帯の株式間で傾きを比較しやすくする
    y = series.tail(period).values
    y_normalized = y / np.linalg.norm(y)
    x = np.arange(len(y_normalized))

    # polyfitで傾き（1次の係数）を計算
    slope = np.polyfit(x, y_normalized, 1)[0]
    return slope

def calculate_rs_rating(stock_data: pd.DataFrame, benchmark_data: pd.DataFrame) -> pd.Series:
    """
    Mansfieldの相対強度（RS）を計算します。
    RS Ratingは、市場全体（ベンチマーク）に対する個別株のパフォーマンスを測定します。

    計算方法:
    1. 相対パフォーマンスライン (RSライン) を計算: (株価の終値 / ベンチマークの終値)
    2. RSラインの52週移動平均を計算します。
    3. 現在のRSラインがその移動平均を上回っているか、またその値自体を評価します。

    ここでは、RSラインを計算し、それを0から100のスケールに変換して「レーティング」とします。
    過去1年間のRSラインの値のパーセンタイルランクを計算することで、これを実現します。

    Args:
        stock_data (pd.DataFrame): 株式のOHLCVデータ。
        benchmark_data (pd.DataFrame): ベンチマークのOHLCVデータ。

    Returns:
        pd.Series: 計算されたRS Rating (0-100)。
    """
    # 両方のデータフレームで日付インデックスを確実に一致させる
    common_index = stock_data.index.intersection(benchmark_data.index)
    _stock = stock_data.loc[common_index]
    _benchmark = benchmark_data.loc[common_index]

    rs_line = (_stock['Close'] / _benchmark['Close'])

    # 過去252取引日（約1年）のパーセンタイルランクとしてRS Ratingを計算
    rs_rating = rs_line.rolling(window=252, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    )
    return rs_rating.fillna(0)


def calculate_all_indicators(stock_data: pd.DataFrame, benchmark_data: pd.DataFrame) -> pd.DataFrame:
    """
    株価データに、ステージ分析に必要なすべてのテクニカル指標を計算して追加します。

    Args:
        stock_data (pd.DataFrame): 株式のOHLCVデータ。
        benchmark_data (pd.DataFrame): ベンチマークのOHLCVデータ。

    Returns:
        pd.DataFrame: 指標が追加された株価データフレーム。
    """
    df = stock_data.copy()

    # 1. 移動平均線 (MA)
    df['ma50'] = ta.sma(df['Close'], length=50)
    df['ma200'] = ta.sma(df['Close'], length=200)

    # 2. 平均出来高
    df['volume_ma50'] = ta.sma(df['Volume'], length=50)

    # 3. MAの傾き
    # ma50の各ポイントで、過去10日間の傾きを計算
    df['ma50_slope'] = df['ma50'].rolling(window=10).apply(
        lambda x: calculate_ma_slope(pd.Series(x), period=10), raw=False
    )
    df['ma50_slope'] = df['ma50_slope'].fillna(0)

    # 4. VWAP (出来高加重平均価格)
    # pandas-taのvwapは期間を必要としないため、日々のVWAPを計算します。
    # スコアリングでは、価格がVWAPを上回っているかどうかが重要です。
    df['vwap'] = ta.vwap(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume'])
    df['vwap_slope'] = df['vwap'].rolling(window=10).apply(
        lambda x: calculate_ma_slope(pd.Series(x), period=10), raw=False
    )
    df['vwap_slope'] = df['vwap_slope'].fillna(0)


    # 5. 相対強度 (RS Rating)
    df['rs_rating'] = calculate_rs_rating(df, benchmark_data)
    df['rs_rating_ma10'] = ta.sma(df['rs_rating'], length=10)

    # 6. ATR (Average True Range) とそれに基づくボラティリティ指標
    df['atr'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    # 価格がMA50からATRの何倍離れているかを計算
    # ゼロ除算を避けるために、atrが0の場合はnp.nanを設定し、後で0に置換
    df['atr_ma_distance_multiple'] = np.where(
        df['atr'] > 0,
        abs(df['Close'] - df['ma50']) / df['atr'],
        0
    )


    # NaN値を削除
    df.dropna(inplace=True)

    return df

if __name__ == '__main__':
    # モジュールのテスト用
    test_ticker = 'QS'
    print(f"テスト用に {test_ticker} とベンチマークのデータを取得中...")
    stock_df, benchmark_df = fetch_stock_data(test_ticker)

    if stock_df is not None and benchmark_df is not None:
        print("すべてのテクニカル指標を計算中...")
        indicators_df = calculate_all_indicators(stock_df, benchmark_df)

        print(f"\n{test_ticker} の計算済み指標データ (直近5日):")
        # 表示する列を絞り込む
        display_columns = [
            'Close', 'ma50', 'volume_ma50', 'ma50_slope', 'rs_rating',
            'atr', 'atr_ma_distance_multiple'
        ]
        print(indicators_df[display_columns].tail())
    else:
        print(f"{test_ticker} のデータ取得に失敗しました。")