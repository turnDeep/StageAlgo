"""
基礎テクニカル指標計算モジュール
全ての理論で使用される基本的な指標を計算
"""
import pandas as pd
import numpy as np


def calculate_sma(df: pd.DataFrame, column: str = 'Close', period: int = 50) -> pd.Series:
    """単純移動平均(SMA)を計算"""
    return df[column].rolling(window=period, min_periods=period).mean()


def calculate_ema(df: pd.DataFrame, column: str = 'Close', period: int = 21) -> pd.Series:
    """指数移動平均(EMA)を計算"""
    return df[column].ewm(span=period, adjust=False, min_periods=period).mean()


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average True Range (ATR)を計算
    Wilder推奨の14日期間
    """
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period, min_periods=period).mean()
    
    return atr


def calculate_obv(df: pd.DataFrame) -> pd.Series:
    """
    On-Balance Volume (OBV)を計算
    出来高の累積により買い圧力/売り圧力を測定
    """
    obv = [0]
    for i in range(1, len(df)):
        if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
            obv.append(obv[-1] + df['Volume'].iloc[i])
        elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
            obv.append(obv[-1] - df['Volume'].iloc[i])
        else:
            obv.append(obv[-1])
    
    return pd.Series(obv, index=df.index, name='OBV')


def calculate_52week_metrics(df: pd.DataFrame) -> tuple:
    """
    52週高値・安値および現在価格との距離を計算
    
    Returns:
        tuple: (52週高値, 52週安値, 高値からの距離%, 安値からの距離%)
    """
    high_52w = df['High'].rolling(window=252, min_periods=50).max()
    low_52w = df['Low'].rolling(window=252, min_periods=50).min()
    
    current_price = df['Close']
    
    # 高値からの距離 (負の値 = 高値を下回る)
    dist_from_high = ((current_price - high_52w) / high_52w * 100)
    
    # 安値からの距離 (正の値 = 安値を上回る)
    dist_from_low = ((current_price - low_52w) / low_52w * 100)
    
    return high_52w, low_52w, dist_from_high, dist_from_low


def calculate_ma_slope(series: pd.Series, period: int = 20) -> pd.Series:
    """
    移動平均の傾きを計算
    
    Args:
        series: 移動平均のシリーズ
        period: 傾き計算に使用する期間
        
    Returns:
        傾きのシリーズ (正規化済み)
    """
    def slope_calc(x):
        if len(x) < 2:
            return 0.0
        # 正規化して線形回帰
        y = x / np.linalg.norm(x) if np.linalg.norm(x) > 0 else x
        x_vals = np.arange(len(y))
        slope = np.polyfit(x_vals, y, 1)[0]
        return slope
    
    return series.rolling(window=period, min_periods=period).apply(slope_calc, raw=False)


def calculate_all_basic_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    全ての基礎指標を一括計算
    
    フェーズ1: 基礎データ計算に対応
    - 移動平均線 (50日/150日/200日SMA, 21日/8日EMA)
    - 価格レベル (52週高値/安値、距離)
    - ATR関連
    - 出来高関連
    """
    result_df = df.copy()
    
    # 1. 移動平均線
    result_df['SMA_50'] = calculate_sma(df, 'Close', 50)
    result_df['SMA_150'] = calculate_sma(df, 'Close', 150)
    result_df['SMA_200'] = calculate_sma(df, 'Close', 200)
    result_df['EMA_21'] = calculate_ema(df, 'Close', 21)
    result_df['EMA_8'] = calculate_ema(df, 'Close', 8)
    
    # 週足相当の移動平均（参考用）
    result_df['SMA_10W'] = calculate_sma(df, 'Close', 50)  # 10週 ≈ 50日
    result_df['SMA_30W'] = calculate_sma(df, 'Close', 150)  # 30週 ≈ 150日
    result_df['SMA_40W'] = calculate_sma(df, 'Close', 200)  # 40週 ≈ 200日
    
    # 2. 移動平均の傾き (20日基準)
    result_df['SMA_50_Slope'] = calculate_ma_slope(result_df['SMA_50'], 20)
    result_df['SMA_150_Slope'] = calculate_ma_slope(result_df['SMA_150'], 20)
    result_df['SMA_200_Slope'] = calculate_ma_slope(result_df['SMA_200'], 20)
    
    # 3. 52週高値・安値
    high_52w, low_52w, dist_high, dist_low = calculate_52week_metrics(df)
    result_df['High_52W'] = high_52w
    result_df['Low_52W'] = low_52w
    result_df['Dist_From_52W_High_Pct'] = dist_high
    result_df['Dist_From_52W_Low_Pct'] = dist_low
    
    # 4. 現在価格と各MAの乖離率
    result_df['Deviation_From_SMA50_Pct'] = ((df['Close'] - result_df['SMA_50']) / result_df['SMA_50'] * 100)
    result_df['Deviation_From_SMA150_Pct'] = ((df['Close'] - result_df['SMA_150']) / result_df['SMA_150'] * 100)
    result_df['Deviation_From_SMA200_Pct'] = ((df['Close'] - result_df['SMA_200']) / result_df['SMA_200'] * 100)
    
    # 5. ATR関連
    result_df['ATR_14'] = calculate_atr(df, 14)
    result_df['ATR_Pct'] = (result_df['ATR_14'] / df['Close'] * 100)
    
    # 6. 出来高関連
    result_df['Volume_SMA_50'] = calculate_sma(df, 'Volume', 50)
    result_df['Relative_Volume'] = df['Volume'] / result_df['Volume_SMA_50']
    
    # 7. OBV
    result_df['OBV'] = calculate_obv(df)
    
    return result_df


if __name__ == '__main__':
    # テスト用
    from data_fetcher import fetch_stock_data
    
    print("指標計算のテストを開始...")
    stock_df, _ = fetch_stock_data('AAPL', period='2y')
    
    if stock_df is not None:
        indicators_df = calculate_all_basic_indicators(stock_df)
        print("\n計算された指標 (最新5日):")
        cols_to_show = ['Close', 'SMA_50', 'SMA_150', 'SMA_200', 'ATR_14', 
                        'Relative_Volume', 'Dist_From_52W_High_Pct']
        print(indicators_df[cols_to_show].tail())
        print(f"\n総行数: {len(indicators_df)}")
        print(f"NaN値を除外後: {indicators_df.dropna().shape[0]}")
