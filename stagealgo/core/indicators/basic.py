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


def calculate_all_basic_indicators(df: pd.DataFrame, interval: str = '1d') -> pd.DataFrame:
    """
    全ての基礎指標を一括計算（日足・週足対応）
    
    Args:
        df (pd.DataFrame): 株価データ
        interval (str): '1d' or '1wk'

    Returns:
        pd.DataFrame: 指標が追加されたDataFrame
    """
    result_df = df.copy()

    # yfinanceが重複列を返すことがあるため、重複を削除してデータ品質を確保
    if result_df.columns.has_duplicates:
        result_df = result_df.loc[:, ~result_df.columns.duplicated()]

    # --- 時間軸に応じたパラメータ設定 ---
    if interval == '1wk':
        # 週足用の期間
        ma_short, ma_mid, ma_long = 10, 30, 40
        slope_period = 10  # 10週
        atr_period = 10  # 10週
        vol_sma_period = 40  # 40週
        high_low_window = 52  # 52週
        ema_short, ema_long = 5, 13 # 週足用のEMA期間
    else:
        # 日足用の期間
        ma_short, ma_mid, ma_long = 50, 150, 200
        slope_period = 20  # 20日
        atr_period = 14  # 14日
        vol_sma_period = 50  # 50日
        high_low_window = 252  # 52週 ≈ 252日
        ema_short, ema_long = 8, 21

    # 1. 移動平均線
    result_df[f'SMA_{ma_short}'] = calculate_sma(result_df, 'Close', ma_short)
    result_df[f'SMA_{ma_mid}'] = calculate_sma(result_df, 'Close', ma_mid)
    result_df[f'SMA_{ma_long}'] = calculate_sma(result_df, 'Close', ma_long)
    result_df[f'EMA_{ema_long}'] = calculate_ema(result_df, 'Close', ema_long)
    result_df[f'EMA_{ema_short}'] = calculate_ema(result_df, 'Close', ema_short)

    # 2. 移動平均の傾き
    result_df[f'SMA_{ma_short}_Slope'] = calculate_ma_slope(result_df[f'SMA_{ma_short}'], slope_period)
    result_df[f'SMA_{ma_mid}_Slope'] = calculate_ma_slope(result_df[f'SMA_{ma_mid}'], slope_period)
    result_df[f'SMA_{ma_long}_Slope'] = calculate_ma_slope(result_df[f'SMA_{ma_long}'], slope_period)

    # 3. 52週高値・安値
    high_52w = result_df['High'].rolling(window=high_low_window, min_periods=high_low_window // 2).max()
    low_52w = result_df['Low'].rolling(window=high_low_window, min_periods=high_low_window // 2).min()
    result_df['High_52W'] = high_52w
    result_df['Low_52W'] = low_52w
    result_df['Dist_From_52W_High_Pct'] = ((result_df['Close'] - high_52w) / high_52w * 100)
    result_df['Dist_From_52W_Low_Pct'] = ((result_df['Close'] - low_52w) / low_52w * 100)

    # 4. 現在価格と各MAの乖離率
    result_df[f'Deviation_From_SMA_{ma_short}_Pct'] = ((result_df['Close'] - result_df[f'SMA_{ma_short}']) / result_df[f'SMA_{ma_short}'] * 100)
    result_df[f'Deviation_From_SMA_{ma_mid}_Pct'] = ((result_df['Close'] - result_df[f'SMA_{ma_mid}']) / result_df[f'SMA_{ma_mid}'] * 100)
    result_df[f'Deviation_From_SMA_{ma_long}_Pct'] = ((result_df['Close'] - result_df[f'SMA_{ma_long}']) / result_df[f'SMA_{ma_long}'] * 100)

    # 5. ATR関連
    result_df[f'ATR_{atr_period}'] = calculate_atr(result_df, atr_period)
    result_df['ATR_Pct'] = (result_df[f'ATR_{atr_period}'] / result_df['Close'] * 100)

    # 6. 出来高関連
    result_df[f'Volume_SMA_{vol_sma_period}'] = calculate_sma(result_df, 'Volume', vol_sma_period)
    result_df['Relative_Volume'] = result_df['Volume'] / result_df[f'Volume_SMA_{vol_sma_period}']

    # 7. OBV
    result_df['OBV'] = calculate_obv(result_df)

    return result_df


if __name__ == '__main__':
    # テスト用
    from data_fetcher import fetch_stock_data
    
    print("--- 日足データの指標計算テスト ---")
    stock_df_daily, _ = fetch_stock_data('AAPL', interval='1d')
    
    if stock_df_daily is not None:
        indicators_df_daily = calculate_all_basic_indicators(stock_df_daily, interval='1d')
        print("\n計算された日足指標 (最新5日):")
        cols_to_show = ['Close', 'SMA_50', 'SMA_150', 'SMA_200', 'ATR_14', 
                        'Relative_Volume', 'Dist_From_52W_High_Pct']
        print(indicators_df_daily[cols_to_show].tail())

    print("\n--- 週足データの指標計算テスト ---")
    stock_df_weekly, _ = fetch_stock_data('AAPL', interval='1wk')

    if stock_df_weekly is not None:
        indicators_df_weekly = calculate_all_basic_indicators(stock_df_weekly, interval='1wk')
        print("\n計算された週足指標 (最新5週):")
        cols_to_show = ['Close', 'SMA_10', 'SMA_30', 'SMA_40', 'ATR_10',
                        'Relative_Volume', 'Dist_From_52W_High_Pct']
        print(indicators_df_weekly[cols_to_show].tail())
