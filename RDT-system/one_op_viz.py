import yfinance as yf
import pandas as pd
import pandas_ta as ta
import mplfinance as mpf
import numpy as np
import matplotlib.pyplot as plt

def calculate_wma(series, length):
    """Calculates Weighted Moving Average (WMA)."""
    weights = np.arange(1, length + 1)
    sum_weights = weights.sum()
    return series.rolling(window=length).apply(lambda x: np.dot(x, weights) / sum_weights, raw=True)

def calculate_tsv_approximation(df, length=13, ma_length=7, ma_type='EMA'):
    """
    Calculates Time Segmented Volume (TSV) approximation.
    """
    price_change = df['Close'].diff()
    signed_volume = df['Volume'] * price_change
    tsv_raw = signed_volume.rolling(window=length).sum()
    
    if ma_type == 'EMA':
        tsv_smoothed = tsv_raw.ewm(span=ma_length, adjust=False).mean()
    elif ma_type == 'SMA':
        tsv_smoothed = tsv_raw.rolling(window=ma_length).mean()
    else:
        tsv_smoothed = tsv_raw.rolling(window=ma_length).mean()
        
    return tsv_smoothed

def calculate_stochrsi_1op(df, rsi_length=14, stoch_length=14, k_smooth=5, d_smooth=5):
    """
    Calculates StochRSI with HEAVIER WMA smoothing (5, 5) to mimic 1OP cycles.
    """
    rsi = ta.rsi(df['Close'], length=rsi_length)
    rsi_low = rsi.rolling(window=stoch_length).min()
    rsi_high = rsi.rolling(window=stoch_length).max()
    
    denominator = rsi_high - rsi_low
    denominator = denominator.replace(0, np.nan) 
    
    stoch_raw = ((rsi - rsi_low) / denominator) * 100
    stoch_raw = stoch_raw.fillna(50) 
    
    k_line = calculate_wma(stoch_raw, k_smooth)
    d_line = calculate_wma(k_line, d_smooth)
    
    return k_line, d_line

def detect_cycle_phases(df):
    """
    Detects Bullish and Bearish Cycle Phases based on StochRSI Crosses.
    
    Logic:
    - Bullish Phase starts when K crosses above D (and K < 50 ideally, but strict crosses are key).
    - Ends when K crosses below D.
    
    - Bearish Phase starts when K crosses below D.
    - Ends when K crosses above D.
    """
    k = df['Fast_K'].values
    d = df['Slow_D'].values
    
    bullish_phase = np.zeros(len(df), dtype=bool)
    bearish_phase = np.zeros(len(df), dtype=bool)
    
    # State: 0 = Neutral/Unknown, 1 = Bullish, -1 = Bearish
    state = 0
    
    for i in range(1, len(df)):
        # Check Crosses
        cross_up = (k[i-1] <= d[i-1]) and (k[i] > d[i])
        cross_down = (k[i-1] >= d[i-1]) and (k[i] < d[i])
        
        if cross_up:
            state = 1
        elif cross_down:
            state = -1
            
        if state == 1:
            bullish_phase[i] = True
        elif state == -1:
            bearish_phase[i] = True
            
    return bullish_phase, bearish_phase

def main():
    ticker = "SPY"
    print(f"Fetching data for {ticker}...")
    df = yf.download(ticker, period="1y", interval="1d", progress=False)
    
    if df.empty:
        print("No data fetched.")
        return

    if isinstance(df.columns, pd.MultiIndex):
        try:
            df.columns = df.columns.droplevel(1)
        except:
            pass
    df.columns = [c.capitalize() for c in df.columns]

    # Indicators
    print("Calculating Indicators...")
    # TSV with 12/7
    df['TSV'] = calculate_tsv_approximation(df, length=12, ma_length=7, ma_type='EMA')
    # StochRSI with 14/14/5/5 for smoothness
    df['Fast_K'], df['Slow_D'] = calculate_stochrsi_1op(df, rsi_length=14, stoch_length=14, k_smooth=5, d_smooth=5)

    # Clean data
    plot_df = df.dropna().copy()
    
    if plot_df.empty:
        print("Not enough data.")
        return

    # Detect Cycles
    print("Detecting Cycle Phases...")
    bull_mask, bear_mask = detect_cycle_phases(plot_df)
    
    plot_df['Bullish_Phase'] = bull_mask
    plot_df['Bearish_Phase'] = bear_mask

    # Setup Fill Data
    y_max = plot_df['High'].max() * 1.05
    y_min = plot_df['Low'].min() * 0.95
    
    # Visualization
    print("Generating Chart...")
    
    apds = [
        mpf.make_addplot(plot_df['TSV'], panel=1, color='teal', width=1.5, ylabel='TSV (1OP Proxy)'),
        mpf.make_addplot(plot_df['Fast_K'], panel=2, color='cyan', width=1.5, ylabel='StochRSI (WMA 5,5)'),
        mpf.make_addplot(plot_df['Slow_D'], panel=2, color='orange', width=1.5),
        
        # Background Colors for PHASES
        mpf.make_addplot(pd.Series(y_max, index=plot_df.index), panel=0, color='g', alpha=0.0, secondary_y=False,
                         fill_between=dict(y1=y_max, y2=y_min, where=plot_df['Bullish_Phase'].values, color='skyblue', alpha=0.15)),
                         
        mpf.make_addplot(pd.Series(y_max, index=plot_df.index), panel=0, color='r', alpha=0.0, secondary_y=False,
                         fill_between=dict(y1=y_max, y2=y_min, where=plot_df['Bearish_Phase'].values, color='lightcoral', alpha=0.15)),
    ]
    
    mc = mpf.make_marketcolors(up='green', down='red', inherit=True)
    s = mpf.make_mpf_style(marketcolors=mc, gridstyle=':', y_on_right=True)
    
    output_file = "SPY_1OP_Approximation_Setup.png"
    
    fig, axlist = mpf.plot(
        plot_df,
        type='candle',
        style=s,
        addplot=apds,
        volume=False,
        panel_ratios=(3, 1, 1),
        title=f"{ticker} 1OP Cycle Approximation\nBlue Zone = Bullish Cycle (Stoch K > D) | Red Zone = Bearish Cycle (Stoch K < D)",
        returnfig=True,
        figsize=(14, 10)
    )
    
    # Reference Lines
    ax_tsv = axlist[2] 
    ax_tsv.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    
    ax_stoch = axlist[4]
    ax_stoch.axhline(80, color='red', linestyle='--', linewidth=0.8)
    ax_stoch.axhline(20, color='green', linestyle='--', linewidth=0.8)
    
    plt.savefig(output_file, bbox_inches='tight')
    print(f"Chart saved to {output_file}")

if __name__ == "__main__":
    main()
