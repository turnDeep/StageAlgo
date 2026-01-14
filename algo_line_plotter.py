import yfinance as yf
import numpy as np
import pandas as pd
from findiff import Diff
import matplotlib.pyplot as plt
import trendln

def calculate_anchored_vwap(df, start_index):
    """
    Calculates Anchored VWAP from a specific start index.
    """
    subset = df.iloc[start_index:].copy()
    subset['Cum_Vol'] = subset['Volume'].cumsum()
    subset['Cum_Vol_Price'] = (subset['Close'] * subset['Volume']).cumsum()
    subset['VWAP'] = subset['Cum_Vol_Price'] / subset['Cum_Vol']

    # Realign with original index
    vwap_series = pd.Series(index=df.index, dtype=float)
    vwap_series.iloc[start_index:] = subset['VWAP']
    return vwap_series

def main():
    # 1. Data Fetching
    ticker_symbol = '^IXIC'
    print(f"Fetching data for {ticker_symbol}...")
    df = yf.download(ticker_symbol, period='1y', interval='1d')

    # Handle MultiIndex columns if present (common in recent yfinance)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    # Ensure necessary columns exist
    if 'High' not in df.columns or 'Volume' not in df.columns:
        print("Error: Missing High or Volume data.")
        return

    # Use integer index for calculations to match findiff and trendln easier
    close_prices = df['Close'].values
    high_prices = df['High'].values
    volumes = df['Volume'].values
    dates = df.index

    # 2. Custom Algo Anchor Detection (Section 5)
    print("Detecting Algo Anchors using findiff...")

    # Define differential operators
    # Assuming unit spacing (dx=1) for daily data
    d_dx = Diff(0, grid=1.0, acc=4)
    d2_dx2 = Diff(0, grid=1.0, acc=4) ** 2

    # Calculate derivatives of High prices
    first_deriv = d_dx(high_prices)
    second_deriv = d2_dx2(high_prices)

    # Find peaks: Zero crossing of 1st derivative (positive to negative) AND negative 2nd derivative
    # sign_change: +1 to -1 means diff is -2
    sign = np.sign(first_deriv)
    sign_change = np.diff(sign)

    # Indices where sign changes from +1 to -1 (peak candidates)
    # np.diff returns array of length N-1. Index i corresponds to change between i and i+1.
    # So if sign_change[i] is -2, it means sign[i]=1 and sign[i+1]=-1. Peak is around i.
    # We will mark i+1 as the peak index for simplicity or i. Let's use i.
    peak_candidates_indices = np.where(sign_change < 0)[0]

    algo_anchors = []

    # Calculate 20-day SMA of Volume for filtering
    volume_sma = df['Volume'].rolling(window=20).mean().values

    for idx in peak_candidates_indices:
        # Check 2nd derivative condition (concave down)
        # We need to be careful with indices. np.diff reduces size by 1.
        # Let's check idx and idx+1
        if idx < len(second_deriv) and second_deriv[idx] < 0:
            # Check Volume Condition: > 1.5 * 20-day SMA
            # Need to handle NaN in SMA (start of data)
            if not np.isnan(volume_sma[idx]):
                if volumes[idx] > volume_sma[idx] * 1.5:
                    algo_anchors.append(idx)

    print(f"Found {len(algo_anchors)} Algo Anchors.")

    # 3. Anchored VWAP (Section 3)
    # Find global highest high
    global_high_idx = np.argmax(high_prices)
    print(f"Global High found at index {global_high_idx} ({dates[global_high_idx].date()})")

    anchored_vwap = calculate_anchored_vwap(df, global_high_idx)

    # 4. Trendln Integration & Visualization
    print("Generating chart with trendln...")

    # Setup figure
    fig = plt.figure(figsize=(15, 10))

    # Use trendln to plot support/resistance on the Close price
    # trendln plots on the current active figure/axes
    # It returns pivot points, but also plots if we call plot_support_resistance

    # We use 'Close' for trendlines as is common, or 'High'/'Low'.
    # The user request mentioned "Algo Line" often uses Highs/Lows.
    # trendln usually works on a single series. Let's use High for resistance structure or Close.
    # Given the Algo Line definition involving 'wicks' (Highs), passing High might be better,
    # but standard trendln usage often defaults to Close.
    # Let's stick to High as per "High Price Anchored VWAP" context and "Algo Line" definition (wicks).

    # trendln.plot_support_resistance(h, ...)
    # It plots the provided series 'h' and the lines.

    trendln.plot_support_resistance(
        df['High'], # Using High for resistance lines
        accuracy=4,
        window=125, # Default
    )

    # Get current axes
    ax = plt.gca()

    # Overlay Algo Anchors
    # trendln uses integer x-axis 0..N-1
    anchor_y_values = [high_prices[i] for i in algo_anchors]
    ax.scatter(algo_anchors, anchor_y_values, color='red', s=100, marker='^', label='Algo Anchors (Vol > 1.5x)', zorder=5)

    # Overlay Anchored VWAP
    # We need to plot against integer index to match trendln
    ax.plot(range(len(df)), anchored_vwap.values, color='orange', linewidth=2, linestyle='--', label='High Anchored VWAP')

    # Improve formatting
    ax.set_title(f"NASDAQ (^IXIC) Algo Line Analysis\nCustom Anchors (Findiff + Vol) & High Anchored VWAP", fontsize=16)
    ax.set_ylabel("Price")
    ax.set_xlabel("Date")

    # Format x-axis to show dates instead of integers
    # trendln might have already handled this if we passed a Series with DateTimeIndex?
    # Let's check. If trendln plots the series directly, it might use the index.
    # If the x-axis is currently integers, we replace labels.

    # Check if x-axis is numeric (integers)
    # Usually trendln plots index 0 to N.
    # We will set major ticks to reasonable intervals

    num_points = len(df)
    tick_step = max(1, num_points // 10)
    tick_indices = range(0, num_points, tick_step)
    tick_labels = [dates[i].strftime('%Y-%m-%d') for i in tick_indices]

    ax.set_xticks(tick_indices)
    ax.set_xticklabels(tick_labels, rotation=45)

    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_filename = 'algo_line_analysis.png'
    plt.savefig(output_filename)
    print(f"Chart saved to {output_filename}")

if __name__ == "__main__":
    main()
