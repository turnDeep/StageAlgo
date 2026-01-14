import yfinance as yf
import numpy as np
import pandas as pd
from findiff import Diff
import matplotlib.pyplot as plt
import trendln

def plot_findiff_only(df):
    """
    Plots only the custom Algo Anchors and lines detected using findiff.
    """
    print("Generating findiff only chart...")
    high_prices = df['High'].values
    volumes = df['Volume'].values
    dates = df.index

    # Define differential operators
    # Assuming unit spacing (dx=1) for daily data
    d_dx = Diff(0, grid=1.0, acc=4)
    d2_dx2 = Diff(0, grid=1.0, acc=4) ** 2

    # Calculate derivatives of High prices
    first_deriv = d_dx(high_prices)
    second_deriv = d2_dx2(high_prices)

    # Find peaks
    sign = np.sign(first_deriv)
    sign_change = np.diff(sign)
    peak_candidates_indices = np.where(sign_change < 0)[0]

    algo_anchors = []
    volume_sma = df['Volume'].rolling(window=20).mean().values

    for idx in peak_candidates_indices:
        if idx < len(second_deriv) and second_deriv[idx] < 0:
            if not np.isnan(volume_sma[idx]):
                if volumes[idx] > volume_sma[idx] * 1.5:
                    algo_anchors.append(idx)

    fig, ax = plt.subplots(figsize=(15, 10))
    ax.plot(range(len(df)), df['High'], label='High Price', color='gray', alpha=0.5)

    # Plot anchors
    anchor_y = [high_prices[i] for i in algo_anchors]
    ax.scatter(algo_anchors, anchor_y, color='red', s=100, marker='^', label='Algo Anchors (Findiff + Vol)', zorder=5)

    # Draw lines connecting anchors (Simple logic: connect sequential anchors)
    # The user asked for "lines using findiff only".
    # Logic: Connect significant pivots.
    # For visualization, we will connect the last N anchors to show "potential" trendlines
    # or just let the scatter points stand as the "detected" features.
    # However, "Algo Line" implies lines. Let's draw lines connecting local maxima.
    # A simple approach is to connect every pair of points and check for cuts (No-Cutting Rule),
    # but that is computationally expensive O(N^2).
    # We will implement a simplified version: connect recent anchors to past anchors if they form a valid line.

    # Simplified line drawing for visualization:
    # Connect the top 2 highest anchors
    if len(algo_anchors) >= 2:
        # Sort by price descending
        sorted_anchors = sorted(algo_anchors, key=lambda i: high_prices[i], reverse=True)
        # Draw a line between the top 2 global highs found
        p1 = sorted_anchors[0]
        p2 = sorted_anchors[1]

        # Ensure left-to-right drawing
        if p1 > p2: p1, p2 = p2, p1

        # Extrapolate line
        m = (high_prices[p2] - high_prices[p1]) / (p2 - p1)
        b = high_prices[p1] - m * p1

        # Plot line across the whole chart
        x_vals = np.array(range(len(df)))
        y_vals = m * x_vals + b
        ax.plot(x_vals, y_vals, color='red', linestyle='--', linewidth=1.5, label='Major Algo Line')

    # Formatting
    ax.set_title(f"NASDAQ (^IXIC) - Findiff Only (Algo Anchors)", fontsize=16)
    _format_axis(ax, df)

    output_filename = 'findiff_only.png'
    plt.savefig(output_filename)
    print(f"Chart saved to {output_filename}")
    plt.close(fig)

def plot_trendln_only(df):
    """
    Plots only the trendln support/resistance lines.
    """
    print("Generating trendln only chart...")
    fig = plt.figure(figsize=(15, 10))

    # Use trendln on High prices
    trendln.plot_support_resistance(
        df['High'],
        accuracy=4,
        window=125
    )

    ax = plt.gca()
    ax.set_title(f"NASDAQ (^IXIC) - Trendln Only", fontsize=16)
    _format_axis(ax, df)

    output_filename = 'trendln_only.png'
    plt.savefig(output_filename)
    print(f"Chart saved to {output_filename}")
    plt.close(fig)

def _format_axis(ax, df):
    ax.set_ylabel("Price")
    ax.set_xlabel("Date")
    ax.legend()
    ax.grid(True, alpha=0.3)

    num_points = len(df)
    tick_step = max(1, num_points // 10)
    tick_indices = range(0, num_points, tick_step)
    tick_labels = [df.index[i].strftime('%Y-%m-%d') for i in tick_indices]

    ax.set_xticks(tick_indices)
    ax.set_xticklabels(tick_labels, rotation=45)
    plt.tight_layout()

def main():
    # 1. Data Fetching
    ticker_symbol = '^IXIC'
    print(f"Fetching data for {ticker_symbol}...")
    df = yf.download(ticker_symbol, period='1y', interval='1d')

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    if 'High' not in df.columns or 'Volume' not in df.columns:
        print("Error: Missing High or Volume data.")
        return

    # 2. Generate Findiff Only Chart
    plot_findiff_only(df)

    # 3. Generate Trendln Only Chart
    plot_trendln_only(df)

if __name__ == "__main__":
    main()
