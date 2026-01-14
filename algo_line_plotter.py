import yfinance as yf
import numpy as np
import pandas as pd
from findiff import Diff
import matplotlib.pyplot as plt
import trendln

def get_line_end_index(p1_idx, p2_idx, price_data, tolerance=0.005):
    """
    Determines how far a line connecting p1 and p2 can extend without being cut.
    Returns the index where the line is cut, or len(price_data) if never cut.
    Returns -1 if invalid between p1 and p2.
    """
    x1, y1 = p1_idx, price_data[p1_idx]
    x2, y2 = p2_idx, price_data[p2_idx]

    if x2 == x1: return -1

    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1

    # Check strictly between p1 and p2
    for i in range(p1_idx + 1, p2_idx):
        line_val = slope * i + intercept
        # Strictly no cutting between anchors allowed (with tolerance)
        if price_data[i] > line_val * (1 + tolerance):
            return -1

    # Check extension beyond p2
    for i in range(p2_idx + 1, len(price_data)):
        line_val = slope * i + intercept
        if price_data[i] > line_val * (1 + tolerance):
            return i # Cut at index i

    return len(price_data) # Valid until end

def plot_findiff_only(df):
    """
    Plots only the custom Algo Anchors and lines detected using findiff.
    """
    print("Generating findiff only chart...")
    high_prices = df['High'].values
    volumes = df['Volume'].values
    dates = df.index

    # Define differential operators
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

    # Use ALL local maxima as potential anchors to maximize line detection
    # (Volume filter removed to ensure lines are drawn, can be re-enabled for stricter 'Algo' definition)
    for idx in peak_candidates_indices:
        if idx < len(second_deriv) and second_deriv[idx] < 0:
            algo_anchors.append(idx)

    fig, ax = plt.subplots(figsize=(15, 10))
    ax.plot(range(len(df)), df['High'], label='High Price', color='gray', alpha=0.5)

    # Plot anchors
    anchor_y = [high_prices[i] for i in algo_anchors]
    ax.scatter(algo_anchors, anchor_y, color='red', s=30, marker='^', label='Local Maxima (Findiff)', zorder=5)

    # Find and plot valid Algo Lines
    valid_lines_count = 0
    algo_anchors.sort()

    # Iterate through pairs.
    # To reduce complexity and noise, only connect points within a certain window?
    # No, let's try all forward pairs but limit drawing.

    for i in range(len(algo_anchors)):
        # Optimization: only check N forward points to emulate local scanning?
        # Or check all for long term lines. Let's check all but limit 'similar' lines.
        for j in range(i + 1, len(algo_anchors)):
            p1 = algo_anchors[i]
            p2 = algo_anchors[j]

            end_idx = get_line_end_index(p1, p2, high_prices, tolerance=0.005)

            if end_idx != -1:
                # Valid line at least between p1 and p2
                # Plot from p1 to end_idx
                x1, y1 = p1, high_prices[p1]
                x2, y2 = p2, high_prices[p2]
                slope = (y2 - y1) / (x2 - x1)
                intercept = y1 - slope * x1

                # Create x range for plotting
                # Plot slightly beyond p2 if end_idx > p2
                plot_end = end_idx if end_idx < len(df) else len(df) - 1

                # Only draw if the line has some length or significance
                # e.g., covers at least 5 days
                if plot_end - p1 > 5:
                    x_vals = np.array(range(p1, plot_end + 1))
                    y_vals = slope * x_vals + intercept

                    # Color coding:
                    # If it reaches the end of chart (active), make it Red/Solid
                    # If it was cut (historical), make it Orange/Dashed
                    if end_idx >= len(df) - 1:
                         ax.plot(x_vals, y_vals, color='red', linestyle='-', linewidth=1.5, alpha=0.8)
                    else:
                         ax.plot(x_vals, y_vals, color='orange', linestyle='--', linewidth=1, alpha=0.4)

                    valid_lines_count += 1

    print(f"Found and plotted {valid_lines_count} valid lines.")

    # Formatting
    ax.set_title(f"NASDAQ (^IXIC) - Findiff Only\n{len(algo_anchors)} Anchors, {valid_lines_count} Lines", fontsize=16)
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
    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], color='gray', alpha=0.5, label='Price'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='red', markersize=10, label='Anchor'),
        Line2D([0], [0], color='red', linestyle='-', label='Active Line'),
        Line2D([0], [0], color='orange', linestyle='--', label='Broken Line')
    ]
    ax.legend(handles=custom_lines)
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
