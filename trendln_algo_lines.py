import trendln
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

def plot_algo_lines_custom(ax, hist, support_data, resistance_data, title, ticker_label):
    """
    Plots Algo Lines on the given Matplotlib Axes.
    """

    # 1. Plot Price
    ax.plot(hist.index, hist.values, label=f'{ticker_label} Price', color='black', alpha=0.6, linewidth=1.2)

    # Unpack data
    # format: (minimaIdxs, pmin, mintrend, minwindows)
    minimaIdxs, pmin, mintrend, minwindows = support_data
    maximaIdxs, pmax, maxtrend, maxwindows = resistance_data

    # Plot pivots
    if len(minimaIdxs) > 0:
        ax.scatter(hist.index[minimaIdxs], hist.values[minimaIdxs], color='green', s=15, alpha=0.7, zorder=3, label='Local Minima')
    if len(maximaIdxs) > 0:
        ax.scatter(hist.index[maximaIdxs], hist.values[maximaIdxs], color='red', s=15, alpha=0.7, zorder=3, label='Local Maxima')

    # Helper to draw lines
    def draw_lines_from_data(trend_list, color, line_label_prefix):
        plotted_label = False

        for i, line_data in enumerate(trend_list):
            # line_data structure: (points_indices, (slope, intercept, error))
            points_indices = line_data[0]
            params = line_data[1]

            slope = params[0]
            intercept = params[1]

            if len(points_indices) < 2:
                continue

            start_idx = np.min(points_indices)
            end_idx = np.max(points_indices)

            # x values (indices)
            x_vals = np.array([start_idx, end_idx])

            # y values
            y_vals = slope * x_vals + intercept

            if end_idx < len(hist):
                x_dates = [hist.index[start_idx], hist.index[end_idx]]

                lbl = f'{line_label_prefix} Trend' if not plotted_label else "_nolegend_"

                ax.plot(x_dates, y_vals, color=color, linestyle='--', linewidth=1.5, label=lbl, zorder=2)
                plotted_label = True

    # Draw Support
    draw_lines_from_data(mintrend, 'green', 'Support')

    # Draw Resistance
    draw_lines_from_data(maxtrend, 'red', 'Resistance')

    # Formatting
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.grid(True, which='major', linestyle=':', alpha=0.6)
    # ax.legend(loc='best', fontsize='small') # Can be crowded

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    # plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')

def main():
    tickers = {
        'S&P 500': '^GSPC',
        'NASDAQ': '^IXIC'
    }
    periods = ['1y', '6mo', '3mo']

    # 2 rows (tickers), 3 columns (periods)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Flatten axes if needed but here we can access by [row, col]

    for i, (name, ticker) in enumerate(tickers.items()):
        print(f"Processing {name} ({ticker})...")
        for j, period in enumerate(periods):
            print(f"  Fetching {period} data...")
            try:
                hist = yf.Ticker(ticker).history(period=period)
                if hist.empty:
                    print(f"    No data found for {ticker} {period}")
                    continue

                # Use 'Close' for calculation
                h = hist['Close']

                # Calculate
                print(f"    Calculating trendln (accuracy=8)...")
                # accuracy=8 as requested/suggested for better handling.

                # trendln returns ((minimaIdxs, pmin, mintrend, minwindows), (maximaIdxs, pmax, maxtrend, maxwindows))
                # Note: calc_support_resistance returns (support, resistance) tuple.
                support, resistance = trendln.calc_support_resistance(h, accuracy=8)

                # Plot
                ax = axes[i, j]
                title = f"{name} - {period}"
                plot_algo_lines_custom(ax, h, support, resistance, title, name)

                # Rotate dates for better visibility
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')

            except Exception as e:
                print(f"    Error processing {ticker} {period}: {e}")
                # Plot empty or text error
                axes[i, j].text(0.5, 0.5, f"Error: {e}", ha='center', va='center')

    plt.tight_layout()
    output_file = 'algo_lines_chart.png'
    plt.savefig(output_file)
    print(f"Chart saved to {output_file}")

if __name__ == "__main__":
    main()
