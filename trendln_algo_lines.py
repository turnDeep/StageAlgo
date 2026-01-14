import trendln
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

def plot_algo_lines_custom(ax, hist, support_data, resistance_data, title, ticker_label, max_lines=5):
    """
    Custom plotting function for trendln results, with filtering to reduce clutter.
    Adapted from user provided snippet.
    """

    # 1. Plot the base price chart
    # Use index numbers for X-axis to ensure linearity with trendln's logic
    # We will format ticks later or just let it be for now (or map back to dates if needed for display)
    # The user snippet used hist.index (dates). However, trendln is linear in integer steps.
    # To match the user's request exactly, I will use hist.index as they did.
    ax.plot(hist.index, hist.values, label=f'{ticker_label} Price', color='black', alpha=0.6, linewidth=1.2)

    # Unpack data
    minimaIdxs, pmin, mintrend, minwindows = support_data
    maximaIdxs, pmax, maxtrend, maxwindows = resistance_data

    # Plot pivots
    if len(minimaIdxs) > 0:
        ax.scatter(hist.index[minimaIdxs], hist.values[minimaIdxs], color='green', s=15, alpha=0.7, zorder=3, label='Local Minima')
    if len(maximaIdxs) > 0:
        ax.scatter(hist.index[maximaIdxs], hist.values[maximaIdxs], color='red', s=15, alpha=0.7, zorder=3, label='Local Maxima')

    # Internal function to draw lines
    def draw_lines_from_data(trend_list, color, line_label_prefix):
        plotted_label = False

        # FILTERING: Take only the top N lines
        # trendln sorts by score/error by default, so taking the first N is the best strategy.
        filtered_list = trend_list[:max_lines]

        for i, line_data in enumerate(filtered_list):
            points_indices = line_data[0]
            params = line_data[1]

            slope = params[0]
            intercept = params[1]

            if len(points_indices) < 2:
                continue

            start_idx = np.min(points_indices)
            end_idx = np.max(points_indices)

            # Calculate Y values using linear equation in index space
            x_vals = np.array([start_idx, end_idx])
            y_vals = slope * x_vals + intercept

            # Map X indices to Dates for plotting
            if end_idx < len(hist):
                # Ensure indices are within bounds
                try:
                    x_dates = [hist.index[start_idx], hist.index[end_idx]]

                    lbl = f'{line_label_prefix} Trend' if not plotted_label else "_nolegend_"

                    ax.plot(x_dates, y_vals, color=color, linestyle='--', linewidth=1.5, label=lbl, zorder=2)
                    plotted_label = True
                except IndexError:
                    continue

    # 2. Draw Support Lines (Green)
    draw_lines_from_data(mintrend, 'green', 'Support')

    # 3. Draw Resistance Lines (Red)
    draw_lines_from_data(maxtrend, 'red', 'Resistance')

    # Formatting
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.grid(True, which='major', linestyle=':', alpha=0.6)

    # Only show legend if there are labels
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc='best', fontsize='x-small')

    # Date formatting
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')

def run_analysis():
    print("Starting Algo Line Analysis...")

    tickers = {
        '^GSPC': 'S&P 500',
        '^IXIC': 'NASDAQ'
    }
    periods = ['1y', '6mo', '3mo']

    # Create 2x3 grid
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    plt.subplots_adjust(hspace=0.4, wspace=0.2)

    # Constants
    ACCURACY_PARAM = 8 # As requested/found in memory

    row = 0
    for ticker_symbol, ticker_name in tickers.items():
        col = 0
        for period in periods:
            print(f"Processing {ticker_name} ({ticker_symbol}) for {period}...")

            try:
                # Fetch data
                data = yf.Ticker(ticker_symbol).history(period=period)
                if data.empty:
                    print(f"No data for {ticker_symbol} {period}")
                    continue

                h_close = data['Close']

                # Calculate Support/Resistance
                # trendln.calc_support_resistance returns ((minimaIdxs, pmin, mintrend, minwindows), (maximaIdxs, pmax, maxtrend, maxwindows))
                # Using accuracy=8 as recommended for individual stocks, but also safe/good for indices to prevent odd-order errors.
                # Explicitly naming arguments to avoid confusion
                support_res, resistance_res = trendln.calc_support_resistance(h_close, accuracy=ACCURACY_PARAM)

                # Plot
                ax = axes[row, col]
                plot_title = f"{ticker_name} ({period})"

                # Pass data to custom plotter
                # Note: calc_support_resistance returns tuple of tuples
                plot_algo_lines_custom(
                    ax,
                    h_close,
                    support_res,    # (minimaIdxs, pmin, mintrend, minwindows)
                    resistance_res, # (maximaIdxs, pmax, maxtrend, maxwindows)
                    plot_title,
                    ticker_symbol,
                    max_lines=3 # LIMIT TO TOP 3 LINES TO REDUCE CLUTTER
                )

            except Exception as e:
                print(f"Error processing {ticker_symbol} {period}: {e}")
                import traceback
                traceback.print_exc()

            col += 1
        row += 1

    print("Saving chart to algo_lines_chart_cleaned.png...")
    plt.tight_layout()
    plt.savefig("algo_lines_chart_cleaned.png")
    print("Done.")

if __name__ == "__main__":
    run_analysis()
