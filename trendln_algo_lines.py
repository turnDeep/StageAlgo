import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

def run_analysis():
    print("Starting Pine Script Algo Line Analysis...")

    tickers = {
        '^GSPC': 'S&P 500',
        '^IXIC': 'NASDAQ'
    }
    periods = ['1y', '6mo', '3mo']

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    plt.subplots_adjust(hspace=0.4, wspace=0.2)

    row = 0
    for ticker_symbol, ticker_name in tickers.items():
        col = 0
        for period in periods:
            print(f"Processing {ticker_name} ({ticker_symbol}) for {period}...")

            try:
                data = yf.Ticker(ticker_symbol).history(period=period)
                if data.empty:
                    print(f"No data for {ticker_symbol} {period}")
                    continue

                h_close = data['Close']
                h_high = data['High']
                h_low = data['Low']

                ax = axes[row, col]

                # --- INLINED LOGIC FOR DUAL SERIES (Method 1: Pivot Span) ---

                # 1. High Pivots (Resistance)
                values_high = h_high.values
                high_pivots = []
                left, right, count = 5, 5, 5
                length = 150

                for i in range(left, len(values_high) - right):
                    window = values_high[i-left : i+right+1]
                    if values_high[i] == np.max(window):
                         high_pivots.append((i, values_high[i]))

                recent_highs = high_pivots[-count:] if len(high_pivots) > count else high_pivots
                high_line = None
                if len(recent_highs) >= 2:
                    far_idx, far_val = recent_highs[0]
                    near_idx, near_val = recent_highs[-1]
                    diff = near_idx - far_idx
                    if diff != 0:
                        slope = (near_val - far_val) / diff
                        intercept = far_val - slope * far_idx
                        x2 = len(values_high) - 1
                        x1 = x2 - (length - 1)
                        y1 = slope * x1 + intercept
                        y2 = slope * x2 + intercept
                        high_line = (x1, y1, x2, y2)

                # 2. Low Pivots (Support)
                values_low = h_low.values
                low_pivots = []
                for i in range(left, len(values_low) - right):
                    window = values_low[i-left : i+right+1]
                    if values_low[i] == np.min(window):
                         low_pivots.append((i, values_low[i]))

                recent_lows = low_pivots[-count:] if len(low_pivots) > count else low_pivots
                low_line = None
                if len(recent_lows) >= 2:
                    far_idx, far_val = recent_lows[0]
                    near_idx, near_val = recent_lows[-1]
                    diff = near_idx - far_idx
                    if diff != 0:
                        slope = (near_val - far_val) / diff
                        intercept = far_val - slope * far_idx
                        x2 = len(values_low) - 1
                        x1 = x2 - (length - 1)
                        y1 = slope * x1 + intercept
                        y2 = slope * x2 + intercept
                        low_line = (x1, y1, x2, y2)

                # Plotting
                x_indices = np.arange(len(h_close))

                # Plot Close Price
                ax.plot(x_indices, h_close.values, label='Close Price', color='black', alpha=0.7, linewidth=1.2)

                # Draw High Trend (Resistance)
                if high_line:
                    hx1, hy1, hx2, hy2 = high_line
                    ax.plot([hx1, hx2], [hy1, hy2], color='#ff7b00', linestyle='--', linewidth=2, label='Resistance')

                # Draw Low Trend (Support)
                if low_line:
                    lx1, ly1, lx2, ly2 = low_line
                    ax.plot([lx1, lx2], [ly1, ly2], color='#ff7b00', linestyle='--', linewidth=2, label='Support')

                # Fill area between lines
                if high_line and low_line:
                    ax.fill_between([high_line[0], high_line[2]],
                                    [high_line[1], high_line[3]],
                                    [low_line[1], low_line[3]],
                                    color='#ff7b00', alpha=0.1)

                ax.set_title(f"{ticker_name} ({period})", fontsize=10, fontweight='bold')
                ax.grid(True, which='major', linestyle=':', alpha=0.6)

                # Ticks (Map integer index to Date)
                tick_indices = np.linspace(0, len(h_close)-1, 5, dtype=int)
                tick_labels = [h_close.index[i].strftime('%Y-%m-%d') for i in tick_indices]
                ax.set_xticks(tick_indices)
                ax.set_xticklabels(tick_labels, rotation=20, ha='right', fontsize=8)

                ax.legend(loc='best', fontsize='x-small')

            except Exception as e:
                print(f"Error processing {ticker_symbol} {period}: {e}")
                import traceback
                traceback.print_exc()

            col += 1
        row += 1

    print("Saving chart to algo_lines_chart.png...")
    plt.tight_layout()
    plt.savefig("algo_lines_chart.png")
    print("Done.")

if __name__ == "__main__":
    run_analysis()
