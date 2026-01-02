import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import os

# Add alpha_synthesis to path to import data_loader
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_loader import AlphaSynthesisDataLoader

class ZigZagPlotter:
    def __init__(self, ticker, threshold_pct=3.0):
        self.ticker = ticker
        self.threshold_pct = threshold_pct
        self.df = None
        self.pivots = [] # List of {'date': date, 'price': price, 'type': 'high'/'low'}

    def fetch_data(self):
        """Fetches historical data using AlphaSynthesisDataLoader."""
        print(f"Fetching data for {self.ticker}...")
        loader = AlphaSynthesisDataLoader()
        try:
            self.df, _ = loader.fetch_data(self.ticker)

            if self.df is None or self.df.empty:
                print(f"No data found for {self.ticker}")
                return False

            # Ensure we have a DatetimeIndex
            if not isinstance(self.df.index, pd.DatetimeIndex):
                self.df.index = pd.to_datetime(self.df.index)

            return True
        except Exception as e:
            print(f"Error fetching data: {e}")
            return False
        finally:
            loader.close()

    def calculate_zigzag(self):
        """
        Calculates ZigZag pivots based on percentage threshold.
        Algorithm:
        1. Track current trend (1 = Up, -1 = Down)
        2. Track last extreme price and index
        3. If price reverses by > threshold, confirm last extreme as pivot, switch trend.
        """
        if self.df is None or self.df.empty:
            return

        dates = self.df.index
        highs = self.df['High'].values
        lows = self.df['Low'].values
        closes = self.df['Close'].values

        # Initialize
        # We need a starting point. Let's assume neutral or start with first bar.

        # State variables
        current_trend = 0 # 0: Unknown, 1: Up, -1: Down
        last_pivot_idx = 0
        last_pivot_price = closes[0]

        # Temporary extreme tracking
        temp_extreme_idx = 0
        temp_extreme_price = closes[0]

        self.pivots = []

        # Start scanning
        for i in range(1, len(closes)):
            curr_high = highs[i]
            curr_low = lows[i]
            curr_close = closes[i]

            if current_trend == 0:
                # establish initial trend
                change_from_start = (curr_close - temp_extreme_price) / temp_extreme_price * 100
                if change_from_start > self.threshold_pct:
                    current_trend = 1 # Up
                    temp_extreme_idx = i
                    temp_extreme_price = curr_high
                    # The start was a low
                    self.pivots.append({'date': dates[0], 'price': lows[0], 'type': 'low', 'idx': 0})
                elif change_from_start < -self.threshold_pct:
                    current_trend = -1 # Down
                    temp_extreme_idx = i
                    temp_extreme_price = curr_low
                    # The start was a high
                    self.pivots.append({'date': dates[0], 'price': highs[0], 'type': 'high', 'idx': 0})

            elif current_trend == 1: # Uptrend
                if curr_high > temp_extreme_price:
                    # New high in uptrend, update extreme
                    temp_extreme_price = curr_high
                    temp_extreme_idx = i
                elif curr_low < temp_extreme_price * (1 - self.threshold_pct/100):
                    # Reversal detected!
                    # The previous extreme high is now a confirmed pivot
                    self.pivots.append({'date': dates[temp_extreme_idx], 'price': temp_extreme_price, 'type': 'high', 'idx': temp_extreme_idx})

                    # Switch to Downtrend
                    current_trend = -1
                    temp_extreme_price = curr_low
                    temp_extreme_idx = i

            elif current_trend == -1: # Downtrend
                if curr_low < temp_extreme_price:
                    # New low in downtrend, update extreme
                    temp_extreme_price = curr_low
                    temp_extreme_idx = i
                elif curr_high > temp_extreme_price * (1 + self.threshold_pct/100):
                    # Reversal detected!
                    # The previous extreme low is now a confirmed pivot
                    self.pivots.append({'date': dates[temp_extreme_idx], 'price': temp_extreme_price, 'type': 'low', 'idx': temp_extreme_idx})

                    # Switch to Uptrend
                    current_trend = 1
                    temp_extreme_price = curr_high
                    temp_extreme_idx = i

        # Add the last extreme as a pending pivot
        if current_trend == 1:
             self.pivots.append({'date': dates[temp_extreme_idx], 'price': temp_extreme_price, 'type': 'high', 'idx': temp_extreme_idx})
        elif current_trend == -1:
             self.pivots.append({'date': dates[temp_extreme_idx], 'price': temp_extreme_price, 'type': 'low', 'idx': temp_extreme_idx})

    def plot(self):
        """Plots the price and ZigZag lines."""
        if self.df is None or not self.pivots:
            print("No data or pivots to plot.")
            return

        plt.figure(figsize=(14, 8))

        # Plot Candles (Simplified as High/Low bars or just Close line for clarity with ZigZag)
        plt.plot(self.df.index, self.df['Close'], color='lightgray', label='Price (Close)', linewidth=1)

        # Plot ZigZag Lines
        pivot_dates = [p['date'] for p in self.pivots]
        pivot_prices = [p['price'] for p in self.pivots]

        plt.plot(pivot_dates, pivot_prices, color='blue', marker='o', linestyle='-', linewidth=1.5, markersize=4, label='ZigZag')

        # Annotate Contractions (High -> Low)
        for i in range(len(self.pivots) - 1):
            p1 = self.pivots[i]
            p2 = self.pivots[i+1]

            if p1['type'] == 'high' and p2['type'] == 'low':
                # This is a contraction/pullback
                change_pct = (p2['price'] - p1['price']) / p1['price'] * 100

                # Midpoint for text
                mid_date = p1['date'] + (p2['date'] - p1['date']) / 2
                mid_price = (p1['price'] + p2['price']) / 2

                # Annotate
                text = f"{change_pct:.1f}%"
                plt.annotate(text, xy=(mid_date, mid_price),
                             xytext=(0, -10), textcoords='offset points',
                             ha='center', va='top', color='red', weight='bold', fontsize=10,
                             arrowprops=dict(arrowstyle='-', color='red', alpha=0.3))

        plt.title(f"{self.ticker} - Price Structure & Contraction Waves (ZigZag {self.threshold_pct}%)")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True, which='both', linestyle='--', alpha=0.5)

        filename = f"{self.ticker}_zigzag.png"
        plt.savefig(filename)
        print(f"Plot saved to {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ZigZag Plotter for VCP Visualization")
    parser.add_argument("-t", "--ticker", type=str, required=True, help="Stock Ticker")
    parser.add_argument("--threshold", type=float, default=5.0, help="ZigZag Threshold % (default: 5.0)")
    args = parser.parse_args()

    plotter = ZigZagPlotter(args.ticker, threshold_pct=args.threshold)
    if plotter.fetch_data():
        plotter.calculate_zigzag()
        plotter.plot()
