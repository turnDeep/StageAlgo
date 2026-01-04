import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
import argparse
import sys
import os
from scipy.signal import argrelextrema
from datetime import timedelta

# Add alpha_synthesis to path to import data_loader
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_loader import AlphaSynthesisDataLoader

class ZigZagPlotter:
    def __init__(self, ticker, order=5):
        self.ticker = ticker
        self.order = order
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

    def calculate_zigzag_scipy(self):
        """
        Calculates ZigZag pivots using scipy.signal.argrelextrema.
        """
        if self.df is None or self.df.empty:
            return

        # 1. Find local maxs and mins
        highs = self.df['High'].values
        lows = self.df['Low'].values
        dates = self.df.index

        # Find indexes of local extrema
        # order=N means strictly greater than N neighbors on each side
        high_idxs = argrelextrema(highs, np.greater, order=self.order)[0]
        low_idxs = argrelextrema(lows, np.less, order=self.order)[0]

        # Create a combined list of candidates
        candidates = []
        for idx in high_idxs:
            candidates.append({'idx': idx, 'date': dates[idx], 'price': highs[idx], 'type': 'high'})
        for idx in low_idxs:
            candidates.append({'idx': idx, 'date': dates[idx], 'price': lows[idx], 'type': 'low'})

        # Sort by index (time)
        candidates.sort(key=lambda x: x['idx'])

        if not candidates:
            self.pivots = []
            return

        # 2. Filter for Alternating High/Low (ZigZag Logic)
        final_pivots = []
        if not candidates:
            return

        # Simple stack-based filtering to ensure alternating high/low
        stack = [candidates[0]]

        for current in candidates[1:]:
            last = stack[-1]

            if last['type'] == current['type']:
                # Same type, keep the more extreme one
                if last['type'] == 'high':
                    if current['price'] > last['price']:
                        stack.pop()
                        stack.append(current)
                else: # low
                    if current['price'] < last['price']:
                        stack.pop()
                        stack.append(current)
            else:
                # Different type (High -> Low or Low -> High)
                stack.append(current)

        self.pivots = stack

    def plot(self):
        """Plots using mplfinance."""
        if self.df is None or not self.pivots:
            print("No data or pivots to plot.")
            return

        # Pre-calc 50-day SMA on full dataframe BEFORE slicing
        self.df['VolumeSMA50'] = self.df['Volume'].rolling(window=50).mean()

        # Filter data for display (last 1 year)
        one_year_ago = self.df.index[-1] - timedelta(days=365)
        plot_df = self.df.loc[one_year_ago:].copy()

        if plot_df.empty:
             plot_df = self.df

        # Filter and Map Pivots for mplfinance
        # mplfinance removes gaps, so we need to map dates to integer indices (0 to N-1)
        start_date = plot_df.index[0]
        display_pivots = [p for p in self.pivots if p['date'] >= start_date]

        # Add prior pivot to connect line
        prior_pivots = [p for p in self.pivots if p['date'] < start_date]
        if prior_pivots:
            display_pivots.insert(0, prior_pivots[-1])

        # Map dates to row numbers in plot_df
        # Create a mapping dictionary
        date_to_idx = {date: i for i, date in enumerate(plot_df.index)}

        zigzag_indices = []
        zigzag_prices = []

        annotates = [] # list of (idx, price, text)

        for i in range(len(display_pivots) - 1):
            p1 = display_pivots[i]
            p2 = display_pivots[i+1]

            # Map p1
            if p1['date'] in date_to_idx:
                idx1 = date_to_idx[p1['date']]
            elif p1['date'] < start_date:
                idx1 = 0
            else:
                continue # Should not happen if filtered

            # Map p2
            if p2['date'] in date_to_idx:
                idx2 = date_to_idx[p2['date']]
            else:
                continue

            # Store line points
            if not zigzag_indices or zigzag_indices[-1] != idx1:
                zigzag_indices.append(idx1)
                zigzag_prices.append(p1['price'])

            zigzag_indices.append(idx2)
            zigzag_prices.append(p2['price'])

            # Contraction Annotation
            if p1['type'] == 'high' and p2['type'] == 'low':
                 mid_idx = (idx1 + idx2) / 2
                 mid_price = (p1['price'] + p2['price']) / 2
                 change_pct = (p2['price'] - p1['price']) / p1['price'] * 100
                 text = f"{change_pct:.1f}%"
                 annotates.append((mid_idx, mid_price, text))

        # Add labels for pivots
        pivot_labels = []
        for p in display_pivots:
            if p['date'] in date_to_idx:
                idx = date_to_idx[p['date']]
                date_str = p['date'].strftime('%m-%d')
                pivot_labels.append((idx, p['price'], date_str))

        # Custom Style
        mc = mpf.make_marketcolors(up='#2ca02c', down='#d62728',
                                   edge={'up':'#2ca02c', 'down':'#d62728'},
                                   wick={'up':'#2ca02c', 'down':'#d62728'},
                                   volume={'up':'#2ca02c', 'down':'#d62728'},
                                   ohlc='black')

        s = mpf.make_mpf_style(marketcolors=mc, gridstyle='--', gridaxis='both')

        filename = f"{self.ticker}_zigzag.png"

        # Plot
        fig, axlist = mpf.plot(plot_df, type='candle', volume=True, style=s,
                               title=f"{self.ticker} - Price Structure & VCP (Last 1 Year)",
                               ylabel='Price', ylabel_lower='Volume',
                               returnfig=True, figsize=(14, 10),
                               datetime_format='%m/%d', xrotation=0)

        ax_main = axlist[0]
        # With volume=True, axlist usually: [Main, MainTwin, Volume, VolumeTwin] (size 4)
        # Or [Main, Volume] (size 2) if twins are not created.
        # Typically mpf creates 2 axes. Let's use index 2 (if available) or -2.
        # But wait, `returnfig=True` returns (fig, axlist).
        # Let's target the known volume axis.
        # If len(axlist) > 2, usually axlist[-2] is the volume axis (before volume twin).
        # If len(axlist) == 2, axlist[1] is volume.
        # Safe bet: axlist[-2] if len >= 4, else axlist[-1]?
        # Actually, simpler: `mpf.plot` with `volume=True` typically returns 2 axes in the list if no other panels.
        # Let's assume axlist[-2] for standard "Main + Volume" setup with twins potentially existing.
        # If twins don't exist, axlist might be just 2 elements.

        ax_vol = None
        if len(axlist) >= 2:
             # Typically axlist[0] = price, axlist[2] = volume (if twins exist).
             # If no twins, axlist[0] = price, axlist[1] = volume.
             # Let's iterate to find the one with the correct ylim? No.
             # The standard return is a list of axes.
             # Let's try grabbing the one that is NOT the main one.
             ax_vol = axlist[2] if len(axlist) > 2 else axlist[1]

        # Overlay ZigZag Line
        ax_main.plot(zigzag_indices, zigzag_prices, color='blue', marker='o', linestyle='-', linewidth=1.5, markersize=4, label='ZigZag')

        # Overlay Annotations (Contraction %)
        for (x, y, text) in annotates:
            ax_main.annotate(text, xy=(x, y),
                             xytext=(0, -15), textcoords='offset points',
                             ha='center', va='top', color='blue', weight='bold', fontsize=11,
                             arrowprops=dict(arrowstyle='-', color='blue', alpha=0.3))

        # Overlay Pivot Date Labels
        for (x, y, text) in pivot_labels:
            ax_main.text(x, y, text, fontsize=9, ha='right', va='bottom', color='black', fontweight='bold')

        ax_main.legend(loc='upper left')

        # --- Annotate Relative Volume on Last Bar ---
        last_idx = len(plot_df) - 1
        last_vol = plot_df['Volume'].iloc[-1]
        last_sma = plot_df['VolumeSMA50'].iloc[-1]

        if pd.notna(last_sma) and last_sma > 0 and ax_vol:
            vol_pct = (last_vol / last_sma) * 100
            vol_text = f"{vol_pct:.0f}%"

            # Place text above the bar
            ax_vol.text(last_idx, last_vol, vol_text,
                        ha='center', va='bottom', fontsize=10, fontweight='bold', color='black')

        # Save
        plt.savefig(filename)
        print(f"Plot saved to {filename}")
        plt.close(fig) # Close to free memory

        print(f"\nDetected Pivots for {self.ticker} (Last 1 Year):")
        for p in display_pivots:
             if p['date'] >= start_date:
                print(f"{p['date'].strftime('%Y-%m-%d')} : {p['type'].upper()} @ {p['price']:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ZigZag Plotter using mplfinance")
    parser.add_argument("-t", "--ticker", type=str, required=True, help="Stock Ticker")
    parser.add_argument("--order", type=int, default=3, help="Order for argrelextrema (default: 3)")
    args = parser.parse_args()

    plotter = ZigZagPlotter(args.ticker, order=args.order)
    if plotter.fetch_data():
        plotter.calculate_zigzag_scipy()
        plotter.plot()
