import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
import argparse
import sys
import os
from scipy.signal import argrelextrema
from datetime import timedelta
import csv

# Add alpha_synthesis to path to import data_loader
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_loader import AlphaSynthesisDataLoader
from indicators import check_trend_template

class ZigZagPlotter:
    def __init__(self, ticker, order=5):
        self.ticker = ticker
        self.order = order
        self.df = None
        self.pivots = [] # List of {'date': date, 'price': price, 'type': 'high'/'low'}
        self.avwap_series = None
        self.vcp_metrics = {}

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

    def calculate_anchored_vwap_252_high(self):
        """
        Calculates AVWAP anchored to the highest high of the last 252 days.
        Uses OHLC4 as input price.
        """
        if self.df is None or len(self.df) < 252:
            return

        df = self.df.copy()

        # 1. OHLC4 Calculation
        average_price = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
        pv = average_price * df['Volume']

        # 2. Identify Anchor (Highest High in last 252 days relative to end of data)
        # Note: The report says "Highest high of past 252 days".
        # If we want a static chart for the current state, we find the max high in the last 252 rows.
        last_252 = df.tail(252)
        anchor_idx = last_252['High'].idxmax()

        # 3. Calculate AVWAP from anchor
        # Create a mask for data from anchor onwards
        mask = df.index >= anchor_idx

        cum_pv = pv[mask].cumsum()
        cum_vol = df['Volume'][mask].cumsum()

        avwap = cum_pv / cum_vol

        # Reindex to match original df, filling pre-anchor with NaN
        self.avwap_series = avwap.reindex(df.index)

        # Store metric
        self.vcp_metrics['avwap_current'] = self.avwap_series.iloc[-1]
        self.vcp_metrics['price_vs_avwap'] = (df['Close'].iloc[-1] - self.avwap_series.iloc[-1]) / self.avwap_series.iloc[-1]
        self.vcp_metrics['anchor_date'] = anchor_idx.strftime('%Y-%m-%d')

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
        high_idxs = argrelextrema(highs, np.greater, order=self.order)[0]
        low_idxs = argrelextrema(lows, np.less, order=self.order)[0]

        candidates = []
        for idx in high_idxs:
            candidates.append({'idx': idx, 'date': dates[idx], 'price': highs[idx], 'type': 'high'})
        for idx in low_idxs:
            candidates.append({'idx': idx, 'date': dates[idx], 'price': lows[idx], 'type': 'low'})

        candidates.sort(key=lambda x: x['idx'])

        if not candidates:
            self.pivots = []
            return

        # 2. Filter for Alternating High/Low (ZigZag Logic)
        stack = [candidates[0]]

        for current in candidates[1:]:
            last = stack[-1]

            if last['type'] == current['type']:
                if last['type'] == 'high':
                    if current['price'] > last['price']:
                        stack.pop()
                        stack.append(current)
                else:
                    if current['price'] < last['price']:
                        stack.pop()
                        stack.append(current)
            else:
                stack.append(current)

        self.pivots = stack

    def analyze_vcp_logic(self):
        """
        Analyzes VCP characteristics: Trend, Contraction, Tightness, Dry Up.
        """
        if self.df is None:
            return

        # 1. Trend Template
        trend_pass = check_trend_template(self.df)
        self.vcp_metrics['trend_template'] = trend_pass

        # 2. Contraction Analysis (Wave Depths)
        # We look at the last few High -> Low waves in pivots
        # Extract pairs of High -> Low
        contractions = []
        pivots_rev = list(reversed(self.pivots))

        # Find High-Low pairs from most recent back
        for i in range(len(pivots_rev) - 1):
            p2 = pivots_rev[i]   # More recent
            p1 = pivots_rev[i+1] # Older

            # We want High (p1) -> Low (p2)
            if p1['type'] == 'high' and p2['type'] == 'low':
                depth = (p2['price'] - p1['price']) / p1['price']
                contractions.append(abs(depth)) # Store as positive magnitude

            if len(contractions) >= 4: # Analyze last 3-4 contractions
                break

        # Reverse to chronological order (Oldest -> Newest)
        contractions = list(reversed(contractions))
        self.vcp_metrics['contractions'] = [round(c * 100, 2) for c in contractions]

        # Check if contractions are decreasing (at least vaguely)
        # Strict VCP: c1 > c2 > c3.
        # We'll allow some tolerance or just check if the last one is small.
        is_contracting = False
        if len(contractions) >= 2:
            # Check if the last contraction is smaller than the max of previous ones
            if contractions[-1] < max(contractions[:-1]) and contractions[-1] < 0.10: # Last one < 10%
                is_contracting = True
        elif len(contractions) == 1:
             if contractions[0] < 0.15: # Single base?
                 is_contracting = True

        self.vcp_metrics['is_contracting'] = is_contracting

        # 3. Tightness (Last 10 days)
        # (High - Low) / Close rolling mean
        df_sub = self.df.tail(20)
        hl_range = (df_sub['High'] - df_sub['Low']) / df_sub['Close']
        recent_volatility = hl_range.rolling(window=10).mean().iloc[-1]
        is_tight = recent_volatility < 0.04 # 4% threshold from report

        self.vcp_metrics['tightness_val'] = round(recent_volatility * 100, 2)
        self.vcp_metrics['is_tight'] = is_tight

        # 4. Volume Dry Up
        vol_sma50 = self.df['Volume'].rolling(window=50).mean().iloc[-1]
        curr_vol = self.df['Volume'].iloc[-1]
        is_dry_up = curr_vol < (vol_sma50 * 0.7) # 70% threshold

        self.vcp_metrics['vol_vs_sma'] = round(curr_vol / vol_sma50, 2) if vol_sma50 > 0 else 0
        self.vcp_metrics['is_dry_up'] = is_dry_up

        # Overall VCP Score/Status
        # Base criteria: Trend + Tightness + DryUp (Contraction is harder to be strict on automation)
        self.vcp_metrics['vcp_qualified'] = (trend_pass and is_tight and is_dry_up)


    def export_to_csv(self, filename=None):
        if not filename:
            filename = f"{self.ticker}_vcp_metrics.csv"

        print(f"Exporting metrics to {filename}...")
        try:
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Metric', 'Value'])
                for k, v in self.vcp_metrics.items():
                    writer.writerow([k, v])
        except Exception as e:
            print(f"Error exporting CSV: {e}")

    def plot(self):
        """Plots using mplfinance."""
        if self.df is None or not self.pivots:
            print("No data or pivots to plot.")
            return

        self.df['VolumeSMA50'] = self.df['Volume'].rolling(window=50).mean()

        # Filter last 1 year
        one_year_ago = self.df.index[-1] - timedelta(days=365)
        plot_df = self.df.loc[one_year_ago:].copy()

        if plot_df.empty:
             plot_df = self.df

        # --- Prepare Plot ---
        # Map dates to indices
        date_to_idx = {date: i for i, date in enumerate(plot_df.index)}

        # ZigZag Lines
        zigzag_indices = []
        zigzag_prices = []
        annotates = []

        display_pivots = [p for p in self.pivots if p['date'] >= plot_df.index[0]]
        # Add prior connector
        prior_pivots = [p for p in self.pivots if p['date'] < plot_df.index[0]]
        if prior_pivots:
            display_pivots.insert(0, prior_pivots[-1])

        for i in range(len(display_pivots) - 1):
            p1 = display_pivots[i]
            p2 = display_pivots[i+1]

            idx1 = date_to_idx.get(p1['date'], 0 if p1['date'] < plot_df.index[0] else None)
            idx2 = date_to_idx.get(p2['date'])

            if idx1 is None or idx2 is None: continue

            if not zigzag_indices or zigzag_indices[-1] != idx1:
                zigzag_indices.append(idx1)
                zigzag_prices.append(p1['price'])
            zigzag_indices.append(idx2)
            zigzag_prices.append(p2['price'])

            # Contraction Text
            if p1['type'] == 'high' and p2['type'] == 'low':
                 mid_idx = (idx1 + idx2) / 2
                 change_pct = (p2['price'] - p1['price']) / p1['price'] * 100
                 text = f"{change_pct:.1f}%"
                 annotates.append((mid_idx, (p1['price']+p2['price'])/2, text))

        # AVWAP Line
        avwap_line = []
        if self.avwap_series is not None:
            # Realign avwap series to plot_df
            avwap_subset = self.avwap_series.loc[plot_df.index]
            avwap_line = avwap_subset.values

        # Style
        mc = mpf.make_marketcolors(up='#2ca02c', down='#d62728',
                                   edge={'up':'#2ca02c', 'down':'#d62728'},
                                   wick={'up':'#2ca02c', 'down':'#d62728'},
                                   volume={'up':'#2ca02c', 'down':'#d62728'},
                                   ohlc='black')
        s = mpf.make_mpf_style(marketcolors=mc, gridstyle='--', gridaxis='both')

        # Additional Plots: AVWAP, ZigZag
        # mpf.make_addplot expects data same length as plot_df
        add_plots = []

        # AVWAP
        if len(avwap_line) > 0:
            add_plots.append(mpf.make_addplot(avwap_line, color='purple', width=1.5, linestyle='-'))

        # Metrics Box Text
        info_text = (
            f"Trend Template: {'PASS' if self.vcp_metrics.get('trend_template') else 'FAIL'}\n"
            f"Contractions: {self.vcp_metrics.get('contractions', [])}\n"
            f"Tightness ({self.vcp_metrics.get('tightness_val')}%): {'YES' if self.vcp_metrics.get('is_tight') else 'NO'}\n"
            f"Vol DryUp ({self.vcp_metrics.get('vol_vs_sma')}x): {'YES' if self.vcp_metrics.get('is_dry_up') else 'NO'}\n"
            f"VCP Qualified: {'YES' if self.vcp_metrics.get('vcp_qualified') else 'NO'}"
        )

        filename = f"{self.ticker}_zigzag.png"

        # Plot
        fig, axlist = mpf.plot(plot_df, type='candle', volume=True, style=s,
                               title=f"{self.ticker} - Price Structure & VCP (Anchor: {self.vcp_metrics.get('anchor_date', 'N/A')})",
                               ylabel='Price', ylabel_lower='Volume',
                               addplot=add_plots,
                               returnfig=True, figsize=(14, 10),
                               datetime_format='%m/%d', xrotation=0)

        ax_main = axlist[0]

        # Overlay ZigZag manually (since it's point-to-point, not series)
        ax_main.plot(zigzag_indices, zigzag_prices, color='blue', marker='o', linestyle='-', linewidth=1.0, markersize=3, alpha=0.7)

        for (x, y, text) in annotates:
            ax_main.annotate(text, xy=(x, y), xytext=(0, -10), textcoords='offset points',
                             ha='center', color='blue', fontsize=9)

        # Add Metrics Box
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        ax_main.text(0.02, 0.95, info_text, transform=ax_main.transAxes, fontsize=10,
                     verticalalignment='top', bbox=props)

        # Add AVWAP Legend
        ax_main.text(0.02, 0.80, "Purple Line: Anchored VWAP (252d High)", transform=ax_main.transAxes,
                     fontsize=9, color='purple', verticalalignment='top')

        plt.savefig(filename)
        print(f"Plot saved to {filename}")
        plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ZigZag VCP Plotter")
    parser.add_argument("-t", "--ticker", type=str, required=True, help="Stock Ticker")
    parser.add_argument("--order", type=int, default=5, help="Order for argrelextrema")
    args = parser.parse_args()

    plotter = ZigZagPlotter(args.ticker, order=args.order)
    if plotter.fetch_data():
        plotter.calculate_anchored_vwap_252_high()
        plotter.calculate_zigzag_scipy()
        plotter.analyze_vcp_logic()
        plotter.plot()
        plotter.export_to_csv()
