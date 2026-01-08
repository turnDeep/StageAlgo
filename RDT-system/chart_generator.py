import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
from data_fetcher import RDTDataFetcher
from indicators import RDTIndicators
import os

class RDTChartGenerator:
    def __init__(self):
        self.fetcher = RDTDataFetcher()

    def generate_chart(self, ticker, output_filename=None):
        print(f"Generating chart for {ticker}...")

        # 1. Fetch Data (2y to ensure enough for indicators)
        spy_df = self.fetcher.fetch_spy(period="2y")
        df = self.fetcher.fetch_single(ticker, period="2y")

        if df is None or spy_df is None:
            print(f"Error: Could not fetch data for {ticker} or SPY.")
            return

        # 2. Calculate Indicators
        # This adds 'RRS', 'Vol_SMA_20', etc.
        df = RDTIndicators.calculate_all(df, spy_df)

        # 3. Slice Data (Last 6 Months)
        # Ensure datetime index
        df.index = pd.to_datetime(df.index)
        spy_df.index = pd.to_datetime(spy_df.index)

        # Calculate start date for 6 months
        if len(df) > 0:
            end_date = df.index[-1]
            start_date = end_date - pd.DateOffset(months=6)
            plot_df = df.loc[start_date:].copy()
        else:
            print("Error: DataFrame is empty.")
            return

        if plot_df.empty:
            print("Error: Plot DataFrame is empty after slicing.")
            return

        # 4. Prepare Plots
        apds = []

        # --- Panel 0: Main (Candles) ---
        # SPY overlay removed per user request

        # --- Panel 1: RRS (Real Relative Strength) ---
        # RRS Line (Orange)
        # Calculate symmetric limits to center 0
        rrs_max = plot_df['RRS'].abs().max()
        if pd.isna(rrs_max) or rrs_max == 0:
            rrs_max = 1.0
        rrs_ylim = (-rrs_max * 1.1, rrs_max * 1.1)

        zero_line = pd.Series(0, index=plot_df.index)
        apds.append(mpf.make_addplot(plot_df['RRS'], panel=1, color='orange', width=1.5, ylabel='RRS', ylim=rrs_ylim))
        apds.append(mpf.make_addplot(zero_line, panel=1, color='gray', linestyle='--', width=0.8))

        # --- Panel 2: Volume + RVol Overlay ---
        # Overlay: RVol (Blue Line) on Volume Panel (2) with Secondary Y-Axis
        # We use secondary_y=True because RVol (ratio ~1.0) scale is different from Volume (millions).
        rvol_line = pd.Series(1.5, index=plot_df.index)
        apds.append(mpf.make_addplot(plot_df['RVol'], panel=2, color='blue', width=1.2, secondary_y=True, ylabel='RVol'))
        apds.append(mpf.make_addplot(rvol_line, panel=2, color='gray', linestyle='--', width=0.8, secondary_y=True))

        # 5. Styling
        mc = mpf.make_marketcolors(up='green', down='red', edge='inherit', wick='inherit', volume='inherit')
        s = mpf.make_mpf_style(marketcolors=mc, gridstyle=':', y_on_right=True, facecolor='white')

        # 6. Plotting
        if output_filename is None:
            output_filename = f"{ticker}_chart.png"

        # Omit title arg to avoid validation error for None
        fig, axes = mpf.plot(
            plot_df,
            type='candle',
            style=s,
            addplot=apds,
            volume=True,
            volume_panel=2,
            panel_ratios=(3, 1, 1),
            returnfig=True,
            figsize=(10, 8),
            scale_padding={'left': 0.1, 'top': 0.1, 'right': 1.0, 'bottom': 0.1},
            tight_layout=True
        )

        # Set Title on the Main Axis (Top Panel)
        axes[0].set_title(f'{ticker} - D1', loc='left', fontsize=12)

        # Enable left-side ticks for Main Panel (axes[0]) and RRS Panel (axes[2])
        # Note: mplfinance axes list order with 3 panels (Main, RRS, Volume)
        # axes[0] = Main Panel
        # axes[2] = Panel 1 (RRS)
        # axes[4] = Panel 2 (Volume)
        # We assume standard structure; if secondary axes exist, indices might shift,
        # but with y_on_right=True, the primary axes are usually the ones we want to enable 'labelleft' on.

        # Main Panel: Enable left ticks
        axes[0].tick_params(axis='y', labelleft=True)

        # RRS Panel: Disable left ticks and labels strictly (as per latest request)
        # We target axes[2] (Primary RRS) and axes[3] (Secondary RRS if present)
        # to remove any lingering grid numbers like 0.050, 0.025 etc.
        if len(axes) > 2:
            axes[2].tick_params(axis='y', which='both', left=False, labelleft=False)

        if len(axes) > 3:
            axes[3].tick_params(axis='y', which='both', left=False, labelleft=False)

        # Save
        fig.savefig(output_filename, bbox_inches='tight')
        print(f"Chart saved to {output_filename}")
        plt.close(fig)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--ticker", type=str, required=True, help="Ticker symbol")
    parser.add_argument("-o", "--output", type=str, help="Output filename")
    args = parser.parse_args()

    generator = RDTChartGenerator()
    generator.generate_chart(args.ticker, args.output)
