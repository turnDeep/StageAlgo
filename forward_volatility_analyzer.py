import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import sys

class ForwardVolatilityAnalyzer:
    def __init__(self, ticker='QQQ', start_date=None, end_date=None):
        self.ticker = ticker
        self.start_date = start_date if start_date else (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')
        self.end_date = end_date if end_date else datetime.now().strftime('%Y-%m-%d')
        self.df = None

    def fetch_data(self):
        """
        Fetches QQQ price data and Volatility Index data.
        Note: Since historical Nasdaq-100 Term Structure indices (^VXN3M, ^VXN6M) are
        often unavailable on public sources like Yahoo Finance, we use S&P 500 Volatility
        indices (^VIX, ^VIX3M) as a robust proxy for the broad market term structure.
        """
        print(f"Fetching data for {self.ticker}, ^VIX, and ^VIX3M...")

        tickers = [self.ticker, '^VIX', '^VIX3M']
        try:
            # Group by ticker to handle multi-index
            raw_data = yf.download(tickers, start=self.start_date, end=self.end_date, group_by='ticker')

            # Extract Close prices
            data = pd.DataFrame()
            data['Price'] = raw_data[self.ticker]['Close']
            data['IV1'] = raw_data['^VIX']['Close']   # 30-day IV
            data['IV2'] = raw_data['^VIX3M']['Close'] # 3-month IV (approx 90 days)

            # Forward fill missing data (holidays can differ slightly between equity and index markets)
            data = data.ffill().dropna()
            self.df = data
            print(f"Data fetched successfully. Records: {len(self.df)}")

        except Exception as e:
            print(f"Error fetching data: {e}")
            sys.exit(1)

    def calculate_forward_volatility(self):
        """
        Calculates Forward Volatility between T1 (30 days) and T2 (90 days).
        Formula: sigma_fwd = sqrt( (T2*sigma2^2 - T1*sigma1^2) / (T2 - T1) )
        """
        if self.df is None:
            return

        # Time periods in years
        t1 = 30 / 365.0
        t2 = 91 / 365.0 # VIX3M is approx 3 months

        # Convert VIX points to decimals (e.g., 20 -> 0.20)
        sigma1 = self.df['IV1'] / 100.0
        sigma2 = self.df['IV2'] / 100.0

        # Calculate Variance
        # V_total = sigma^2 * T
        v1 = (sigma1 ** 2) * t1
        v2 = (sigma2 ** 2) * t2

        # Forward Variance
        v_fwd = v2 - v1

        # Handle negative forward variance (extreme backwardation anomaly or noise)
        # In reality, forward variance should be positive. If negative, it implies extreme arbitrage opportunity or bad data.
        # We clip it to 0 or use absolute value for calculation stability, though 0 implies no volatility.
        v_fwd = v_fwd.clip(lower=0.000001)

        # Forward Volatility
        sigma_fwd = np.sqrt(v_fwd / (t2 - t1))

        # Convert back to percentage
        self.df['FwdVol'] = sigma_fwd * 100.0

        # Calculate Spread (Spot vs Forward)
        # Positive = Contango (Normal), Negative = Backwardation (Fear)
        self.df['VolSpread'] = self.df['FwdVol'] - self.df['IV1']

    def generate_signals(self):
        """
        Generates trading signals based on the logic provided.
        """
        if self.df is None:
            return

        # 1. Event/Risk Filter: Fwd Vol is exceptionally high
        # We use a Bollinger Band like approach: FwdVol > MA(20) + 2*STD(20)
        window = 20
        self.df['FwdVol_MA'] = self.df['FwdVol'].rolling(window=window).mean()
        self.df['FwdVol_STD'] = self.df['FwdVol'].rolling(window=window).std()
        self.df['Risk_Threshold'] = self.df['FwdVol_MA'] + 2 * self.df['FwdVol_STD']

        self.df['Signal_Risk_Avoid'] = self.df['FwdVol'] > self.df['Risk_Threshold']

        # 2. Term Structure Signal (Trend Reversal/Fear)
        # If Spot IV > Forward Vol (Inverted/Backwardation), Market is pricing immediate fear.
        # This is a bearish signal or "Exit Longs".
        self.df['Signal_Bearish_Regime'] = self.df['IV1'] > self.df['FwdVol']

        # 3. Breakout Confirmation
        # Simple Price Breakout: Price > 20-day High
        self.df['Price_20d_High'] = self.df['Price'].rolling(window=20).max()
        price_breakout = self.df['Price'] >= self.df['Price_20d_High']

        # Vol Confirmation: Fwd Vol Rising (10-day slope positive)
        fwd_vol_slope = self.df['FwdVol'].diff(5) > 0 # Simple check if higher than 5 days ago

        self.df['Signal_Strong_Buy'] = price_breakout & fwd_vol_slope & (~self.df['Signal_Bearish_Regime']) & (~self.df['Signal_Risk_Avoid'])

    def plot_analysis(self, output_filename='qqq_forward_volatility.png'):
        if self.df is None:
            return

        # Filter for plotting (remove initial NaN from rolling window)
        plot_df = self.df.dropna()

        fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True, gridspec_kw={'height_ratios': [2, 1, 1]})

        # --- Panel 1: Price and Buy Signals ---
        ax1 = axes[0]
        ax1.plot(plot_df.index, plot_df['Price'], label=f'{self.ticker} Price', color='black', alpha=0.7)

        # Highlight Strong Buy Signals
        buy_signals = plot_df[plot_df['Signal_Strong_Buy']]
        ax1.scatter(buy_signals.index, buy_signals['Price'], marker='^', color='green', s=100, label='Strong Breakout (Vol Confirmed)', zorder=5)

        # Highlight Risk Zones (Background)
        # We define risk zones where Risk_Avoid is True OR Bearish_Regime is True
        risk_zones = plot_df['Signal_Risk_Avoid'] | plot_df['Signal_Bearish_Regime']

        # Fill strictly high risk areas
        # Note: matplotlib fill_between works best with series, so we use where condition
        ax1.fill_between(plot_df.index, plot_df['Price'].min(), plot_df['Price'].max(),
                         where=plot_df['Signal_Bearish_Regime'], color='red', alpha=0.1, label='Bearish Regime (Backwardation)')

        ax1.set_title(f'{self.ticker} Price & Forward Volatility Signals', fontsize=14)
        ax1.set_ylabel('Price')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)

        # --- Panel 2: Volatility Term Structure ---
        ax2 = axes[1]
        ax2.plot(plot_df.index, plot_df['IV1'], label='Spot IV (VIX 30d)', color='blue', linewidth=1, alpha=0.6)
        ax2.plot(plot_df.index, plot_df['FwdVol'], label='Forward Vol (30d->90d)', color='orange', linewidth=2)
        ax2.plot(plot_df.index, plot_df['Risk_Threshold'], label='+2 Sigma Threshold', color='gray', linestyle='--', alpha=0.5)

        # Fill where Spot > Forward (Inversion)
        ax2.fill_between(plot_df.index, plot_df['IV1'], plot_df['FwdVol'],
                         where=(plot_df['IV1'] > plot_df['FwdVol']),
                         color='red', alpha=0.3, label='Inversion (Fear)')

        ax2.set_ylabel('Volatility (%)')
        ax2.set_title('Term Structure: Spot vs Forward Volatility', fontsize=12)
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)

        # --- Panel 3: Volatility Spread (Fwd - Spot) ---
        ax3 = axes[2]
        ax3.plot(plot_df.index, plot_df['VolSpread'], label='Vol Spread (Fwd - Spot)', color='purple')
        ax3.axhline(0, color='black', linestyle='--', alpha=0.5)

        # Highlight extremely high spread (complacency?) or low spread (fear)
        ax3.fill_between(plot_df.index, 0, plot_df['VolSpread'], where=(plot_df['VolSpread'] < 0), color='red', alpha=0.3)
        ax3.fill_between(plot_df.index, 0, plot_df['VolSpread'], where=(plot_df['VolSpread'] > 0), color='green', alpha=0.1)

        ax3.set_ylabel('Spread (pts)')
        ax3.set_title('Volatility Spread (Positive = Contango/Normal, Negative = Backwardation/Fear)', fontsize=12)
        ax3.legend(loc='upper left')
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_filename)
        print(f"Chart saved to {output_filename}")

if __name__ == "__main__":
    analyzer = ForwardVolatilityAnalyzer(ticker='QQQ')
    analyzer.fetch_data()
    analyzer.calculate_forward_volatility()
    analyzer.generate_signals()
    analyzer.plot_analysis()
