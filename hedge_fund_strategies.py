import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict

# Apply style
plt.style.use('bmh')

class HedgeFundStrategies:
    def __init__(self, ticker: str = "QQQ", start_date: str = "2010-01-01"):
        """
        Initialize with ticker and date range.
        Using QQQ as the primary example as requested.
        """
        self.ticker = ticker
        self.start_date = start_date
        self.data = self._fetch_data()

    def _fetch_data(self) -> pd.DataFrame:
        """Fetch OHLCV data from Yahoo Finance."""
        print(f"Fetching data for {self.ticker}...")
        df = yf.download(self.ticker, start=self.start_date, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
             df.columns = df.columns.get_level_values(0)

        # Calculate basic indicators
        df['Returns'] = df['Close'].pct_change()
        df['RSI'] = self._calculate_rsi(df['Close'], 14)
        return df.dropna()

    def _calculate_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def run_empirical_kelly(self, initial_capital: float = 100000, simulations: int = 10000):
        """
        Method 1: Empirical Kelly Criterion with Monte Carlo.
        Strategy: Buy when RSI < 30 (Oversold), Sell when RSI > 70 (Overbought).
        """
        print("\n--- Method 1: Empirical Kelly Analysis ---")
        df = self.data.copy()

        # Simple Strategy Logic
        df['Signal'] = 0
        df.loc[df['RSI'] < 30, 'Signal'] = 1  # Buy
        df.loc[df['RSI'] > 70, 'Signal'] = -1 # Sell

        # Generate Trade Log
        trades = []
        position = 0
        entry_price = 0

        for i in range(len(df)):
            price = df['Close'].iloc[i]
            signal = df['Signal'].iloc[i]

            if position == 0 and signal == 1:
                position = 1
                entry_price = price
            elif position == 1 and signal == -1:
                # Close Long
                ret = (price - entry_price) / entry_price
                trades.append(ret)
                position = 0

        if not trades:
            print("No trades generated with this strategy.")
            return

        trade_returns = np.array(trades)
        win_rate = np.sum(trade_returns > 0) / len(trade_returns)
        avg_win = np.mean(trade_returns[trade_returns > 0]) if np.any(trade_returns > 0) else 0
        avg_loss = np.mean(trade_returns[trade_returns < 0]) if np.any(trade_returns < 0) else 0

        print(f"Total Trades: {len(trades)}")
        print(f"Win Rate: {win_rate:.2%}")
        print(f"Avg Win: {avg_win:.2%}, Avg Loss: {avg_loss:.2%}")

        # Theoretical Kelly
        # f* = (p * b - q) / b
        # where b = avg_win / abs(avg_loss)
        if avg_loss == 0:
            kelly = 0 # Prevent division by zero
        else:
            b = avg_win / abs(avg_loss)
            p = win_rate
            q = 1 - p
            kelly = (p * b - q) / b

        print(f"Theoretical Full Kelly: {kelly:.2%}")

        # Monte Carlo Simulation for Drawdowns
        max_drawdowns = []

        for _ in range(simulations):
            # Resample trades with replacement
            sim_trades = np.random.choice(trade_returns, size=len(trade_returns), replace=True)
            equity_curve = np.cumprod(1 + sim_trades * kelly) # Using Full Kelly for stress test

            # Calculate Max Drawdown
            peak = np.maximum.accumulate(equity_curve)
            drawdown = (equity_curve - peak) / peak
            max_drawdowns.append(drawdown.min())

        max_drawdowns = np.array(max_drawdowns)
        dd_95 = np.percentile(max_drawdowns, 5) # 5th percentile (e.g., -0.40)

        print(f"Monte Carlo 95% Worst Drawdown (at Full Kelly): {dd_95:.2%}")

        # Risk Adjustment
        # Target Drawdown Limit (e.g., 20%)
        target_dd = -0.20
        adjustment_factor = target_dd / dd_95 if dd_95 < 0 else 0
        safe_kelly = kelly * adjustment_factor

        print(f"Adjusted 'Safe' Kelly (Target MaxDD 20%): {safe_kelly:.2%}")

        return trade_returns, max_drawdowns

    def run_calibration_analysis(self, future_days: int = 5):
        """
        Method 2: Calibration Surface Analysis.
        Analyze RSI Deciles vs Future Win Rate.
        Does 'Extreme Oversold' (Strong Signal) actually mean 'High Probability Reversal'?
        """
        print("\n--- Method 2: Calibration Analysis ---")
        df = self.data.copy()

        # Calculate Future Returns (N days ahead)
        df['Future_Return'] = df['Close'].shift(-future_days) / df['Close'] - 1
        df['Win'] = (df['Future_Return'] > 0).astype(int)

        # Bin RSI into Deciles
        df['RSI_Bin'] = pd.cut(df['RSI'], bins=range(0, 110, 10), labels=range(0, 100, 10))

        # Group by Bin
        calibration = df.groupby('RSI_Bin')[['Win', 'Future_Return']].agg(['mean', 'count'])
        calibration.columns = ['Win_Rate', 'Count', 'Avg_Return', 'Count2']
        calibration = calibration.drop(columns=['Count2'])

        print(f"{future_days}-Day Forward Returns by RSI Decile:")
        print(calibration)

        # Check for Longshot Bias / Signal Quality
        # Ideally, Low RSI (0-30) should have High Win Rate (Mean Reversion)
        # If Low RSI has Low Win Rate, it's a "Value Trap" or "Momentum Crash"
        return calibration

    def run_maker_taker_simulation(self, threshold_rsi: int = 30):
        """
        Method 3: Maker vs Taker Profitability.
        Scenario: We want to buy the dip when RSI < threshold.
        Taker: Buy Market at Next Open.
        Maker: Buy Limit at Next Open * 0.995 (0.5% discount).
        """
        print("\n--- Method 3: Maker vs Taker Simulation ---")
        df = self.data.copy()

        # Identify Setup Days (Close RSI < Threshold)
        setup_days = df[df['RSI'] < threshold_rsi].index

        taker_results = []
        maker_results = []

        limit_discount = 0.005 # 0.5% discount
        exit_days = 5 # Hold for 5 days fixed for simplicity

        for date in setup_days:
            idx = df.index.get_loc(date)
            if idx + exit_days >= len(df):
                continue

            # Next day (Execution day)
            next_day = df.iloc[idx + 1]
            exit_day = df.iloc[idx + exit_days]

            # Taker Execution (Buy Open)
            taker_entry = next_day['Open']
            taker_return = (exit_day['Close'] - taker_entry) / taker_entry
            taker_results.append(taker_return)

            # Maker Execution (Buy Limit)
            limit_price = next_day['Open'] * (1 - limit_discount)

            # Check if Low price hit the limit
            if next_day['Low'] <= limit_price:
                # Filled
                maker_return = (exit_day['Close'] - limit_price) / limit_price
                maker_results.append(maker_return)
            else:
                # Not Filled (Opportunity Cost)
                maker_results.append(0.0)

        if not taker_results:
            print("No setups found.")
            return

        taker_avg = np.mean(taker_results)
        maker_avg = np.mean(maker_results)
        maker_fill_rate = np.sum(np.array(maker_results) != 0) / len(maker_results)

        print(f"Strategy: Buy when RSI < {threshold_rsi}, Hold 5 Days")
        print(f"Taker (Market Buy Open) Avg Return: {taker_avg:.4%}")
        print(f"Maker (Limit Buy -0.5%) Avg Return: {maker_avg:.4%}")
        print(f"Maker Fill Rate: {maker_fill_rate:.2%}")

        if maker_avg > taker_avg:
            print("-> MAKER (Providing Liquidity) yielded better risk-adjusted returns.")
        else:
            print("-> TAKER (Taking Liquidity) yielded better returns (Aggressive entry paid off).")

if __name__ == "__main__":
    strategy = HedgeFundStrategies(ticker="QQQ", start_date="2010-01-01")

    # 1. Empirical Kelly
    strategy.run_empirical_kelly()

    # 2. Calibration
    strategy.run_calibration_analysis()

    # 3. Maker vs Taker
    strategy.run_maker_taker_simulation()
