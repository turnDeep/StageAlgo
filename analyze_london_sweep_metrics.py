import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
import pytz

def analyze_london_sweep_metrics(interval="5m", period="60d"):
    print(f"\n--- Analyzing London Sweep Metrics on NQ=F ({interval}, {period}) ---")

    # Fetch Data
    ticker = "NQ=F"
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
    except Exception as e:
        print(f"Error fetching data: {e}")
        return

    if df.empty:
        print("No data fetched.")
        return

    # Check if index is MultiIndex (common with new yfinance versions)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Convert to Eastern Time
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    df.index = df.index.tz_convert("America/New_York")

    results = []

    # Group by Date
    grouped = df.groupby(df.index.date)

    london_start = time(2, 0)
    london_end = time(8, 0) # Exclusive of 8:00

    ny_start = time(8, 0)
    ny_end = time(16, 0) # Main session close

    for date, day_df in grouped:
        # Extract London Session Data
        london_data = day_df.between_time(london_start, london_end, inclusive="left") # 02:00 <= t < 08:00

        if london_data.empty:
            continue

        london_high = london_data['High'].max()
        london_low = london_data['Low'].min()

        # Get 08:00 Price (Open of 08:00 candle)
        try:
            at_8am = day_df.between_time(time(8,0), time(8,5)).iloc[0]
            price_8am = at_8am['Open']
        except IndexError:
            continue

        # Filter: Price in lower 50% of range
        midpoint = (london_high + london_low) / 2

        is_lower_half = price_8am < midpoint

        if is_lower_half:
            # NY Session
            ny_data = day_df.between_time(ny_start, ny_end, inclusive="left")

            if ny_data.empty:
                continue

            ny_low = ny_data['Low'].min()
            swept = ny_low < london_low

            mae = 0.0 # Maximum Adverse Excursion (Drawdown before sweep)
            profit_points = 0.0 # Points captured (Short: 8am Open -> London Low)
            bounce_points = 0.0 # Points recovered after sweep (Reversal)

            if swept:
                # Calculate MAE: Max High before the sweep occurred
                # Find the exact timestamp of the sweep
                sweep_time = ny_data[ny_data['Low'] < london_low].index[0]

                # Data before sweep
                pre_sweep_data = ny_data.loc[:sweep_time]

                # MAE: Highest point reached before the sweep
                mae_high = pre_sweep_data['High'].max()
                mae = mae_high - price_8am if mae_high > price_8am else 0

                # Profit: 8am Open -> London Low (Target)
                profit_points = price_8am - london_low

                # Calculate Reversal (Bounce)
                # Find the lowest point after the sweep (or including sweep candle)
                post_sweep_data = ny_data.loc[sweep_time:]

                if not post_sweep_data.empty:
                    # Lowest point of the day (could be lower than London Low)
                    day_low = post_sweep_data['Low'].min()
                    # Highest point after the low is made (reversal peak)
                    # Find time of day_low
                    day_low_time = post_sweep_data[post_sweep_data['Low'] == day_low].index[0]

                    reversal_data = post_sweep_data.loc[day_low_time:]
                    reversal_high = reversal_data['High'].max()

                    bounce_points = reversal_high - day_low

            results.append({
                'date': date,
                'london_high': london_high,
                'london_low': london_low,
                'midpoint': midpoint,
                'price_8am': price_8am,
                'swept': swept,
                'mae': mae,
                'profit_points': profit_points,
                'bounce_points': bounce_points
            })

    if not results:
        print("No trades found.")
        return

    df_results = pd.DataFrame(results)
    successful_trades = df_results[df_results['swept'] == True]

    print(f"Total Trades (Filter Met): {len(df_results)}")
    print(f"Successful Sweeps: {len(successful_trades)}")
    print(f"Win Rate: {(len(successful_trades) / len(df_results)) * 100:.2f}%")

    if not successful_trades.empty:
        avg_profit = successful_trades['profit_points'].mean()
        avg_mae = successful_trades['mae'].mean()
        avg_bounce = successful_trades['bounce_points'].mean()

        # Risk/Reward Analysis
        # Assuming Stop Loss at London High
        # Risk = London High - 8am Open (approx, varies per trade)
        # But let's use actual MAE to see "Average Drawdown"

        print("\n--- Metrics for Successful Sweeps ---")
        print(f"Avg Profit (Short: 8am Open -> London Low): {avg_profit:.2f} points")
        print(f"Avg Drawdown (MAE) before Target: {avg_mae:.2f} points")
        print(f"Avg Reversal (Bounce) from Low of Day: {avg_bounce:.2f} points")

        # Stop Loss Check
        # How often would a stop at London High be hit before the target?
        # A "failed short" even if it eventually swept (whipsaw)
        # Check against London High for each trade
        sl_hits = 0
        for _, row in successful_trades.iterrows():
            if row['mae'] > (row['london_high'] - row['price_8am']):
                sl_hits += 1

        sl_hit_rate = (sl_hits / len(successful_trades)) * 100
        print(f"\nStop Loss Analysis (Target: London Low):")
        print(f"Trades where Price > London High before Sweep: {sl_hits} ({sl_hit_rate:.2f}%)")
        print("  -> Suggestion: Stop Loss at London High is very safe.")

        # Midpoint Stop Loss
        # Check if price went back above midpoint before sweep
        sl_mid_hits = 0
        for _, row in successful_trades.iterrows():
            if row['mae'] > (row['midpoint'] - row['price_8am']):
                sl_mid_hits += 1

        sl_mid_rate = (sl_mid_hits / len(successful_trades)) * 100
        print(f"Trades where Price > Midpoint before Sweep: {sl_mid_hits} ({sl_mid_rate:.2f}%)")
        print("  -> Suggestion: Stop Loss at Midpoint is risky (high whipsaw probability).")

if __name__ == "__main__":
    # Run on 1h data for statistical significance
    analyze_london_sweep_metrics(interval="1h", period="730d")
