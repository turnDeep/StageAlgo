import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
import pytz

def verify_london_sweep(interval="5m", period="60d"):
    print(f"\n--- Verifying London Sweep Setup on NQ=F ({interval}, {period}) ---")

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
    london_end = time(8, 0) # Exclusive of 8:00 for range calculation usually, but setup says "02:00-08:00 forms range"
    # User says: "London time: (EST 02:00-08:00) forms the morning range."
    # So we take High/Low from 02:00 up to (but not including) 08:00?
    # Or inclusive? Usually "range formed by session" implies the session's high/low.
    # Let's assume 02:00:00 to 07:59:59.

    ny_start = time(8, 0)
    ny_end = time(16, 0) # Or 17:00. Let's use 16:00 (close) or 17:00 (futures close). Futures often trade almost 24h but settle at 16:15 or 17:00. Let's use 16:00 for core NY session.

    for date, day_df in grouped:
        # Extract London Session Data
        london_data = day_df.between_time(london_start, london_end, inclusive="left") # 02:00 <= t < 08:00

        if london_data.empty:
            continue

        london_high = london_data['High'].max()
        london_low = london_data['Low'].min()

        # Get 08:00 Price (Open of 08:00 candle)
        # We need the candle exactly at 08:00 or close to it
        try:
            at_8am = day_df.between_time(time(8,0), time(8,5)).iloc[0]
            price_8am = at_8am['Open'] # User says "08:00 filter: check where price is". Open of 8am candle is best proxy.
        except IndexError:
            # 08:00 candle missing
            continue

        # Filter: Price in lower 50% of range
        midpoint = (london_high + london_low) / 2

        is_lower_half = price_8am < midpoint

        if is_lower_half:
            # Trigger!
            # Check for sweep in NY Session
            # NY Session: 08:00 onwards. Let's look until end of day or 16:00.
            # The "sweep" can happen anytime after 08:00.
            ny_data = day_df.between_time(ny_start, ny_end, inclusive="left")

            if ny_data.empty:
                continue

            ny_low = ny_data['Low'].min()

            swept = ny_low < london_low

            results.append({
                'date': date,
                'london_high': london_high,
                'london_low': london_low,
                'midpoint': midpoint,
                'price_8am': price_8am,
                'swept': swept
            })

    # Calculate Stats
    if not results:
        print("No trades found matching the criteria.")
        return

    df_results = pd.DataFrame(results)
    total_trades = len(df_results)
    wins = df_results['swept'].sum()
    win_rate = (wins / total_trades) * 100

    print(f"Total Days Analyzed: {len(grouped)}")
    print(f"Total Trades (Filter Met): {total_trades}")
    print(f"London Low Sweeps: {wins}")
    print(f"Win Rate: {win_rate:.2f}%")

    # Context
    print(f"Data Range: {df.index[0]} to {df.index[-1]}")

if __name__ == "__main__":
    verify_london_sweep(interval="5m", period="60d")
    # Attempting 1h for longer history (approx)
    print("\n" + "="*30 + "\n")
    verify_london_sweep(interval="1h", period="730d")
