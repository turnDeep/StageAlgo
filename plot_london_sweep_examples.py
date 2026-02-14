import yfinance as yf
import pandas as pd
import mplfinance as mpf
from datetime import time
import os

def plot_london_sweep_examples(num_examples=5, period="60d", interval="5m"):
    print(f"\n--- Generating {num_examples} London Sweep Example Charts ({interval}, {period}) ---")

    # 1. Fetch Data
    ticker = "NQ=F"
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
    except Exception as e:
        print(f"Error fetching data: {e}")
        return

    if df.empty:
        print("No data fetched.")
        return

    # MultiIndex handling
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Timezone conversion
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert("America/New_York")

    # 2. Find Candidates
    candidates = []

    grouped = df.groupby(df.index.date)

    london_start = time(2, 0)
    london_end = time(8, 0)
    ny_start = time(8, 0)
    ny_end = time(16, 0)

    # Iterate in reverse order to get most recent first
    dates = sorted(list(grouped.groups.keys()), reverse=True)

    for date in dates:
        day_df = grouped.get_group(date)

        # Check sufficient data for the day (02:00 to 16:00)
        day_start_ts = day_df.index[0].replace(hour=2, minute=0)
        day_end_ts = day_df.index[0].replace(hour=16, minute=0)

        # Slice for the trading day view
        day_view = day_df.between_time(time(2,0), time(16,0))
        if day_view.empty:
            continue

        # London Session
        london_data = day_df.between_time(london_start, london_end, inclusive="left")
        if london_data.empty:
            continue

        london_high = london_data['High'].max()
        london_low = london_data['Low'].min()
        midpoint = (london_high + london_low) / 2

        # 08:00 Price check
        try:
            at_8am = day_df.between_time(time(8,0), time(8,5)).iloc[0]
            price_8am = at_8am['Open']
        except IndexError:
            continue

        if price_8am < midpoint:
            # Check Sweep
            ny_data = day_df.between_time(ny_start, ny_end, inclusive="left")
            if ny_data.empty:
                continue

            ny_low = ny_data['Low'].min()

            if ny_low < london_low:
                # Calculate Bounce (Reversal)
                # Find sweep time
                sweep_idx = ny_data[ny_data['Low'] < london_low].index[0]
                post_sweep = ny_data.loc[sweep_idx:]

                if post_sweep.empty:
                    continue

                day_low = post_sweep['Low'].min()
                # Find bounce high after the low
                low_time = post_sweep['Low'].idxmin()
                reversal_data = post_sweep.loc[low_time:]
                reversal_high = reversal_data['High'].max()

                bounce = reversal_high - day_low

                candidates.append({
                    'date': date,
                    'df': day_view,
                    'london_high': london_high,
                    'london_low': london_low,
                    'sweep_time': sweep_idx,
                    'bounce': bounce
                })

    print(f"Found {len(candidates)} successful sweep days.")

    # 3. Plot top N candidates
    # Sort candidates by date (descending) is default, but maybe prioritize 'bounce' size?
    # User asked for "examples". Recent ones are best.
    # Let's take the first N (most recent)
    selected = candidates[:num_examples]

    if not os.path.exists("charts"):
        os.makedirs("charts")

    for i, cand in enumerate(selected):
        date_str = cand['date'].strftime("%Y-%m-%d")
        filename = f"charts/london_sweep_example_{i+1}_{date_str}.png"

        print(f"Generating chart for {date_str} (Bounce: {cand['bounce']:.2f} pts)...")

        # Prepare plot data
        plot_df = cand['df']

        # Add Horizontal Lines for London High/Low
        hlines = [cand['london_high'], cand['london_low']]

        # Add Vertical Line for 08:00 (NY Open)
        vlines = [cand['date'].strftime("%Y-%m-%d 08:00")]

        # Customize style
        s = mpf.make_mpf_style(base_mpf_style='yahoo', rc={'figure.figsize': (12, 8)})

        # Add Title
        title = f"NQ=F London Sweep & Reverse Setup ({date_str})\nLondon Low Sweep -> {cand['bounce']:.0f}pt Bounce"

        # Plot
        # Highlights: London Session background?
        # Ideally, shade 02:00-08:00. MPF `vspan` doesn't exist directly but `fill_between` might work or just rely on the VLine.
        # Let's stick to simple H-lines and V-line.

        mpf.plot(
            plot_df,
            type='candle',
            style=s,
            title=title,
            hlines=dict(hlines=hlines, colors=['g', 'r'], linestyle='-.', linewidths=1.5),
            vlines=dict(vlines=vlines, colors=['b'], linestyle=':', linewidths=1),
            savefig=filename,
            volume=False
        )
        print(f"Saved: {filename}")

if __name__ == "__main__":
    plot_london_sweep_examples(num_examples=5, period="60d", interval="5m")
