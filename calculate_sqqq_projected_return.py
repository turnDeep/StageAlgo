import yfinance as yf
import pandas as pd
from datetime import time, timedelta

def calculate_sqqq_projected_return(period="60d", interval="5m"):
    print(f"--- Calculating Projected Annual Return for SQQQ Strategy (3x NQ Inverse) ---")

    # 1. Fetch Data
    nq_symbol = "NQ=F"
    try:
        nq_df = yf.download(nq_symbol, period=period, interval=interval, progress=False)
    except Exception as e:
        print(f"Error fetching data: {e}")
        return

    # Process Data
    if isinstance(nq_df.columns, pd.MultiIndex):
        nq_df.columns = nq_df.columns.get_level_values(0)
    if nq_df.index.tz is None:
        nq_df.index = nq_df.index.tz_localize("UTC")
    nq_df.index = nq_df.index.tz_convert("America/New_York")

    grouped = nq_df.groupby(nq_df.index.date)

    london_start = time(2, 0)
    london_end = time(8, 0)
    ny_start = time(8, 0)
    ny_end = time(16, 0)

    trades = []

    for date, day_df in grouped:
        london_data = day_df.between_time(london_start, london_end, inclusive="left")
        if london_data.empty:
            continue

        london_high = london_data['High'].max()
        london_low = london_data['Low'].min()
        midpoint = (london_high + london_low) / 2

        try:
            at_8am = day_df.between_time(time(8,0), time(8,5)).iloc[0]
            price_8am = at_8am['Open']
        except IndexError:
            continue

        if price_8am >= midpoint:
            continue

        # Short Trade (Entry at 08:00)
        entry = price_8am
        target = london_low
        stop = london_high

        trade_data = day_df.between_time(ny_start, ny_end)
        if trade_data.empty:
            continue

        exit_price = trade_data.iloc[-1]['Close'] # Default EOD
        bars = trade_data.iloc[1:]

        outcome = "Running"

        for i in range(len(bars)):
            bar = bars.iloc[i]
            if bar['High'] >= stop:
                exit_price = stop
                outcome = "Stop Hit"
                break
            if bar['Low'] <= target:
                exit_price = target
                outcome = "Target Hit"
                break

        # Calculate NQ Drop Percentage (Short Return)
        # Short Profit % = (Entry - Exit) / Entry
        # Note: If Entry > Exit, price fell (good for short). PnL > 0.
        nq_return_pct = ((entry - exit_price) / entry) * 100

        # SQQQ Return (3x Inverse Leveraged)
        # Simply 3x the NQ move
        sqqq_return_pct = nq_return_pct * 3

        trades.append({
            'date': date,
            'nq_return_pct': nq_return_pct,
            'sqqq_return_pct': sqqq_return_pct,
            'outcome': outcome
        })

    if not trades:
        print("No trades found.")
        return

    df = pd.DataFrame(trades)

    total_sqqq_return = df['sqqq_return_pct'].sum() # Assuming compounding not active intra-trade
    avg_sqqq_return = df['sqqq_return_pct'].mean()

    # Calculate Compounded Annual Return? Or Simple?
    # Usually "Asset Increase" implies compounding.
    # Let's calculate simple sum first, then compounded.

    # Simple Sum Annualized
    scaler = 252 / len(grouped) # approx 4.2x
    projected_simple_annual = total_sqqq_return * scaler

    # Compounded Annualized
    # Product of (1 + r)
    growth_factor = 1.0
    for r in df['sqqq_return_pct']:
        growth_factor *= (1 + (r / 100))

    # Scale growth factor to year
    # (Growth)^(Scaler) - 1
    projected_compounded_annual = (growth_factor ** scaler) - 1
    projected_compounded_pct = projected_compounded_annual * 100

    print(f"\n--- SQQQ Strategy Projected Returns (3x Leverage, Pre-Market Access) ---")
    print(f"Trades (60 Days): {len(df)}")
    print(f"Win Rate: {(len(df[df['sqqq_return_pct'] > 0]) / len(df)) * 100:.2f}%")
    print(f"Avg SQQQ Return per Trade: {avg_sqqq_return:.2f}%")
    print(f"Total SQQQ Return (60 Days, Simple Sum): {total_sqqq_return:.2f}%")
    print(f"Projected Annual Return (Simple Sum): {projected_simple_annual:.2f}%")
    print(f"Projected Annual Return (Compounded): {projected_compounded_pct:.2f}%")

if __name__ == "__main__":
    calculate_sqqq_projected_return()
