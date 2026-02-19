import yfinance as yf
import pandas as pd
from datetime import time, timedelta

def calculate_annual_returns(period="60d", interval="5m"):
    print(f"--- Calculating Annual Returns (NQ=F, 60d) with 0.132% Fee ---")

    # 1. Fetch Data
    nq_symbol = "NQ=F"
    try:
        nq_df = yf.download(nq_symbol, period=period, interval=interval, progress=False)
    except Exception as e:
        print(f"Error fetching data: {e}")
        return

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

        # Short Trade
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

        gross_pnl_pts = entry - exit_price

        # Calculate Commission (0.132% of Notional Value)
        # NQ Contract Notional = Price * 20
        # If the user meant "0.132% of Trade Value" (CFD style):
        # Fee = (Entry + Exit) * 0.00132
        # BUT maybe they meant "0.132% of Margin"? Or "0.132% per trade"?
        # Standard futures are flat fee. 0.132% suggests CFD or Prop Firm or Crypto.
        # Let's assume total round trip cost rate is 0.132% of the notional value traded.
        # Or per side? Let's assume PER SIDE is standard for % based fees.
        # So Cost = (Entry * 0.00132) + (Exit * 0.00132)
        # Wait, if it's 0.132% *Notional*, that's huge.
        # Example: 24,000 * 0.00132 = 31.68 points PER SIDE. Total ~63 points.
        # If it's 0.132% *Total Round Trip*, that's ~31.68 points.
        # Given "Average Profit" is ~9 points, this fee wipes out the strategy completely.

        # Let's calculate both "Points Cost" and "Dollar Cost".
        # Assuming NQ multiplier $20.

        fee_rate = 0.00132
        # Assume fee is applied to the Notional Value (Price) in Points term
        # Fee in Points = (Entry * fee_rate) + (Exit * fee_rate) ?
        # Or just Entry * fee_rate?
        # Let's assume Total Round Trip cost is 0.132% of the Entry Price (approx).
        # Cost = Entry * 0.00132

        cost_pts = entry * fee_rate

        net_pnl_pts = gross_pnl_pts - cost_pts

        trades.append({
            'date': date,
            'gross_pnl': gross_pnl_pts,
            'cost': cost_pts,
            'net_pnl': net_pnl_pts,
            'entry': entry,
            'exit': exit_price
        })

    if not trades:
        print("No trades found.")
        return

    df = pd.DataFrame(trades)

    total_gross = df['gross_pnl'].sum()
    total_cost = df['cost'].sum()
    total_net = df['net_pnl'].sum()

    avg_gross = df['gross_pnl'].mean()
    avg_cost = df['cost'].mean()
    avg_net = df['net_pnl'].mean()

    win_rate = (len(df[df['gross_pnl'] > 0]) / len(df)) * 100

    print(f"\n--- Strategy Performance (60 Days) ---")
    print(f"Trades: {len(df)}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Avg Gross Profit: {avg_gross:.2f} pts")
    print(f"Avg Commission Cost (0.132%): {avg_cost:.2f} pts")
    print(f"Avg Net Profit: {avg_net:.2f} pts")
    print(f"Total Net PnL: {total_net:.2f} pts")

    # Projection (Annualized)
    # 60 days approx -> 365 days? Or trading days?
    # Let's use simple scaling: Annual = Total * (252 / 40) ?
    # 60 calendar days is approx 40 trading days. Let's count unique dates in df.
    num_days = len(grouped) # This is days with data, not just trade days
    # Let's use (252 / num_days_in_period) scaler
    scaler = 252 / num_days

    projected_net_pts = total_net * scaler
    projected_gross_pts = total_gross * scaler

    # Dollar Value (NQ = $20/pt)
    projected_net_usd = projected_net_pts * 20

    # ROI Calculation
    # Assumption: Account Size $30,000 (Conservative for 1 NQ)
    account_size = 30000
    roi = (projected_net_usd / account_size) * 100

    print(f"\n--- Annual Projection ---")
    print(f"Projected Net Points: {projected_net_pts:.2f}")
    print(f"Projected Net USD (at $20/pt): ${projected_net_usd:.2f}")
    print(f"Projected ROI (on $30k account): {roi:.2f}%")

    # Flat Fee Comparison (e.g. $5 round trip = 0.25 pts)
    flat_fee_pts = 0.25
    total_net_flat = total_gross - (len(df) * flat_fee_pts)
    proj_net_flat_pts = total_net_flat * scaler
    proj_net_flat_usd = proj_net_flat_pts * 20
    roi_flat = (proj_net_flat_usd / account_size) * 100

    print(f"\n--- Comparison with Flat Fee ($5/trade) ---")
    print(f"Projected Net USD: ${proj_net_flat_usd:.2f}")
    print(f"Projected ROI: {roi_flat:.2f}%")

if __name__ == "__main__":
    calculate_annual_returns()
