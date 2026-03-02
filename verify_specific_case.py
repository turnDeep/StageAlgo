import yfinance as yf
import pandas as pd
from datetime import time

def verify_case_study():
    print("--- Verifying Case Study: 2026-02-17 ---")

    # Symbols
    nq_symbol = "NQ=F"
    es_symbol = "ES=F"

    # Fetch Data for Feb 17, 2026
    start_date = "2026-02-17"
    end_date = "2026-02-18"

    try:
        nq_df = yf.download(nq_symbol, start=start_date, end=end_date, interval="5m", progress=False)
        es_df = yf.download(es_symbol, start=start_date, end=end_date, interval="5m", progress=False)
    except Exception as e:
        print(f"Error fetching data: {e}")
        return

    # Process NQ
    if isinstance(nq_df.columns, pd.MultiIndex):
        nq_df.columns = nq_df.columns.get_level_values(0)
    if nq_df.index.tz is None:
        nq_df.index = nq_df.index.tz_localize("UTC")
    nq_df.index = nq_df.index.tz_convert("America/New_York")

    # Process ES
    if isinstance(es_df.columns, pd.MultiIndex):
        es_df.columns = es_df.columns.get_level_values(0)
    if es_df.index.tz is None:
        es_df.index = es_df.index.tz_localize("UTC")
    es_df.index = es_df.index.tz_convert("America/New_York")

    # Define Times
    london_start = time(2, 0)
    london_end = time(8, 0)
    check_time = time(8, 0)
    manipulation_time_start = time(9, 30) # 09:30
    manipulation_time_end = time(10, 0)   # Short window for manipulation

    # Analyze NQ
    day_nq = nq_df[nq_df.index.date == pd.Timestamp("2026-02-17").date()]
    london_nq = day_nq.between_time(london_start, london_end, inclusive="left")

    if london_nq.empty:
        print("No London Data for NQ")
        return

    london_high_nq = london_nq['High'].max()
    london_low_nq = london_nq['Low'].min()
    midpoint_nq = (london_high_nq + london_low_nq) / 2

    try:
        price_8am_nq = day_nq.between_time(time(8,0), time(8,5)).iloc[0]['Open']
    except IndexError:
        print("No 8am candle for NQ")
        return

    print(f"NQ London High: {london_high_nq}")
    print(f"NQ London Low: {london_low_nq}")
    print(f"NQ Midpoint: {midpoint_nq}")
    print(f"NQ 8am Price: {price_8am_nq}")

    is_lower_half = price_8am_nq < midpoint_nq
    print(f"Condition Met (Price < Midpoint): {is_lower_half}")

    # Check Sweep at 9:30
    ny_start_nq = day_nq.between_time(time(9,30), time(10,0)) # The manipulation window mentioned
    min_930_nq = ny_start_nq['Low'].min()
    swept_nq = min_930_nq < london_low_nq
    print(f"NQ 9:30 Low: {min_930_nq}")
    print(f"NQ Swept London Low? {swept_nq}")

    # Check ES SMT
    day_es = es_df[es_df.index.date == pd.Timestamp("2026-02-17").date()]
    london_es = day_es.between_time(london_start, london_end, inclusive="left")

    if london_es.empty:
        print("No London Data for ES")
        return

    london_low_es = london_es['Low'].min()

    ny_start_es = day_es.between_time(time(9,30), time(10,0))
    min_930_es = ny_start_es['Low'].min()
    swept_es = min_930_es < london_low_es

    print(f"ES London Low: {london_low_es}")
    print(f"ES 9:30 Low: {min_930_es}")
    print(f"ES Swept London Low? {swept_es}")

    # SMT Divergence Logic
    # Bullish SMT: One sweeps low, other does not.
    if swept_nq != swept_es:
        print(f"SMT Divergence CONFIRMED: NQ Swept={swept_nq}, ES Swept={swept_es}")
    else:
        print(f"No SMT Divergence (Both Swept or Neither Swept): NQ={swept_nq}, ES={swept_es}")

if __name__ == "__main__":
    verify_case_study()
