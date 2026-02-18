import yfinance as yf
import pandas as pd
from datetime import time, timedelta

def verify_smt_divergence():
    print("--- Verifying SMT Divergence Case Study: 2026-02-17 ---")

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

    # Process Data
    for df in [nq_df, es_df]:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        df.index = df.index.tz_convert("America/New_York")

    # Define Times
    london_start = time(2, 0)
    london_end = time(8, 0)

    pre_open_start = time(8, 0)
    pre_open_end = time(9, 30)

    manipulation_start = time(9, 30)
    manipulation_end = time(10, 0)

    # Filter Day Data
    day_nq = nq_df[nq_df.index.date == pd.Timestamp("2026-02-17").date()]
    day_es = es_df[es_df.index.date == pd.Timestamp("2026-02-17").date()]

    if day_nq.empty or day_es.empty:
        print("Missing data for 2026-02-17")
        return

    # --- Step 1: Analyze London Low ---
    london_nq = day_nq.between_time(london_start, london_end, inclusive="left")
    london_es = day_es.between_time(london_start, london_end, inclusive="left")

    if london_nq.empty or london_es.empty:
        print("Missing London Session Data")
        return

    london_low_nq = london_nq['Low'].min()
    london_low_es = london_es['Low'].min()

    print(f"\n[London Low (02:00-08:00)]")
    print(f"NQ London Low: {london_low_nq}")
    print(f"ES London Low: {london_low_es}")

    # --- Step 2: Analyze Pre-Open Low (08:00-09:30) ---
    pre_open_nq = day_nq.between_time(pre_open_start, pre_open_end, inclusive="left")
    pre_open_es = day_es.between_time(pre_open_start, pre_open_end, inclusive="left")

    pre_low_nq = pre_open_nq['Low'].min()
    pre_low_es = pre_open_es['Low'].min()

    print(f"\n[Pre-Open Low (08:00-09:30)]")
    print(f"NQ Pre-Open Low: {pre_low_nq}")
    print(f"ES Pre-Open Low: {pre_low_es}")

    # --- Step 3: Analyze 9:30 Manipulation Low ---
    manip_nq = day_nq.between_time(manipulation_start, manipulation_end)
    manip_es = day_es.between_time(manipulation_start, manipulation_end)

    if manip_nq.empty or manip_es.empty:
        print("Missing 9:30 Session Data")
        return

    manip_low_nq = manip_nq['Low'].min()
    manip_low_es = manip_es['Low'].min()

    print(f"\n[9:30 Manipulation Low (09:30-10:00)]")
    print(f"NQ 9:30 Low: {manip_low_nq}")
    print(f"ES 9:30 Low: {manip_low_es}")

    # --- Step 4: Compare Sweeps (SMT Detection) ---
    print("\n[SMT Analysis: London Low vs 9:30 Low]")

    nq_swept_london = manip_low_nq < london_low_nq
    es_swept_london = manip_low_es < london_low_es

    print(f"Did NQ sweep London Low ({london_low_nq})? -> {nq_swept_london} (Low: {manip_low_nq})")
    print(f"Did ES sweep London Low ({london_low_es})? -> {es_swept_london} (Low: {manip_low_es})")

    if nq_swept_london and not es_swept_london:
        print("\n*** BULLISH SMT DETECTED! ***")
        print("Reason: NQ swept the London Low (Weakness), but ES held above its London Low (Strength).")
        print("This confirms the bullish divergence (SMT) at the lows.")

    elif es_swept_london and not nq_swept_london:
        print("\n*** BULLISH SMT DETECTED! ***")
        print("Reason: ES swept the London Low (Weakness), but NQ held above its London Low (Strength).")
        print("This confirms the bullish divergence (SMT) at the lows.")

    else:
        print("\nNo SMT Divergence at London Low (Both swept or neither swept).")

    # --- Step 5: Compare Pre-Open Low vs 9:30 Low (Common short-term SMT) ---
    print("\n[SMT Analysis: Pre-Open Low vs 9:30 Low]")

    nq_swept_pre = manip_low_nq < pre_low_nq
    es_swept_pre = manip_low_es < pre_low_es

    print(f"Did NQ sweep Pre-Open Low ({pre_low_nq})? -> {nq_swept_pre} (Low: {manip_low_nq})")
    print(f"Did ES sweep Pre-Open Low ({pre_low_es})? -> {es_swept_pre} (Low: {manip_low_es})")

    if nq_swept_pre != es_swept_pre:
        print("\n*** BULLISH SMT DETECTED (Short Term)! ***")
        if nq_swept_pre and not es_swept_pre:
            print("Reason: NQ swept the Pre-Open Low, but ES held higher.")
        else:
            print("Reason: ES swept the Pre-Open Low, but NQ held higher.")
    else:
        print("\nNo SMT Divergence at Pre-Open Low.")

    # --- Step 6: Check ORG High (Opening Range Gap High / Old Range High?) ---
    # Not sure what "ORG High" means exactly from context, could be Previous Day High or Session High?
    # Or maybe "Opening Range Gap" from 9:30-9:35 candle?
    # Let's check the High of the 9:30 candle vs previous highs.
    # Often SMT at Highs means Bearish Divergence (Target), but SMT at Lows means Bullish Entry.

    print("\n[Highs Check (for context)]")
    london_high_nq = london_nq['High'].max()
    london_high_es = london_es['High'].max()
    print(f"NQ London High: {london_high_nq}")
    print(f"ES London High: {london_high_es}")

if __name__ == "__main__":
    verify_smt_divergence()
