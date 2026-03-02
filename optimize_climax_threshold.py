import pandas as pd
import numpy as np

BREADTH_FILE = 'market_breadth_history.csv'

TARGET_DATES = pd.to_datetime([
    '2016-01-04',
    '2016-01-11',
    '2018-12-17',
    '2020-02-24',
    '2020-03-02',
    '2020-03-09',
    '2022-09-19',
    '2025-03-31'
])

def load_breadth():
    df = pd.read_csv(BREADTH_FILE, parse_dates=['Date'], index_col='Date')
    df = df.sort_index()
    return df

def check_match(df_weekly, threshold, method_name):
    # Climax Days: Ratio > Threshold
    climax_days = df_weekly[df_weekly['New_Lows_Ratio'] > threshold].index

    # Check overlap
    matched_targets = []
    false_positives = []

    # We allow +/- 1 week tolerance?
    # Or exact match?
    # The user provided specific dates. Let's aim for exact match first (same week).

    target_set = set(TARGET_DATES.date)
    found_set = set(climax_days.date)

    matches = target_set.intersection(found_set)
    fps = found_set - target_set
    fns = target_set - found_set

    score = len(matches) - (len(fps) * 0.5) # Penalize FP less severely? Or heavily?
    # If the user listed ALL climaxes they care about, then FPs are bad.
    # If the user listed SOME examples, FPs might be okay.
    # Usually "detect only these" implies minimizing FPs.

    print(f"--- Method: {method_name}, Threshold: {threshold:.1f}% ---")
    print(f"Matches ({len(matches)}/{len(TARGET_DATES)}): {sorted([d.strftime('%Y-%m-%d') for d in matches])}")
    print(f"False Positives ({len(fps)}): {sorted([d.strftime('%Y-%m-%d') for d in fps])[:5]}...")
    print(f"Missed ({len(fns)}): {sorted([d.strftime('%Y-%m-%d') for d in fns])}")
    return len(matches), len(fps)

def main():
    df = load_breadth()

    # Aggregation Methods
    # 1. Resample to W-MON (Start of week)
    # Aggregations: 'mean', 'max', 'last'

    methods = ['mean', 'max']
    thresholds = np.arange(10.0, 40.0, 1.0) # 10% to 40% step 1%

    best_config = None
    best_score = -999

    for method in methods:
        # Resample
        # Note: resample('W-MON') labels with the next Monday usually? Or previous?
        # 'W-MON' with closed='left', label='left' -> Start of week date?
        # Default pandas resample 'W-MON' puts date at the END of the period.
        # User dates are Mondays. So maybe W-MON label='left'?
        # Let's try to match the user's date format.
        # 2016-01-04 is a Monday.

        # Resample Logic:
        # We want the week starting on Monday 2016-01-04 to be labeled 2016-01-04.
        resampled = df.resample('W-MON', closed='left', label='left')['New_Lows_Ratio'].agg(method)

        for th in thresholds:
            matches, fps = check_match(pd.DataFrame(resampled), th, method)
            score = matches - fps

            if score > best_score:
                best_score = score
                best_config = (method, th)

    print("\n==========================================")
    print(f"BEST CONFIGURATION: Method={best_config[0].upper()}, Threshold={best_config[1]:.1f}%")
    print("==========================================")

if __name__ == "__main__":
    main()
