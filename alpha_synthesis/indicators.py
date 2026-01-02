import pandas as pd
import numpy as np

def check_trend_template(df):
    """
    Checks Minervini Trend Template criteria.
    """
    if len(df) < 260: # Need ~1 year of data
        return False

    current_close = df['Close'].iloc[-1]
    sma_50 = df['Close'].rolling(window=50).mean().iloc[-1]
    sma_150 = df['Close'].rolling(window=150).mean().iloc[-1]
    sma_200 = df['Close'].rolling(window=200).mean().iloc[-1]

    # 200 SMA trending up (compare to 1 month ago ~ 20 trading days)
    sma_200_1m_ago = df['Close'].rolling(window=200).mean().iloc[-21]

    low_52w = df['Low'].tail(252).min()
    high_52w = df['High'].tail(252).max()

    # Conditions
    # 1. Price > 150MA > 200MA
    c1 = current_close > sma_150 and sma_150 > sma_200
    # 2. 200MA trending up
    c2 = sma_200 > sma_200_1m_ago
    # 3. Price > 50MA
    c3 = current_close > sma_50
    # 4. 30% above 52 week low
    c4 = current_close >= (low_52w * 1.30)
    # 5. Within 25% of 52 week high
    c5 = current_close >= (high_52w * 0.75)

    return c1 and c2 and c3 and c4 and c5

def calculate_smr_rating_value(financials):
    """
    Calculates SMR Rating (0-100) based on approximate logic.
    Sales Growth, Profit Margin, ROE.
    Returns None if data insufficient.
    """
    if financials is None or financials.empty:
        return None

    try:
        # yfinance financials format: columns are dates, rows are metrics
        # e.g., 'Total Revenue', 'Net Income', 'Stockholders Equity'
        # Check rows existence

        row_map = {k.lower(): k for k in financials.index}

        rev_key = row_map.get('total revenue')
        ni_key = row_map.get('net income')
        equity_key = row_map.get('stockholders equity') or row_map.get('total stockholders equity')

        if not rev_key or not ni_key:
            return 50 # Default if keys missing

        # Need at least current period
        latest_date = financials.columns[0]

        # Sales Growth
        if len(financials.columns) < 2:
            sales_growth = 0
        else:
            prev_date = financials.columns[1]
            rev_curr = financials.loc[rev_key, latest_date]
            rev_prev = financials.loc[rev_key, prev_date]
            sales_growth = (rev_curr - rev_prev) / abs(rev_prev) if rev_prev != 0 else 0

        # Profit Margin
        rev_curr = financials.loc[rev_key, latest_date]
        net_income = financials.loc[ni_key, latest_date]
        margin = net_income / rev_curr if rev_curr != 0 else 0

        # ROE
        equity = 1
        if equity_key:
            equity = financials.loc[equity_key, latest_date]
        roe = net_income / equity if equity != 0 else 0

        # Scoring (Arbitrary simplified model based on description)
        # Sales Growth (40%), Margin (30%), ROE (30%)
        # Normalize: Growth > 20% = 100, Margin > 20% = 100, ROE > 15% = 100

        s_score = min(max(sales_growth / 0.20, 0), 1) * 100
        m_score = min(max(margin / 0.20, 0), 1) * 100
        r_score = min(max(roe / 0.15, 0), 1) * 100

        final_score = (s_score * 0.4) + (m_score * 0.3) + (r_score * 0.3)
        return final_score

    except Exception:
        return None

def get_smr_rating_grade(score):
    """
    Maps 0-100 score to SMR Rating A-E.
    """
    if score is None:
        return 'E'
    if score >= 80:
        return 'A'
    elif score >= 60:
        return 'B'
    elif score >= 40:
        return 'C'
    elif score >= 20:
        return 'D'
    else:
        return 'E'

def calculate_rs(df):
    """
    Calculates RS Score (Relative Strength) using price performance.
    Formula: 40% weight on 3-month perf.
    We'll use a simplified version: (C - C_63)/C_63 * 0.4 + ...
    """
    try:
        # 3 month (63 days), 6 month (126 days), 12 month (252 days)
        p_now = df['Close'].iloc[-1]
        p_3m = df['Close'].iloc[-63] if len(df) >= 63 else df['Close'].iloc[0]
        p_6m = df['Close'].iloc[-126] if len(df) >= 126 else df['Close'].iloc[0]
        p_12m = df['Close'].iloc[-252] if len(df) >= 252 else df['Close'].iloc[0]

        roc_3m = (p_now - p_3m) / p_3m
        roc_6m = (p_now - p_6m) / p_6m
        roc_12m = (p_now - p_12m) / p_12m

        # 40% weight on 3m. 20% on 6m, 20% on 12m?
        raw_rs = (roc_3m * 0.4) + (roc_6m * 0.3) + (roc_12m * 0.3)

        # Normalize roughly to 0-100 scale?
        # A good stock doubles (100%), so let's just multiply by 100 for now.
        return raw_rs * 100
    except Exception:
        return 0

def calculate_anchored_vwap(df):
    """
    Calculates Anchored VWAP from the highest high of the last 252 days.
    Uses OHLC4.
    Returns the latest AVWAP value.
    """
    try:
        lookback = 252
        if len(df) < lookback:
            lookback = len(df)

        # Find index of highest high in lookback
        high_idx = df['High'].tail(lookback).idxmax()

        # Slice from anchor
        df_anchor = df.loc[high_idx:].copy()

        if df_anchor.empty:
            return None

        # OHLC4
        ohlc4 = (df_anchor['Open'] + df_anchor['High'] + df_anchor['Low'] + df_anchor['Close']) / 4

        # VWAP = Cumulative(Price * Volume) / Cumulative(Volume)
        pv = ohlc4 * df_anchor['Volume']
        cum_pv = pv.cumsum()
        cum_vol = df_anchor['Volume'].cumsum()

        avwap = cum_pv / cum_vol
        return avwap.iloc[-1]

    except Exception:
        return None

def check_vcp(df):
    """
    Checks for VCP characteristics.
    1. Volatility contraction (Depth decreases).
    2. Volume dry up.
    Returns (True/False, details_dict).
    """
    try:
        # Use last 100 days for pattern
        df_sub = df.tail(100).copy()
        if len(df_sub) < 50:
            return False, {}

        # 1. Tightness Check
        # Weekly range is tight? Or daily average range is low?
        high_low_range_pct = (df_sub['High'] - df_sub['Low']) / df_sub['Close']

        # Rolling 10 day average of daily range
        recent_volatility = high_low_range_pct.rolling(window=10).mean().iloc[-1]

        # Threshold: < 4% average daily movement
        is_tight = recent_volatility < 0.04

        # 2. Volume Dry Up Check
        vol_sma_50 = df_sub['Volume'].rolling(50).mean().iloc[-1]
        curr_vol = df_sub['Volume'].iloc[-1]

        # Threshold: < 70% of 50SMA
        is_dry_up = curr_vol < (vol_sma_50 * 0.7)

        return (is_tight and is_dry_up), {
            "tightness": is_tight,
            "dry_up": is_dry_up,
            "volatility": round(recent_volatility, 3)
        }

    except Exception as e:
        return False, {"error": str(e)}
