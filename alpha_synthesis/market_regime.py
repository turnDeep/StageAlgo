import pandas_datareader.data as web
import datetime
import yfinance as yf
import pandas as pd

def check_macro_environment():
    """
    Checks the market regime.
    Returns True if Risk-ON (Stage 2 environment), False otherwise.
    """
    print("Checking Macro Environment...")
    is_risk_on = True # Default

    # 1. Yield Curve (10Y - 2Y) via FRED
    try:
        start = datetime.datetime.now() - datetime.timedelta(days=365*2)
        # T10Y2Y: 10-Year Treasury Constant Maturity Minus 2-Year Treasury Constant Maturity
        # BAMLC0A0CM: ICE BofA US High Yield Index Option-Adjusted Spread
        # Note: pandas_datareader might fail if FRED API is down or changed.
        try:
            macro_data = web.DataReader(['T10Y2Y', 'BAMLC0A0CM'], 'fred', start)

            if not macro_data.empty:
                latest_yield_curve = macro_data['T10Y2Y'].iloc[-1]
                high_yield_spread = macro_data['BAMLC0A0CM'].iloc[-1]

                print(f"  Yield Curve (10Y-2Y): {latest_yield_curve:.2f}")
                print(f"  High Yield Spread: {high_yield_spread:.2f}")

                # Simple logic: If spread is blowing out > 5.0, risk off.
                if high_yield_spread > 5.0:
                    print("  [Warning] High Yield Spread > 5.0. Risk OFF.")
                    is_risk_on = False
            else:
                print("  [Info] FRED data empty.")

        except Exception as fred_err:
             print(f"  [Info] FRED fetch failed (likely connection/API issue): {fred_err}")
             # Proceeding without FRED data

    except Exception as e:
        print(f"  [Error] General error in FRED check: {e}")

    # 2. Market Health (SPY > 200MA) via yfinance
    try:
        spy = yf.Ticker("SPY")
        hist = spy.history(period="1y")
        if not hist.empty:
            sma_200 = hist['Close'].rolling(window=200).mean().iloc[-1]
            current_price = hist['Close'].iloc[-1]

            print(f"  SPY Price: {current_price:.2f}, 200 SMA: {sma_200:.2f}")

            if current_price < sma_200:
                print("  [Warning] SPY below 200 SMA. Risk OFF.")
                is_risk_on = False
    except Exception as e:
        print(f"  [Error] Failed to fetch SPY data: {e}")

    # 3. VIX Check via yfinance
    try:
        vix = yf.Ticker("^VIX")
        hist_vix = vix.history(period="1y")
        if not hist_vix.empty:
            vix_current = hist_vix['Close'].iloc[-1]
            vix_ma = hist_vix['Close'].rolling(window=50).mean().iloc[-1]
            print(f"  VIX: {vix_current:.2f}, VIX 50MA: {vix_ma:.2f}")

            # If VIX is significantly elevated above MA (e.g. > 1.5x) or absolute high level > 30
            if vix_current > 30 or (vix_ma > 0 and vix_current > (vix_ma * 1.5)):
                 print("  [Warning] VIX is elevated. Risk OFF.")
                 is_risk_on = False
    except Exception as e:
         print(f"  [Error] Failed to fetch VIX data: {e}")

    return is_risk_on
