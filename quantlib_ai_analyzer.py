import QuantLib as ql
import yfinance as yf
from curl_cffi import requests as cffi_requests
import pandas as pd
import numpy as np
import json
import math
from datetime import datetime, timedelta
import argparse

class QuantLibAnalyzer:
    def __init__(self, ticker):
        self.ticker = ticker
        self.risk_free_rate = 0.045
        self.day_count = ql.Actual365Fixed()
        self.calendar = ql.UnitedStates(ql.UnitedStates.NYSE)

    def fetch_data(self):
        """Fetch stock data using yfinance."""
        try:
            self.stock = yf.Ticker(self.ticker)
            self.hist = self.stock.history(period="1y")
            if self.hist.empty:
                return False

            self.current_price = self.hist['Close'].iloc[-1]
            self.current_date = self.hist.index[-1].date()
            self.evaluation_date = ql.Date(self.current_date.day, self.current_date.month, self.current_date.year)
            ql.Settings.instance().evaluationDate = self.evaluation_date

            # Dividend Yield
            self.dividend_yield = 0.0
            try:
                info = self.stock.info
                if 'dividendYield' in info and info['dividendYield'] is not None:
                    self.dividend_yield = info['dividendYield']
            except:
                pass

            return True
        except Exception as e:
            print(f"Error fetching data for {self.ticker}: {e}")
            return False

    def calculate_cycle_metrics(self):
        """Calculate Volatility Cycle and Trend."""
        self.hist['LogReturn'] = np.log(self.hist['Close'] / self.hist['Close'].shift(1))
        self.hist['HV_20d'] = self.hist['LogReturn'].rolling(window=20).std() * np.sqrt(252) * 100

        self.hv_current = self.hist['HV_20d'].iloc[-1]
        self.hv_max_1y = self.hist['HV_20d'].max()
        self.hv_min_1y = self.hist['HV_20d'].min()

        hv_range = self.hv_max_1y - self.hv_min_1y
        if self.hv_current < self.hv_min_1y + (hv_range * 0.2):
            self.cycle_phase = "Contraction (Base Forming)"
        elif self.hv_current > self.hv_max_1y * 0.8:
            self.cycle_phase = "Expansion (Climax/Panic)"
        else:
            hv_slope = self.hist['HV_20d'].iloc[-1] - self.hist['HV_20d'].iloc[-10]
            if hv_slope < 0:
                self.cycle_phase = "Transition (Cooling Down)"
            else:
                self.cycle_phase = "Transition (Heating Up)"

        sma50 = self.hist['Close'].rolling(window=50).mean().iloc[-1]
        sma200 = self.hist['Close'].rolling(window=200).mean().iloc[-1]

        if self.current_price > sma50 > sma200:
            self.trend = "Stage 2 (Uptrend)"
        elif self.current_price < sma50 < sma200:
            self.trend = "Stage 4 (Downtrend)"
        elif sma50 < sma200 and self.current_price > sma50:
            self.trend = "Stage 1 (Recovery)"
        else:
            self.trend = "Neutral/Choppy"

    def calculate_iv_and_gamma(self):
        """Calculate IV, Skew, and Gamma Exposure Profile."""
        self.iv_current = None
        self.gamma_data = {
            "Zero_Gamma_Level": None,
            "Max_Positive_GEX_Strike": None,
            "Max_Negative_GEX_Strike": None,
            "GEX_Profile": "N/A"
        }

        options_dates = self.stock.options
        if not options_dates:
            self.iv_current = self.hv_current / 100.0
            self.expiry_date = self.current_date + timedelta(days=30)
            self.iv_source = "HV Proxy"
            self.calculate_probabilities() # Calc probs using HV
            return

        target_expiry_str = options_dates[1] if len(options_dates) > 1 else options_dates[0]
        self.expiry_date = datetime.strptime(target_expiry_str, "%Y-%m-%d").date()

        chain = self.stock.option_chain(target_expiry_str)
        calls = chain.calls
        puts = chain.puts

        if calls.empty:
            self.iv_current = self.hv_current / 100.0
            self.iv_source = "HV Proxy"
            self.calculate_probabilities()
            return

        # 1. Determine IV (ATM)
        strikes = calls['strike'].values
        if len(strikes) > 0:
            atm_strike = min(strikes, key=lambda x: abs(x - self.current_price))
            atm_call = calls[calls['strike'] == atm_strike].iloc[0]
            self.iv_current = atm_call['impliedVolatility']
            self.iv_source = "Market"
            if self.iv_current < 0.01:
                self.iv_current = self.hv_current / 100.0
                self.iv_source = "HV Proxy (Market Data Invalid)"
        else:
            self.iv_current = self.hv_current / 100.0
            self.iv_source = "HV Proxy"

        # 2. Calculate Gamma Profile (GEX)
        if not puts.empty:
            self.calculate_gex(calls, puts)

        # 3. Calculate Probabilities using the determined IV
        self.calculate_probabilities()

    def calculate_gex(self, calls, puts):
        """Calculate Net Gamma Exposure per strike."""
        # Setup QuantLib Engine
        ql_expiry = ql.Date(self.expiry_date.day, self.expiry_date.month, self.expiry_date.year)
        spot_handle = ql.QuoteHandle(ql.SimpleQuote(self.current_price))
        rate_handle = ql.YieldTermStructureHandle(ql.FlatForward(self.evaluation_date, self.risk_free_rate, self.day_count))
        div_handle = ql.YieldTermStructureHandle(ql.FlatForward(self.evaluation_date, self.dividend_yield, self.day_count))
        vol_handle = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(self.evaluation_date, self.calendar, self.iv_current, self.day_count))

        bs_process = ql.BlackScholesMertonProcess(spot_handle, div_handle, rate_handle, vol_handle)
        engine = ql.AnalyticEuropeanEngine(bs_process)
        exercise = ql.EuropeanExercise(ql_expiry)

        gex_by_strike = {}
        all_strikes = sorted(list(set(calls['strike']).union(set(puts['strike']))))

        for k in all_strikes:
            # Get OI
            call_oi = calls[calls['strike'] == k]['openInterest'].sum()
            put_oi = puts[puts['strike'] == k]['openInterest'].sum()

            # Gamma Calc
            payoff = ql.PlainVanillaPayoff(ql.Option.Call, k)
            option = ql.VanillaOption(payoff, exercise)
            option.setPricingEngine(engine)
            try:
                gamma = option.gamma()
            except:
                gamma = 0

            net_gex = (put_oi * gamma) - (call_oi * gamma)
            gex_by_strike[k] = net_gex

        # Find Key Levels
        strikes = sorted(gex_by_strike.keys())
        gex_vals = [gex_by_strike[k] for k in strikes]

        if not gex_vals: return

        self.gamma_data["Max_Positive_GEX_Strike"] = strikes[np.argmax(gex_vals)] # Resistance/Pin
        self.gamma_data["Max_Negative_GEX_Strike"] = strikes[np.argmin(gex_vals)] # Acceleration Zone

        # Zero Gamma Flip
        # Find where sign changes
        flip_price = None
        current_idx = np.searchsorted(strikes, self.current_price)

        # Look left and right for nearest flip
        left_flip = None
        for i in range(current_idx-1, 0, -1):
             if (gex_vals[i] > 0 and gex_vals[i+1] < 0) or (gex_vals[i] < 0 and gex_vals[i+1] > 0):
                 left_flip = (strikes[i] + strikes[i+1]) / 2
                 break

        right_flip = None
        for i in range(current_idx, len(strikes)-1):
             if (gex_vals[i] > 0 and gex_vals[i+1] < 0) or (gex_vals[i] < 0 and gex_vals[i+1] > 0):
                 right_flip = (strikes[i] + strikes[i+1]) / 2
                 break

        if left_flip and right_flip:
            flip_price = left_flip if abs(self.current_price - left_flip) < abs(self.current_price - right_flip) else right_flip
        elif left_flip:
            flip_price = left_flip
        elif right_flip:
            flip_price = right_flip

        self.gamma_data["Zero_Gamma_Level"] = flip_price if flip_price else "None detected nearby"

        total_gex = sum(gex_vals)
        if total_gex > 0:
            self.gamma_data["GEX_Profile"] = "Positive (Stabilizing)"
        else:
            self.gamma_data["GEX_Profile"] = "Negative (Volatile/Accelerator)"

    def calculate_probabilities(self):
        # ... (Same as before, using self.iv_current)
        ql_expiry = ql.Date(self.expiry_date.day, self.expiry_date.month, self.expiry_date.year)
        spot_handle = ql.QuoteHandle(ql.SimpleQuote(self.current_price))
        rate_handle = ql.YieldTermStructureHandle(ql.FlatForward(self.evaluation_date, self.risk_free_rate, self.day_count))
        div_handle = ql.YieldTermStructureHandle(ql.FlatForward(self.evaluation_date, self.dividend_yield, self.day_count))
        vol_handle = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(self.evaluation_date, self.calendar, self.iv_current, self.day_count))

        bs_process = ql.BlackScholesMertonProcess(spot_handle, div_handle, rate_handle, vol_handle)
        engine = ql.AnalyticEuropeanEngine(bs_process)
        exercise = ql.EuropeanExercise(ql_expiry)

        target_up = self.current_price * 1.10
        target_down = self.current_price * 0.90

        bin_up = ql.VanillaOption(ql.CashOrNothingPayoff(ql.Option.Call, target_up, 1.0), exercise)
        bin_down = ql.VanillaOption(ql.CashOrNothingPayoff(ql.Option.Put, target_down, 1.0), exercise)

        bin_up.setPricingEngine(engine)
        bin_down.setPricingEngine(engine)

        time_to_expiry = self.day_count.yearFraction(self.evaluation_date, ql_expiry)
        disc = math.exp(-self.risk_free_rate * time_to_expiry)

        self.prob_up_10pct = bin_up.NPV() / disc
        self.prob_down_10pct = bin_down.NPV() / disc

    def run(self):
        if not self.fetch_data():
            return None
        self.calculate_cycle_metrics()
        self.calculate_iv_and_gamma() # Merged IV and Gamma calculation

        return {
            "Ticker": self.ticker,
            "Price": round(self.current_price, 2),
            "Volatility_Analysis": {
                "Cycle_Phase": self.cycle_phase,
                "HV_Current": round(self.hv_current, 2),
                "HV_1y_Max": round(self.hv_max_1y, 2),
                "HV_1y_Min": round(self.hv_min_1y, 2),
                "IV_Current": round(self.iv_current * 100, 2),
                "IV_Source": self.iv_source,
                "IV_Premium_Ratio": round((self.iv_current * 100) / self.hv_current, 2) if self.hv_current else 0
            },
            "Gamma_Analysis": self.gamma_data, # New Section
            "Probability_Analysis": {
                "Prob_Up_10pct": round(self.prob_up_10pct * 100, 2),
                "Prob_Down_10pct": round(self.prob_down_10pct * 100, 2),
                "Skew_Bias_Pts": round((self.prob_down_10pct - self.prob_up_10pct) * 100, 2)
            },
            "Technical_Context": {
                "Trend": self.trend,
                "Dividend_Yield": f"{self.dividend_yield*100:.2f}%"
            }
        }

def generate_analysis_prompt(json_data):
    return f"""
# Role
You are an elite **QuantLib Strategist** (Gamma/Volatility Expert) and **Technical Analyst**.
You possess the ability to see "Invisible Levels" (Gamma Flip, GEX Pins) that traditional chartists miss.

# Context
You are analyzing a JSON dataset containing advanced QuantLib metrics, including **Gamma Exposure (GEX)** profiles and **Volatility Cycles**.
Your mission is to provide a "Hybrid Trading Strategy" that identifies specific price levels where Dealer Hedging Flows will likely accelerate or stop the price.

# Input Data (JSON)
{json.dumps(json_data, indent=2)}

# Analysis Framework (The "4-Dimensional View")

## 1. Gamma Profile (The "Hidden Force")
- **Zero Gamma Level (Flip):** If price crosses this, volatility regime flips.
    - *Above Flip:* Dealer buys dips / sells rips (Stabilizing).
    - *Below Flip:* Dealer sells dips / buys rips (Acceleration/Crash Risk).
- **Max Positive GEX:** The "Magnet" or "Ceiling" where price tends to stick.
- **Max Negative GEX:** The "Danger Zone" where price movement accelerates.

## 2. Volatility Cycle (Time)
- **Transition/Contraction:** Safe for entry.
- **Expansion:** Dangerous, high risk of reversal.

## 3. Risk Perception (IV Premium)
- **Ratio < 1.0:** Smart money is calm.
- **Ratio > 1.2:** Smart money is buying protection (Fear).

## 4. Probability Skew (Direction)
- **High Downside Bias:** Mathematical headwind.

# Output Format (Japanese)

## 1. Executive Summary
Briefly summarize the market's "Gamma Environment" (Stable vs Volatile).

## 2. Deep Dive Strategy (Per Ticker)
**[Ticker Symbol]** - *[Gamma-Based Catchphrase]*
- **Gamma Levels (The "Invisible Lines"):**
    - **Flip Level:** $[Value] (Bullish above / Bearish below)
    - **Magnet/Wall:** $[Value] (Resistance/Pin)
    - **Acceleration Zone:** $[Value] (Breakout/Crash Trigger)
- **Trend & Cycle:**
    - [Cycle_Phase] / [Trend]
- **Quant Analysis:**
    - Explain how Gamma positioning interacts with the current trend. (e.g., "Price is approaching the Zero Gamma Flip; a breakout here triggers dealer buying.")
- **Actionable Strategy:**
    - **Trigger:** "Buy if price breaks $[Level]..."
    - **Logic:** "...because dealers will be forced to cover shorts (Gamma Squeeze)."

## 3. Comparative Matrix
| Ticker | Zero Gamma Flip | Max GEX Level | Strategy Focus |

## 4. Final Wisdom
A quote about the power of Gamma and Volatility.
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QuantLib AI Batch Analyzer")
    parser.add_argument("-t", "--tickers", nargs='+', type=str, help="List of tickers to analyze (space separated)")
    args = parser.parse_args()

    if args.tickers:
        tickers = [t.upper() for t in args.tickers]
    else:
        # Default list if no arguments provided
        tickers = ['NVDA', 'HL', 'HYMC', 'AXTI', 'MU', 'LITE', 'TTMI', 'CDE', 'APLD', 'ALM', 'IREN', 'ALTO']

    results = []

    print(f"--- Starting Hybrid Gamma Analysis for: {', '.join(tickers)} ---")
    for t in tickers:
        print(f"Processing {t}...")
        analyzer = QuantLibAnalyzer(t)
        data = analyzer.run()
        if data:
            results.append(data)

    with open("market_analysis_data.json", "w") as f:
        json.dump(results, f, indent=2)

    prompt = generate_analysis_prompt(results)

    with open("openai_prompt.txt", "w") as f:
        f.write(prompt)

    print("\n--- Generation Complete ---")
    print("Prompt with Gamma Logic saved to 'openai_prompt.txt'")
    print(prompt[:500] + "...")
