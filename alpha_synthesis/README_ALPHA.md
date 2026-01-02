# Alpha-Synthesis System

## Overview
Alpha-Synthesis is a technical trading system logic designed to identify high-probability trade setups by integrating Market Regime, Fundamental/Momentum Selection, and Pattern Recognition.

## Components

### 1. Market Regime (`market_regime.py`)
- Checks the 10Y-2Y Treasury Yield Curve.
- Monitors High Yield Spreads (via FRED).
- Analyzes S&P 500 trend (vs 200 SMA).
- Monitors VIX levels.
- **Output:** Determines if the market is "Risk-ON" or "Risk-OFF".

### 2. Data Loader (`data_loader.py`)
- Uses `yfinance` combined with `curl_cffi` to impersonate a Chrome browser, reducing the risk of rate limiting or blocking.
- Implements random jitter (sleep) between requests.
- Fetches Price History and Financials.

### 3. Indicators (`indicators.py`)
- **Trend Template:** Minervini-style trend criteria (Price > 150 > 200 MA, etc.).
- **SMR Rating:** Simplified Sales, Margin, ROE scoring based on fundamentals. Graded A-E.
- **RS Score:** Relative Strength calculation based on price momentum.
- **Anchored VWAP (AVWAP):** Calculated from the highest high of the last 252 days using OHLC4.
- **VCP (Volatility Contraction Pattern):** Detects price tightness and volume dry-up.

### 4. Runner (`run_alpha.py`)
- Orchestrates the workflow.
- Loads `stock.csv`.
- Analyzes ALL tickers in the list using parallel processing.
- Runs the analysis and saves results to `alpha_synthesis_results_SMR_AB_YYYYMMDD.csv`.

## Usage
1. Ensure dependencies are installed:
   ```bash
   pip install -r requirements.txt
   # Ensure curl_cffi, pandas_datareader, scipy are included
   ```
2. Run the system:
   ```bash
   python3 alpha_synthesis/run_alpha.py
   ```

## Note on External Data
- The system attempts to fetch economic data from FRED via `pandas_datareader`. If this fails (due to API changes or network issues), it degrades gracefully and relies on Market Health (SPY) and Volatility (VIX) checks.
