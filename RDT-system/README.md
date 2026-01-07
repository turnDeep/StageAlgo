# RDT-System

This directory contains the screener and analysis tools for the **RDT (Risk/Reward/Trend)** system. The screener identifies stocks with high "Real Relative Strength" (RRS), strong institutional backing (fuel), and explosive potential.

## Components

### 1. `screener.py`
The main entry point for running the screener.
- Reads tickers from `../stock.csv`.
- Fetches daily data for each ticker and SPY (Benchmark).
- Calculates RDT indicators.
- Filters stocks based on strict criteria.
- Outputs results to `results/YYYYMMDD-rdt-screener.csv`.

**Usage:**
```bash
python3 RDT-system/screener.py
```

### 2. `indicators.py`
Contains the core mathematical logic for RDT indicators.

#### Key Indicators:
1.  **RRS (Real Relative Strength / VARS)**:
    -   Measures "True" strength by normalizing price performance with ATR (Volatility).
    -   Formula: `Excess_Move = Delta_Stock - (Delta_SPY * (Stock_ATR / SPY_ATR))`
    -   `RRS = Excess_Move / Stock_ATR`
    -   **Threshold**: > 1.0 (Stock outperformed its beta-adjusted expectation by more than 1 ATR).

2.  **RVol (Fuel / Relative Volume)**:
    -   Measures institutional participation.
    -   Formula: `Volume / SMA_Volume(20)`
    -   **Threshold**: > 1.5 (150% of average volume).

3.  **ADR% (Potential)**:
    -   Measures the average daily range as a percentage of price.
    -   Formula: `Mean(High/Low - 1) * 100` over 20 days.
    -   **Threshold**: > 4%.

4.  **Trend Structure**:
    -   Ensures the stock is in a confirmed uptrend.
    -   **Criteria**: Price > SMA(50) > SMA(100) > SMA(200).

5.  **Liquidity Filter**:
    -   Ensures tradability.
    -   **Criteria**: 10-day Average Volume > 1,000,000 shares.
    -   **Price**: > $5.

### 3. `data_fetcher.py`
Handles data retrieval using `yfinance` with robustness improvements (retries, rate limiting).
-   Fetches SPY data first for benchmark calculations.
-   Fetches stock data in batches for efficiency.

## Output
Results are saved in `RDT-system/results/`:
-   `YYYYMMDD-rdt-screener.csv`: Contains only stocks that passed ALL criteria.
-   `YYYYMMDD-rdt-all.csv`: Contains data for all processed tickers (useful for debugging).

## Note on Time-Based RVol
The current implementation runs on Daily (End-of-Day) data. In this context, "Time-Based RVol" converges to standard Daily Relative Volume. If intraday data were used, the logic would compare cumulative volume at time *t* vs historical average at *t*.
