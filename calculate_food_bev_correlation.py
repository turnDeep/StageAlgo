import yfinance as yf
import polars as pl
import pandas as pd
import warnings

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def main():
    # Define representative tickers for each sector
    # Food: General Mills, Kraft Heinz, McCormick, Campbell Soup, Conagra
    food_tickers = ['GIS', 'KHC', 'MKC', 'CPB', 'CAG']
    # Beverage: Coca-Cola, PepsiCo, Monster, Keurig Dr Pepper, Constellation Brands
    bev_tickers = ['KO', 'PEP', 'MNST', 'KDP', 'STZ']

    all_tickers = food_tickers + bev_tickers
    print(f"Fetching data for: {all_tickers}")

    # Download data (10 years)
    # auto_adjust=True ensures 'Close' is adjusted for splits/dividends
    try:
        pdf = yf.download(all_tickers, period="10y", auto_adjust=True, progress=False)
    except Exception as e:
        print(f"Error downloading data: {e}")
        return

    # yfinance returns a MultiIndex DataFrame if multiple tickers are requested.
    # Structure is usually Level 0: Price Type (Close, Open...), Level 1: Ticker
    # Or if only one price type is returned, it might be just Tickers.

    # Handle column structure
    if isinstance(pdf.columns, pd.MultiIndex):
        # Check if 'Close' is a level
        if 'Close' in pdf.columns.get_level_values(0):
            pdf = pdf['Close']
        elif 'Adj Close' in pdf.columns.get_level_values(0):
             pdf = pdf['Adj Close']

    # Ensure index is datetime and reset it to make it a column
    pdf = pdf.reset_index()

    # Convert to Polars
    try:
        df = pl.from_pandas(pdf)
    except Exception as e:
        print(f"Error converting to Polars: {e}")
        return

    # Clean column names (remove any residual object types or whitespace)
    df.columns = [str(c) for c in df.columns]

    print(f"Data shape: {df.shape}")
    print(f"Columns: {df.columns}")

    # Calculate Daily Returns
    # Formula: (Price_t / Price_t-1) - 1

    # Identify valid ticker columns (exclude Date)
    # Note: Column names in Polars match the pandas columns (tickers)
    valid_tickers = [c for c in df.columns if c != 'Date']

    # Calculate returns for each ticker
    return_exprs = [pl.col(t).pct_change().alias(f"{t}_ret") for t in valid_tickers]

    # Create a new DataFrame with just Date and Returns
    df_ret = df.select(['Date'] + return_exprs)

    # Filter columns that actually exist (in case some tickers failed)
    available_food = [f"{t}_ret" for t in food_tickers if f"{t}_ret" in df_ret.columns]
    available_bev = [f"{t}_ret" for t in bev_tickers if f"{t}_ret" in df_ret.columns]

    if not available_food or not available_bev:
        print("Error: Could not find data for one or both sectors.")
        return

    print(f"Using Food Tickers: {[c.replace('_ret','') for c in available_food]}")
    print(f"Using Beverage Tickers: {[c.replace('_ret','') for c in available_bev]}")

    # Aggregate Returns into Sector Indices (Equal Weighted Mean)
    # We calculate the row-wise mean of the returns for each sector
    df_sector = df_ret.with_columns([
        pl.concat_list(available_food).list.mean().alias("Food_Index_Ret"),
        pl.concat_list(available_bev).list.mean().alias("Bev_Index_Ret")
    ])

    # Drop nulls (e.g. first day of returns, or days where a sector had no data)
    # This ensures we correlate on valid overlapping days
    df_clean = df_sector.select(["Date", "Food_Index_Ret", "Bev_Index_Ret"]).drop_nulls()

    if df_clean.height < 10:
        print("Not enough overlapping data points to calculate correlation.")
        return

    # Calculate Pearson Correlation Coefficient
    correlation = df_clean.select(pl.corr("Food_Index_Ret", "Bev_Index_Ret")).item()

    print("\n" + "="*30)
    print("ANALYSIS RESULTS")
    print("="*30)
    print(f"Analysis Period: {df_clean['Date'].min()} to {df_clean['Date'].max()}")
    print(f"Number of overlapping trading days: {df_clean.height}")
    print("-" * 30)
    print(f"Correlation Coefficient (Daily Returns): {correlation:.4f}")
    print("="*30)

    # Interpretation
    if correlation > 0.7:
        strength = "Strong Positive"
    elif correlation > 0.3:
        strength = "Moderate Positive"
    elif correlation > -0.3:
        strength = "Weak/None"
    elif correlation > -0.7:
        strength = "Moderate Negative"
    else:
        strength = "Strong Negative"

    print(f"Interpretation: {strength} Correlation")

if __name__ == "__main__":
    main()
