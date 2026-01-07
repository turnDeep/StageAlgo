import yfinance as yf
from curl_cffi import requests as curl_requests
import pandas as pd
import time
import random

class RDTDataFetcher:
    def __init__(self):
        self.session = curl_requests.Session(impersonate="chrome")

    def fetch_spy(self, period="2y"):
        """Fetches SPY data for RRS calculation."""
        try:
            # Using simple yf.download for SPY as it's a single liquid ticker
            # But stick to consistency with session
            obj = yf.Ticker("SPY", session=self.session)
            hist = obj.history(period=period)
            if hist.empty:
                print("Failed to fetch SPY data")
                return None
            return hist
        except Exception as e:
            print(f"Error fetching SPY: {e}")
            return None

    def fetch_batch(self, tickers, period="2y"):
        """
        Fetches data for a list of tickers.
        Returns a dictionary {ticker: dataframe}.
        Using yf.download for speed on batches.
        """
        if not tickers:
            return {}

        try:
            # yfinance download is faster for batches but rate limits can be tricky.
            # RDT verification needs robustness.
            # Using threads=True (default)
            data = yf.download(tickers, period=period, group_by='ticker', threads=True, progress=False, ignore_tz=True)

            result = {}
            if len(tickers) == 1:
                # If single ticker, structure is just columns.
                # Need to check if data is empty or valid.
                if not data.empty:
                    # yf.download for single ticker returns DataFrame with columns like 'Open', 'High'...
                    # For consistency with group_by='ticker' (MultiIndex), we might want to standardize?
                    # But fetch_batch is expected to return {ticker: df}.
                    result[tickers[0]] = data
            else:
                # MultiIndex columns: (Ticker, PriceType) -> Swap to (Ticker) -> DF
                # Actually group_by='ticker' makes the top level index the Ticker.
                for ticker in tickers:
                    try:
                        df = data[ticker]
                        # Check if valid (not all NaNs)
                        if not df.empty and not df.isnull().all().all():
                            result[ticker] = df
                    except KeyError:
                        pass

            return result
        except Exception as e:
            print(f"Batch fetch error: {e}")
            return {}

    def fetch_single(self, ticker, period="2y"):
        """Fetches single ticker using Session (more robust for individual detailed fetches if needed)."""
        try:
            obj = yf.Ticker(ticker, session=self.session)
            hist = obj.history(period=period)
            if hist.empty:
                return None
            return hist
        except Exception as e:
            # print(f"Error fetching {ticker}: {e}")
            return None
